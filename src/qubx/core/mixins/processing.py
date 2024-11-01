from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import FunctionType


from threading import Thread
from multiprocessing.pool import ThreadPool
import traceback

import pandas as pd

from qubx import lookup, logger
from qubx.core.account import AccountProcessor
from qubx.core.helpers import BasicScheduler, CachedMarketDataHolder, process_schedule_spec, set_parameters_to_object
from qubx.core.loggers import LogsWriter, StrategyLogging
from qubx.core.basics import (
    TargetPosition,
    TriggerEvent,
    Deal,
    Instrument,
    Order,
    Position,
    Signal,
    dt_64,
    td_64,
    CtrlChannel,
    BatchEvent,
    SW,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.interfaces import (
    IBrokerServiceProvider,
    IPositionGathering,
    IStrategy,
    ITradingServiceProvider,
    PositionsTracker,
    IStrategyContext,
    SubscriptionType,
    IProcessingManager,
    ITimeProvider,
    IUniverseManager,
)
from qubx.core.series import Trade, Quote, Bar, OHLCV, OrderBook
from qubx.data.readers import DataReader
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.trackers.sizers import FixedSizer
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_seconds_to_str


class ProcessingManager(IProcessingManager, IStrategyContext):
    MAX_NUMBER_OF_STRATEGY_FAILURES = 10

    __strategy: IStrategy
    __initial_instruments: list[Instrument]
    __logging: StrategyLogging
    __broker: IBrokerServiceProvider
    __universe_manager: IUniverseManager
    __time_provider: ITimeProvider
    __position_tracker: PositionsTracker
    __position_gathering: IPositionGathering
    __cache: CachedMarketDataHolder
    __scheduler: BasicScheduler

    __handlers: dict[str, Callable[["ProcessingManager", Instrument | str, Any], TriggerEvent | None]]
    __strategy_name: str

    __fit_is_running: bool = False
    __init_fit_was_called: bool = False
    __init_fit_args: tuple[dt_64 | None, dt_64 | None]
    __fails_counter: int = 0
    __is_simulation: bool
    __pool: ThreadPool | None

    # TODO: refactor
    _trig_interval_in_bar_nsec: int
    _trig_bar_freq_nsec: int
    _current_bar_trigger_processed: bool = False

    def __init__(
        self,
        strategy: IStrategy,
        instruments: list[Instrument],
        logging: StrategyLogging,
        broker: IBrokerServiceProvider,
        universe_manager: IUniverseManager,
        time_provider: ITimeProvider,
        position_tracker: PositionsTracker,
        position_gathering: IPositionGathering,
        cache: CachedMarketDataHolder,
    ):
        self.__strategy = strategy
        self.__initial_instruments = instruments
        self.__logging = logging
        self.__broker = broker
        self.__universe_manager = universe_manager
        self.__time_provider = time_provider
        self.__is_simulation = broker.is_simulated_trading
        self.__position_gathering = position_gathering
        self.__position_tracker = position_tracker
        self.__cache = cache
        self.__scheduler = broker.get_scheduler()

        self.__pool = ThreadPool(2) if not self.__is_simulation else None
        self.__handlers = {
            n.split("__handle_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) == FunctionType and n.startswith("__handle_")
        }
        self.__strategy_name = strategy.__class__.__name__
        self.__init_fit_args = (None, self.time())

    def set_fit_schedule(self, schedule: str) -> None:
        rule = process_schedule_spec(schedule)
        if rule["type"] != "cron":
            raise ValueError("Only cron type is supported for fit schedule")
        self.__scheduler.schedule_event(rule["schedule"], "fit")

    def set_event_schedule(self, schedule: str) -> None:
        rule = process_schedule_spec(schedule)
        if rule["type"] != "cron":
            raise ValueError("Only cron type is supported for event schedule")
        self.__scheduler.schedule_event(rule["schedule"], "time")

    def process_data(self, symbol: str, d_type: str, data: Any) -> bool:
        handler = self.__handlers.get(d_type)
        with SW("StrategyContext.handler"):
            _strategy_trigger_on_event = handler(self, symbol, data) if handler else None

        # - check if it still didn't call on_fit() for first time
        if not self.__init_fit_was_called:
            self.__process_fit(None, self.__init_fit_args)

        if _strategy_trigger_on_event:
            # - if fit was not called - skip on_event call
            if not self.__init_fit_was_called:
                logger.debug(
                    f"{self.__strategy_name}::on_event() is SKIPPED for now because on_fit() was not called yet!"
                )
                return False

            # - if strategy still fitting - skip on_event call
            if self.__fit_is_running:
                logger.warning(f"{self.__strategy_name}::on_event() is SKIPPED for now because is being still fitting!")
                return False

            signals: List[Signal] | Signal | None = None
            with SW("StrategyContext.on_event"):
                try:
                    signals = self.__strategy.on_event(self, _strategy_trigger_on_event)
                    self._fails_counter = 0
                except Exception as strat_error:
                    # - probably we need some cooldown interval after exception to prevent flooding
                    logger.error(f"Strategy {self.__strategy_name} raised an exception: {strat_error}")
                    logger.opt(colors=False).error(traceback.format_exc())

                    #  - we stop execution after let's say maximal number of errors in a row
                    self.__fails_counter += 1
                    if self.__fails_counter >= self.MAX_NUMBER_OF_STRATEGY_FAILURES:
                        logger.error("STRATEGY FAILURES IN THE ROW EXCEEDED MAX ALLOWED NUMBER - STOPPING ...")
                        return True

            # - process and execute signals if they are provided
            if signals:
                # process signals by tracker and turn convert them into positions
                positions_from_strategy = self.__process_and_log_target_positions(
                    self.__position_tracker.process_signals(self, self.__process_signals(signals))
                )

                # gathering in charge of positions
                self.__position_gathering.alter_positions(self, positions_from_strategy)

        # - notify poition and portfolio loggers
        self.__logging.notify(self.time())

        return False

    @SW.watch("StrategyContext.on_fit")
    def __invoke_on_fit(self, current_fit_time: dt_64, prev_fit_time: dt_64 | None) -> None:
        try:
            logger.debug(
                f"Invoking <green>{self.__strategy.__class__.__name__}</green> on_fit('{current_fit_time}', '{prev_fit_time}')"
            )
            self.__strategy.on_fit(self, current_fit_time, prev_fit_time)
            logger.debug(f"<green>{self.__strategy.__class__.__name__}</green> is fitted")
        except Exception as strat_error:
            logger.error(
                f"[{self.time()}]: Strategy {self.__strategy.__class__.__name__} "
                f"on_fit('{current_fit_time}', '{prev_fit_time}') raised an exception: {strat_error}"
            )
            logger.opt(colors=False).error(traceback.format_exc())
        finally:
            self.__fit_is_running = False
            self.__init_fit_was_called = True

    def __process_and_log_target_positions(
        self, target_positions: List[TargetPosition] | TargetPosition | None
    ) -> list[TargetPosition]:
        if target_positions is None:
            return []
        if isinstance(target_positions, TargetPosition):
            target_positions = [target_positions]
        self.__logging.save_signals_targets(target_positions)
        return target_positions

    def __process_signals_from_target_positions(
        self, target_positions: list[TargetPosition] | TargetPosition | None
    ) -> None:
        if target_positions is None:
            return
        if isinstance(target_positions, TargetPosition):
            target_positions = [target_positions]
        signals = [pos.signal for pos in target_positions]
        self.__process_signals(signals)

    def __process_signals(self, signals: list[Signal] | Signal | None) -> List[Signal]:
        if isinstance(signals, Signal):
            signals = [signals]
        elif signals is None:
            return []

        for signal in signals:
            # set strategy group name if not set
            if not signal.group:
                signal.group = self.__strategy_name

            # set reference prices for signals
            if signal.reference_price is None:
                q = self.quote(signal.instrument)
                if q is None:
                    continue
                signal.reference_price = q.mid_price()

        return signals

    def _run_in_thread_pool(self, func: Callable, args=()):
        # For simulation we don't need to call function in thread
        if self.__is_simulation:
            func(*args)
        else:
            assert self.__pool
            self.__pool.apply_async(func, args)

    ###########################################################################
    # - Handlers for different types of incoming data
    ###########################################################################
    def __handle_fit(self, _: Instrument | None, data: Any) -> None:
        """
        When scheduled fit event is happened - we need to invoke strategy on_fit method
        """
        if not self.__cache.is_data_ready():
            return

        # times are in seconds here
        prev_fit_time, now_fit_time = data

        # - we need to run this in separate thread
        self.__fit_is_running = True
        self._run_in_thread_pool(
            self.__invoke_on_fit,
            (dt_64(now_fit_time, "s"), dt_64(prev_fit_time, "s") if prev_fit_time else None),
        )

    def __handle_hist_bars(self, instrument: Instrument, bars: list[Bar]) -> None:
        for b in bars:
            self.__handle_hist_bar(instrument, b)

    def __handle_hist_bar(self, instrument: Instrument | None, bar: Bar) -> None:
        """
        Handles the processing of a single historical bar.

        This method updates the cache with the provided historical bar data.
        Note that a historical bar cannot trigger strategy logic.

        Args:
            symbol (str): The symbol associated with the bar.
            bar (Bar): The historical bar data to be processed.

        Returns:
            TriggerEvent | None: Always returns None as historical bars do not trigger events.
        """
        assert instrument is not None
        self.__cache.update_by_bar(instrument, bar)
        return None

    def __handle_hist_quote(self, instrument: Instrument, quote: Quote | BatchEvent) -> None:
        if isinstance(quote, BatchEvent):
            for q in quote.data:
                self.__cache.update_by_quote(instrument, q)
        else:
            self.__cache.update_by_quote(instrument, quote)

    def __handle_hist_trade(self, instrument: Instrument, trade: Trade | BatchEvent) -> None:
        if isinstance(trade, BatchEvent):
            for t in trade.data:
                self.__cache.update_by_trade(instrument, t)
        else:
            self.__cache.update_by_trade(instrument, trade)

    def __handle_bar(self, instrument: Instrument, bar: Bar) -> TriggerEvent | None:
        # - processing current bar's update
        self._cache.update_by_bar(symbol, bar)

        # - update tracker and handle alterd positions if need
        self.positions_gathering.alter_positions(
            self,
            self.__process_and_log_target_positions(
                self.positions_tracker.update(self, self._symb_to_instr[symbol], bar)
            ),
        )

        # - check if it's time to trigger the on_event if it's configured
        return self._check_if_need_trigger_on_bar(symbol, bar)

    def _check_if_need_trigger_on_bar(self, symbol: str, bar: Bar | None):
        if self._trig_on_bar:
            t = self.trading_service.time().item()
            _time_to_trigger = t % self._trig_bar_freq_nsec >= self._trig_interval_in_bar_nsec
            if _time_to_trigger:
                # we want to trigger only first one - not every
                if not self._current_bar_trigger_processed:
                    self._current_bar_trigger_processed = True
                    if bar is None:
                        bar = self._cache.get_ohlcv(symbol)[0]
                    return TriggerEvent(self.time(), "bar", self._symb_to_instr.get(symbol), bar)
            else:
                self._current_bar_trigger_processed = False
        return None

    def _processing_trade(self, symbol: str, trade: Trade | BatchEvent) -> TriggerEvent | None:
        is_batch_event = isinstance(trade, BatchEvent)
        if is_batch_event:
            for t in trade.data:
                self._cache.update_by_trade(symbol, t)
        else:
            self._cache.update_by_trade(symbol, trade)

        target_positions = self.positions_tracker.update(
            self, self._symb_to_instr[symbol], trade.data[-1] if is_batch_event else trade
        )
        self.__process_signals_from_target_positions(target_positions)

        # - update tracker and handle alterd positions if need
        self.positions_gathering.alter_positions(
            self,
            self.__process_and_log_target_positions(target_positions),
        )

        event_type = "trade" if not is_batch_event else "batch:trade"
        return TriggerEvent(self.time(), event_type, self._symb_to_instr.get(symbol), trade)

    def _processing_orderbook(self, symbol: str, orderbook: OrderBook) -> TriggerEvent | None:
        quote = orderbook.to_quote()
        self._cache.update_by_quote(symbol, quote)
        target_positions = self.positions_tracker.update(self, self._symb_to_instr[symbol], quote)
        self.__process_signals_from_target_positions(target_positions)
        self.positions_gathering.alter_positions(
            self,
            self.__process_and_log_target_positions(target_positions),
        )
        return TriggerEvent(self.time(), "orderbook", self._symb_to_instr.get(symbol), orderbook)

    def _processing_quote(self, symbol: str, quote: Quote) -> TriggerEvent | None:
        self._cache.update_by_quote(symbol, quote)

        target_positions = self.positions_tracker.update(self, self._symb_to_instr[symbol], quote)
        self.__process_signals_from_target_positions(target_positions)

        # - update tracker and handle alterd positions if need
        self.positions_gathering.alter_positions(self, self.__process_and_log_target_positions(target_positions))

        # - TODO: here we can apply throttlings or filters
        #  - let's say we can skip quotes if bid & ask is not changed
        #  - or we can collect let's say N quotes before sending to strategy
        if self._trig_on_quote:
            return TriggerEvent(self.time(), "quote", self._symb_to_instr.get(symbol), quote)

        return self._check_if_need_trigger_on_bar(symbol, None)
        # return None

    @_SW.watch("StrategyContext")
    def _processing_order(self, symbol: str, order: Order) -> TriggerEvent | None:
        logger.debug(
            f"[<red>{order.id}</red> / {order.client_id}] : {order.type} {order.side} {order.quantity} of {symbol} { (' @ ' + str(order.price)) if order.price else '' } -> [{order.status}]"
        )
        # - check if we want to trigger any strat's logic on order
        return None

    def _processing_event(self, symbol: str, event_data: Dict) -> TriggerEvent | None:
        """
        Processing external events
        """
        return TriggerEvent(self.time(), "event", self._symb_to_instr.get(symbol), event_data)

    @_SW.watch("StrategyContext")
    def _processing_deals(self, symbol: str, deals: List[Deal]) -> TriggerEvent | None:
        # - log deals in storage
        instr = self._symb_to_instr.get(symbol)
        self._logging.save_deals(symbol, deals)
        if instr is not None:
            for d in deals:
                # - notify position gatherer and tracker
                self.positions_gathering.on_execution_report(self, instr, d)
                self.positions_tracker.on_execution_report(self, instr, d)
                logger.debug(f"Executed {d.amount} @ {d.price} of {symbol} for order <red>{d.order_id}</red>")
        else:
            logger.debug(f"Execution report for unknown instrument {symbol}")
        return None
