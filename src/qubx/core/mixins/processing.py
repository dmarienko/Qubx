import traceback
from multiprocessing.pool import ThreadPool
from types import FunctionType
from typing import Any, Callable, List, Tuple

import pandas as pd

from qubx import logger
from qubx.core.basics import (
    SW,
    DataType,
    Deal,
    Instrument,
    MarketEvent,
    Order,
    Signal,
    TargetPosition,
    TriggerEvent,
    dt_64,
)
from qubx.core.helpers import BasicScheduler, CachedMarketDataHolder, extract_price, process_schedule_spec
from qubx.core.interfaces import (
    IAccountProcessor,
    IMarketManager,
    IPositionGathering,
    IProcessingManager,
    IStrategy,
    IStrategyContext,
    ISubscriptionManager,
    ITimeProvider,
    PositionsTracker,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.series import Bar, OrderBook, Quote, Trade


class ProcessingManager(IProcessingManager):
    MAX_NUMBER_OF_STRATEGY_FAILURES = 10

    _context: IStrategyContext
    _strategy: IStrategy
    _logging: StrategyLogging
    _market_data: IMarketManager
    _subscription_manager: ISubscriptionManager
    _time_provider: ITimeProvider
    _account: IAccountProcessor
    _position_tracker: PositionsTracker
    _position_gathering: IPositionGathering
    _cache: CachedMarketDataHolder
    _scheduler: BasicScheduler

    _handlers: dict[str, Callable[["ProcessingManager", Instrument, str, Any], TriggerEvent | None]]
    _strategy_name: str

    _trigger_on_time_event: bool = False
    _fit_is_running: bool = False
    _init_fit_was_called: bool = False
    _fails_counter: int = 0
    _is_simulation: bool
    _pool: ThreadPool | None
    _trig_bar_freq_nsec: int | None = None
    _cur_sim_step: int | None = None

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        logging: StrategyLogging,
        market_data: IMarketManager,
        subscription_manager: ISubscriptionManager,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        position_tracker: PositionsTracker,
        position_gathering: IPositionGathering,
        cache: CachedMarketDataHolder,
        scheduler: BasicScheduler,
        is_simulation: bool,
    ):
        self._context = context
        self._strategy = strategy
        self._logging = logging
        self._market_data = market_data
        self._subscription_manager = subscription_manager
        self._time_provider = time_provider
        self._account = account
        self._is_simulation = is_simulation
        self._position_gathering = position_gathering
        self._position_tracker = position_tracker
        self._cache = cache
        self._scheduler = scheduler

        self._pool = ThreadPool(2) if not self._is_simulation else None
        self._handlers = {
            n.split("_handle_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) is FunctionType and n.startswith("_handle_")
        }
        self._strategy_name = strategy.__class__.__name__
        self._trig_bar_freq_nsec = None

    def set_fit_schedule(self, schedule: str) -> None:
        rule = process_schedule_spec(schedule)
        if rule["type"] != "cron":
            raise ValueError("Only cron type is supported for fit schedule")
        self._scheduler.schedule_event(rule["schedule"], "fit")

    def set_event_schedule(self, schedule: str) -> None:
        rule = process_schedule_spec(schedule)
        if not rule or "type" not in rule:
            raise ValueError(f"Can't recognoize schedule format: '{schedule}'")

        if rule["type"] != "cron":
            raise ValueError("Only cron type is supported for event schedule")

        self._scheduler.schedule_event(rule["schedule"], "time")
        self._trigger_on_time_event = True

    def get_event_schedule(self, event_id: str) -> str | None:
        return self._scheduler.get_schedule_for_event(event_id)

    def process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool) -> bool:
        self._logging.notify(self._time_provider.time())

        handler = self._handlers.get(d_type)
        with SW("StrategyContext.handler"):
            if not d_type:
                event = None
            elif is_historical:
                event = self._process_hist_event(instrument, d_type, data)
            elif handler:
                event = handler(self, instrument, d_type, data)
            else:
                event = self._process_custom_event(instrument, d_type, data)

        # - check if it still didn't call on_fit() for first time
        if not self._init_fit_was_called and not self._fit_is_running:
            self._handle_fit(None, "fit", (None, self._time_provider.time()))
            return False

        if not event:
            return False

        # - if fit was not called - skip on_event call
        if not self._init_fit_was_called:
            logger.debug(f"{self._strategy_name}::on_event() is SKIPPED for now because on_fit() was not called yet!")
            return False

        # - if strategy still fitting - skip on_event call
        if self._fit_is_running:
            logger.warning(f"{self._strategy_name}::on_event() is SKIPPED for now because is being still fitting!")
            return False

        signals: list[Signal] | Signal | None = None
        with SW("StrategyContext.on_event"):
            try:
                if isinstance(event, MarketEvent):
                    signals = self._wrap_signal_list(self._strategy.on_market_data(self._context, event))

                if signals is None:
                    signals = []

                if isinstance(event, TriggerEvent) or (isinstance(event, MarketEvent) and event.is_trigger):
                    _trigger_event = event.to_trigger() if isinstance(event, MarketEvent) else event
                    _signals = self._wrap_signal_list(self._strategy.on_event(self._context, _trigger_event))
                    signals.extend(_signals)

                if isinstance(event, Order):
                    _signals = self._wrap_signal_list(self._strategy.on_order_update(self._context, event))
                    signals.extend(_signals)

                self._subscription_manager.commit()  # apply pending operations

                self._fails_counter = 0
            except Exception as strat_error:
                # - probably we need some cooldown interval after exception to prevent flooding
                logger.error(f"Strategy {self._strategy_name} raised an exception: {strat_error}")
                logger.opt(colors=False).error(traceback.format_exc())

                #  - we stop execution after let's say maximal number of errors in a row
                self._fails_counter += 1
                if self._fails_counter >= self.MAX_NUMBER_OF_STRATEGY_FAILURES:
                    logger.error("STRATEGY FAILURES IN THE ROW EXCEEDED MAX ALLOWED NUMBER - STOPPING ...")
                    return True

        # - process and execute signals if they are provided
        if signals:
            # fmt: off
            positions_from_strategy = self.__process_and_log_target_positions(
                self._position_tracker.process_signals(
                    self._context,
                    self.__process_signals(signals)
                )
            )
            self._position_gathering.alter_positions(self._context, positions_from_strategy)
            # fmt: on

        # - notify poition and portfolio loggers
        self._logging.notify(self._time_provider.time())

        return False

    def is_fitted(self) -> bool:
        return self._init_fit_was_called

    @SW.watch("StrategyContext.on_fit")
    def __invoke_on_fit(self) -> None:
        try:
            logger.debug(f"Invoking <green>{self._strategy_name}</green> on_fit")
            self._strategy.on_fit(self._context)
            self._subscription_manager.commit()  # apply pending operations
            logger.debug(f"<green>{self._strategy_name}</green> is fitted")
        except Exception as strat_error:
            logger.error(f"Strategy {self._strategy_name} on_fit raised an exception: {strat_error}")
            logger.opt(colors=False).error(traceback.format_exc())
        finally:
            self._fit_is_running = False
            self._init_fit_was_called = True

    def __process_and_log_target_positions(
        self, target_positions: List[TargetPosition] | TargetPosition | None
    ) -> list[TargetPosition]:
        if target_positions is None:
            return []
        if isinstance(target_positions, TargetPosition):
            target_positions = [target_positions]
        self._logging.save_signals_targets(target_positions)
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
                signal.group = self._strategy_name

            # set reference prices for signals
            if signal.reference_price is None:
                q = self._market_data.quote(signal.instrument)
                if q is None:
                    continue
                signal.reference_price = q.mid_price()

        return signals

    def _run_in_thread_pool(self, func: Callable, args=()):
        # For simulation we don't need to call function in thread
        if self._is_simulation:
            func(*args)
        else:
            assert self._pool
            self._pool.apply_async(func, args)

    def _wrap_signal_list(self, signals: List[Signal] | Signal | None) -> List[Signal]:
        if signals is None:
            signals = []
        elif isinstance(signals, Signal):
            signals = [signals]
        return signals

    def __update_base_data(
        self, instrument: Instrument, event_type: str, data: Any, is_historical: bool = False
    ) -> bool:
        """
        Updates the base data cache with the provided data.

        Returns:
            bool: True if the data is base data and the strategy should be triggered, False otherwise.
        """
        is_base_data = self.__is_base_data(data)
        # update cached ohlc is this is base subscription or if we are in simulation and subscribed to ohlc
        # and receive quotes
        _update_ohlc = is_base_data or (
            not is_historical
            and self._is_simulation
            and DataType.OHLC == self._subscription_manager.get_base_subscription()
            and isinstance(data, Quote)
        )
        self._cache.update(instrument, event_type, data, update_ohlc=_update_ohlc)
        # update trackers, gatherers on base data and on Quote (always)
        if not is_historical and (is_base_data or isinstance(data, Quote)):
            _data = data if not isinstance(data, OrderBook) else data.to_quote()
            self._account.update_position_price(self._time_provider.time(), instrument, extract_price(_data))
            target_positions = self.__process_and_log_target_positions(
                self._position_tracker.update(self._context, instrument, _data)
            )
            self.__process_signals_from_target_positions(target_positions)
            self._position_gathering.alter_positions(self._context, target_positions)
        return is_base_data and not self._trigger_on_time_event

    def __is_base_data(self, data: Any) -> bool:
        _sub_type = self._subscription_manager.get_base_subscription()
        sub_type, sub_params = DataType.from_str(_sub_type)
        timeframe = sub_params.get("timeframe")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # TODO - think about refactoring we need to get rid of it !!!
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if self._is_simulation and DataType.OHLC == sub_type and timeframe:
            # in simulate we transform OHLC into quotes, so we need to check
            # if this is the final quote of a bar which should be considered as base data
            if self._trig_bar_freq_nsec is None:
                self._trig_bar_freq_nsec = pd.Timedelta(timeframe).as_unit("ns").asm8.item()
            t = self._time_provider.time().item()
            assert self._trig_bar_freq_nsec is not None
            # shifting by 1sec in ns
            _sim_step = (t + 1e9) // self._trig_bar_freq_nsec
            if self._cur_sim_step is None:
                self._cur_sim_step = _sim_step
                return False
            if _sim_step > self._cur_sim_step:
                self._cur_sim_step = _sim_step
                return True
            return False

        # TODO: handle batched events
        return (
            (sub_type == DataType.OHLC and isinstance(data, Bar))
            or (sub_type == DataType.OHLC_QUOTES and isinstance(data, Quote))  # TEMPORARY: just to pass test
            or (sub_type == DataType.OHLC_TRADES and isinstance(data, Trade))  # TEMPORARY: just to pass test
            or (sub_type == DataType.QUOTE and isinstance(data, Quote))
            or (sub_type == DataType.ORDERBOOK and isinstance(data, OrderBook))
            or (sub_type == DataType.TRADE and isinstance(data, Trade))
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ###########################################################################
    # - Handlers for different types of incoming data
    ###########################################################################

    # it's important that we call it with _process to not include in the handlers map
    def _process_custom_event(
        self, instrument: Instrument | None, event_type: str, event_data: Any
    ) -> MarketEvent | None:
        if instrument is not None:
            self.__update_base_data(instrument, event_type, event_data)

        elif instrument is None and isinstance(event_data, dict):
            for _instrument, data in event_data.items():
                if isinstance(_instrument, Instrument):
                    self.__update_base_data(_instrument, event_type, data)

        return MarketEvent(self._time_provider.time(), event_type, instrument, event_data)

    def _process_hist_event(self, instrument: Instrument, event_type: str, event_data: Any) -> None:
        if not isinstance(event_data, list):
            event_data = [event_data]
        if DataType.OHLC == event_type:
            # - update ohlc using the list directly, this allows to update
            # multiple timeframes with different data (1h can have more bars than 1m)
            _, sub_params = DataType.from_str(event_type)
            timeframe = sub_params.get("timeframe", self._cache.default_timeframe)
            self._cache.update_by_bars(instrument, timeframe, event_data)
        else:
            for data in event_data:
                self.__update_base_data(instrument, event_type, data, is_historical=True)

    def _handle_event(self, instrument: Instrument, event_type: str, event_data: Any) -> TriggerEvent:
        return TriggerEvent(self._time_provider.time(), event_type, instrument, event_data)

    def _handle_time(self, instrument: Instrument, event_type: str, data: dt_64) -> TriggerEvent:
        return TriggerEvent(self._time_provider.time(), event_type, instrument, data)

    def _handle_service_time(self, instrument: Instrument, event_type: str, data: dt_64) -> TriggerEvent | None:
        """It is used by simulation as a dummy to trigger actual time events."""
        pass

    def _handle_fit(self, instrument: Instrument | None, event_type: str, data: Tuple[dt_64 | None, dt_64]) -> None:
        """
        When scheduled fit event is happened - we need to invoke strategy on_fit method
        """
        if not self._cache.is_data_ready():
            return
        self._fit_is_running = True
        self._run_in_thread_pool(self.__invoke_on_fit)

    def _handle_ohlc(self, instrument: Instrument, event_type: str, bar: Bar) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, bar)
        return MarketEvent(self._time_provider.time(), event_type, instrument, bar, is_trigger=base_update)

    def _handle_trade(self, instrument: Instrument, event_type: str, trade: Trade) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, trade)
        return MarketEvent(self._time_provider.time(), event_type, instrument, trade, is_trigger=base_update)

    def _handle_orderbook(self, instrument: Instrument, event_type: str, orderbook: OrderBook) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, orderbook)
        return MarketEvent(self._time_provider.time(), event_type, instrument, orderbook, is_trigger=base_update)

    def _handle_quote(self, instrument: Instrument, event_type: str, quote: Quote) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, quote)
        return MarketEvent(self._time_provider.time(), event_type, instrument, quote, is_trigger=base_update)

    @SW.watch("StrategyContext.order")
    def _handle_order(self, instrument: Instrument, event_type: str, order: Order) -> Order:
        self._account.process_order(order)
        return order

    @SW.watch("StrategyContext")
    def _handle_deals(self, instrument: Instrument, event_type: str, deals: list[Deal]) -> TriggerEvent | None:
        self._account.process_deals(instrument, deals)
        self._logging.save_deals(instrument, deals)
        if instrument is None:
            logger.debug(f"Execution report for unknown instrument {instrument}")
            return
        for d in deals:
            # - notify position gatherer and tracker
            self._position_gathering.on_execution_report(self._context, instrument, d)
            self._position_tracker.on_execution_report(self._context, instrument, d)
            logger.debug(f"Executed {d.amount} @ {d.price} of {instrument} for order <red>{d.order_id}</red>")
        return None
