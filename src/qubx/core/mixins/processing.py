import traceback

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import FunctionType
from multiprocessing.pool import ThreadPool

from qubx import logger
from qubx.core.helpers import BasicScheduler, CachedMarketDataHolder, process_schedule_spec
from qubx.core.loggers import StrategyLogging
from qubx.core.series import Trade, Quote, Bar, OrderBook
from qubx.core.basics import (
    SW,
    Deal,
    Order,
    dt_64,
    Signal,
    Instrument,
    TriggerEvent,
    TargetPosition,
)
from qubx.core.interfaces import (
    IMarketDataProvider,
    IPositionGathering,
    IStrategy,
    ISubscriptionManager,
    PositionsTracker,
    IStrategyContext,
    SubscriptionType,
    IProcessingManager,
    ITimeProvider,
)


class ProcessingManager(IProcessingManager):
    MAX_NUMBER_OF_STRATEGY_FAILURES = 10

    __context: IStrategyContext
    __strategy: IStrategy
    __logging: StrategyLogging
    __market_data: IMarketDataProvider
    __subscription_manager: ISubscriptionManager
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

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        logging: StrategyLogging,
        market_data: IMarketDataProvider,
        subscription_manager: ISubscriptionManager,
        time_provider: ITimeProvider,
        position_tracker: PositionsTracker,
        position_gathering: IPositionGathering,
        cache: CachedMarketDataHolder,
        scheduler: BasicScheduler,
        is_simulation: bool,
    ):
        self.__context = context
        self.__strategy = strategy
        self.__logging = logging
        self.__market_data = market_data
        self.__subscription_manager = subscription_manager
        self.__time_provider = time_provider
        self.__is_simulation = is_simulation
        self.__position_gathering = position_gathering
        self.__position_tracker = position_tracker
        self.__cache = cache
        self.__scheduler = scheduler

        self.__pool = ThreadPool(2) if not self.__is_simulation else None
        self.__handlers = {
            n.split("_handle_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) == FunctionType and n.startswith("_handle_")
        }
        self.__strategy_name = strategy.__class__.__name__
        self.__init_fit_args = (None, self.__time_provider.time())

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

    def process_data(self, instrument: Instrument, d_type: str, data: Any) -> bool:
        self.__logging.notify(self.__time_provider.time())

        handler = self.__handlers.get(d_type)
        with SW("StrategyContext.handler"):
            if handler:
                trigger_event = handler(self, instrument, data)
            else:
                trigger_event = self._handle_event(instrument, d_type, data)

        # - check if it still didn't call on_fit() for first time
        if not self.__init_fit_was_called:
            self._handle_fit(None, self.__init_fit_args)

        if not trigger_event:
            return False

        # - if fit was not called - skip on_event call
        if not self.__init_fit_was_called:
            logger.debug(f"{self.__strategy_name}::on_event() is SKIPPED for now because on_fit() was not called yet!")
            return False

        # - if strategy still fitting - skip on_event call
        if self.__fit_is_running:
            logger.warning(f"{self.__strategy_name}::on_event() is SKIPPED for now because is being still fitting!")
            return False

        signals: list[Signal] | Signal | None = None
        with SW("StrategyContext.on_event"):
            try:
                signals = self.__strategy.on_event(self.__context, trigger_event)
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
            # fmt: off
            positions_from_strategy = self.__process_and_log_target_positions(
                self.__position_tracker.process_signals(
                    self.__context,
                    self.__process_signals(signals)
                )
            )
            self.__position_gathering.alter_positions(self.__context, positions_from_strategy)
            # fmt: on

        # - notify poition and portfolio loggers
        self.__logging.notify(self.__time_provider.time())

        return False

    @SW.watch("StrategyContext.on_fit")
    def __invoke_on_fit(self, current_fit_time: dt_64, prev_fit_time: dt_64 | None) -> None:
        try:
            logger.debug(
                f"Invoking <green>{self.__strategy_name}</green> on_fit('{current_fit_time}', '{prev_fit_time}')"
            )
            self.__strategy.on_fit(self.__context, current_fit_time, prev_fit_time)
            logger.debug(f"<green>{self.__strategy_name}</green> is fitted")
        except Exception as strat_error:
            logger.error(
                f"Strategy {self.__strategy_name} on_fit('{current_fit_time}', '{prev_fit_time}') "
                f"raised an exception: {strat_error}"
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
                q = self.__market_data.quote(signal.instrument)
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

    def __update_base_data(self, instrument: Instrument, data: Any, is_historical: bool = False) -> None:
        """
        Updates the base data cache with the provided data.
        """
        sub_type, _ = self.__subscription_manager.get_base_subscription()
        # TODO: handle batched events
        is_base_data = (
            (sub_type == SubscriptionType.OHLC and isinstance(data, Bar))
            or (sub_type == SubscriptionType.QUOTE and isinstance(data, Quote))
            or (sub_type == SubscriptionType.ORDERBOOK and isinstance(data, OrderBook))
            or (sub_type == SubscriptionType.TRADE and isinstance(data, Trade))
        )
        self.__cache.update(instrument, data, update_ohlc=is_base_data)
        if not is_historical and is_base_data:
            _data = data if not isinstance(data, OrderBook) else data.to_quote()
            target_positions = self.__process_and_log_target_positions(
                self.__position_tracker.update(self.__context, instrument, _data)
            )
            self.__process_signals_from_target_positions(target_positions)
            self.__position_gathering.alter_positions(self.__context, target_positions)

    ###########################################################################
    # - Handlers for different types of incoming data
    ###########################################################################
    def _handle_fit(self, _: Instrument | None, data: Any) -> None:
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

    def _handle_hist_bars(self, instrument: Instrument, bars: list[Bar]) -> None:
        for b in bars:
            self._handle_hist_bar(instrument, b)

    def _handle_hist_bar(self, instrument: Instrument, bar: Bar) -> None:
        self.__update_base_data(instrument, bar, is_historical=True)

    def _handle_hist_quote(self, instrument: Instrument, quote: Quote) -> None:
        self.__update_base_data(instrument, quote, is_historical=True)

    def _handle_hist_trade(self, instrument: Instrument, trade: Trade) -> None:
        self.__update_base_data(instrument, trade, is_historical=True)

    def _handle_bar(self, instrument: Instrument, bar: Bar) -> TriggerEvent:
        self.__update_base_data(instrument, bar)
        return TriggerEvent(self.__time_provider.time(), SubscriptionType.OHLC, instrument, bar)

    def _handle_trade(self, instrument: Instrument, trade: Trade) -> TriggerEvent:
        self.__update_base_data(instrument, trade)
        return TriggerEvent(self.__time_provider.time(), SubscriptionType.TRADE, instrument, trade)

    def _handle_orderbook(self, instrument: Instrument, orderbook: OrderBook) -> TriggerEvent:
        self.__update_base_data(instrument, orderbook)
        return TriggerEvent(self.__time_provider.time(), SubscriptionType.ORDERBOOK, instrument, orderbook)

    def _handle_quote(self, instrument: Instrument, quote: Quote) -> TriggerEvent:
        self.__update_base_data(instrument, quote)
        return TriggerEvent(self.__time_provider.time(), SubscriptionType.QUOTE, instrument, quote)

    def _handle_event(self, instrument: Instrument, event_type: str, event_data: Any) -> TriggerEvent:
        return TriggerEvent(self.__time_provider.time(), event_type, instrument, event_data)

    @SW.watch("StrategyContext.order")
    def _handle_order(self, instrument: Instrument, order: Order) -> TriggerEvent | None:
        logger.debug(
            f"[<red>{order.id}</red> / {order.client_id}] : {order.type} {order.side} {order.quantity} "
            f"of {instrument.symbol} { (' @ ' + str(order.price)) if order.price else '' } -> [{order.status}]"
        )
        # - check if we want to trigger any strat's logic on order
        return None

    @SW.watch("StrategyContext")
    def _handle_deals(self, instrument: Instrument, deals: list[Deal]) -> TriggerEvent | None:
        # - log deals in storage
        self.__logging.save_deals(instrument, deals)
        if instrument is None:
            logger.debug(f"Execution report for unknown instrument {instrument}")
            return
        for d in deals:
            # - notify position gatherer and tracker
            self.__position_gathering.on_execution_report(self.__context, instrument, d)
            self.__position_tracker.on_execution_report(self.__context, instrument, d)
            logger.debug(f"Executed {d.amount} @ {d.price} of {instrument} for order <red>{d.order_id}</red>")
        return None
