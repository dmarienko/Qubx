import traceback
import pandas as pd

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
    MarketEvent,
    Order,
    dt_64,
    Signal,
    Instrument,
    TriggerEvent,
    TargetPosition,
    SubscriptionType,
)
from qubx.core.interfaces import (
    IMarketDataProvider,
    IPositionGathering,
    IStrategy,
    IBrokerServiceProvider,
    ISubscriptionManager,
    PositionsTracker,
    IStrategyContext,
    IProcessingManager,
    ITimeProvider,
)


class ProcessingManager(IProcessingManager):
    MAX_NUMBER_OF_STRATEGY_FAILURES = 10

    __context: IStrategyContext
    __strategy: IStrategy
    __broker: IBrokerServiceProvider
    __logging: StrategyLogging
    __market_data: IMarketDataProvider
    __subscription_manager: ISubscriptionManager
    __time_provider: ITimeProvider
    __position_tracker: PositionsTracker
    __position_gathering: IPositionGathering
    __cache: CachedMarketDataHolder
    __scheduler: BasicScheduler

    __handlers: dict[str, Callable[["ProcessingManager", Instrument, str, Any], TriggerEvent | None]]
    __strategy_name: str

    __trigger_on_time_event: bool = False
    __fit_is_running: bool = False
    __init_fit_was_called: bool = False
    __fails_counter: int = 0
    __is_simulation: bool
    __pool: ThreadPool | None
    _trig_bar_freq_nsec: int | None = None
    _cur_sim_step: int | None = None

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        broker: IBrokerServiceProvider,
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
        self.__broker = broker
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
        self._trig_bar_freq_nsec = None

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
        self.__trigger_on_time_event = True

    def process_data(self, instrument: Instrument, d_type: str, data: Any) -> bool:
        self.__logging.notify(self.__time_provider.time())

        handler = self.__handlers.get(d_type)
        with SW("StrategyContext.handler"):
            if not d_type:
                event = None
            elif handler:
                event = handler(self, instrument, d_type, data)
            else:
                event = self._process_custom_event(instrument, d_type, data)

        # - check if it still didn't call on_fit() for first time
        if not self.__init_fit_was_called:
            self._handle_fit(None, "fit", (None, self.__time_provider.time()))
            return False

        if not event or event.data is None:
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
                if isinstance(event, MarketEvent):
                    signals = self._wrap_signal_list(self.__strategy.on_market_data(self.__context, event))

                if signals is None:
                    signals = []

                if isinstance(event, TriggerEvent) or (isinstance(event, MarketEvent) and event.is_trigger):
                    _signals = self._wrap_signal_list(self.__strategy.on_event(self.__context, event))
                    signals.extend(_signals)

                self.__broker.commit()  # apply pending broker operations

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
    def __invoke_on_fit(self) -> None:
        try:
            logger.debug(f"Invoking <green>{self.__strategy_name}</green> on_fit")
            self.__strategy.on_fit(self.__context)
            self.__broker.commit()  # apply pending broker operations
            logger.debug(f"<green>{self.__strategy_name}</green> is fitted")
        except Exception as strat_error:
            logger.error(f"Strategy {self.__strategy_name} on_fit raised an exception: {strat_error}")
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
            and self.__is_simulation
            and SubscriptionType.OHLC == self.__subscription_manager.get_base_subscription()[0]
            and isinstance(data, Quote)
        )
        self.__cache.update(instrument, event_type, data, update_ohlc=_update_ohlc)
        # update trackers, gatherers on base data and on Quote (always)
        if not is_historical and is_base_data or isinstance(data, Quote):
            _data = data if not isinstance(data, OrderBook) else data.to_quote()
            target_positions = self.__process_and_log_target_positions(
                self.__position_tracker.update(self.__context, instrument, _data)
            )
            self.__process_signals_from_target_positions(target_positions)
            self.__position_gathering.alter_positions(self.__context, target_positions)
        return is_base_data and not self.__trigger_on_time_event

    def __is_base_data(self, data: Any) -> bool:
        sub_type, sub_params = self.__subscription_manager.get_base_subscription()
        timeframe = sub_params.get("timeframe")
        if self.__is_simulation and SubscriptionType.OHLC == sub_type and timeframe:
            # in simulate we transform OHLC into quotes, so we need to check
            # if this is the final quote of a bar which should be considered as base data
            if self._trig_bar_freq_nsec is None:
                self._trig_bar_freq_nsec = pd.Timedelta(timeframe).as_unit("ns").asm8.item()
            t = self.__time_provider.time().item()
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
            (sub_type == SubscriptionType.OHLC and isinstance(data, Bar))
            or (sub_type == SubscriptionType.QUOTE and isinstance(data, Quote))
            or (sub_type == SubscriptionType.ORDERBOOK and isinstance(data, OrderBook))
            or (sub_type == SubscriptionType.TRADE and isinstance(data, Trade))
        )

    ###########################################################################
    # - Handlers for different types of incoming data
    ###########################################################################

    # it's important that we call it with _process to not include in the handlers map
    def _process_custom_event(self, instrument: Instrument, event_type: str, event_data: Any) -> MarketEvent | None:
        if event_type.startswith("hist_"):
            return self._process_hist_event(instrument, event_type, event_data)
        self.__update_base_data(instrument, event_type, event_data)
        return MarketEvent(self.__time_provider.time(), event_type, instrument, event_data)

    def _process_hist_event(self, instrument: Instrument, event_type: str, event_data: Any) -> None:
        event_type = event_type[5:]
        if isinstance(event_data, list):
            for data in event_data:
                self.__update_base_data(instrument, event_type, data, is_historical=True)
        else:
            self.__update_base_data(instrument, event_type, event_data, is_historical=True)

    def _handle_event(self, instrument: Instrument, event_type: str, event_data: Any) -> TriggerEvent:
        return TriggerEvent(self.__time_provider.time(), event_type, instrument, event_data)

    def _handle_time(self, instrument: Instrument, event_type: str, data: dt_64) -> TriggerEvent:
        return TriggerEvent(self.__time_provider.time(), event_type, instrument, data)

    def _handle_service_time(self, instrument: str, event_type: str, data: dt_64) -> TriggerEvent | None:
        """It is used by simulation as a dummy to trigger actual time events."""
        pass

    def _handle_fit(self, instrument: Instrument | None, event_type: str, data: Tuple[dt_64 | None, dt_64]) -> None:
        """
        When scheduled fit event is happened - we need to invoke strategy on_fit method
        """
        if not self.__cache.is_data_ready():
            return
        self.__fit_is_running = True
        self._run_in_thread_pool(self.__invoke_on_fit)

    def _handle_ohlc(self, instrument: Instrument, event_type: str, bar: Bar) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, bar)
        return MarketEvent(self.__time_provider.time(), event_type, instrument, bar, is_trigger=base_update)

    def _handle_trade(self, instrument: Instrument, event_type: str, trade: Trade) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, trade)
        return MarketEvent(self.__time_provider.time(), event_type, instrument, trade, is_trigger=base_update)

    def _handle_orderbook(self, instrument: Instrument, event_type: str, orderbook: OrderBook) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, orderbook)
        return MarketEvent(self.__time_provider.time(), event_type, instrument, orderbook, is_trigger=base_update)

    def _handle_quote(self, instrument: Instrument, event_type: str, quote: Quote) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, quote)
        return MarketEvent(self.__time_provider.time(), event_type, instrument, quote, is_trigger=base_update)

    @SW.watch("StrategyContext.order")
    def _handle_order(self, instrument: Instrument, event_type: str, order: Order) -> TriggerEvent | None:
        logger.debug(
            f"[<red>{order.id}</red> / {order.client_id}] : {order.type} {order.side} {order.quantity} "
            f"of {instrument.symbol} { (' @ ' + str(order.price)) if order.price else '' } -> [{order.status}]"
        )
        # - check if we want to trigger any strat's logic on order
        return None

    @SW.watch("StrategyContext")
    def _handle_deals(self, instrument: Instrument, event_type: str, deals: list[Deal]) -> TriggerEvent | None:
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
