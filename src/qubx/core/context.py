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
)
from qubx.core.loggers import StrategyLogging
from qubx.core.strategy import (
    IBrokerServiceProvider,
    IPositionGathering,
    IStrategy,
    ITradingServiceProvider,
    PositionsTracker,
    StrategyContext,
    SubscriptionType,
)
from qubx.core.series import Trade, Quote, Bar, OHLCV
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.trackers.sizers import FixedSizer
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_seconds_to_str


_SW = Stopwatch()


def _dict_with_exception(dct, f):
    if f not in dct:
        raise ValueError(f"Configuration {dct} must contain field '{f}'")
    return dct[f]


class StrategyContextImpl(StrategyContext):
    """
    Main implementation of StrategyContext interface.
    """

    MAX_NUMBER_OF_STRATEGY_FAILURES = 10

    strategy: IStrategy  # strategy instance
    trading_service: ITradingServiceProvider  # service for exchange API: orders managemewnt
    broker_provider: IBrokerServiceProvider  # market data provider
    instruments: List[Instrument]  # list of instruments this strategy trades
    positions: Dict[str, Position]  # positions of the strategy (instrument -> position)
    acc: AccountProcessor
    positions_tracker: PositionsTracker  # position tracker
    positions_gathering: IPositionGathering  # position adjuster (executor)

    # - loggers
    _logging: StrategyLogging  # recording all activities for the strat: execs, positions, portfolio

    # - cached marked data anb scheduler
    _cache: CachedMarketDataHolder  # market data cache
    _scheduler: BasicScheduler

    # - configuration
    _market_data_subcription_type: str = "unknown"
    _market_data_subcription_params: dict = dict()
    _thread_data_loop: Thread | None = None  # market data loop

    _trig_interval_in_bar_nsec: int
    _trig_bar_freq_nsec: int
    _trig_on_bar: bool = False
    _trig_on_time: bool = False
    _trig_on_quote: bool = False
    _trig_on_trade: bool = False
    _trig_on_book: bool = False
    _current_bar_trigger_processed: bool = False
    _is_initilized: bool = False
    _symb_to_instr: Dict[str, Instrument]
    __strategy_id: str
    __order_id: int
    __fails_counter = 0
    __handlers: Dict[str, Callable[["StrategyContext", str, Any], TriggerEvent | None]]
    __fit_is_running: bool = False  # during fitting working it stops calling on_event method
    __init_fit_was_called: bool = False  # true if initial fit was already run
    __init_fit_args: Tuple = (None, None)  # arguments for initial on_fit() method call
    __pool: ThreadPool | None = None  # thread pool used for running aux tasks (fit etc)

    def __init__(
        self,
        # - strategy with parameters
        strategy: IStrategy,
        config: Dict[str, Any] | None,
        # - - - - - - - - - - - - - - - - - - - - -
        # - data provider and exchange service
        broker_connector: IBrokerServiceProvider,
        instruments: List[Instrument],
        base_currency: str = "USDT",
        initial_capital: float = 0,
        reserves: Dict[str, float] | None = None,
        # - - - - - - - - - - - - - - - - - - - - -
        # - data subscription - - - - - - - - - - -
        md_subscription: Dict[str, Any] = dict(type="ohlc", timeframe="1Min", nback=60),
        # - - - - - - - - - - - - - - - - - - - - -
        # - when need to trigger and fit strategy - - - - - - -
        trigger_spec: str | list[str] = "bar: -1Sec",  # 1 sec before subscription bar is closed
        fit_spec: str | None = None,
        # - - - - - - - - - - - - - - - - - - - - -
        # - how to write logs - - - - - - - - - -
        logs_writer: LogsWriter | None = None,
        positions_log_freq: str = "1Min",
        portfolio_log_freq: str = "5Min",
        num_exec_records_to_write=1,  # in live let's write every execution
        # - - - - - - - - - - - - - - - - - - - - -
        # - signals executor configuration - - - -
        position_gathering: IPositionGathering | None = None,
    ) -> None:
        # - initialization
        self.broker_provider = broker_connector
        self.trading_service = broker_connector.get_trading_service()
        self.acc = AccountProcessor(
            account_id=self.trading_service.get_account_id(),
            base_currency=base_currency,
            reserves=reserves,
            initial_capital=initial_capital,
        )
        self.trading_service.set_account(self.acc)

        self.config = config
        self.instruments = instruments
        self.positions = {}
        self.strategy_name = strategy.__class__.__name__
        self.__fit_is_running = False
        self.__init_fit_was_called = False
        self.__pool = None
        self.__fails_counter = 0

        # - for fast access to instrument by it's symbol
        self._symb_to_instr = {i.symbol: i for i in instruments}

        # - get scheduling service from broker
        self._scheduler = self.broker_provider.get_scheduler()

        # - instantiate logging functional
        self._logs_writer = logs_writer
        self._logging = StrategyLogging(logs_writer, positions_log_freq, portfolio_log_freq, num_exec_records_to_write)

        # - position adjuster
        self.positions_gathering = position_gathering if position_gathering else SimplePositionGatherer()

        # - extract data and event handlers
        self.__handlers = {
            n.split("_processing_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) == FunctionType and n.startswith("_processing_")
        }

        # - create strategy instance and populate custom paramameters
        self._instantiate_strategy(strategy, config)

        # - process market data configuration
        self.__check_how_to_listen_to_market_data(md_subscription)

        # - process trigger and fit configurations
        self.__check_how_to_trigger_and_fit_strategy(trigger_spec, fit_spec)

        # - run cron scheduler
        self._scheduler.run()

    def _instantiate_strategy(self, strategy: IStrategy, config: Dict[str, Any] | None):
        # - set parameters to strategy (not sure we will do it here)
        self.strategy = strategy
        if isinstance(strategy, type):
            self.strategy = strategy()
        self.strategy.ctx = self

        # - set strategy custom parameters
        set_parameters_to_object(self.strategy, **config if config else {})
        self._is_initilized = False
        self.__strategy_id = self.strategy.__class__.__name__ + "_"
        self.__order_id = self.time().item() // 100_000_000

        # - here we need to get tracker
        _track = self.strategy.tracker(self)

        # - by default we use default position tracker with fixed 1:1 sizer
        #   so any signal is coinsidered as raw position size
        self.positions_tracker = _track if _track else PositionsTracker(FixedSizer(1.0, amount_in_quote=False))

    def __check_how_to_listen_to_market_data(self, md_config: dict):
        self._market_data_subcription_type = _dict_with_exception(md_config, "type").lower()
        match self._market_data_subcription_type:
            case "ohlc":
                timeframe = _dict_with_exception(md_config, "timeframe")
                self._market_data_subcription_params = {
                    "timeframe": timeframe,
                    "nback": md_config.get("nback", 1),
                }
                self._cache = CachedMarketDataHolder(timeframe)

            case "trade" | "trades" | "tas":
                timeframe = md_config.get("timeframe", "1Sec")
                self._market_data_subcription_params = {
                    "nback": md_config.get("nback", 1),
                }
                self._cache = CachedMarketDataHolder("1Sec")

            case "quote" | "quotes":
                self._cache = CachedMarketDataHolder("1Sec")

            case "ob" | "orderbook":
                self._cache = CachedMarketDataHolder("1Sec")

            case _:
                raise ValueError(
                    f"{self._market_data_subcription_type} is not a valid value for market data subcription type !!!"
                )

    def __check_how_to_trigger_and_fit_strategy(
        self, trigger_schedule: str | list[str] | None, fit_schedue: str | None
    ):
        _td2ns = lambda x: x.as_unit("ns").asm8.item()

        self._trig_interval_in_bar_nsec = 0

        if not trigger_schedule:
            raise ValueError("trigger parameter can't be empty !")
        if not isinstance(trigger_schedule, list):
            trigger_schedule = [trigger_schedule]

        t_rules = [process_schedule_spec(tr) for tr in trigger_schedule]

        f_rules = process_schedule_spec(fit_schedue)

        if not t_rules:
            raise ValueError(f"Couldn't recognize 'trigger' parameter specification: {trigger_schedule} !")

        # - we can use it as reference if bar timeframe is not secified
        _default_data_timeframe = pd.Timedelta(self._cache.default_timeframe)

        # - check trigger spec - - - - -
        for t_rule in t_rules:
            match t_rule["type"]:
                # it triggers on_event method when new bar is formed.
                # in this case it won't arm scheduler but just ruled by actual market data updates
                # - TODO: so probably need to drop this in favor to cron tasks
                # it's also possible to specify delay
                # "bar: -1s"      - it uses default timeframe and wake up 1 sec before every bar's close
                # "bar.5m: -5sec" - 5 sec before 5min bar's close
                # "bar.5m: 1sec"  - 1 sec after 5min bar closed (in next bar)
                case "bar":
                    _r_tf = t_rule.get("timeframe")
                    _bar_timeframe = _default_data_timeframe if not _r_tf else pd.Timedelta(_r_tf)
                    _inside_bar_delay: pd.Timedelta = t_rule.get("delay", pd.Timedelta(0))

                    if abs(pd.Timedelta(_inside_bar_delay)) > pd.Timedelta(_bar_timeframe):
                        raise ValueError(
                            f"Delay must be less or equal to bar's timeframe for bar trigger: you used {_inside_bar_delay} delay for {_bar_timeframe}"
                        )

                    # for positive delay - trigger strategy when this interval passed after new bar's open
                    if _inside_bar_delay >= pd.Timedelta(0):
                        self._trig_interval_in_bar_nsec = _td2ns(_inside_bar_delay)

                    # for negative delay - trigger strategy when time is closer to bar's closing time more than this interval
                    else:
                        self._trig_interval_in_bar_nsec = _td2ns(_bar_timeframe + _inside_bar_delay)

                    self._trig_bar_freq_nsec = _td2ns(_bar_timeframe)
                    self._trig_on_bar = True

                    logger.debug(
                        f"Triggering strategy on every {convert_seconds_to_str(self._trig_bar_freq_nsec/1e9)} bar after {convert_seconds_to_str(self._trig_interval_in_bar_nsec/1e9)}"
                    )

                case "cron":
                    if "schedule" not in t_rule:
                        raise ValueError(f"cron trigger type is specified but cron schedule not found !")

                    self._scheduler.schedule_event(t_rule["schedule"], "time_event")

                case "quote":
                    self._trig_on_quote = True
                    raise ValueError(f"quote trigger NOT IMPLEMENTED YET")

                case "trade":
                    self._trig_on_trade = True
                    logger.debug("Triggering strategy on every trade event")

                case "orderbook" | "ob":
                    self._trig_on_book = True
                    raise ValueError(f"orderbook trigger NOT IMPLEMENTED YET")

                case _:
                    raise ValueError(f"Wrong trigger type {t_rule['type']}")

        # - check fit spec - - - - -
        _last_fit_data_can_be_used = pd.Timestamp(self.time())
        match f_rules.get("type"):
            case "cron":
                if "schedule" not in f_rules:
                    raise ValueError(f"cron fit trigger type is specified but cron schedule not found !")

                self._scheduler.schedule_event(f_rules["schedule"], "fit_event")
                _last_fit_data_can_be_used = self._scheduler.get_event_last_time("fit_event")

            case "bar":
                raise ValueError("Raw 'bar' type is not supported for fitting spec yet, please use cron type !")

            case _:
                # if schedule is not specified just do not arm the task
                # only initial fit will be called
                pass

        # - we can't just call on_fit right now because not all market data may be ready
        # - so we just mark it as not called yet
        self.__init_fit_was_called = False
        self.__init_fit_args = (None, _last_fit_data_can_be_used)

    def process_data(self, symbol: str, d_type: str, data: Any) -> bool:
        # logger.debug(f" <cyan>({self.time()})</cyan> DATA : <yellow>{(symbol, d_type, data) }</yellow>")

        # - process data if handler is registered
        handler = self.__handlers.get(d_type)
        _SW.start("StrategyContext.handler")
        _strategy_trigger_on_event = handler(self, symbol, data) if handler else None
        _SW.stop("StrategyContext.handler")

        # - check if it still didn't call on_fit() for first time
        if not self.__init_fit_was_called:
            self._processing_fit_event(None, self.__init_fit_args)

        if _strategy_trigger_on_event:

            # - if fit was not called - skip on_event call
            if not self.__init_fit_was_called:
                logger.warning(
                    f"[{self.time()}] {self.strategy.__class__.__name__}::on_event() is SKIPPED for now because on_fit() was not called yet !"
                )
                return False

            # - if strategy still fitting - skip on_event call
            if self.__fit_is_running:
                logger.warning(
                    f"[{self.time()}] {self.strategy.__class__.__name__}::on_event() is SKIPPED for now because is being still fitting !"
                )
                return False

            signals: List[Signal] | Signal | None = None
            try:
                _SW.start("strategy.on_event")
                signals = self.strategy.on_event(self, _strategy_trigger_on_event)
                self._fails_counter = 0
            except Exception as strat_error:
                # - probably we need some cooldown interval after exception to prevent flooding
                logger.error(
                    f"[{self.time()}]: Strategy {self.strategy.__class__.__name__} raised an exception: {strat_error}"
                )
                logger.opt(colors=False).error(traceback.format_exc())

                #  - we stop execution after let's say maximal number of errors in a row
                self.__fails_counter += 1
                if self.__fails_counter >= StrategyContextImpl.MAX_NUMBER_OF_STRATEGY_FAILURES:
                    logger.error("STRATEGY FAILURES IN THE ROW EXCEEDED MAX ALLOWED NUMBER - STOPPING ...")
                    return True
            finally:
                _SW.stop("strategy.on_event")

            # - process and execute signals if they are provided
            if signals:
                # process signals by tracker and turn convert them into positions
                positions_from_strategy = self.__process_and_log_target_positions(
                    self.positions_tracker.process_signals(self, self.__process_signals(signals))
                )

                # gathering in charge of positions
                self.positions_gathering.alter_positions(self, positions_from_strategy)

        # - notify poition and portfolio loggers
        self._logging.notify(self.time())

        return False

    def _process_incoming_data_loop(self, channel: CtrlChannel):
        logger.info("(StrategyContext) Start processing market data")

        while channel.control.is_set():
            # - start loop latency measurement
            _SW.start("StrategyContext._process_incoming_data")

            # - waiting for incoming market data
            symbol, d_type, data = channel.receive()
            if self.process_data(symbol, d_type, data):
                _SW.stop("StrategyContext._process_incoming_data")
                channel.stop()
                break

            _SW.stop("StrategyContext._process_incoming_data")

        logger.info("(StrategyContext) Market data processing stopped")

    def _invoke_on_fit(self, current_fit_time: str | pd.Timestamp, prev_fit_time: str | pd.Timestamp | None):
        try:
            self.__fit_is_running = True
            logger.debug(
                f"Invoking <green>{self.strategy.__class__.__name__}</green> on_fit('{current_fit_time}', '{prev_fit_time}')"
            )
            _SW.start("strategy.on_fit")
            self.strategy.on_fit(self, current_fit_time, prev_fit_time)
            logger.debug(f"<green>{self.strategy.__class__.__name__}</green> is fitted")
        except Exception as strat_error:
            logger.error(
                f"[{self.time()}]: Strategy {self.strategy.__class__.__name__} on_fit('{current_fit_time}', '{prev_fit_time}') raised an exception: {strat_error}"
            )
            logger.opt(colors=False).error(traceback.format_exc())
        finally:
            self.__fit_is_running = False
            self.__init_fit_was_called = True
            _SW.stop("strategy.on_fit")
        return None

    def _processing_time_event(self, symbol: str, data: Any) -> TriggerEvent | None:
        """
        When scheduled time event is happened - we need to invoke strategy on_event method
        """
        return TriggerEvent(self.time(), "time", None, data)

    def _processing_fit_event(self, symbol: str | None, data: Any) -> TriggerEvent | None:
        """
        When scheduled fit event is happened - we need to invoke strategy on_fit method
        """
        if not self._cache.is_data_ready():
            return None

        # times are in seconds here
        prev_fit_time, now_fit_time = data

        # - we need to run this in separate thread
        self.__fit_is_running = True
        self._run_in_thread_pool(
            self._invoke_on_fit,
            (pd.Timestamp(now_fit_time, unit="s"), pd.Timestamp(prev_fit_time, unit="s") if prev_fit_time else None),
        )

        return None

    def _processing_hist_bar(self, symbol: str, bar: Bar) -> TriggerEvent | None:
        # - processing single historical bar
        #   here it just updates cache - historical bar can't trigger strategy logic
        self._cache.update_by_bar(symbol, bar)
        return None

    def _processing_hist_quote(self, symbol: str, quote: Quote | BatchEvent) -> TriggerEvent | None:
        if isinstance(quote, BatchEvent):
            for q in quote.data:
                self._cache.update_by_quote(symbol, q)
        else:
            self._cache.update_by_quote(symbol, quote)
        return None

    def _processing_hist_trade(self, symbol: str, trade: Trade | BatchEvent) -> TriggerEvent | None:
        if isinstance(trade, BatchEvent):
            for t in trade.data:
                self._cache.update_by_trade(symbol, t)
        else:
            self._cache.update_by_trade(symbol, trade)
        return None

    def _processing_hist_bars(self, symbol: str, bars: List[Bar]) -> TriggerEvent | None:
        # - processing historical bars as list
        for b in bars:
            self._processing_hist_bar(symbol, b)
        return None

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

    def __process_signals(self, signals: list[Signal] | Signal | None) -> List[Signal]:
        if isinstance(signals, Signal):
            signals = [signals]
        elif signals is None:
            return []

        for signal in signals:
            # set strategy group name if not set
            if not signal.group:
                signal.group = self.strategy_name

            # set reference prices for signals
            if signal.reference_price is None:
                q = self.quote(signal.instrument.symbol)
                if q is None:
                    continue
                signal.reference_price = q.mid_price()

        return signals

    def __process_signals_from_target_positions(
        self, target_positions: List[TargetPosition] | TargetPosition | None
    ) -> None:
        if target_positions is None:
            return
        if isinstance(target_positions, TargetPosition):
            target_positions = [target_positions]
        signals = [pos.signal for pos in target_positions]
        self.__process_signals(signals)

    def __process_and_log_target_positions(
        self, target_positions: List[TargetPosition] | TargetPosition | None
    ) -> List[TargetPosition]:

        if isinstance(target_positions, TargetPosition):
            target_positions = [target_positions]
        elif target_positions is None:
            return []

        self._logging.save_signals_targets(target_positions)
        return target_positions

    @_SW.watch("StrategyContext")
    def _processing_bar(self, symbol: str, bar: Bar) -> TriggerEvent | None:
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

        if self._trig_on_trade:
            event_type = "trade" if not is_batch_event else "batch:trade"
            return TriggerEvent(self.time(), event_type, self._symb_to_instr.get(symbol), trade)
        return None

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

    def ohlc(self, instrument: str | Instrument, timeframe: str) -> OHLCV:
        return self._cache.get_ohlcv(instrument if isinstance(instrument, str) else instrument.symbol, timeframe)

    def _create_and_update_positions(self, instruments: list[Instrument]):
        for instrument in instruments:
            symb = instrument.symbol
            self.positions[symb] = self.trading_service.get_position(instrument)

            # - check if we need any aux instrument for calculating pnl ?
            # TODO: test edge cases for aux symbols
            aux = lookup.find_aux_instrument_for(instrument, self.trading_service.get_base_currency())
            if aux is not None:
                instrument._aux_instrument = aux
                instruments.append(aux)
                self.positions[aux.symbol] = self.trading_service.get_position(aux)

    def start(self, blocking: bool = False):
        if self._is_initilized:
            raise ValueError("Strategy is already started !")

        # - create incoming market data processing
        databus = self.broker_provider.get_communication_channel()
        databus.register(self)

        self.__add_instruments(self.instruments)

        # - initialize strategy (should we do that after any first market data received ?)
        if not self._is_initilized:
            try:
                self.strategy.on_start(self)
                self._is_initilized = True
            except Exception as strat_error:
                logger.error(
                    f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_start: {strat_error}"
                )
                logger.error(traceback.format_exc())
                return

        # - for live we run loop
        if not self.broker_provider.is_simulated_trading:
            self._thread_data_loop = Thread(target=self._process_incoming_data_loop, args=(databus,), daemon=True)
            self._thread_data_loop.start()
            logger.info("(StrategyContext) strategy is started in thread")
            if blocking:
                self._thread_data_loop.join()

    def stop(self):
        if self._thread_data_loop:
            self.broker_provider.get_communication_channel().stop()
            self._thread_data_loop.join()
            try:
                self.strategy.on_stop(self)
            except Exception as strat_error:
                logger.error(
                    f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_stop: {strat_error}"
                )
                logger.error(traceback.format_exc())
            self._thread_data_loop = None

        # - close logging
        self._logging.close()
        self.get_latencies_report()

    def time(self) -> dt_64:
        return self.trading_service.time()

    def _generate_order_client_id(self, symbol: str) -> str:
        self.__order_id += 1
        return self.__strategy_id + symbol + "_" + str(self.__order_id)

    @_SW.watch("StrategyContext")
    def trade(
        self,
        instr_or_symbol: Instrument | str,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **options,
    ) -> Order:
        instrument: Instrument | None = (
            self._symb_to_instr.get(instr_or_symbol) if isinstance(instr_or_symbol, str) else instr_or_symbol
        )
        if instrument is None:
            raise ValueError(f"Can't find instrument for symbol {instr_or_symbol}")

        # - adjust size
        size_adj = instrument.round_size_down(abs(amount))
        if size_adj < instrument.min_size:
            raise ValueError(f"Attempt to trade size {abs(amount)} less than minimal allowed {instrument.min_size} !")

        side = "buy" if amount > 0 else "sell"
        type = "market"
        if price is not None:
            price = instrument.round_price_down(price) if amount > 0 else instrument.round_price_up(price)
            type = "limit"
            if (stp_type := options.get("stop_type")) is not None:
                type = f"stop_{stp_type}"

        logger.debug(
            f"(StrategyContext) sending {type} {side} for {size_adj} of <green>{instrument.symbol}</green> @ {price} ..."
        )
        client_id = self._generate_order_client_id(instrument.symbol)

        order = self.trading_service.send_order(
            instrument, side, type, size_adj, price, time_in_force=time_in_force, client_id=client_id, **options
        )

        return order

    @_SW.watch("StrategyContext")
    def cancel(self, instr_or_symbol: Instrument | str):
        instrument: Instrument | None = (
            self._symb_to_instr.get(instr_or_symbol) if isinstance(instr_or_symbol, str) else instr_or_symbol
        )
        if instrument is None:
            raise ValueError(f"Can't find instrument for symbol {instr_or_symbol}")
        for o in self.trading_service.get_orders(instrument.symbol):
            self.trading_service.cancel_order(o.id)

    def cancel_order(self, order_id: str):
        if order_id:
            self.trading_service.cancel_order(order_id)

    def quote(self, symbol: str) -> Quote | None:
        return self.broker_provider.get_quote(symbol)

    def get_capital(self) -> float:
        return self.trading_service.get_capital()

    def get_reserved(self, instrument: Instrument) -> float:
        return self.trading_service.get_account().get_reserved(instrument)

    @_SW.watch("StrategyContext")
    def get_historical_ohlcs(self, instrument: Instrument | str, timeframe: str, length: int) -> OHLCV | None:
        """
        Helper for historical ohlc data
        """
        instr = self._symb_to_instr.get(instrument) if isinstance(instrument, str) else instrument

        if instr is None:
            logger.warning(f"Can't find instrument for {instrument} symbol !")
            return None

        # - first check if we can use cached series
        rc = self.ohlc(instr, timeframe)
        if len(rc) >= length:
            return rc

        # - send request for historical data
        bars = self.broker_provider.get_historical_ohlcs(instr.symbol, timeframe, length)
        r = self._cache.update_by_bars(instr.symbol, timeframe, bars)
        return r

    @_SW.watch("StrategyContext")
    def set_universe(self, instruments: list[Instrument]) -> None:
        for instr in instruments:
            self._symb_to_instr[instr.symbol] = instr

        new_set = set(instruments)
        prev_set = set(self.instruments)
        rm_instr = list(prev_set - new_set)
        add_instr = list(new_set - prev_set)

        self.__add_instruments(add_instr)
        self.__remove_instruments(rm_instr)

        if add_instr or rm_instr:
            self.strategy.on_universe_change(self, add_instr, rm_instr)

        # set new instruments
        self.instruments = instruments

    def __remove_instruments(self, instruments: list[Instrument]) -> None:
        """
        Remove symbols from universe. Steps:
        - close all open positions
        - unsubscribe from market data
        - remove from data cache

        We are still keeping the symbols in the positions dictionary.
        """
        # - close all open positions
        exit_targets = [
            TargetPosition.zero(self, instr.signal(0, group="Universe", comment="Universe change"))
            for instr in instruments
            if instr.symbol in self.positions and abs(self.positions[instr.symbol].quantity) > instr.min_size
        ]
        self.positions_gathering.alter_positions(self, exit_targets)

        # - if still open positions close them manually
        for instr in instruments:
            pos = self.positions.get(instr.symbol)
            if pos and abs(pos.quantity) > instr.min_size:
                self.trade(instr, -pos.quantity)

        # - unsubscribe from market data
        for instr in instruments:
            self.unsubscribe(self._market_data_subcription_type, instr)

        # - remove from data cache
        for instr in instruments:
            self._cache.remove(instr.symbol)

    def __add_instruments(self, instruments: list[Instrument]) -> None:
        # - create positions for instruments
        self._create_and_update_positions(instruments)

        # - get actual positions from exchange
        _symbols = []
        for instr in instruments:
            # process instruments - need to find convertors etc
            self._cache.init_ohlcv(instr.symbol)
            _symbols.append(instr.symbol)

        # - subscribe to market data
        logger.debug(
            f"(StrategyContext) Subscribing to {self._market_data_subcription_type} updates using {self._market_data_subcription_params} for \n\t{_symbols} "
        )
        self.broker_provider.subscribe(
            self._market_data_subcription_type, instruments, **self._market_data_subcription_params
        )

        # - initialize strategy loggers
        self._logging.initialize(self.time(), self.positions, self.trading_service.get_account().get_balances())

    @_SW.watch("StrategyContext")
    def subscribe(self, subscription_type: str, instr_or_symbol: Instrument | str, **kwargs) -> bool:
        """
        Subscribe to market data updates
        """
        instrument: Instrument | None = (
            self._symb_to_instr.get(instr_or_symbol) if isinstance(instr_or_symbol, str) else instr_or_symbol
        )
        if instrument is None:
            raise ValueError(f"Can't find instrument for symbol {instr_or_symbol}")

        return self.broker_provider.subscribe(subscription_type, [instrument], **kwargs)

    @_SW.watch("StrategyContext")
    def unsubscribe(self, subscription_type: str, instr_or_symbol: Instrument | str) -> bool:
        """
        Unsubscribe from market data updates
        """
        instrument: Instrument | None = (
            self._symb_to_instr.get(instr_or_symbol) if isinstance(instr_or_symbol, str) else instr_or_symbol
        )
        if instrument is None:
            raise ValueError(f"Can't find instrument for symbol {instr_or_symbol}")

        return self.broker_provider.unsubscribe(subscription_type, [instrument])

    def has_subscription(self, subscription_type: str, instr_or_symbol: Instrument | str) -> bool:
        """
        Check if subscription is active
        """
        instrument: Instrument | None = (
            self._symb_to_instr.get(instr_or_symbol) if isinstance(instr_or_symbol, str) else instr_or_symbol
        )
        if instrument is None:
            raise ValueError(f"Can't find instrument for symbol {instr_or_symbol}")

        return self.broker_provider.has_subscription(subscription_type, instrument)

    def _run_in_thread_pool(self, func: Callable, args=()):
        """
        For the simulation we don't need to call function in thread
        """
        if self.broker_provider.is_simulated_trading:
            func(*args)
        else:
            if self.__pool is None:
                self.__pool = ThreadPool(2)
            self.__pool.apply_async(func, args)
