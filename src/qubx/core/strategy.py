"""
 # All interfaces related to strategy etc
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import FunctionType
from collections import defaultdict
from dataclasses import dataclass
from threading import Thread
from multiprocessing.pool import ThreadPool
import traceback

import pandas as pd

from qubx import lookup, logger
from qubx.core.account import AccountProcessor
from qubx.core.helpers import BasicScheduler, CachedMarketDataHolder, process_schedule_spec
from qubx.core.loggers import LogsWriter, StrategyLogging
from qubx.core.lookups import InstrumentsLookup
from qubx.core.basics import Deal, Instrument, Order, Position, Signal, dt_64, td_64, CtrlChannel
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_seconds_to_str


@dataclass
class TriggerEvent:
    time: dt_64
    type: str
    instrument: Optional[Instrument]
    data: Optional[Any] 


class IDataProvider:

    def subscribe(self, subscription_type: str, symbols: List[str], **kwargs) -> bool:
        raise NotImplementedError("subscribe")

    def get_communication_channel(self) -> CtrlChannel:
        raise NotImplementedError("get_communication_channel")

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> List[Bar]:
        raise NotImplementedError("get_historical_ohlcs")

    def get_quote(self, symbol: str) -> Quote | None:
        raise NotImplementedError("get_quote")


class IExchangeServiceProvider:
    acc: AccountProcessor

    def time(self) -> dt_64:
        """
        Returns current time
        """
        raise NotImplementedError("time is not implemented")

    def get_account(self) -> AccountProcessor:
        return self.acc

    def get_name(self) -> str:
        raise NotImplementedError("get_name is not implemented")

    def get_capital(self) -> float:
        return self.acc.get_capital()

    def send_order(self, instrument: Instrument, order_side: str, order_type: str, amount: float, price: float | None = None, 
        client_id: str | None = None, time_in_force: str='gtc') -> Order | None:
        raise NotImplementedError("send_order is not implemented")

    def cancel_order(self, order_id: str) -> Order | None:
        raise NotImplementedError("cancel_order is not implemented")

    def get_orders(self, symbol: str | None = None) -> List[Order]:
        raise NotImplementedError("get_orders is not implemented")

    def get_position(self, instrument: Instrument) -> Position:
        raise NotImplementedError("get_position is not implemented")

    def get_base_currency(self) -> str:
        raise NotImplementedError("get_basic_currency is not implemented")


class PositionsTracker:
    ctx: 'StrategyContext'

    def __init__(self, ctx: 'StrategyContext') -> None:
        self.ctx = ctx
    

class IStrategy:
    ctx: 'StrategyContext'

    def on_start(self, ctx: 'StrategyContext'):
        """
        This method is called strategy is started
        """
        pass

    def on_fit(self, ctx: 'StrategyContext', fit_time: str | pd.Timestamp, previous_fit_time: str | pd.Timestamp | None = None):
        """
        This method is called when it's time to fit model
        :param fit_time: last time of fit data to use 
        :param previous_fit_time: last time of fit data used in previous fit call 
        """
        return None

    def on_event(self, ctx: 'StrategyContext', event: TriggerEvent) -> Optional[List[Signal]]:
        return None

    def on_stop(self, ctx: 'StrategyContext'):
        pass

    def tracker(self, ctx: 'StrategyContext') -> PositionsTracker | None:
        pass


def _dict_with_exception(dct, f):
    if f not in dct:
        raise ValueError(f"Configuration {dct} must contain field '{f}'")
    return dct[f]


def _round_down_at_min_qty(x: float, min_size: float) -> float:
    return (int(x / min_size)) * min_size


_SW = Stopwatch()

 
class StrategyContext:
    MAX_NUMBER_OF_STRATEGY_FAILURES = 10

    strategy: IStrategy                         # strategy instance
    exchange_service: IExchangeServiceProvider  # service for exchange API: orders managemewnt
    data_provider: IDataProvider                # market data provider 
    instruments: List[Instrument]               # list of instruments this strategy trades
    positions: Dict[str, Position]              # positions of the strategy (instrument -> position)

    # - loggers
    _logging: StrategyLogging                    # recording all activities for the strat: execs, positions, portfolio

    # - cached marked data anb scheduler
    _cache: CachedMarketDataHolder # market data cache
    _scheduler: BasicScheduler

    # - configuration
    _market_data_subcription_type:str = 'unknown'
    _market_data_subcription_params: dict = dict()
    _thread_data_loop: Thread | None = None            # market data loop

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
    __handlers: Dict[str, Callable[['StrategyContext', str, Any], TriggerEvent | None]]
    __fit_is_running: bool = False                       # during fitting working it stops calling on_event method
    __init_fit_was_called: bool = False                  # true if initial fit was already run
    __init_fit_args: Tuple = (None, None)                # arguments for initial on_fit() method call
    __pool: ThreadPool | None = None                     # thread pool used for running aux tasks (fit etc)

    def __init__(self, 
            # - strategy with parameters
            strategy: IStrategy, config: Dict[str, Any] | None,
            # - - - - - - - - - - - - - - - - - - - - -

            # - data provider and exchange service
            data_provider: IDataProvider,
            exchange_service: IExchangeServiceProvider, 
            instruments: List[Instrument],
            # - - - - - - - - - - - - - - - - - - - - -

            # - data subscription - - - - - - - - - - -
            md_subscription: Dict[str,Any] = dict(type='ohlc', timeframe='1Min', nback=60),
            # - - - - - - - - - - - - - - - - - - - - -

            # - when need to trigger and fit strategy - - - - - - -
            trigger_spec: str = 'bar: -1Sec',       # 1 sec before subscription bar is closed
            fit_spec: str | None = None,
            # - - - - - - - - - - - - - - - - - - - - -

            # - how to write logs - - - - - - - - - -
            logs_writer: LogsWriter | None = None,
            positions_log_freq: str = '1Min',
            portfolio_log_freq: str = '5Min',
            num_exec_records_to_write = 1      # in live let's write every execution 
        ) -> None:

        # - initialization
        self.exchange_service = exchange_service
        self.data_provider = data_provider
        self.config = config
        self.instruments = instruments
        self.positions = {}
        self.__fit_is_running = False         
        self.__init_fit_was_called = False     
        self.__pool = None 

        # - for fast access to instrument by it's symbol
        self._symb_to_instr = {i.symbol: i for i in instruments}

        # - create scheduler
        self._scheduler = BasicScheduler(self.data_provider.get_communication_channel(), lambda: self.time().item())

        # - instantiate logging functional
        self._logging = StrategyLogging(logs_writer, positions_log_freq, portfolio_log_freq, num_exec_records_to_write)

        # - extract data and event handlers
        self.__handlers = {
            n.split('_processing_')[1]: f for n, f in self.__class__.__dict__.items() 
            if type(f) == FunctionType and n.startswith('_processing_') 
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

        # TODO: - trackers - - - - - - - - - - - - -
        # - here we need to instantiate trackers 
        # - need to think how to do it properly !!!
        # TODO: - trackers - - - - - - - - - - - - -

        # - set strategy custom parameters 
        self.populate_parameters_to_strategy(self.strategy, **config if config else {})
        self._is_initilized = False
        self.__strategy_id = self.strategy.__class__.__name__ + "_"
        self.__order_id = self.time().item() // 100_000_000

    def __check_how_to_listen_to_market_data(self, md_config: dict):
        self._market_data_subcription_type = _dict_with_exception(md_config, 'type').lower()
        match self._market_data_subcription_type:
            case 'ohlc':
                timeframe = _dict_with_exception(md_config, 'timeframe')
                self._market_data_subcription_params = {
                    'timeframe': timeframe,
                    'nback': md_config.get('nback', 1), 
                }
                self._cache = CachedMarketDataHolder(timeframe) 

            case 'trade' | 'trades' | 'tas':
                self._cache = CachedMarketDataHolder('1Sec') 

            case 'quote' | 'quotes':
                self._cache = CachedMarketDataHolder('1Sec') 

            case 'ob' | 'orderbook':
                self._cache = CachedMarketDataHolder('1Sec') 

            case _:
                raise ValueError(f"{self._market_data_subcription_type} is not a valid value for market data subcription type !!!")

    def __check_how_to_trigger_and_fit_strategy(self, trigger_schedule: str | None, fit_schedue: str | None):
        _td2ns = lambda x: x.as_unit('ns').asm8.item()

        self._trig_interval_in_bar_nsec = 0
        
        if not trigger_schedule:
            raise ValueError("trigger parameter can't be empty !")
        t_rules = process_schedule_spec(trigger_schedule)
        f_rules = process_schedule_spec(fit_schedue)

        if t_rules is None:
            raise ValueError(f"Couldn't recognize 'trigger' parameter specification: {trigger_schedule} !")

        # - we can use it as reference if bar timeframe is not secified
        _default_data_timeframe = pd.Timedelta(self._cache.default_timeframe)

        # - check trigger spec - - - - -
        match t_rules['type']:
            # it triggers on_event method when new bar is formed.
            # in this case it won't arm scheduler but just ruled by actual market data updates 
            # - TODO: so probably need to drop this in favor to cron tasks
            # it's also possible to specify delay
            # "bar: -1s"      - it uses default timeframe and wake up 1 sec before every bar's close
            # "bar.5m: -5sec" - 5 sec before 5min bar's close
            # "bar.5m: 1sec"  - 1 sec after 5min bar closed (in next bar)
            case 'bar':
                _r_tf = t_rules.get('timeframe')
                _bar_timeframe = _default_data_timeframe if not _r_tf else pd.Timedelta(_r_tf)
                _inside_bar_delay: pd.Timedelta = t_rules.get('delay', pd.Timedelta(0))

                if abs(pd.Timedelta(_inside_bar_delay)) > pd.Timedelta(_bar_timeframe):
                    raise ValueError(f"Delay must be less or equal to bar's timeframe for bar trigger: you used {_inside_bar_delay} delay for {_bar_timeframe}")

                # for positive delay - trigger strategy when this interval passed after new bar's open
                if _inside_bar_delay >= pd.Timedelta(0): 
                    self._trig_interval_in_bar_nsec = _td2ns(_inside_bar_delay)

                # for negative delay - trigger strategy when time is closer to bar's closing time more than this interval
                else:
                    self._trig_interval_in_bar_nsec = _td2ns(_bar_timeframe + _inside_bar_delay)

                self._trig_bar_freq_nsec = _td2ns(_bar_timeframe)
                self._trig_on_bar = True

                logger.debug(f"Triggering strategy on every {convert_seconds_to_str(self._trig_bar_freq_nsec/1e9)} bar after {convert_seconds_to_str(self._trig_interval_in_bar_nsec/1e9)}")

            case 'cron':
                if 'schedule' not in t_rules:
                    raise ValueError(f"cron trigger type is specified but cron schedule not found !")

                self._scheduler.schedule_event(t_rules['schedule'], 'time_event')

            case 'quote': 
                self._trig_on_quote = True
                raise ValueError(f"quote trigger NOT IMPLEMENTED YET")

            case 'trade': 
                self._trig_on_trade = True
                raise ValueError(f"trade trigger NOT IMPLEMENTED YET")

            case 'orderbook' | 'ob': 
                self._trig_on_book = True
                raise ValueError(f"orderbook trigger NOT IMPLEMENTED YET")

            case _: 
                raise ValueError(f"Wrong trigger type {t_rules['type']}")

        # - check fit spec - - - - -
        _last_fit_data_can_be_used = pd.Timestamp(self.time())
        match f_rules.get('type'):
            case 'cron':
                if 'schedule' not in f_rules:
                    raise ValueError(f"cron fit trigger type is specified but cron schedule not found !")

                self._scheduler.schedule_event(f_rules['schedule'], 'fit_event')
                _last_fit_data_can_be_used = self._scheduler.get_event_last_time('fit_event')

            case 'bar':
                raise ValueError("Raw 'bar' type is not supported for fitting spec yet, please use cron type !")

            case _: 
                # if schedule is not specified just do not arm the task 
                # only initial fit will be called
                pass

        # - we can't just call on_fit right now because not all market data may be ready
        # - so we just mark it as not called yet
        self.__init_fit_was_called = False
        self.__init_fit_args = (None, _last_fit_data_can_be_used) 
   
    def _process_incoming_data(self, channel: CtrlChannel):
        _fails_counter = 0 
        logger.info("(StrategyContext) Start processing market data")

        while channel.control.is_set():
            # - start loop latency measurement 
            _SW.start('StrategyContext._process_incoming_data')

            # - waiting for incoming market data
            symbol, d_type, data = channel.queue.get()

            # - process data if handler is registered
            handler = self.__handlers.get(d_type)
            _SW.start('StrategyContext.handler')
            _strategy_trigger_on_event = handler(self, symbol, data) if handler else None
            _SW.stop('StrategyContext.handler')

            # - check if it still didn't call on_fit() for first time
            if not self.__init_fit_was_called:
                self._processing_fit_event(None, self.__init_fit_args)

            if _strategy_trigger_on_event:

                # - if fit was not called - skip on_event call
                if not self.__init_fit_was_called:
                    _SW.stop('StrategyContext._process_incoming_data')
                    logger.warning(f"[{self.time()}] {self.strategy.__class__.__name__}::on_event() is SKIPPED for now because on_fit() was not called yet !")
                    continue

                # - if strategy still fitting - skip on_event call
                if self.__fit_is_running:
                    _SW.stop('StrategyContext._process_incoming_data')
                    logger.warning(f"[{self.time()}] {self.strategy.__class__.__name__}::on_event() is SKIPPED for now because is being still fitting !")
                    continue

                try:
                    _SW.start('strategy.on_event')
                    self.strategy.on_event(self, _strategy_trigger_on_event)
                    _fails_counter = 0
                except Exception as strat_error:
                    # - probably we need some cooldown interval after exception to prevent flooding
                    logger.error(f"[{self.time()}]: Strategy {self.strategy.__class__.__name__} raised an exception: {strat_error}")
                    logger.opt(colors=False).error(traceback.format_exc())

                    #  - we stop execution after let's say maximal number of errors in a row
                    if (_fails_counter := _fails_counter + 1) >= StrategyContext.MAX_NUMBER_OF_STRATEGY_FAILURES:
                        logger.error("STRATEGY FAILURES IN THE ROW EXCEEDED MAX ALLOWED NUMBER - STOPPING ...")
                        channel.stop()
                        break
                finally:
                    _SW.stop('strategy.on_event')

            # - stop loop latency measurement 
            _SW.stop('StrategyContext._process_incoming_data')

            # - notify poition and portfolio loggers
            self._logging.notify(self.time())

        _SW.stop('StrategyContext._process_incoming_data')
        logger.info("(StrategyContext) Market data processing stopped")

    def _invoke_on_fit(self, current_fit_time: str | pd.Timestamp, prev_fit_time: str | pd.Timestamp | None):
        try:
            self.__fit_is_running = True
            logger.debug(f"[{self.time()}]: Invoking {self.strategy.__class__.__name__} on_fit('{current_fit_time}', '{prev_fit_time}')")
            _SW.start('strategy.on_fit')
            self.strategy.on_fit(self, current_fit_time, prev_fit_time)
            logger.debug(f"[{self.time()}]: {self.strategy.__class__.__name__} is fitted")
        except Exception as strat_error:
            logger.error(f"[{self.time()}]: Strategy {self.strategy.__class__.__name__} on_fit('{current_fit_time}', '{prev_fit_time}') raised an exception: {strat_error}")
            logger.opt(colors=False).error(traceback.format_exc())
        finally:
            self.__fit_is_running = False
            self.__init_fit_was_called = True
            _SW.stop('strategy.on_fit')
        return None

    def _processing_time_event(self, symbol: str, data: Any) -> TriggerEvent | None:
        """
        When scheduled time event is happened - we need to invoke strategy on_event method
        """
        return TriggerEvent(self.time(), 'time', None, data)

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
        self._run_in_thread_pool(self._invoke_on_fit, (
            pd.Timestamp(now_fit_time, unit='s'), 
            pd.Timestamp(prev_fit_time, unit='s') if prev_fit_time else None
        ))

        return None

    def _processing_hist_bar(self, symbol: str, bar: Bar) -> TriggerEvent | None:
        # - processing single historical bar
        #   here it just updates cache - historical bar can't trigger strategy logic
        self._cache.update_by_bar(symbol, bar)
        return None

    def _processing_hist_bars(self, symbol: str, bars: List[Bar]) -> TriggerEvent | None:
        # - processing historical bars as list
        for b in bars:
            self._processing_hist_bar(symbol, b)
        return None

    @_SW.watch('StrategyContext')
    def _processing_bar(self, symbol: str, bar: Bar) -> TriggerEvent | None:
        # - processing current bar's update
        self._cache.update_by_bar(symbol, bar)

        # - check if it's time to trigger the on_event if it's configured
        if self._trig_on_bar:
            t = self.exchange_service.time().item()
            _time_to_trigger = t % self._trig_bar_freq_nsec >= self._trig_interval_in_bar_nsec
            if _time_to_trigger: 
                # we want to trigger only first one - not every
                if not self._current_bar_trigger_processed:
                    self._current_bar_trigger_processed = True
                    return TriggerEvent(self.time(), 'bar', self._symb_to_instr.get(symbol), bar)
            else:
                self._current_bar_trigger_processed = False
        return None

    def _processing_trade(self, symbol: str, trade: Trade) -> TriggerEvent | None:
        if self._trig_on_trade:
            return TriggerEvent(self.time(), 'trade', symbol, trade)
        return None

    def _processing_quote(self, symbol: str, quote: Quote) -> TriggerEvent | None:
        # - TODO: here we can apply throttlings or filters
        #  - let's say we can skip quotes if bid & ask is not changed
        #  - or we can collect let's say N quotes before sending to strategy
        if self._trig_on_quote:
            return TriggerEvent(self.time(), 'quote', symbol, quote)
        return None

    @_SW.watch('StrategyContext')
    def _processing_order(self, symbol: str, order: Order) -> TriggerEvent | None:
        logger.debug(f"[{order.id} / {order.client_id}] : {order.type} {order.side} {order.quantity} of {symbol} { (' @ ' + str(order.price)) if order.price else '' } -> [{order.status}]")
        # - check if we want to trigger any strat's logic on order
        return None

    @_SW.watch('StrategyContext')
    def _processing_deals(self, symbol: str, deals: List[Deal]) -> TriggerEvent | None:
        # - log deals in storage
        self._logging.save_deals(symbol, deals)
        for d in deals:
            logger.debug(f"Executed {d.amount} @ {d.price} of {symbol} for order {d.order_id}")
        return None

    def ohlc(self, instrument: str | Instrument, timeframe: str) -> OHLCV:
        return self._cache.get_ohlcv(instrument if isinstance(instrument, str) else instrument.symbol, timeframe)

    def _create_and_update_positions(self):
        for instrument in self.instruments:
            symb = instrument.symbol
            self.positions[symb] = self.exchange_service.get_position(instrument)
            # - check if we need any aux instrument for calculating pnl ?
            aux = lookup.find_aux_instrument_for(instrument, self.exchange_service.get_base_currency())
            if aux is not None:
                instrument._aux_instrument = aux
                self.instruments.append(aux)
                self.positions[aux.symbol] = self.exchange_service.get_position(aux)

    def start(self):
        if self._is_initilized :
            raise ValueError("Strategy is already started !")

        # - create positions for instruments
        self._create_and_update_positions()

        # - get actual positions from exchange
        _symbols = []
        for instr in self.instruments:
            # process instruments - need to find convertors etc
            self._cache.init_ohlcv(instr.symbol)
            _symbols.append(instr.symbol)

        # - create incoming market data processing
        # self._thread_data_loop = AsyncioThreadRunner(self.data_provider.get_communication_channel())
        self._thread_data_loop = Thread(
            target=self._process_incoming_data, 
            args=(self.data_provider.get_communication_channel(),), 
            daemon=True
        )
        # self._thread_data_loop.add(self._process_incoming_market_data)

        # - subscribe to market data
        logger.info(f"(StrategyContext) Subscribing to {self._market_data_subcription_type} updates using {self._market_data_subcription_params} for \n\t{_symbols} ")
        self.data_provider.subscribe(self._market_data_subcription_type, _symbols, **self._market_data_subcription_params)

        # - initialize strategy loggers
        self._logging.initialize(
            self.time(), self.positions, self.exchange_service.get_account().get_balances()
        )

        # - initialize strategy (should we do that after any first market data received ?)
        if not self._is_initilized:
            try:
                self.strategy.on_start(self)
                self._is_initilized = True
            except Exception as strat_error:
                logger.error(f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_start: {strat_error}")
                logger.error(traceback.format_exc())
                return

        self._thread_data_loop.start()
        logger.info("(StrategyContext) strategy is started")

    def stop(self):
        if self._thread_data_loop:
            self.data_provider.get_communication_channel().stop()
            self._thread_data_loop.join()
            try:
                self.strategy.on_stop(self)
            except Exception as strat_error:
                logger.error(f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_stop: {strat_error}")
                logger.error(traceback.format_exc())
            self._thread_data_loop = None

        # - close logging
        self._logging.close()
        self.get_latencies_report()

    def get_latencies_report(self):
        for l in _SW.latencies.keys():
            logger.info(f"\t<w>{l}</w> took <r>{_SW.latency_sec(l):.7f}</r> secs")

    def populate_parameters_to_strategy(self, strategy: IStrategy, **kwargs):
        _log_info = ""
        for k,v in kwargs.items():
            if k.startswith('_'):
                raise ValueError("Internal variable can't be set from external parameter !")
            if hasattr(strategy, k):
                strategy.__dict__[k] = v
                _log_info += f"\n\tset <green>{k}</green> <- <red>{v}</red>"
        if _log_info:
            logger.info("(StrategyContext) set strategy parameters:" + _log_info)

    def time(self) -> dt_64:
        return self.exchange_service.time()

    def _generate_order_client_id(self, symbol: str) -> str:
        self.__order_id += 1
        return self.__strategy_id + symbol + '_' + str(self.__order_id)

    @_SW.watch('StrategyContext')
    def trade(self, instr_or_symbol: Instrument | str, amount:float, price: float|None=None, time_in_force='gtc') -> Order:
        _SW.start('send_order')
        instrument: Instrument | None = self._symb_to_instr.get(instr_or_symbol) if isinstance(instr_or_symbol, str) else instr_or_symbol
        if instrument is None:
            raise ValueError(f"Can't find instrument for symbol {instr_or_symbol}")

        # - adjust size
        size_adj = _round_down_at_min_qty(abs(amount), instrument.min_size)
        if size_adj == 0:
            raise ValueError(f"Attempt to trade size {abs(amount)} less than minimal allowed {instrument.min_size} !")

        side = 'buy' if amount > 0 else 'sell'
        type = 'limit' if price is not None else 'market'
        logger.info(f"(StrategyContext) sending {type} {side} for {size_adj} of {instrument.symbol} ...")
        client_id = self._generate_order_client_id(instrument.symbol)
        order = self.exchange_service.send_order(instrument, side, type, size_adj, price, time_in_force=time_in_force, client_id=client_id) 

        return order

    @_SW.watch('StrategyContext')
    def cancel(self, instr_or_symbol: Instrument | str):
        instrument: Instrument | None = self._symb_to_instr.get(instr_or_symbol) if isinstance(instr_or_symbol, str) else instr_or_symbol
        if instrument is None:
            raise ValueError(f"Can't find instrument for symbol {instr_or_symbol}")
        for o in self.exchange_service.get_orders(instrument.symbol):
            self.exchange_service.cancel_order(o.id)

    def quote(self, symbol: str) -> Quote | None:
        return self.data_provider.get_quote(symbol)

    def get_capital(self) -> float:
        return self.exchange_service.get_capital()

    def get_reserved(self, instrument: Instrument) -> float:
        return self.exchange_service.get_account().get_reserved(instrument)

    @_SW.watch('StrategyContext')
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
        bars = self.data_provider.get_historical_ohlcs(instr.symbol, timeframe, length) 
        r = self._cache.update_by_bars(instr.symbol, timeframe, bars)
        return r 

    def _run_in_thread_pool(self, func: Callable, args=()):
        """
        For backtester we need to override this method and just call function
        """
        if self.__pool is None:
            self.__pool = ThreadPool(2)
        self.__pool.apply_async(func, args)


if __name__ == '__main__':
    from qubx.utils.runner import create_strategy_context
    import time, sys
    filename, accounts, paths = '../experiments/configs/zero_test_scheduler.yaml', "../experiments/configs/.env", ['../']

    # - create context
    ctx = create_strategy_context(filename, accounts, paths)
    if ctx is None:
        sys.exit(0)

    # - run main loop
    try:
        ctx.start()

        # - just wake up every 60 sec and check if it's OK 
        while True: 
            time.sleep(60)

    except KeyboardInterrupt:
        ctx.stop()
        time.sleep(1)
        sys.exit(0)