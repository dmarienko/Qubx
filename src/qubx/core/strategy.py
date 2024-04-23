"""
 # All interfaces related to strategy etc
"""
from collections import defaultdict
from threading import Thread
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from types import FunctionType
import numpy as np
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from qubx import lookup, logger
from qubx.core.account import AccountProcessor
from qubx.core.loggers import ExecutionsLogger, LogsWriter, PortfolioLogger, PositionsDumper
from qubx.core.lookups import InstrumentsLookup
from qubx.core.basics import Deal, Instrument, Order, Position, Signal, dt_64, td_64, CtrlChannel
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV
from qubx.utils.time import convert_tf_str_td64


@dataclass
class EventFromExchange:
    time: dt_64
    type: str
    data: Optional[Any] 


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

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> List[Bar] | None:
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

    def schedule_trigger(self, trigger_id: str, when: str):
        raise NotImplementedError("schedule_trigger is not implemented")

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
        pass

    def on_event(self, ctx: 'StrategyContext', event: TriggerEvent) -> Optional[List[Signal]]:
        return None

    def on_stop(self, ctx: 'StrategyContext'):
        pass

    def tracker(self, ctx: 'StrategyContext') -> PositionsTracker | None:
        pass


def _dict_with_exc(dct, f):
    if f not in dct:
        raise ValueError(f"Configuration {dct} must contain field '{f}'")
    return dct[f]


def _round_down_at_min_qty(x: float, min_size: float) -> float:
    return (int(x / min_size)) * min_size


class CachedMarketDataHolder: 
    _min_timeframe: dt_64
    _last_bar: Dict[str, Optional[Bar]]
    _ohlcvs: Dict[str, Dict[td_64, OHLCV]]

    def __init__(self, minimal_timeframe: str) -> None:
        self._min_timeframe = convert_tf_str_td64(minimal_timeframe)
        self._ohlcvs = dict()
        self._last_bar = defaultdict(lambda: None)

    def init_ohlcv(self, symbol: str, max_size=np.inf):
        self._ohlcvs[symbol] = {self._min_timeframe: OHLCV(symbol, self._min_timeframe, max_size)}
    
    def get_ohlcv(self, symbol: str, timeframe: str, max_size=np.inf) -> OHLCV:
        tf = convert_tf_str_td64(timeframe) 

        if symbol not in self._ohlcvs:
           self._ohlcvs[symbol] = {}

        if tf not in self._ohlcvs[symbol]: 
            # - check requested timeframe
            new_ohlc = OHLCV(symbol, tf, max_size)
            if tf < self._min_timeframe:
                logger.warning(f"[{symbol}] Request for timeframe {timeframe} that is smaller then minimal {self._min_timeframe}")
            else:
                # - first try to resample from smaller frame
                if (basis := self._ohlcvs[symbol].get(self._min_timeframe)):
                    for b in basis[::-1]:
                        new_ohlc.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume, b.bought_volume)
                
            self._ohlcvs[symbol][tf] = new_ohlc

        return self._ohlcvs[symbol][tf]

    def update_by_bar(self, symbol: str, bar: Bar):
        _last_bar = self._last_bar[symbol]
        v_tot_inc = bar.volume
        v_buy_inc = bar.bought_volume

        if _last_bar is not None:
            if _last_bar.time == bar.time: # just current bar updated
                v_tot_inc -= _last_bar.volume
                v_buy_inc -= _last_bar.bought_volume

            if _last_bar.time > bar.time: # update is too late - skip it
                return

        if symbol in self._ohlcvs:
            self._last_bar[symbol] = bar
            for ser in self._ohlcvs[symbol].values():
                ser.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, v_tot_inc, v_buy_inc)

    def update_by_quote(self, symbol: str, quote: Quote):
        series = self._ohlcvs.get(symbol)
        if series:
            for ser in series.values():
                ser.update(quote.time, quote.mid_price(), 0)

    def update_by_trade(self, symbol: str, trade: Trade):
        series = self._ohlcvs.get(symbol)
        if series:
            total_vol = trade.size 
            bought_vol = total_vol if trade.taker >= 1 else 0.0 
            for ser in series.values():
                ser.update(trade.time, trade.price, total_vol, bought_vol)

 
class StrategyContext:
    MAX_NUMBER_OF_FAILURES = 10

    strategy: IStrategy 
    exchange_service: IExchangeServiceProvider
    data_provider: IDataProvider
    instruments: List[Instrument]
    positions: Dict[str, Position]

    # - loggers
    positions_dumper: PositionsDumper | None = None
    portfolio_logger: PortfolioLogger | None = None
    executions_logger: ExecutionsLogger | None = None

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

    _cache: CachedMarketDataHolder # market data cache

    def __init__(self, 
            # - strategy with parameters
            strategy: IStrategy, config: Dict[str, Any] | None,
            # - - - - - - - - - - - - - - - - - - - - -

            # - data provider and exchange service
            data_provider: IDataProvider,
            exchange_service: IExchangeServiceProvider, 
            instruments: List[Instrument],
            # - - - - - - - - - - - - - - - - - - - - -

            # - context's parameters - - - - - - - - - -
            trigger: Dict[str, Any] = dict(type='ohlc', timeframe='1Min', nback=60),
            md_subscription: Dict[str,Any] = dict(type='ohlc', timeframe='1Min'),
            # - - - - - - - - - - - - - - - - - - - - -

            # - how to write logs - - - - - - - - - -
            logs_writer: LogsWriter | None = None,
            positions_log_freq: str = '1Min',
            portfolio_log_freq: str = '5Min',
            num_exec_records_to_write = 1      # in live let's write every execution 
        ) -> None:

        # - set parameters to strategy (not sure we will do it here)
        self.strategy = strategy
        if isinstance(strategy, type):
            self.strategy = strategy()
        self.strategy.ctx = self

        # TODO: - trackers - - - - - - - - - - - - -
        # - here we need to instantiate trackers 
        # - need to think how to do it properly !!!

        # - set strategy custom parameters 
        self.populate_parameters_to_strategy(self.strategy, **config if config else {})

        # - other initialization
        self.exchange_service = exchange_service
        self.data_provider = data_provider
        self.config = config
        self.instruments = instruments
        self.positions = {}

        # - for fast access to instrument by it's symbol
        self._symb_to_instr = {i.symbol: i for i in instruments}
 
        # - process trigger configuration
        self._check_how_to_trigger_strategy(trigger)

        # - process market data configuration
        self._check_how_to_listen_to_market_data(md_subscription)

        # - instantiate loggers
        if logs_writer:
            if positions_log_freq:
                # - store current positions
                self.positions_dumper = PositionsDumper(logs_writer, positions_log_freq)

            if portfolio_log_freq:
                # - store portfolio log
                self.portfolio_logger = PortfolioLogger(logs_writer, portfolio_log_freq)

            # - store executions
            if num_exec_records_to_write >= 1:
                self.executions_logger = ExecutionsLogger(logs_writer, num_exec_records_to_write)

        # - states 
        self._is_initilized = False
        self.__strategy_id = self.strategy.__class__.__name__ + "_"
        self.__order_id = self.time().item() // 100_000_000

        # - extract data and event handlers
        self.__handlers = {
            n.split('_processing_')[1]: f for n, f in self.__class__.__dict__.items() 
            if type(f) == FunctionType and n.startswith('_processing_') 
        }

    def _check_how_to_listen_to_market_data(self, md_config: dict):
        self._market_data_subcription_type = _dict_with_exc(md_config, 'type').lower()
        match self._market_data_subcription_type:
            case 'ohlc':
                timeframe = _dict_with_exc(md_config, 'timeframe')
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

    def _check_how_to_trigger_strategy(self, trigger_config: dict):
        # - check how it's configured to be triggered
        self._trig_interval_in_bar_nsec = 0

        match (_trigger := _dict_with_exc(trigger_config, 'type').lower()):
            case 'bar': 
                self._trig_on_bar = True

                _bar_timeframe = pd.Timedelta(_dict_with_exc(trigger_config, 'timeframe'))
                _inside_bar_delay = pd.Timedelta(_dict_with_exc(trigger_config, 'delay'))

                if abs(pd.Timedelta(_inside_bar_delay)) > pd.Timedelta(_bar_timeframe):
                    raise ValueError(f"Delay must be less or equal to bar's timeframe for {_trigger} trigger: you set delay {_inside_bar_delay} for {_bar_timeframe}")

                # for positive delay - trigger strategy when this interval passed after new bar's open
                if _inside_bar_delay >= pd.Timedelta(0): 
                    self._trig_interval_in_bar_nsec = _inside_bar_delay.asm8.item()
                # for negative delay - trigger strategy when time is closer to bar's closing time more than this interval
                else:
                    self._trig_interval_in_bar_nsec = (_bar_timeframe + _inside_bar_delay).asm8.item()
                self._trig_bar_freq_nsec = _bar_timeframe.asm8.item()

            case 'time': 
                self._trig_on_time = True
                _time_to_trigger = _dict_with_exc(trigger_config, 'when')

                # - schedule periodic timer
                self.exchange_service.schedule_trigger('time', _time_to_trigger)

            case 'quote': 
                self._trig_on_quote = True
                raise ValueError(f"{_trigger} NOT IMPLEMENTED")

            case 'trade': 
                self._trig_on_trade = True
                raise ValueError(f"{_trigger} NOT IMPLEMENTED")

            case 'orderbook' | 'ob': 
                self._trig_on_book = True
                raise ValueError(f"{_trigger} NOT IMPLEMENTED")

            case _: 
                raise ValueError(f"Wrong trigger type {_trigger}")

    def _process_incoming_data(self, channel: CtrlChannel):
        _fails_counter = 0 
        logger.info("(StrategyContext) Start processing market data")

        while channel.control.is_set():
            # - waiting for incoming market data
            symbol, d_type, data = channel.queue.get()

            # - process data if handler is registered
            handler = self.__handlers.get(d_type)
            _strategy_trigger_event = handler(self, symbol, data) if handler else None
            if _strategy_trigger_event:
                try:
                    self.strategy.on_event(self, _strategy_trigger_event)
                    _fails_counter = 0
                except Exception as strat_error:
                    # - TODO: probably we need some cooldown interval after exception to prevent flooding
                    logger.error(f"[{self.time()}]: Strategy {self.strategy.__class__.__name__} raised an exception: {strat_error}")
                    logger.opt(colors=False).error(traceback.format_exc())

                    #  - we stop execution after let's say maximal number of errors in a row
                    if (_fails_counter := _fails_counter + 1) >= StrategyContext.MAX_NUMBER_OF_FAILURES:
                        logger.error("STRATEGY FAILURES IN THE ROW EXCEEDED MAX ALLOWED NUMBER - STOPPING ...")
                        channel.stop()
                        break

                # - notify poition and portfolio loggers
                self._notify_loggers()

        logger.info("(StrategyContext) Market data processing stopped")

    def _notify_loggers(self):
        # - notify position and loggers
        if self.positions_dumper:
            self.positions_dumper.store(self.time())

        if self.portfolio_logger:
            self.portfolio_logger.store(self.time())

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

    def _processing_bar(self, symbol: str, bar: Bar) -> TriggerEvent | None:
        # - processing current bar's update
        self._cache.update_by_bar(symbol, bar)
        if self._trig_on_bar:
            t = self.exchange_service.time().item()
            _time_to_trigger = t % self._trig_bar_freq_nsec >= self._trig_interval_in_bar_nsec
            if _time_to_trigger: 
                # we want to trigger only first one - not every
                if not self._current_bar_trigger_processed:
                    self._current_bar_trigger_processed = True
                    return TriggerEvent(self.time(), 'bar', symbol, bar)
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

    def _processing_event(self, symbol: str, event: EventFromExchange) -> TriggerEvent | None:
        if self._trig_on_time and event.type == 'time':
            return TriggerEvent(self.time(), 'time', symbol, event)
        return None

    def _processing_order(self, symbol: str, order: Order) -> TriggerEvent | None:
        logger.debug(f"[{order.id} / {order.client_id}] : {order.type} {order.side} {order.quantity} of {symbol} { (' @ ' + str(order.price)) if order.price else '' } -> [{order.status}]")
        # - check if we want to trigger any strat's logic on order
        return None

    def _processing_deals(self, symbol: str, deals: List[Deal]) -> TriggerEvent | None:
        if self.executions_logger:
            self.executions_logger.record_deals(symbol, deals)
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

    def start(self, join=False):
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

        # - attach positions to loggers
        if self.positions_dumper:
            self.positions_dumper.attach_positions(*list(self.positions.values()))

        if self.portfolio_logger:
            self.portfolio_logger.attach_positions(*list(self.positions.values()))

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
        logger.info("(StrategyContext) Market data processor started")
        # if join:
            # self._thread_data_loop.join()

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

        # - close 
        if self.portfolio_logger:
            self.portfolio_logger.close()

        if self.executions_logger:
            self.executions_logger.close()

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

    def trade(self, instr_or_symbol: Instrument | str, amount:float, price: float|None=None, time_in_force='gtc') -> Order:
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