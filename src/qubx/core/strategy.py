"""
 # All interfaces related to strategy etc
"""
from collections import defaultdict
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from qubx import lookup, logger
from qubx.core.lookups import InstrumentsLookup
from qubx.core.basics import Instrument, Position, Signal, TransactionCostsCalculator, dt_64, td_64, AsyncioThreadRunner, CtrlChannel
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


class IExchangeServiceProvider:

    def time(self) -> dt_64:
        """
        Returns current time
        """
        raise NotImplementedError("time is not implemented")

    def get_name(self) -> str:
        raise NotImplementedError("get_name is not implemented")

    def schedule_trigger(self, trigger_id: str, when: str):
        raise NotImplementedError("schedule_trigger is not implemented")

    def get_quote(self, symbol: str) -> Quote | None:
        pass

    def get_capital(self) -> float:
        raise NotImplementedError("get_capital is not implemented")

    def get_position(self, instrument: Instrument) -> Position:
        raise NotImplementedError("get_position is not implemented")

    def sync_position_and_orders(self, position: Position) -> Position:
        raise NotImplementedError("sync_position is not implemented")

    def get_base_currency(self) -> str:
        raise NotImplementedError("get_basic_currency is not implemented")


class IStrategy:
    ctx: 'StrategyContext'

    def on_start(self, ctx: 'StrategyContext'):
        pass

    def on_event(self, ctx: 'StrategyContext', event: TriggerEvent) -> Optional[List[Signal]]:
        return None

    def on_stop(self, ctx: 'StrategyContext'):
        pass


def _dict_with_exc(dct, f):
    if f not in dct:
        raise ValueError(f"Configuration {dct} must contain field '{f}'")
    return dct[f]


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

    _t_mdata_processor: Optional[AsyncioThreadRunner] = None
    _market_data_subcription_type:str = 'unknown'
    _market_data_subcription_params: dict = dict()

    _trig_interval_in_bar_nsec: int
    _trig_bar_freq_nsec: int
    _trig_on_bar: bool = False
    _trig_on_time: bool = False
    _trig_on_quote: bool = False
    _trig_on_trade: bool = False
    _trig_on_book: bool = False
    _current_bar_trigger_processed: bool = False
    _is_initilized: bool = False

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

        ) -> None:

        # - set parameters to strategy (not sure we will do it here)
        self.strategy = strategy
        if isinstance(strategy, type):
            self.strategy = strategy()
        self.populate_parameters_to_strategy(self.strategy, **config if config else {})

        # - other initialization
        self.exchange_service = exchange_service
        self.data_provider = data_provider
        self.config = config
        self.instruments = instruments
        self.positions = {}
 
        # - process trigger configuration
        self._check_how_to_trigger_strategy(trigger)

        # - process market data configuration
        self._check_how_to_listen_to_market_data(md_subscription)

        # - states 
        self._is_initilized = False

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

    async def _process_incoming_market_data(self, channel: CtrlChannel):
        _fails_counter = 0 
        logger.info("Start processing market data")

        while channel.control.is_set():
            _strategy_event: TriggerEvent = None

            # - waiting for incoming market data
            symbol, data = channel.queue.get()

            # - processing quote
            if isinstance(data, Quote):
                _strategy_event = self._update_ctx_by_quote(symbol, data)

            # - processing trade
            if isinstance(data, Trade):
                _strategy_event = self._update_ctx_by_trade(symbol, data)

            # - processing ohlc bar
            if isinstance(data, Bar):
                _strategy_event = self._update_ctx_by_bar(symbol, data)

            # - any events from exchange: may be timer, news, alt data, etc
            if isinstance(data, EventFromExchange):
                _strategy_event = self._update_ctx_by_bar(symbol, data)

            if _strategy_event:
                try:
                    self.strategy.on_event(self, _strategy_event)
                    _fails_counter = 0
                except Exception as strat_error:
                    # - TODO: probably we need some cooldown interval after exception to prevent flooding
                    logger.error(f"[{self.time()}]: Strategy {self.strategy.__class__.__name__} raised an exception: {strat_error}")
                    logger.error(traceback.format_exc())

                    #  - we stop execution after let's say maximal number of errors in a row
                    if (_fails_counter := _fails_counter + 1) >= StrategyContext.MAX_NUMBER_OF_FAILURES:
                        logger.error("STRATEGY FAILURES IN THE ROW EXCEEDED MAX ALLOWED NUMBER - STOPPING ...")
                        channel.stop()
                        break

        logger.info("Market data processing finished")

    def _update_ctx_by_bar(self, symbol: str, bar: Bar) -> TriggerEvent:
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

    def _update_ctx_by_trade(self, symbol: str, trade: Trade) -> TriggerEvent:
        if self._trig_on_trade:
            return TriggerEvent(self.time(), 'trade', symbol, trade)
        return None

    def _update_ctx_by_quote(self, symbol: str, quote: Quote) -> TriggerEvent:
        # - TODO: here we can apply throttlings or filters
        #  - let's say we can skip quotes if bid & ask is not changed
        #  - or we can collect let's say N quotes before sending to strategy
        if self._trig_on_quote:
            return TriggerEvent(self.time(), 'quote', symbol, quote)
        return None

    def _update_ctx_by_exchange_event(self, symbol: str, event: EventFromExchange) -> TriggerEvent:
        if self._trig_on_time and event.type == 'time':
            return TriggerEvent(self.time(), 'time', symbol, event)
        return None

    def ohlc(self, instrument: str | Instrument, timeframe: str) -> OHLCV:
        return self._cache.get_ohlcv(instrument if isinstance(instrument, str) else instrument.symbol, timeframe)

    def _create_synced_position(self):
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
        self._create_synced_position()

        # - get actual positions from exchange
        _symbols = []
        for instr in self.instruments:
            # process instruments - need to find convertors etc
            self._cache.init_ohlcv(instr.symbol)
            _symbols.append(instr.symbol)

        # - create incoming market data processing
        self._t_mdata_processor = AsyncioThreadRunner(self.data_provider.get_communication_channel())
        self._t_mdata_processor.add(self._process_incoming_market_data)

        # - subscribe to market data
        logger.info(f"Subscribing to {self._market_data_subcription_type} updates using {self._market_data_subcription_params} for \n\t{_symbols} ")
        self.data_provider.subscribe(self._market_data_subcription_type, _symbols, **self._market_data_subcription_params)

        # - initialize strategy
        if not self._is_initilized:
            try:
                self.strategy.on_start(self)
                self._is_initilized = True
            except Exception as strat_error:
                logger.error(f"Strategy {self.strategy.__class__.__name__} raised an exception in on_start: {strat_error}")
                logger.error(traceback.format_exc())
                return

        self._t_mdata_processor.start()
        logger.info("> Data processor started")
        if join:
            self._t_mdata_processor.join()

    def stop(self):
        if self._t_mdata_processor:
            self.data_provider.get_communication_channel().stop()
            self._t_mdata_processor.stop()
            try:
                self.strategy.on_stop(self)
            except Exception as strat_error:
                logger.error(f"Strategy {self.strategy.__class__.__name__} raised an exception in on_stop: {strat_error}")
                logger.error(traceback.format_exc())
            self._t_mdata_processor = None

    def populate_parameters_to_strategy(self, strategy: IStrategy, **kwargs):
        for k,v in kwargs.items():
            if k.startswith('_'):
                raise ValueError("Internal variable can't be set from external parameter !")
            if hasattr(strategy, k):
                strategy.__dict__[k] = v
                logger.info(f"Set {k} -> {v}")

    def time(self) -> dt_64:
        return self.exchange_service.time()

    def quote(self, symbol: str) -> Optional[Quote]:
        return self.exchange_service.get_quote(symbol)

    def get_capital(self) -> float:
        # TODO: here we need to get actual capital from account 
        # return min(self.capital, self.exchange_service.get_available_capital())
        return self.capital * self.leverage