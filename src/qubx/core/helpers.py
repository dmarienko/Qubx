from collections import defaultdict
from typing import Dict, Optional
import numpy as np
from croniter import croniter

from qubx import logger
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_tf_str_td64
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV


_SW = Stopwatch()

class CachedMarketDataHolder: 
    """
    Collected cached data updates from StrategyContext
    """
    _min_timeframe: np.timedelta64
    _last_bar: Dict[str, Optional[Bar]]
    _ohlcvs: Dict[str, Dict[np.timedelta64, OHLCV]]

    def __init__(self, minimal_timeframe: str) -> None:
        self._min_timeframe = convert_tf_str_td64(minimal_timeframe)
        self._ohlcvs = dict()
        self._last_bar = defaultdict(lambda: None)

    def init_ohlcv(self, symbol: str, max_size=np.inf):
        self._ohlcvs[symbol] = {self._min_timeframe: OHLCV(symbol, self._min_timeframe, max_size)}
    
    @_SW.watch('CachedMarketDataHolder')
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

    @_SW.watch('CachedMarketDataHolder')
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

    @_SW.watch('CachedMarketDataHolder')
    def update_by_quote(self, symbol: str, quote: Quote):
        series = self._ohlcvs.get(symbol)
        if series:
            for ser in series.values():
                ser.update(quote.time, quote.mid_price(), 0)

    @_SW.watch('CachedMarketDataHolder')
    def update_by_trade(self, symbol: str, trade: Trade):
        series = self._ohlcvs.get(symbol)
        if series:
            total_vol = trade.size 
            bought_vol = total_vol if trade.taker >= 1 else 0.0 
            for ser in series.values():
                ser.update(trade.time, trade.price, total_vol, bought_vol)


class Scheduler:
    
    def add_cron_schedule(self, schedule: str, name: str):
        if not croniter.is_valid(schedule):
            raise ValueError(f"Specified schedule {schedule} is not valid for {name} !")

        logger.debug(f"Adding schedule {schedule} for {name}")