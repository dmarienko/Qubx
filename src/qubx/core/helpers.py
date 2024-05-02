from collections import defaultdict
import re
from typing import Dict, List, Optional
import numpy as np
from croniter import croniter
import pandas as pd

from qubx import logger
from qubx.core.basics import CtrlChannel
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_tf_str_td64, convert_seconds_to_str
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


S = lambda s: [x for x in re.split(r"[, ]", s) if x]
HMS = lambda s: list(map(int, s.split(':') if s.count(':') == 2 else [*s.split(':'), 0]))
AS_INT = lambda d, k: int(d.get(k, 0)) 

def _mk_cron(time: str, by: list | None) -> str:
    h,m,s = HMS(time)
    assert h < 24, f'Wrong value for hour {h}'
    assert m < 60, f'Wrong value for minute {m}'
    assert s < 60, f'Wrong value for seconds {s}'
    b = ','.join(by) if by else '*'
    c = f'{m} {h} * * {b}'
    return c if s == 0 else c + f' {s}'


def _make_shift(_b, _w, _d, _h, _m, _s):
    D0 = pd.Timedelta(0)
    AS_TD = lambda d: pd.Timedelta(d)
    P, N = D0, D0 
    
    # return AS_TD(f'{_b*4}W') + AS_TD(f'{_w}W') + AS_TD(f'{_d}D') + AS_TD(f'{_h}h') + AS_TD(f'{_m}Min') + AS_TD(f'{_s}Sec')
    for t in [
        AS_TD(f'{_b*4}W'), AS_TD(f'{_w}W'), AS_TD(f'{_d}D'), 
        AS_TD(f'{_h}h'), AS_TD(f'{_m}Min'), AS_TD(f'{_s}Sec')]:
        if t > D0:
            P += t
        else:
            N += t
    return P, N


class BasicScheduler:
    """
    Basic scheduler functionality
    """

    SPEC_REGEX = re.compile(
        r"((?P<type>[A-Za-z]+)(\.?(?P<timeframe>[0-9A-Za-z]+))?\ *:)?"
        r"\ *"
        r"((?P<spec>"
            r"(?P<time>((\d+:\d+(:\d+)?)\ *,?\ *)+)?"
            r"((\ *@\ *)(?P<by>([A-Za-z0-9-,\ ]+)))?"
            r"(("
            r'((?P<months>[-+]?\d+)(months|month|bm|mo))?'
            r'((?P<weeks>[-+]?\d+)(weeks|week|w))?'
            r'((?P<days>[-+]?\d+)(days|day|d))?'
            r'((?P<hours>[-+]?\d+)(hours|hour|h))?'
            r'((?P<minutes>[-+]?\d+)(mins|min|m))?'
            r'((?P<seconds>[-+]?\d+)(sec|s))?'
            r")(\ *)?)*"
            r".*"
        r"))?", re.IGNORECASE
    )
    channel: CtrlChannel

    def initialize(self, channel: CtrlChannel):
        self.channel = channel

    def parse_schedule_spec(self, schedule: str) -> Dict[str, str]:
        m = BasicScheduler.SPEC_REGEX.match(schedule)
        return {k: v for k, v in m.groupdict().items() if v} if m else {}

    def _process_spec_dict(self, spec: dict) -> List[Dict]:
        config = []
        _T, _S = spec.get('type'), spec.get('spec')
        _F = spec.get('timeframe')
        _t, _by = S(spec.get('time', '')), S(spec.get('by', ''))
        _b, _w, _d = AS_INT(spec, 'months'), AS_INT(spec, 'weeks'), AS_INT(spec, 'days')
        _h, _m, _s = AS_INT(spec, 'hours'), AS_INT(spec, 'minutes'), AS_INT(spec, 'seconds')
        _has_intervals = (_b != 0) or (_w != 0) or (_d != 0) or (_h != 0) or (_m != 0) or (_s != 0) 
        _s_pos, _s_neg = _make_shift(_b, _w, _d, _h, _m, _s)
        _shift = _s_pos + _s_neg

        match _T:
            case 'cron':
                if not _S or croniter.is_valid(_S):
                    config.append(dict(type='cron', args=_S))
                else:
                    raise ValueError(f"Wrong specification for cron type: {_S}")

            case 'time':
                for t in _t:
                    config.append(dict(type='cron', args=_mk_cron(t, _by)))
            
            case None:
                if _t: # - if time specified
                    for t in _t:
                        config.append(dict(type='cron', args=_mk_cron(t, _by)))
                else:
                    # - check if it's valid cron
                    if _S:
                        if croniter.is_valid(_S):
                            config.append(dict(type='cron', args=_S))
                        else:
                            if _has_intervals:
                                _F = convert_seconds_to_str(int(_s_pos.as_unit('s').to_timedelta64().item().total_seconds())) if not _F else _F  
                                config.append(dict(type='bar', args=_S, timeframe=_F, delay=_s_neg))
            case _:
                config.append(dict(type=_T, args=_S, timeframe=_F, delay=_shift))

        return config

    def schedule_event(self, schedule: str, event_name: str):
        if not croniter.is_valid(schedule):
            raise ValueError(f"Specified schedule {schedule} is not valid for {event_name} !")

        logger.debug(f"Adding schedule {schedule} for {event_name}")
    