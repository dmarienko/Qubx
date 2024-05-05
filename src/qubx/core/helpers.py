from collections import defaultdict
import re, sched, time
from typing import Any, Callable, Dict, List, Optional, Tuple
from croniter import croniter
import numpy as np
import pandas as pd
from threading import Thread

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
    default_timeframe: np.timedelta64
    _last_bar: Dict[str, Bar | None]
    _ohlcvs: Dict[str, Dict[np.timedelta64, OHLCV]]
    _updates: Dict[str, Any]

    def __init__(self, default_timeframe: str) -> None:
        self.default_timeframe = convert_tf_str_td64(default_timeframe)
        self._ohlcvs = dict()
        self._last_bar = defaultdict(lambda: None)
        self._updates = dict()

    def init_ohlcv(self, symbol: str, max_size=np.inf):
        self._ohlcvs[symbol] = {self.default_timeframe: OHLCV(symbol, self.default_timeframe, max_size)}

    def is_data_ready(self) -> bool:
        """
        Check if all symbols in this cache have at least one update
        """
        for v in self._ohlcvs.keys():
            if v not in self._updates:
                return False
        return True
    
    @_SW.watch('CachedMarketDataHolder')
    def get_ohlcv(self, symbol: str, timeframe: str, max_size=np.inf) -> OHLCV:
        tf = convert_tf_str_td64(timeframe) 

        if symbol not in self._ohlcvs:
           self._ohlcvs[symbol] = {}

        if tf not in self._ohlcvs[symbol]: 
            # - check requested timeframe
            new_ohlc = OHLCV(symbol, tf, max_size)
            if tf < self.default_timeframe:
                logger.warning(f"[{symbol}] Request for timeframe {timeframe} that is smaller then minimal {self.default_timeframe}")
            else:
                # - first try to resample from smaller frame
                if (basis := self._ohlcvs[symbol].get(self.default_timeframe)):
                    for b in basis[::-1]:
                        new_ohlc.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume, b.bought_volume)
                
            self._ohlcvs[symbol][tf] = new_ohlc

        return self._ohlcvs[symbol][tf]

    @_SW.watch('CachedMarketDataHolder')
    def update_by_bars(self, symbol: str, timeframe: str, bars: List[Bar]) -> OHLCV:
        """
        Substitute or create new series based on provided historical bars
        """
        if symbol not in self._ohlcvs:
           self._ohlcvs[symbol] = {}

        tf = convert_tf_str_td64(timeframe) 
        new_ohlc = OHLCV(symbol, tf)
        for b in bars:
            new_ohlc.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume, b.bought_volume)
            self._updates[symbol] = b

        self._ohlcvs[symbol][tf] = new_ohlc
        return new_ohlc

    @_SW.watch('CachedMarketDataHolder')
    def update_by_bar(self, symbol: str, bar: Bar):
        self._updates[symbol] = bar

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
        self._updates[symbol] = quote

        series = self._ohlcvs.get(symbol)
        if series:
            for ser in series.values():
                ser.update(quote.time, quote.mid_price(), 0)

    @_SW.watch('CachedMarketDataHolder')
    def update_by_trade(self, symbol: str, trade: Trade):
        self._updates[symbol] = trade
        series = self._ohlcvs.get(symbol)
        if series:
            total_vol = trade.size 
            bought_vol = total_vol if trade.taker >= 1 else 0.0 
            for ser in series.values():
                ser.update(trade.time, trade.price, total_vol, bought_vol)


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


def _mk_cron(time: str, by: list | None) -> str:
    HMS = lambda s: list(map(int, s.split(':') if s.count(':') == 2 else [*s.split(':'), 0]))

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


def _parse_schedule_spec(schedule: str) -> Dict[str, str]:
    m = SPEC_REGEX.match(schedule)
    return {k: v for k, v in m.groupdict().items() if v} if m else {}


def process_schedule_spec(spec_str: str | None) -> Dict[str, Any]:
    AS_INT = lambda d, k: int(d.get(k, 0)) 
    S = lambda s: [x for x in re.split(r"[, ]", s) if x]
    config = {}

    if not spec_str:
        return config

    # - parse schedule spec
    spec = _parse_schedule_spec(spec_str)

    # - check how to run it 
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
                config = dict(type='cron', schedule=_S, spec=_S)
            else:
                raise ValueError(f"Wrong specification for cron type: {_S}")

        case 'time':
            for t in _t:
                config = dict(type='cron', schedule=_mk_cron(t, _by), spec=_S)
        
        case None:
            if _t: # - if time specified
                for t in _t:
                    config = dict(type='cron', schedule=_mk_cron(t, _by), spec=_S)
            else:
                # - check if it's valid cron
                if _S:
                    if croniter.is_valid(_S):
                        config = dict(type='cron', schedule=_S, spec=_S)
                    else:
                        if _has_intervals:
                            _F = convert_seconds_to_str(int(_s_pos.as_unit('s').to_timedelta64().item().total_seconds())) if not _F else _F  
                            config = dict(type='bar', schedule=None, timeframe=_F, delay=_s_neg, spec=_S)
        case _:
            config = dict(type=_T, schedule=None, timeframe=_F, delay=_shift, spec=_S)

    return config


_SEC2TS = lambda t: pd.Timestamp(t, unit='s')

class BasicScheduler:
    """
    Basic scheduler functionality. It helps to create scheduled event task
    """
    _chan: CtrlChannel
    _scdlr:  sched.scheduler
    _ns_time_fun: Callable[[], float]
    _crons: Dict[str, croniter]
    _is_started: bool

    def __init__(self, channel: CtrlChannel, time_provider_ns: Callable[[], float]):
        self._chan = channel
        self._ns_time_fun = time_provider_ns
        self._scdlr = sched.scheduler(self.time_sec)
        self._crons = dict()
        self._is_started = False

    def time_sec(self) -> float:
        return self._ns_time_fun() / 1000000000.0

    def schedule_event(self, cron_schedule: str, event_name: str):
        if not croniter.is_valid(cron_schedule):
            raise ValueError(f"Specified schedule {cron_schedule} for {event_name} doesn't have valid cron format !")
        self._crons[event_name] = croniter(cron_schedule, self.time_sec())

        if self._is_started:
            self._arm_schedule(event_name, self.time_sec())

    def get_event_last_time(self, event_name: str) -> pd.Timestamp | None:
        if event_name in self._crons: 
            _iter = self._crons[event_name]
            _c = _iter.get_current()
            _t = pd.Timestamp(_iter.get_prev(), unit='s')
            _iter.set_current(_c, force=True)
            return _t
        return None

    def get_event_next_time(self, event_name: str) -> pd.Timestamp | None:
        if event_name in self._crons: 
            _iter = self._crons[event_name]
            _t = pd.Timestamp(_iter.get_next(start_time=self.time_sec()), unit='s')
            return _t
        return None

    def _arm_schedule(self, event: str, start_time: float) -> bool:
        iter = self._crons[event]
        prev_time = iter.get_prev()
        next_time = iter.get_next(start_time=start_time)
        if next_time:
            self._scdlr.enterabs(
                next_time, 1, self._trigger, (event, prev_time, next_time)
            )
            logger.debug(f"Next ({event}) event scheduled at <red>{_SEC2TS(next_time)}</red>")
            return True
        logger.debug(f"({event}) task is not scheduled")
        return False

    def _trigger(self, event: str, prev_time_sec: float, trig_time: float):
        now = self.time_sec()

        # - send notification to channel
        if self._chan.control.is_set():
            self._chan.queue.put((None, event, (prev_time_sec, trig_time)))

        # - try to arm this event again
        self._arm_schedule(event, now)

    def check_and_run_tasks(self) -> float | None:
        return self._scdlr.run(blocking=False) 

    def run(self):
        if self._is_started:
            logger.warning("Scheduler is already running")
            return

        _has_tasks = False
        for k in self._crons.keys():
            _has_tasks |= self._arm_schedule(k, self.time_sec())

        def _watcher():
            while (r := self.check_and_run_tasks()):
                if not self._chan.control.is_set():
                    break
                _delay = max(min(r/5, 5), 0.1)
                time.sleep(_delay)
            logger.debug("Scheduler is stopped ")                    
            self._is_started = False

        if _has_tasks:
            Thread(target=_watcher).start()
            self._is_started = True

    