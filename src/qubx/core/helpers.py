from collections import defaultdict
import re, sched, time
from typing import Any, Callable, Dict, List, Optional, Tuple
from croniter import croniter
import numpy as np
import pandas as pd
from threading import Thread

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_tf_str_td64, convert_seconds_to_str
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV


_SW = Stopwatch()


class CachedMarketDataHolder:
    """
    Collected cached data updates from StrategyContext
    """

    default_timeframe: np.timedelta64
    _last_bar: dict[Instrument, Bar | None]
    _ohlcvs: dict[Instrument, dict[np.timedelta64, OHLCV]]
    _updates: dict[Instrument, Any]

    def __init__(self, default_timeframe: str | None = None) -> None:
        self._ohlcvs = dict()
        self._last_bar = defaultdict(lambda: None)
        self._updates = dict()
        if default_timeframe:
            self.default_timeframe = convert_tf_str_td64(default_timeframe)

    def update_default_timeframe(self, default_timeframe: str):
        self.default_timeframe = convert_tf_str_td64(default_timeframe)

    def init_ohlcv(self, instrument: Instrument, max_size=np.inf):
        self._ohlcvs[instrument] = {self.default_timeframe: OHLCV(instrument.symbol, self.default_timeframe, max_size)}

    def remove(self, instrument: Instrument) -> None:
        self._ohlcvs.pop(instrument, None)
        self._last_bar.pop(instrument, None)
        self._updates.pop(instrument, None)

    def is_data_ready(self) -> bool:
        """
        Check if at least one symbol had an update.
        """
        for v in self._ohlcvs.keys():
            if v in self._updates:
                return True
        return False

    @_SW.watch("CachedMarketDataHolder")
    def get_ohlcv(self, instrument: Instrument, timeframe: str | None = None, max_size: float | int = np.inf) -> OHLCV:
        tf = convert_tf_str_td64(timeframe) if timeframe else self.default_timeframe

        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {}

        if tf not in self._ohlcvs[instrument]:
            # - check requested timeframe
            new_ohlc = OHLCV(instrument.symbol, tf, max_size)
            if tf < self.default_timeframe:
                logger.warning(
                    f"[{instrument.symbol}] Request for timeframe {timeframe} that is smaller then minimal {self.default_timeframe}"
                )
            else:
                # - first try to resample from smaller frame
                if basis := self._ohlcvs[instrument].get(self.default_timeframe):
                    for b in basis[::-1]:
                        new_ohlc.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume, b.bought_volume)

            self._ohlcvs[instrument][tf] = new_ohlc

        return self._ohlcvs[instrument][tf]

    @_SW.watch("CachedMarketDataHolder")
    def update_by_bars(self, instrument: Instrument, timeframe: str, bars: List[Bar]) -> OHLCV:
        """
        Substitute or create new series based on provided historical bars
        """
        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {}

        tf = convert_tf_str_td64(timeframe)
        new_ohlc = OHLCV(instrument.symbol, tf)
        for b in bars:
            new_ohlc.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume, b.bought_volume)
            self._updates[instrument] = b

        self._ohlcvs[instrument][tf] = new_ohlc
        return new_ohlc

    @_SW.watch("CachedMarketDataHolder")
    def update_by_bar(self, instrument: Instrument, bar: Bar):
        self._updates[instrument] = bar

        _last_bar = self._last_bar[instrument]
        v_tot_inc = bar.volume
        v_buy_inc = bar.bought_volume

        if _last_bar is not None:
            if _last_bar.time == bar.time:  # just current bar updated
                v_tot_inc -= _last_bar.volume
                v_buy_inc -= _last_bar.bought_volume

            if _last_bar.time > bar.time:  # update is too late - skip it
                return

        if instrument in self._ohlcvs:
            self._last_bar[instrument] = bar
            for ser in self._ohlcvs[instrument].values():
                ser.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, v_tot_inc, v_buy_inc)

    @_SW.watch("CachedMarketDataHolder")
    def update_by_quote(self, instrument: Instrument, quote: Quote):
        self._updates[instrument] = quote

        series = self._ohlcvs.get(instrument)
        if series:
            for ser in series.values():
                ser.update(quote.time, quote.mid_price(), 0)

    @_SW.watch("CachedMarketDataHolder")
    def update_by_trade(self, instrument: Instrument, trade: Trade):
        self._updates[instrument] = trade
        series = self._ohlcvs.get(instrument)
        if series:
            total_vol = trade.size
            bought_vol = total_vol if trade.taker >= 1 else 0.0
            for ser in series.values():
                if len(ser) > 0 and ser[0].time > trade.time:
                    continue
                ser.update(trade.time, trade.price, total_vol, bought_vol)


SPEC_REGEX = re.compile(
    r"((?P<type>[A-Za-z]+)(\.?(?P<timeframe>[0-9A-Za-z]+))?\ *:)?"
    r"\ *"
    r"((?P<spec>"
    r"(?P<time>((\d+:\d+(:\d+)?)\ *,?\ *)+)?"
    r"((\ *@\ *)(?P<by>([A-Za-z0-9-,\ ]+)))?"
    r"(("
    r"((?P<months>[-+]?\d+)(months|month|bm|mo))?"
    r"((?P<weeks>[-+]?\d+)(weeks|week|w))?"
    r"((?P<days>[-+]?\d+)(days|day|d))?"
    r"((?P<hours>[-+]?\d+)(hours|hour|h))?"
    r"((?P<minutes>[-+]?\d+)(mins|min|m))?"
    r"((?P<seconds>[-+]?\d+)(sec|s))?"
    r")(\ *)?)*"
    r".*"
    r"))?",
    re.IGNORECASE,
)


def _mk_cron(time: str, by: list | None) -> str:
    HMS = lambda s: list(map(int, s.split(":") if s.count(":") == 2 else [*s.split(":"), 0]))

    h, m, s = HMS(time)
    assert h < 24, f"Wrong value for hour {h}"
    assert m < 60, f"Wrong value for minute {m}"
    assert s < 60, f"Wrong value for seconds {s}"
    b = ",".join(by) if by else "*"
    c = f"{m} {h} * * {b}"
    return c if s == 0 else c + f" {s}"


def _make_shift(_b, _w, _d, _h, _m, _s):
    D0 = pd.Timedelta(0)
    AS_TD = lambda d: pd.Timedelta(d)
    P, N = D0, D0

    # return AS_TD(f'{_b*4}W') + AS_TD(f'{_w}W') + AS_TD(f'{_d}D') + AS_TD(f'{_h}h') + AS_TD(f'{_m}Min') + AS_TD(f'{_s}Sec')
    for t in [
        AS_TD(f"{_b*4}W"),
        AS_TD(f"{_w}W"),
        AS_TD(f"{_d}D"),
        AS_TD(f"{_h}h"),
        AS_TD(f"{_m}Min"),
        AS_TD(f"{_s}Sec"),
    ]:
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
    _T, _S = spec.get("type"), spec.get("spec")
    _F = spec.get("timeframe")
    _t, _by = S(spec.get("time", "")), S(spec.get("by", ""))
    _b, _w, _d = AS_INT(spec, "months"), AS_INT(spec, "weeks"), AS_INT(spec, "days")
    _h, _m, _s = AS_INT(spec, "hours"), AS_INT(spec, "minutes"), AS_INT(spec, "seconds")
    _has_intervals = (_b != 0) or (_w != 0) or (_d != 0) or (_h != 0) or (_m != 0) or (_s != 0)
    _s_pos, _s_neg = _make_shift(_b, _w, _d, _h, _m, _s)
    _shift = _s_pos + _s_neg

    match _T:
        case "cron":
            if not _S or croniter.is_valid(_S):
                config = dict(type="cron", schedule=_S, spec=_S)
            else:
                raise ValueError(f"Wrong specification for cron type: {_S}")

        case "time":
            for t in _t:
                config = dict(type="cron", schedule=_mk_cron(t, _by), spec=_S)

        case None:
            if _t:  # - if time specified
                for t in _t:
                    config = dict(type="cron", schedule=_mk_cron(t, _by), spec=_S)
            else:
                # - check if it's valid cron
                if _S:
                    if croniter.is_valid(_S):
                        config = dict(type="cron", schedule=_S, spec=_S)
                    else:
                        if _has_intervals:
                            _F = (
                                convert_seconds_to_str(int(_s_pos.as_unit("s").to_timedelta64().item().total_seconds()))
                                if not _F
                                else _F
                            )
                            config = dict(type="bar", schedule=None, timeframe=_F, delay=_s_neg, spec=_S)
        case _:
            config = dict(type=_T, schedule=None, timeframe=_F, delay=_shift, spec=_S)

    return config


_SEC2TS = lambda t: pd.Timestamp(t, unit="s")


class BasicScheduler:
    """
    Basic scheduler functionality. It helps to create scheduled event task
    """

    _chan: CtrlChannel
    _scdlr: sched.scheduler
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
            _t = pd.Timestamp(_iter.get_prev(), unit="s")
            _iter.set_current(_c, force=True)
            return _t
        return None

    def get_event_next_time(self, event_name: str) -> pd.Timestamp | None:
        if event_name in self._crons:
            _iter = self._crons[event_name]
            _t = pd.Timestamp(_iter.get_next(start_time=self.time_sec()), unit="s")
            return _t
        return None

    def _arm_schedule(self, event: str, start_time: float) -> bool:
        iter = self._crons[event]
        prev_time = iter.get_prev()
        next_time = iter.get_next(start_time=start_time)
        if next_time:
            self._scdlr.enterabs(next_time, 1, self._trigger, (event, prev_time, next_time))
            # logger.debug(
            #     f"Now is <red>{_SEC2TS(self.time_sec())}</red> next ({event}) at <cyan>{_SEC2TS(next_time)}</cyan>"
            # )
            return True
        logger.debug(f"({event}) task is not scheduled")
        return False

    def _trigger(self, event: str, prev_time_sec: float, trig_time: float):
        now = self.time_sec()

        # - send notification to channel
        self._chan.send((None, event, (prev_time_sec, trig_time)))

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
            while r := self.check_and_run_tasks():
                if not self._chan.control.is_set():
                    break
                _delay = max(min(r / 5, 5), 0.1)
                time.sleep(_delay)
            logger.debug("Scheduler is stopped ")
            self._is_started = False

        if _has_tasks:
            Thread(target=_watcher).start()
            self._is_started = True


def set_parameters_to_object(strategy: Any, **kwargs):
    """
    Set given parameters values to object.
    Parameter can be set only if it's declared as attribute of object and it's not starting with underscore (_).
    """
    _log_info = ""
    for k, v in kwargs.items():
        if k.startswith("_"):
            raise ValueError("Internal variable can't be set from external parameter !")
        if hasattr(strategy, k):
            strategy.__dict__[k] = v
            v_str = str(v).replace(">", "").replace("<", "")
            _log_info += f"\n\tset <green>{k}</green> <- <red>{v_str}</red>"

    if _log_info:
        logger.debug(f"<yellow>{strategy.__class__.__name__}</yellow> new parameters:" + _log_info)
