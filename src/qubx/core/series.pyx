import pandas as pd
import numpy as np
cimport numpy as np
from cython cimport abs
from typing import Union
from qubx.core.utils import time_to_str, time_delta_to_str, recognize_timeframe


cdef extern from "math.h":
    float INFINITY


cdef np.ndarray nans(int dims):
    """
    nans(n) is an n length array of NaNs.
    
    :param dims: array size
    :return: nans matrix 
    """
    return np.nan * np.ones(dims)


cdef inline long long floor_t64(long long time, long long dt):
    """
    Floor timestamp by dt
    """
    return time - time % dt


cpdef long long time_as_nsec(time):
    """
    Tries to recognize input time and convert it to nanosec
    """
    if isinstance(time, np.datetime64):
        return time.astype('<M8[ns]').item()
    elif isinstance(time, pd.Timestamp):
        return time.asm8
    elif isinstance(time, str):
        return np.datetime64(time).astype('<M8[ns]').item()
    return time


cdef class RollingSum:
    """
    Rolling fast summator
    """

    def __init__(self, int period):
        self.period = period
        self.__s = np.zeros(period)
        self.__i = 0
        self.rsum = 0.0
        self.is_init_stage = 1

    cpdef double update(self, double value, short new_item_started):
        if np.isnan(value):
            return np.nan
        sub = self.__s[self.__i]
        if new_item_started:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0
                self.is_init_stage = 0
            sub = self.__s[self.__i]
        self.__s[self.__i] = value
        self.rsum -= sub
        self.rsum += value 
        return self.rsum

    def __str__(self):
        return f"rs[{self.period}] = {self.__s} @ {self.__i} -> {self.is_init_stage}"


cdef class Indexed:

    def __init__(self, max_series_length=INFINITY):
        self.max_series_length = max_series_length
        self.values = list()
        self._is_empty = 1

    def __len__(self) -> int:
        return len(self.values)

    def empty(self) -> bool:
        return self._is_empty

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.values[self._get_index(i)] for i in range(*idx.indices(len(self.values)))]
        return self.values[self._get_index(idx)]

    def _get_index(self, idx: int) -> int:
        n_len = len(self)
        if n_len == 0 or (idx > 0 and idx > (n_len - 1)) or (idx < 0 and abs(idx) > n_len):
            raise IndexError(f"Can't find record at index {idx}")
        return (n_len - idx - 1) if idx >= 0 else abs(1 + idx)

    def add(self, v):
        self.values.append(v)
        self._is_empty = 0
        if len(self.values) >= self.max_series_length:
            self.values.pop(0)

    def update_last(self, v):
        if self.values:
            self.values[-1] = v
        else:
            self.append(v)
        self._is_empty = 0

    def set_values(self, new_values: list):
        self._is_empty = False
        self.values = new_values

    def clear(self):
        self.values.clear()
        self._is_empty = 1


global _plot_func


cdef class TimeSeries:

    def __init__(
        self, str name, timeframe, max_series_length=INFINITY, 
        process_every_update=True, # calculate indicators on every update (tick) - by default
    ) -> None:
        self.name = name
        self.max_series_length = max_series_length
        self.timeframe = recognize_timeframe(timeframe)
        self.times = Indexed(max_series_length)
        self.values = Indexed(max_series_length)
        self.indicators = dict()
        self.calculation_order = []

        # - processing every update
        self._process_every_update = process_every_update
        self._last_bar_update_value = np.nan
        self._last_bar_update_time = -1

    def __len__(self) -> int:
        return len(self.times)

    def _on_attach_indicator(self, indicator: Indicator, indicator_input: TimeSeries):
        self.calculation_order.append((
            id(indicator_input), indicator, id(indicator)
        ))

    def __getitem__(self, idx):
        return self.values[idx]

    def _add_new_item(self, long long time, double value):
        self.times.add(time)
        self.values.add(value)
        self._is_new_item = True

    def _update_last_item(self, long long time, double value):
        self.times.update_last(time)
        self.values.update_last(value)
        self._is_new_item = False

    def update(self, long long time, double value) -> bool:
        item_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(item_start_time, value)

            # - disable first notification because first item may be incomplete
            self._is_new_item = False

        elif time - self.times[0] >= self.timeframe:
            # - add new item
            self._add_new_item(item_start_time, value)

            # - if it's needed to process every tick in indicator
            if self._process_every_update:
                self._update_indicators(item_start_time, value, True)
            else:
                # - it's required to update indicators only on closed (formed) bar
                self._update_indicators(self._last_bar_update_time, self._last_bar_update_value, True)

            # - store last data
            self._last_bar_update_time = item_start_time
            self._last_bar_update_value = value

            return self._is_new_item
        else:
            self._update_last_item(item_start_time, value)

        # - update indicators by new data
        if self._process_every_update:
            self._update_indicators(item_start_time, value, False)

        # - store last data
        self._last_bar_update_time = item_start_time
        self._last_bar_update_value = value

        return self._is_new_item

    cdef _update_indicators(self, long long time, value, short new_item_started):
        mem = dict()              # store calculated values during this update
        mem[id(self)] = value     # initail value - new data from itself
        for input, indicator, iid in self.calculation_order:
            if input not in mem:
                raise ValueError("> No input data - something wrong in calculation order !")
            mem[iid] = indicator.update(time, mem[input], new_item_started)

    def shift(self, int period):
        """
        Returns shifted series by period
        """
        if period < 0:
            raise ValueError("Only positive shift (from past) period is allowed !")
        return lag(self, period)

    def __add__(self, other: Union[TimeSeries, float, int]):
        return plus(self, other)

    def __sub__(self, other: Union[TimeSeries, float, int]):
        return minus(self, other)

    def __mul__(self, other: Union[TimeSeries, float, int]):
        return mult(self, other)

    def __truediv__(self, other: Union[TimeSeries, float, int]):
        return divide(self, other)

    def __lt__(self, other: Union[TimeSeries, float, int]):
        return lt(self, other)

    def __le__(self, other: Union[TimeSeries, float, int]):
        return le(self, other)

    def __gt__(self, other: Union[TimeSeries, float, int]):
        return gt(self, other)

    def __ge__(self, other: Union[TimeSeries, float, int]):
        return ge(self, other)

    def __eq__(self, other: Union[TimeSeries, float, int]):
        return eq(self, other)

    def __ne__(self, other: Union[TimeSeries, float, int]):
        return ne(self, other)

    def __neg__(self):
        return neg(self)

    def __abs__(self):
        return series_abs(self)

    def to_records(self) -> dict:
        ts = [np.datetime64(t, 'ns') for t in self.times[::-1]]
        return dict(zip(ts, self.values[::-1]))

    def to_series(self):
        return pd.Series(self.values.values, index=pd.DatetimeIndex(self.times.values), name=self.name, dtype=float)
        # return pd.Series(self.to_records(), name=self.name, dtype=float)

    def pd(self):
        return self.to_series()

    def get_indicators(self) -> dict:
        return self.indicators

    def plot(self, *args, **kwargs):
        _timeseries_plot_func(self, *args, **kwargs)

    def __str__(self):
        nl = len(self)
        r = f"{self.name}[{time_delta_to_str(self.timeframe)}] | {nl} records\n"
        hd, tl = 3, 3 
        if nl <= hd + tl:
            hd, tl = nl, 0
        
        for n in range(hd):
            r += f"  {time_to_str(self.times[n], 'ns')} {str(self[n])}\n"
        
        if tl > 0:
            r += "   .......... \n"
            for n in range(-tl, 0):
                r += f"  {time_to_str(self.times[n], 'ns')} {str(self[n])}\n"

        return r

    def __repr__(self):
        return repr(self.pd())


def _wrap_indicator(series: TimeSeries, clz, *args, **kwargs):
    aw = ','.join([a.name if isinstance(a, TimeSeries) else str(a) for a in args])
    if kwargs:
        aw += ',' + ','.join([f"{k}={str(v)}" for k,v in kwargs.items()])
    nn = clz.__name__.lower() + "(" + aw + ")"
    inds = series.get_indicators()
    if nn in inds:
        return inds[nn]
    return clz(nn, series, *args, **kwargs) 


cdef class Indicator(TimeSeries):

    def __init__(self, str name, TimeSeries series):
        if not name:
            raise ValueError(f" > Name must not be empty for {self.__class__.__name__}!")
        super().__init__(name, series.timeframe, series.max_series_length)
        series.indicators[name] = self
        self.name = name

        # - we need to make a empty copy and fill it 
        self.series = TimeSeries(series.name, series.timeframe, series.max_series_length)
        self.parent = series 
        
        # - notify the parent series that indicator has been attached
        self._on_attach_indicator(self, series)

        # - recalculate indicator on data as if it would being streamed
        self._initial_data_recalculate(series)

    def _on_attach_indicator(self, indicator: Indicator, indicator_input: TimeSeries):
        self.parent._on_attach_indicator(indicator, indicator_input)

    def _initial_data_recalculate(self, TimeSeries series):
        for t, v in zip(series.times[::-1], series.values[::-1]):
            self.update(t, v, True)

    def update(self, long long time, value, short new_item_started) -> object:
        if new_item_started or len(self) == 0:
            self.series._add_new_item(time, value)
            iv = self.calculate(time, value, new_item_started)
            self._add_new_item(time, iv)
        else:
            self.series._update_last_item(time, value)
            iv = self.calculate(time, value, new_item_started)
            self._update_last_item(time, iv)

        return iv

    def calculate(self, long long time, value, short new_item_started) -> object:
        raise ValueError("Indicator must implement calculate() method")

    @classmethod
    def wrap(clz, series:TimeSeries, *args, **kwargs):
        return _wrap_indicator(series, clz, *args, **kwargs)


cdef class Lag(Indicator):
    cdef int period

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if len(self.series) <= self.period:
            return np.nan
        return self.series[self.period]
     
     
def lag(series:TimeSeries, period: int):
    return Lag.wrap(series, period)


cdef class Abs(Indicator):

    def __init__(self, str name, TimeSeries series):
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        return abs(self.series[0])


def series_abs(series:TimeSeries):
    return Abs.wrap(series)


cdef class Compare(Indicator):
    cdef TimeSeries to_compare 
    cdef double comparable_scalar
    cdef short _cmp_to_series

    def __init__(self, name: str,  original: TimeSeries, comparable: Union[TimeSeries, float, int]):
        if isinstance(comparable, TimeSeries):
            if comparable.timeframe != original.timeframe:
                raise ValueError("Series must be of the same timeframe for performing operation !")
            self.to_compare = comparable
            self._cmp_to_series = 1
        else:
            self.comparable_scalar = comparable
            self._cmp_to_series = 0
        super().__init__(name, original)

    cdef double _operation(self, double a, double b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return +1 if a > b else -1 if a < b else 0

    def _initial_data_recalculate(self, TimeSeries series):
        if self._cmp_to_series:
            r = pd.concat((series.to_series(), self.to_compare.to_series()), axis=1)
            for t, (a, b) in zip(r.index, r.values):
                self.series._add_new_item(t.asm8, a)
                self._add_new_item(t.asm8, self._operation(a, b))
        else:
            r = series.to_series()
            for t, a in zip(r.index, r.values):
                self.series._add_new_item(t.asm8, a)
                self._add_new_item(t.asm8, self._operation(a, self.comparable_scalar))

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if self._cmp_to_series:
            if len(self.to_compare) == 0 or len(self.series) == 0 or time != self.to_compare.times[0]:
                return np.nan
            return self._operation(value, self.to_compare[0])
        else:
            if len(self.series) == 0:
                return np.nan
            return self._operation(value, self.comparable_scalar)


def compare(series0:TimeSeries, series1:TimeSeries):
    return Compare.wrap(series0, series1)


cdef class Plus(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a + b


cdef class Minus(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a - b


cdef class Mult(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a * b


cdef class Divide(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a / b


cdef class EqualTo(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a == b


cdef class NotEqualTo(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a != b


cdef class LessThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a < b


cdef class LessEqualThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a <= b


cdef class GreaterThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a > b


cdef class GreaterEqualThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a >= b


cdef class Neg(Indicator):

    def __init__(self, name: str, series:TimeSeries):
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        return -value


def plus(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Plus.wrap(series0, series1)


def minus(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Minus.wrap(series0, series1)


def mult(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Mult.wrap(series0, series1)


def divide(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Divide.wrap(series0, series1)


def eq(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return EqualTo.wrap(series0, series1)
    

def ne(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return NotEqualTo.wrap(series0, series1)


def lt(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return LessThan.wrap(series0, series1)


def le(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return LessEqualThan.wrap(series0, series1)


def gt(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return GreaterThan.wrap(series0, series1)


def ge(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return GreaterEqualThan.wrap(series0, series1)


def neg(series: TimeSeries):
    return Neg.wrap(series)


cdef class Trade:
    def __init__(self, time, double price, double size, short taker=-1, long long trade_id=0):
        self.time = time_as_nsec(time)
        self.price = price
        self.size = size
        self.taker = taker
        self.trade_id = trade_id

    def __repr__(self):
        return "[%s]\t%.5f (%.1f) <%s> %s" % ( 
            time_to_str(self.time, 'ns'), self.price, self.size, 
            'take' if self.taker == 1 else 'make' if self.taker == 0 else '???',
            str(self.trade_id) if self.trade_id > 0 else ''
        ) 


cdef class Quote:
    def __init__(self, time, double bid, double ask, double bid_size, double ask_size):
        self.time = time_as_nsec(time)
        self.bid = bid
        self.ask = ask
        self.bid_size = bid_size
        self.ask_size = ask_size

    cpdef double mid_price(self):
        return 0.5 * (self.ask + self.bid)

    def __repr__(self):
        return "[%s]\t%.5f (%.1f) | %.5f (%.1f)" % (
            time_to_str(self.time, 'ns'), self.bid, self.bid_size, self.ask, self.ask_size
        )


cdef class Bar:

    def __init__(self, long long time, double open, double high, double low, double close, double volume, double bought_volume=0) -> None:
        self.time = time
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.bought_volume = bought_volume

    cpdef Bar update(self, double price, double volume, double bought_volume=0):
        self.close = price
        self.high = max(price, self.high)
        self.low = min(price, self.low)
        self.volume += volume
        self.bought_volume += bought_volume
        return self

    cpdef dict to_dict(self, unsigned short skip_time=0):
        if skip_time:
            return {
                'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close,
                'volume': self.volume, 'bought_volume': self.bought_volume,
            }
        return {
            'timestamp': np.datetime64(self.time, 'ns'), 
            'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close, 
            'volume': self.volume,
            'bought_volume': self.bought_volume,
        }

    def __repr__(self):
        return "{o:%f | h:%f | l:%f | c:%f | v:%f}" % (self.open, self.high, self.low, self.close, self.volume)


cdef class OHLCV(TimeSeries):

    def __init__(self, str name, timeframe, max_series_length=INFINITY) -> None:
        super().__init__(name, timeframe, max_series_length)
        self.open = TimeSeries('open', timeframe, max_series_length)
        self.high = TimeSeries('high', timeframe, max_series_length)
        self.low = TimeSeries('low', timeframe, max_series_length)
        self.close = TimeSeries('close', timeframe, max_series_length)
        self.volume = TimeSeries('volume', timeframe, max_series_length)
        self.bvolume = TimeSeries('bvolume', timeframe, max_series_length)

    cpdef object append_data(self, 
                    np.ndarray times, 
                    np.ndarray opens,
                    np.ndarray highs,
                    np.ndarray lows,
                    np.ndarray closes,
                    np.ndarray volumes,
                    np.ndarray bvolumes
                ):
        cdef long long t
        cdef short _conv
        cdef short _upd_inds, _has_vol
        cdef Bar b 

        # - check if volume data presented
        _has_vol = len(volumes) > 0
        _has_bvol = len(bvolumes) > 0

        # - check if need to convert time to nanosec
        _conv = 0
        if not isinstance(times[0].item(), long):
            _conv = 1

        # - check if need to update any indicators
        _upd_inds = 0
        if (
            len(self.indicators) > 0 or 
            len(self.open.indicators) > 0 or 
            len(self.high.indicators) > 0 or
            len(self.low.indicators) > 0 or 
            len(self.close.indicators) > 0 or
            len(self.volume.indicators) > 0
        ):
            _upd_inds = 1

        for i in range(len(times)):
            if _conv:
                t = times[i].astype('datetime64[ns]').item()
            else:
                t = times[i].item()

            b = Bar(t, opens[i], highs[i], lows[i], closes[i], 
                    volumes[i] if _has_vol else 0, 
                    bvolumes[i] if _has_bvol else 0)
            self._add_new_item(t, b)

            if _upd_inds:
                self._update_indicators(t, b, True)

        return self

    def _add_new_item(self, long long time, Bar value):
        self.times.add(time)
        self.values.add(value)
        self.open._add_new_item(time, value.open)
        self.high._add_new_item(time, value.high)
        self.low._add_new_item(time, value.low)
        self.close._add_new_item(time, value.close)
        self.volume._add_new_item(time, value.volume)
        self.bvolume._add_new_item(time, value.bought_volume)
        self._is_new_item = True

    def _update_last_item(self, long long time, Bar value):
        self.times.update_last(time)
        self.values.update_last(value)
        self.open._update_last_item(time, value.open)
        self.high._update_last_item(time, value.high)
        self.low._update_last_item(time, value.low)
        self.close._update_last_item(time, value.close)
        self.volume._update_last_item(time, value.volume)
        self.bvolume._update_last_item(time, value.bought_volume)
        self._is_new_item = False

    cpdef short update(self, long long time, double price, double volume=0.0, double bvolume=0.0):
        cdef Bar b
        bar_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(bar_start_time, Bar(bar_start_time, price, price, price, price, volume, bvolume))

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False

        elif time - self.times[0] >= self.timeframe:
            b = Bar(bar_start_time, price, price, price, price, volume, bvolume)

            # - add new item
            self._add_new_item(bar_start_time, b)

            # - update indicators
            self._update_indicators(bar_start_time, b, True)

            return self._is_new_item
        else:
            self._update_last_item(bar_start_time, self[0].update(price, volume, bvolume))

        # - update indicators by new data
        self._update_indicators(bar_start_time, self[0], False)

        return self._is_new_item

    cpdef short update_by_bar(self, long long time, double open, double high, double low, double close, double vol_incr=0.0, double b_vol_incr=0.0):
        cdef Bar b
        cdef Bar l_bar
        bar_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(bar_start_time, Bar(bar_start_time, open, high, low, close, vol_incr, b_vol_incr))

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False

        elif time - self.times[0] >= self.timeframe:
            b = Bar(bar_start_time, open, high, low, close, vol_incr, b_vol_incr)

            # - add new item
            self._add_new_item(bar_start_time, b)

            # - update indicators
            self._update_indicators(bar_start_time, b, True)

            return self._is_new_item
        else:
            l_bar = self[0]
            l_bar.high = max(high, l_bar.high)
            l_bar.low = min(low, l_bar.low)
            l_bar.close = close
            l_bar.volume += vol_incr
            l_bar.bought_volume += b_vol_incr
            self._update_last_item(bar_start_time, l_bar)

        # # - update indicators by new data
        self._update_indicators(bar_start_time, self[0], False)

        return self._is_new_item

    # - TODO: need to check if it's safe to drop value series (series of Bar) to avoid duplicating data
    # def __getitem__(self, idx):
    #     if isinstance(idx, slice):
    #         return [
    #             Bar(self.times[i], self.open[i], self.high[i], self.low[i], self.close[i], self.volume[i])
    #             for i in range(*idx.indices(len(self.times)))
    #         ]
    #     return Bar(self.times[idx], self.open[idx], self.high[idx], self.low[idx], self.close[idx], self.volume[idx])

    cpdef _update_indicators(self, long long time, value, short new_item_started):
        TimeSeries._update_indicators(self, time, value, new_item_started)
        if new_item_started:
            self.open._update_indicators(time, value.open, new_item_started)
        self.close._update_indicators(time, value.close, new_item_started)
        self.high._update_indicators(time, value.high, new_item_started)
        self.low._update_indicators(time, value.low, new_item_started)
        self.volume._update_indicators(time, value.volume, new_item_started)

    def to_series(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'open': self.open.to_series(),
            'high': self.high.to_series(),
            'low': self.low.to_series(),
            'close': self.close.to_series(),
            'volume': self.volume.to_series(),         # total volume
            'bought_volume': self.bvolume.to_series(), # bought volume
        })
        df.index.name = 'timestamp'
        return df

    def to_records(self) -> dict:
        ts = [np.datetime64(t, 'ns') for t in self.times[::-1]]
        bs = [v.to_dict(skip_time=True) for v in self.values[::-1]]
        return dict(zip(ts, bs))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - this should be done in separate module -
def _plot_mpl(series: TimeSeries, *args, **kwargs):
    import matplotlib.pyplot as plt
    include_indicators = kwargs.pop('with_indicators', False)
    no_labels = kwargs.pop('no_labels', False)

    plt.plot(series.pd(), *args, **kwargs, label=series.name)
    if include_indicators:
        for k, vi in series.get_indicators().items():
            plt.plot(vi.pd(), label=k)
    if not no_labels:
        plt.legend(loc=2)

_timeseries_plot_func = _plot_mpl
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 