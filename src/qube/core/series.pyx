import pandas as pd
import numpy as np
cimport numpy as np
from collections import deque
from qube.utils import convert_tf_str_td64

from cython cimport abs


cpdef recognize_time(time):
    return np.datetime64(time, 'ns') if isinstance(time, str) else np.datetime64(time, 'ms')

NS = 1_000_000_000 
UNIX_T0 = np.datetime64('1970-01-01T00:00:00').astype(np.int64)

cdef extern from "math.h":
    float INFINITY


cpdef str time_to_str(long long t, str units = 'ns'):
    return str(np.datetime64(t, units)) #.isoformat()


cpdef str time_delta_to_str(long long d):
    """
    Convert timedelta object to pretty print format

    :param d:
    :return:
    """
    days, seconds = divmod(d, 86400*NS)
    hours, seconds = divmod(seconds, 3600*NS)
    minutes, seconds = divmod(seconds, 60*NS)
    seconds, rem  = divmod(seconds, NS)
    r = ''
    if days > 0:
        r += '%dD' % days
    if hours > 0:
        r += '%dH' % hours
    if minutes > 0:
        r += '%dMin' % minutes
    if seconds > 0:
        r += '%dS' % seconds
    if rem > 0:
        r += '%dmS' % (rem // 1000000)
    return r


cdef nans(dims):
    """
    nans((M,N,P,...)) is an M-by-N-by-P-by-... array of NaNs.
    
    :param dims: dimensions tuple 
    :return: nans matrix 
    """
    return np.nan * np.ones(dims)


cdef inline long long floor_t64(long long time, long long dt):
    """
    Floor timestamp by dt
    """
    return time - time % dt


cpdef recognize_timeframe(timeframe):
    tf = timeframe
    if isinstance(timeframe, str):
        tf = np.int64(convert_tf_str_td64(timeframe).item().total_seconds() * NS)

    elif isinstance(timeframe, (int, float)) and timeframe >= 0:
        tf = timeframe

    elif isinstance(timeframe, np.timedelta64):
        tf = np.int64(timeframe.item().total_seconds() * NS) 

    else:
        raise ValueError('Unknown timeframe type !')
    return tf


cdef class RollingSum:
    """
    Rolling fast summator (TODO: move to utils)
    """
    cdef unsigned int period
    cdef np.ndarray __s
    cdef unsigned int __i
    cdef double rsum
    cdef unsigned short is_init_stage 

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
    cdef list values
    cdef float max_series_length
    cdef unsigned short _is_empty

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

    def clear(self):
        self.values.clear()
        self._is_empty = 1


cdef class TimeSeries:
    cdef public long long timeframe
    cdef public Indexed times
    cdef public Indexed values
    cdef float max_series_length
    cdef unsigned short _is_new_item
    cdef str name
    cdef dict indicators

    def __init__(self, str name, timeframe, max_series_length=INFINITY) -> None:
        self.name = name
        self.max_series_length = max_series_length
        self.timeframe = recognize_timeframe(timeframe)
        self.times = Indexed(max_series_length)
        self.values = Indexed(max_series_length)
        self.indicators = dict()

    def __len__(self) -> int:
        return len(self.times)

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

    def update(self, long long time, double value) -> short:
        item_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(item_start_time, value)

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False

        elif time - self.times[0] >= self.timeframe:
            # first we update indicators
            self._update_indicators(item_start_time, value, True)

            # then add new item
            self._add_new_item(item_start_time, value)
            return self._is_new_item
        else:
            self._update_last_item(item_start_time, value)

        # update indicators by new data
        self._update_indicators(item_start_time, value, False)

        return self._is_new_item

    cdef _update_indicators(self, long long time, value, short new_item_started):
        for i in self.indicators.values():
            i.update(time, value, new_item_started)

    def to_records(self) -> dict:
        ts = [np.datetime64(t, 'ns') for t in self.times[::-1]]
        return dict(zip(ts, self.values[::-1]))

    def to_series(self):
        return pd.Series(self.to_records(), name=self.name)

    def get_indicators(self) -> dict:
        return self.indicators

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


def _wrap_indicator(series: TimeSeries, clz, *args, **kwargs):
    aw = ','.join([str(a) for a in args])
    if kwargs:
        aw += ',' + ','.join([f"{k}={str(v)}" for k,v in kwargs.items()])
    nn = clz.__name__.lower() + "(" + aw + ")"
    inds = series.get_indicators()
    if nn in inds:
        return inds[nn]
    return clz(nn, series, *args, **kwargs) 


cdef class Indicator(TimeSeries):
    cdef TimeSeries series

    def __init__(self, str name, TimeSeries series):
        if not name:
            raise ValueError(f" > Name must not be empty for {self.__class__.__name__}!")
        super().__init__(name, series.timeframe, series.max_series_length)
        series.indicators[name] = self
        self.series = series 
        self._recalculate()

    def _recalculate(self):
        for t, v in zip(self.series.times[::-1], self.series.values[::-1]):
            self.update(t, v, True)

    def update(self, long long time, value, short new_item_started) -> any:
        iv = self.calculate(time, value, new_item_started)
        
        if new_item_started:
            self._add_new_item(time, iv)
        else:
            if len(self) > 0:
                self._update_last_item(time, iv)
            else:
                self._add_new_item(time, iv)

        # update attached indicators
        self._update_indicators(time, iv, self._is_new_item)

        return iv

    def calculate(self, long long time, value, short new_item_started) -> any:
        raise ValueError("Indicator must implement calculate() method")

    @classmethod
    def wrap(clz, series:TimeSeries, *args, **kwargs):
        return _wrap_indicator(series, clz, *args, **kwargs)


cdef class Sma(Indicator):
    cdef unsigned int period
    cdef RollingSum summator

    """
    Simple moving average
    """
    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.summator = RollingSum(period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double r = self.summator.update(value, new_item_started)
        return np.nan if self.summator.is_init_stage else r / self.period


def sma(series:TimeSeries, period: int): 
    return Sma.wrap(series, period)


cdef class Ema(Indicator):
    """
    Exponential moving average
    """
    cdef int period
    cdef np.ndarray __s
    cdef int __i
    cdef double alpha
    cdef double alpha_1
    cdef unsigned short init_mean 
    cdef unsigned short _init_stage

    def __init__(self, str name, TimeSeries series, int period, init_mean=True):
        self.period = period

        # when it's required to initialize this ema by mean on first period
        self.init_mean = init_mean
        if init_mean:
            self.__s = nans(period)
            self.__i = 0

        self._init_stage = 1
        self.alpha = 2.0 / (1.0 + period)
        self.alpha_1 = (1 - self.alpha)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef int prev_bar_idx = 0 if new_item_started else 1 

        if self._init_stage:
            if np.isnan(value): return np.nan

            if new_item_started:
                self.__i += 1
                if self.__i > self.period - 1:
                    self._init_stage = False
                    return self.alpha * value + self.alpha_1 * self[prev_bar_idx]

            if self.__i == self.period - 1:
                self.__s[self.__i] = value 
                return np.nansum(self.__s) / self.period

            self.__s[self.__i] = value 
            return np.nan

        if len(self) == 0:
            return value

        return self.alpha * value + self.alpha_1 * self[prev_bar_idx]


def ema(series:TimeSeries, period: int, init_mean: bool = True):
    return Ema.wrap(series, period, init_mean=init_mean)


cdef class Tema(Indicator):
    cdef int period
    cdef unsigned short init_mean 
    cdef TimeSeries ser0
    cdef Ema ema1
    cdef Ema ema2
    cdef Ema ema3

    def __init__(self, str name, TimeSeries series, int period, init_mean=True):
        self.period = period
        self.init_mean = init_mean
        self.ser0 = TimeSeries('ser0', series.timeframe, series.max_series_length)
        self.ema1 = ema(self.ser0, period, init_mean)
        self.ema2 = ema(self.ema1, period, init_mean)
        self.ema3 = ema(self.ema2, period, init_mean)
        super().__init__(name, series)
        
    cpdef double calculate(self, long long time, double value, short new_item_started):
        self.ser0.update(time, value)
        return 3 * self.ema1[0] - 3 * self.ema2[0] + self.ema3[0]


def tema(series:TimeSeries, period: int, init_mean: bool = True):
    return Tema.wrap(series, period, init_mean=init_mean)


cdef class Dema(Indicator):
    cdef int period
    cdef unsigned short init_mean 
    cdef TimeSeries ser0
    cdef Ema ema1
    cdef Ema ema2

    def __init__(self, str name, TimeSeries series, int period, init_mean=True):
        self.period = period
        self.init_mean = init_mean
        self.ser0 = TimeSeries('ser0', series.timeframe, series.max_series_length)
        self.ema1 = ema(self.ser0, period, init_mean)
        self.ema2 = ema(self.ema1, period, init_mean)
        super().__init__(name, series)
        
    cpdef double calculate(self, long long time, double value, short new_item_started):
        self.ser0.update(time, value)
        return 2 * self.ema1[0] - self.ema2[0]


def dema(series:TimeSeries, period: int, init_mean: bool = True):
    return Dema.wrap(series, period, init_mean=init_mean)


cdef class Kama(Indicator):
    cdef int period
    cdef int fast_span
    cdef int slow_span
    cdef double _S1 
    cdef double _K1 
    cdef _x_past
    cdef RollingSum summator

    def __init__(self, str name, TimeSeries series, int period, int fast_span=2, int slow_span=30):
        self.period = period
        self.fast_span = fast_span
        self.slow_span = slow_span
        self._S1 = 2.0 / (slow_span + 1)
        self._K1 = 2.0 / (fast_span + 1) - self._S1
        self._x_past = deque(nans(period+1), period+1)
        self.summator = RollingSum(period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if new_item_started:
            self._x_past.append(value)
        else:
            self._x_past[-1] = value

        cdef double rs = self.summator.update(abs(value - self._x_past[-2]), new_item_started)
        cdef double er = abs(value - self._x_past[0]) / rs
        cdef double sc = (er * self._K1 + self._S1) ** 2

        if self.summator.is_init_stage:
            if not np.isnan(self._x_past[1]):
                return value
            return np.nan

        return sc * value + (1 - sc) * self[0 if new_item_started else 1]


def kama(series:TimeSeries, period: int, fast_span:int=2, slow_span:int=30):
    return Kama.wrap(series, period, fast_span, slow_span)


cdef class Bar:
    cdef public long long time
    cdef public double open
    cdef public double high
    cdef public double low
    cdef public double close
    cdef public double volume

    def __init__(self, long long time, double open, double high, double low, double close, double volume) -> None:
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    cpdef Bar update(self, double price, double volume):
        self.close = price
        self.high = max(price, self.high)
        self.low = min(price, self.low)
        self.volume += volume
        return self

    cpdef dict to_dict(self, skip_time=False):
        if skip_time:
            return {
                'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close, 'volume': self.volume,
            }
        return {
            'timestamp': np.datetime64(self.time, 'ns'), 'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close, 'volume': self.volume,
        }

    def __repr__(self):
        return "{o:%f | h:%f | l:%f | c:%f | v:%f}" % (self.open, self.high, self.low, self.close, self.volume)


cdef class OHLCV(TimeSeries):
    cdef public TimeSeries open
    cdef public TimeSeries high
    cdef public TimeSeries low
    cdef public TimeSeries close
    cdef public TimeSeries volume

    def __init__(self, timeframe, max_series_length=INFINITY) -> None:
        super().__init__('OHLCV', timeframe, max_series_length)
        self.open = TimeSeries('open', timeframe, max_series_length)
        self.high = TimeSeries('high', timeframe, max_series_length)
        self.low = TimeSeries('low', timeframe, max_series_length)
        self.close = TimeSeries('close', timeframe, max_series_length)
        self.volume = TimeSeries('volume', timeframe, max_series_length)

    def _add_new_item(self, long long time, Bar value):
        self.times.add(time)
        self.values.add(value)
        self.open._add_new_item(time, value.open)
        self.high._add_new_item(time, value.high)
        self.low._add_new_item(time, value.low)
        self.close._add_new_item(time, value.close)
        self.volume._add_new_item(time, value.volume)
        self._is_new_item = True

    def _update_last_item(self, long long time, Bar value):
        self.times.update_last(time)
        self.values.update_last(value)
        self.open._update_last_item(time, value.open)
        self.high._update_last_item(time, value.high)
        self.low._update_last_item(time, value.low)
        self.close._update_last_item(time, value.close)
        self.volume._update_last_item(time, value.volume)
        self._is_new_item = False

    cpdef short update(self, long long time, double price, double volume=0.0):
        cdef Bar b
        bar_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(bar_start_time, Bar(bar_start_time, price, price, price, price, volume))

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False

        elif time - self.times[0] >= self.timeframe:
            # first we update indicators
            b = Bar(bar_start_time, price, price, price, price, volume)
            self._update_indicators(bar_start_time, b, True)

            # then add new item
            self._add_new_item(bar_start_time, b)
            return self._is_new_item
        else:
            self._update_last_item(bar_start_time, self[0].update(price, volume))

        # update indicators by new data
        self._update_indicators(bar_start_time, self[0], False)

        return self._is_new_item

    cpdef _update_indicators(self, long long time, value, short new_item_started):
        TimeSeries._update_indicators(self, time, value, new_item_started)
        if new_item_started:
            self.open._update_indicators(time, value.open, new_item_started)
        self.close._update_indicators(time, value.close, new_item_started)
        self.high._update_indicators(time, value.high, new_item_started)
        self.low._update_indicators(time, value.low, new_item_started)
        self.volume._update_indicators(time, value.volume, new_item_started)

    def to_records(self) -> dict:
        ts = [np.datetime64(t, 'ns') for t in self.times[::-1]]
        bs = [v.to_dict(skip_time=True) for v in self.values[::-1]]
        return dict(zip(ts, bs))

    def to_series(self):
        df = pd.DataFrame.from_dict(self.to_records(), orient='index')
        df.index.name = 'timestamp'
        return df

