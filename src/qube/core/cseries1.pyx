# - experimental 2 -
import numpy as np
cimport numpy as np
from cpython.datetime cimport datetime
from qube.utils import convert_tf_str_td64
import pandas as pd


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


cdef class Indexed:
    cdef list values
    cdef float max_series_length

    def __init__(self, max_series_length=INFINITY):
        self.max_series_length = max_series_length
        self.values = list()

    def __len__(self) -> int:
        return len(self.values)

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
        if len(self.values) >= self.max_series_length:
            self.values.pop(0)

    def update_last(self, v):
        if self.values:
            self.values[-1] = v
        else:
            self.append(v)


cdef class TimeSeries:
    cdef public long long timeframe
    cdef public Indexed times
    cdef public Indexed values
    cdef float max_series_length
    cdef unsigned short _is_new_item
    cdef str name
    cdef indicators

    def __init__(self, str name, timeframe, max_series_length=INFINITY) -> None:
        self.name = name
        self.max_series_length = max_series_length
        self.timeframe = recognize_timeframe(timeframe)
        self.times = Indexed(max_series_length)
        self.values = Indexed(max_series_length)
        self.indicators = list()

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
            # print(f" + [{time_to_str(item_start_time)[:19]}] <NEW> {value}\t| ", end='')
            self._update_indicators(item_start_time, value, True)

            # then add new item
            self._add_new_item(item_start_time, value)
            return self._is_new_item
        else:
            self._update_last_item(item_start_time, value)

        # update indicators by new data
        # print(f" - [{time_to_str(item_start_time)[:19]}] <LST> {value}\t| ", end='')

        self._update_indicators(item_start_time, value, False)

        return self._is_new_item

    cdef _update_indicators(self, long long time, value, short new_item_started):
        for i in self.indicators:
            i.update(time, value, new_item_started)

    def to_records(self) -> dict:
        ts = [np.datetime64(t, 'ns') for t in self.times[::-1]]
        return dict(zip(ts, self.values[::-1]))

    def to_series(self):
        return pd.Series(self.to_records(), name=self.name)

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


cdef class Indicator(TimeSeries):
    def __init__(self, TimeSeries series):
        super().__init__(self.name(), series.timeframe, series.max_series_length)
        series.indicators.append(self)

    def name(self) -> str:
        return 'none'

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
        pass


cdef class Sma(Indicator):
    cdef unsigned int period
    cdef np.ndarray __s
    cdef unsigned int __i
    cdef double _r_sum
    cdef unsigned short _init_stage

    """
    Simple moving average
    """
    def __init__(self, TimeSeries series, int period):
        self.period = period
        self.__s = np.zeros(period)
        self.__i = 0
        self._r_sum = 0.0
        self._init_stage = 1
        super().__init__(series)

    def name(self) -> str:
        return f'sma{self.period}'

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if np.isnan(value):
            return np.nan
        sub = self.__s[self.__i]
        if new_item_started:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0
                self._init_stage = 0
            sub = self.__s[self.__i]
        self.__s[self.__i] = value
        self._r_sum -= sub
        self._r_sum += value 
        return np.nan if self._init_stage else self._r_sum / self.period


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

    def __init__(self, TimeSeries series, int period, init_mean=True):
        self.period = period

        # when it's required to initialize this ema by mean on first period
        self.init_mean = init_mean
        if init_mean:
            self.__s = nans(period)
            self.__i = 0

        self._init_stage = 1
        self.alpha = 2.0 / (1.0 + period)
        self.alpha_1 = (1 - self.alpha)
        super().__init__(series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef int p_idx = 0 if new_item_started else 1 

        # - - - - - - - -
        if self._init_stage:
            if np.isnan(value): return np.nan

            if new_item_started:
                self.__i += 1
                if self.__i > self.period - 1:
                    self._init_stage = False
                    # print(' >>> (STANDARD A)', self[p_idx], self.alpha * value + self.alpha_1 * self[p_idx])
                    return self.alpha * value + self.alpha_1 * self[p_idx]

            if self.__i == self.period - 1:
                self.__s[self.__i] = value 
                # print(' >>> update last in init', self.__s)
                return np.nansum(self.__s) / self.period

            self.__s[self.__i] = value 
            # print(' -> ret NAN | ', self.__s)
            return np.nan
        # - - - - - - - -

        if len(self) == 0:
            return value

        # print(' >>> (STANDARD X)', self[p_idx], self.alpha * value + self.alpha_1 * self[p_idx])
        return self.alpha * value + self.alpha_1 * self[p_idx]
