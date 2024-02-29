import numpy as np
cimport numpy as np
from collections import deque

from qubx.core.series cimport TimeSeries, Indicator, RollingSum, nans


cdef extern from "math.h":
    float INFINITY


cdef class Sma(Indicator):
    """
    Simple Moving Average indicator
    """
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


cdef class Highest(Indicator):
    cdef int period
    cdef queue

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.queue = deque([np.nan] * period, maxlen=period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        """
        Not a most effictive algo but simplest and can handle updated last value
        """
        cdef float r = np.nan

        if not np.isnan(value):
            if new_item_started:
                self.queue.append(value)
            else:
                self.queue[-1] = value

        if not np.isnan(self.queue[0]):
            r = max(self.queue) 

        return r


def highest(series:TimeSeries, period:int):
    return Highest.wrap(series, period)


cdef class Lowest(Indicator):
    cdef int period
    cdef queue

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.queue = deque([np.nan] * period, maxlen=period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        """
        Not a most effictive algo but simplest and can handle updated last value
        """
        cdef float r = np.nan

        if not np.isnan(value):
            if new_item_started:
                self.queue.append(value)
            else:
                self.queue[-1] = value

        if not np.isnan(self.queue[0]):
            r = min(self.queue) 

        return r


def lowest(series:TimeSeries, period:int):
    return Lowest.wrap(series, period)


# - - - - TODO !!!!!!!
cdef class Std(Indicator):
    cdef int period

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        pass


def std(series:TimeSeries, period:int, mean=0):
    return Std.wrap(series, period)
# - - - - TODO !!!!!!!
