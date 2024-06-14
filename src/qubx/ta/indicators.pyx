import numpy as np
import pandas as pd
cimport numpy as np
from scipy.special.cython_special import ndtri
from collections import deque

from qubx.core.series cimport TimeSeries, Indicator, IndicatorOHLC, RollingSum, nans, OHLCV, Bar
from qubx.pandaz.utils import scols, srows


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


cdef double norm_pdf(double x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


cdef double lognorm_pdf(double x, double s):
    return np.exp(-np.log(x) ** 2 / (2 * s ** 2)) / (x * s * np.sqrt(2 * np.pi))


cdef class Pewma(Indicator):
    cdef public TimeSeries std
    cdef double alpha, beta
    cdef int T

    cdef double _mean, _std, _var
    cdef long _i

    def __init__(self, str name, TimeSeries series, double alpha, double beta, int T):
        self.alpha = alpha 
        self.beta = beta
        self.T = T

        # - local variables
        self._i = 0
        self.std = TimeSeries('std', series.timeframe, series.max_series_length)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double x, short new_item_started):
        cdef double diff, p, a_t, incr, v

        if self._i < 1:
            self._mean = x
            self._std = 0.0
            self._var = 0.0
            incr = 0.0
            v = 0.0
        else:
            diff = x - self._mean
            # prob of observing diff
            p = norm_pdf(diff / self._std) if self._std != 0.0 else 0.0  

            # weight to give to this point
            a_t = self.alpha * (1 - self.beta * p) if self._i > self.T else (1.0 - 1.0 / self._i)  
            incr = (1.0 - a_t) * diff
            v = a_t * (self._var + diff * incr)
            self.std.update(time, np.sqrt(v))

        if new_item_started:
            self._mean += incr
            self._i += 1
            self._var = v
            self._std = np.sqrt(v)
            self.std.update(time, self._std)

        return self._mean


def pewma(series:TimeSeries, alpha: float, beta: float, T:int=30):
    """
    Implementation of probabilistic exponential weighted ma (https://sci-hub.shop/10.1109/SSP.2012.6319708)
    See pandas version here: qubx.pandaz.ta::pwma 
    """
    return Pewma.wrap(series, alpha, beta, T)


cdef class PewmaOutliersDetector(Indicator):
    cdef public TimeSeries upper
    cdef public TimeSeries lower
    cdef public TimeSeries outliers
    cdef double alpha, beta, threshold
    cdef int T

    cdef long _i
    cdef double _z_thr, _mean, _variance,_std

    def __init__(self, str name, TimeSeries series, double alpha, double beta, int T, double threshold):
        self.alpha = alpha 
        self.beta = beta
        self.T = T
        self.threshold = threshold

        # - series
        self.upper = TimeSeries('uba', series.timeframe, series.max_series_length)
        self.lower = TimeSeries('lba', series.timeframe, series.max_series_length)
        self.outliers = TimeSeries('outliers', series.timeframe, series.max_series_length)

        # - local variables
        self._i = 0
        self._z_thr = ndtri(1 - threshold / 2)

        self._mean = 0.0
        self._variance = 0.0
        self._std = 0.0
        super().__init__(name, series)

    cdef double _get_alpha(self, double p_t):
        if self._i == 0:
            return 0.0

        if self._i < self.T:
            return 1.0 - 1.0 / self._i

        return self.alpha * (1.0 - self.beta * p_t)

    cdef double _get_mean(self, double x, double alpha_t):
        return alpha_t * self._mean + (1.0 - alpha_t) * x

    cdef double _get_variance(self, double x, double alpha_t):
        return alpha_t * self._variance + (1.0 - alpha_t) * np.square(x)

    cdef double _get_std(self, double variance, double mean):
        return np.sqrt(max(variance - np.square(mean), 0.0))

    cdef double _get_p(self, double x):
        cdef double z_t = 0.0
        cdef double p_t

        if self._i != 1:
            z_t = ((x - self._mean) / self._std) if (self._std != 0 and not np.isnan(x)) else 0.0

        # if self.dist == 'normal':
        p_t = norm_pdf(z_t)
        # elif self.dist == 'cauchy':
        #     p_t = (1 / (np.pi * (1 + np.square(z_t))))
        # elif self.dist == 'student_t':
        #     p_t = (1 + np.square(z_t)) ** (-0.5 * (self.count - 1)) / \
        #           (np.sqrt(self.count - 1) * np.sqrt(np.pi) * np.exp(np.math.lgamma(0.5 * (self.count - 1))))
        # else:
        #     raise ValueError('Invalid distribution type')
        return p_t

    cpdef double calculate(self, long long time, double x, short new_item_started):
        cdef double p_t = self._get_p(x)
        cdef double alpha_t = self._get_alpha(p_t)
        cdef double mean = self._get_mean(x, alpha_t)
        cdef double variance = self._get_variance(x, alpha_t)
        cdef double std = self._get_std(variance, mean)
        cdef double ub = mean + self._z_thr * std
        cdef double lb = mean - self._z_thr * std

        self.upper.update(time, ub)
        self.lower.update(time, lb)
        if new_item_started:
            self._mean = mean
            self._i += 1
            self._variance = variance
            self._std = std

        # - check if it's outlier
        if p_t < self.threshold:
            self.outliers.update(time, x)
        else:
            self.outliers.update(time, np.nan)
        return mean


def pewma_outliers_detector(series:TimeSeries, alpha: float, beta: float, T:int=30, threshold=0.05):
    """
    Outliers detector based on pwma
    """
    return PewmaOutliersDetector.wrap(series, alpha, beta, T, threshold)


cdef class Psar(IndicatorOHLC):
    cdef int _bull
    cdef double _af
    cdef double _psar
    cdef double _lp
    cdef double _hp

    cdef int bull
    cdef double af
    cdef double psar
    cdef double lp
    cdef double hp

    cdef public TimeSeries upper
    cdef public TimeSeries lower

    cdef double iaf
    cdef double maxaf

    def __init__(self, name, series, iaf, maxaf):
        self.iaf = iaf
        self.maxaf = maxaf
        self.upper = TimeSeries('upper', series.timeframe, series.max_series_length)
        self.lower = TimeSeries('lower', series.timeframe, series.max_series_length)
        super().__init__(name, series)

    cdef _store(self):
        self.bull = self._bull
        self.af = self._af
        self.psar = self._psar
        self.lp = self._lp
        self.hp = self._hp

    cdef _restore(self):
        self._bull = self.bull
        self._af = self.af
        self._psar = self.psar
        self._lp = self.lp
        self._hp = self.hp

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        cdef short reverse = 1

        if len(self.series) <= 2:
            self._bull = 1
            self._af = self.iaf
            self._psar = bar.close

            if len(self.series) == 1:
                self._lp = bar.low
                self._hp = bar.high
            self._store()
            return self._psar

        if not new_item_started:
            self._store()
        else:
            self._restore()

        bar1 = self.series[1]
        bar2 = self.series[2]
        cdef double h0 = bar.high
        cdef double l0 = bar.low
        cdef double h1 = bar1.high
        cdef double l1 = bar1.low
        cdef double h2 = bar2.high
        cdef double l2 = bar2.low

        if self.bull:
            self.psar += self.af * (self.hp - self.psar)
        else:
            self.psar += self.af * (self.lp - self.psar)

        reverse = 0
        if self.bull:
            if l0 < self.psar:
                self.bull = 0
                reverse = 1
                self.psar = self.hp
                self.lp = l0
                self.af = self.iaf
        else:
            if h0 > self.psar:
                self.bull = 1
                reverse = 1
                self.psar = self.lp
                self.hp = h0
                self.af = self.iaf

        if not reverse:
            if self.bull:
                if h0 > self.hp:
                    self.hp = h0
                    self.af = min(self.af + self.iaf, self.maxaf)
                if l1 < self.psar:
                    self.psar = l1
                if l2 < self.psar:
                    self.psar = l2
            else:
                if l0 < self.lp:
                    self.lp = l0
                    self.af = min(self.af + self.iaf, self.maxaf)
                if h1 > self.psar:
                    self.psar = h1
                if h2 > self.psar:
                    self.psar = h2

        if self.bull:
            self.lower.update(time, self.psar)
            self.upper.update(time, np.nan)
        else:
            self.upper.update(time, self.psar)
            self.lower.update(time, np.nan)

        return self.psar


def psar(series: OHLCV, iaf: float=0.02, maxaf: float=0.2):
    if not isinstance(series, OHLCV):
        raise ValueError('Series must be OHLCV !')

    return Psar.wrap(series, iaf, maxaf)


# List of smoothing functions
_smoothers = {f.__name__: f for f in [pewma, ema, sma, kama, tema, dema]}


def smooth(TimeSeries series, str smoother, *args, **kwargs) -> Indicator:
    """
    Handy utility function to smooth series
    """
    _sfn = _smoothers.get(smoother)
    if _sfn is None:
        raise ValueError(f"Smoother {smoother} not found!")
    return _sfn(series, *args, **kwargs)


cdef class Atr(IndicatorOHLC):
    cdef short percentage
    cdef TimeSeries tr
    cdef Indicator ma

    def __init__(self, str name, OHLCV series, int period, str smoother, short percentage):
        self.percentage = percentage
        self.tr = TimeSeries("tr", series.timeframe, series.max_series_length)
        self.ma = smooth(self.tr, smoother, period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        if len(self.series) <= 1:
            return np.nan

        cdef double c1 = self.series[1].close
        cdef double h_l = abs(bar.high - bar.low)
        cdef double h_pc = abs(bar.high - c1)
        cdef double l_pc = abs(bar.low - c1)
        self.tr.update(time, max(h_l, h_pc, l_pc))
        return (100 * self.ma[0] / c1) if self.percentage else self.ma[0]


def atr(series: OHLCV, period: int = 14, smoother="sma", percentage: bool = False):
    if not isinstance(series, OHLCV):
        raise ValueError("Series must be OHLCV !")
    return Atr.wrap(series, period, smoother, percentage)


cdef class Swings(IndicatorOHLC):
    cdef double _min_l
    cdef long long _min_t
    cdef double _max_h
    cdef long long _max_t
    cdef OHLCV base
    cdef Indicator trend
    cdef public TimeSeries tops
    cdef public TimeSeries bottoms

    def __init__(self, str name, OHLCV series, trend_indicator, **indicator_args):
        self.base = OHLCV("base", series.timeframe, series.max_series_length)
        self.trend = trend_indicator(self.base, **indicator_args)
        self.tops = TimeSeries("tops", series.timeframe, series.max_series_length)
        self.bottoms = TimeSeries("bottoms", series.timeframe, series.max_series_length)
        self._min_l = +np.inf
        self._max_h = -np.inf
        self._max_t = 0
        self._min_t = 0
        super().__init__(name, series)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        self.base.update_by_bar(time, bar.open, bar.high, bar.low, bar.close, bar.volume)
        cdef int _t = 0

        if len(self.trend.upper) > 0:
            _u = self.trend.upper[0]
            _d = self.trend.lower[0]

            if not np.isnan(_u):
                if self._max_t > 0:
                    self.tops.update(self._max_t, self._max_h)

                if bar.low <= self._min_l:
                    self._min_l = bar.low
                    self._min_t = time

                self._max_h = -np.inf
                self._max_t = 0
                _t = -1
            elif not np.isnan(_d):
                if self._min_t > 0:
                    self.bottoms.update(self._min_t, self._min_l)

                if bar.high >= self._max_h:
                    self._max_h = bar.high
                    self._max_t = time

                self._min_l = +np.inf
                self._min_t = 0
                _t = +1

        return _t

    def get_current_trend_end(self):
        if np.isfinite(self._min_l):
            return pd.Timestamp(self._min_t, 'ns'), self._min_l
        elif np.isfinite(self._max_h):
            return pd.Timestamp(self._max_t, 'ns'), self._max_h
        return (None, None)

    def pd(self) -> pd.DataFrame:
        _t, _d = self.get_current_trend_end()
        tps, bts = self.tops.pd(), self.bottoms.pd()
        if _t is not None:
            if bts.index[-1] < tps.index[-1]:
                bts = srows(bts, pd.Series({_t: _d}))
            else:
                tps = srows(tps, pd.Series({_t: _d}))

        eid = pd.Series(tps.index, tps.index)
        mx = scols(bts, tps, eid, names=["start_price", "end_price", "end"])
        dt = scols(mx["start_price"], mx["end_price"].shift(-1), mx["end"].shift(-1))  # .dropna()
        dt = dt.assign(delta = dt["end_price"] - dt["start_price"])

        eid = pd.Series(bts.index, bts.index)
        mx = scols(tps, bts, eid, names=["start_price", "end_price", "end"])
        ut = scols(mx["start_price"], mx["end_price"].shift(-1), mx["end"].shift(-1))  # .dropna()
        return scols(ut, dt, keys=["DownTrends", "UpTrends"])


def swings(series: OHLCV, trend_indicator, **indicator_args):
    """
    Swing detector based on provided trend indicator.
    """
    if not isinstance(series, OHLCV):
        raise ValueError("Series must be OHLCV !")
    return Swings.wrap(series, trend_indicator, **indicator_args)