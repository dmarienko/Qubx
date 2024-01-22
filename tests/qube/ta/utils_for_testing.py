import types
from typing import Tuple, List

import numpy as np
import pandas as pd

from qube.core.series import TimeSeries, recognize_time
from pytest import approx


N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)


def push(series: TimeSeries, ds: List[Tuple], v=None):
    """
    Update series by data from the input 
    """
    for t, d in ds:
        if isinstance(t, str):
            t = recognize_time(t)
        elif isinstance(t, pd.Timestamp):
            t = t.asm8
        if isinstance(d, (list, tuple)):
            series.update(t, d[0], d[1])
        else:
            series.update(t, d) if v is None else series.update(t, d, v) 


def shift(xs: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = fill
        e[n:] = xs[:-n]
    else:
        e[n:] = fill
        e[:n] = xs[-n:]
    return e


def column_vector(x):
    if isinstance(x, (pd.DataFrame, pd.Series)): x = x.values
    return np.reshape(x, (x.shape[0], -1))


def sink_nans_down(x_in, copy=False) -> Tuple[np.ndarray, np.ndarray]:
    x = np.copy(x_in) if copy else x_in
    n_ix = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        f_n = np.where(~np.isnan(x[:, i]))[0]
        if len(f_n) > 0:
            if f_n[0] != 0:
                x[:, i] = np.concatenate((x[f_n[0]:, i], x[:f_n[0], i]))
            n_ix[i] = f_n[0]
    return x, n_ix


def lift_nans_up(x_in, n_ix, copy=False) -> np.ndarray:
    x = np.copy(x_in) if copy else x_in
    for i in range(0, x.shape[1]):
        f_n = int(n_ix[i])
        if f_n != 0:
            x[:, i] = np.concatenate((nans(f_n), x[:-f_n, i]))
    return x


def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    for i in range(0, x.shape[1]):
        ret = np.nancumsum(x[:, i])
        ret[n:] = ret[n:] - ret[:-n]
        x[:, i] = np.concatenate((nans(n - 1), ret[n - 1:]))
    return x


def nans(dims):
    return np.nan * np.ones(dims)


def apply_to_frame(func, x, *args, **kwargs):
    _keep_names = False
    if 'keep_names' in kwargs:
        _keep_names = kwargs.pop('keep_names')

    if func is None or not isinstance(func, types.FunctionType):
        raise ValueError(str(func) + ' must be callable object')

    xp = column_vector(func(x, *args, **kwargs))
    _name = None
    if not _keep_names:
        _name = func.__name__ + '_' + '_'.join([str(i) for i in args])

    if isinstance(x, pd.DataFrame):
        c_names = x.columns if _keep_names else ['%s_%s' % (c, _name) for c in x.columns]
        return pd.DataFrame(xp, index=x.index, columns=c_names)
    elif isinstance(x, pd.Series):
        return pd.Series(xp.flatten(), index=x.index, name=_name)

    return xp


def sma(x, period):
    """
    Classical simple moving average

    :param x: input data (as np.array or pd.DataFrame/Series)
    :param period: period of smoothing
    :return: smoothed values
    """
    if period <= 0:
        raise ValueError('Period must be positive and greater than zero !!!')

    x = column_vector(x)
    x, ix = sink_nans_down(x, copy=True)
    s = rolling_sum(x, period) / period
    return lift_nans_up(s, ix)


def _calc_ema(x, span, init_mean=True, min_periods=0):
    alpha = 2.0 / (1 + span)
    x = x.astype(np.float64)
    for i in range(0, x.shape[1]):
        nan_start = np.where(~np.isnan(x[:, i]))[0][0]
        x_s = x[:, i][nan_start:]
        a_1 = 1 - alpha
        s = np.zeros(x_s.shape)

        start_i = 1
        if init_mean:
            s += np.nan
            if span - 1 >= len(s):
                x[:, :] = np.nan
                continue
            s[span - 1] = np.mean(x_s[:span])
            start_i = span
        else:
            s[0] = x_s[0]

        for n in range(start_i, x_s.shape[0]):
            s[n] = alpha * x_s[n] + a_1 * s[n - 1]

        if min_periods > 0:
            s[:min_periods - 1] = np.nan

        x[:, i] = np.concatenate((nans(nan_start), s))

    return x


def ema(x, span, init_mean=True, min_periods=0) -> np.ndarray:
    return _calc_ema(column_vector(x), span, init_mean, min_periods)


def tema(x, n: int, init_mean=True):
    e1 = ema(x, n, init_mean=init_mean)
    e2 = ema(e1, n, init_mean=init_mean)
    return 3 * e1 - 3 * e2 + ema(e2, n, init_mean=init_mean)


def kama(xs, period=10, period_fast=2, period_slow=30):
    #Efficiency Ratio
    change = abs(xs - xs.shift(period))
    volatility = (abs(xs - xs.shift())).rolling(period).sum()
    er = change / volatility

    #Smoothing Constant
    sc_fatest = 2/(period_fast + 1)
    sc_slowest = 2/(period_slow + 1)
    sc = (er * (sc_fatest - sc_slowest) + sc_slowest)**2

    #KAMA
    kama=np.zeros_like(xs)
    kama[period-1] = xs[period-1]
    for i in range(period, len(xs)):
        kama[i] = kama[i-1] + sc[i] * (xs[i] - kama[i-1])
    kama[kama==0]=np.nan

    return kama


def dema(x, n: int, init_mean=True):
    e1 = ema(x, n, init_mean=init_mean)
    return 2 * e1 - ema(e1, n, init_mean=init_mean)
