import types
import numpy as np
import pandas as pd


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


def shift(xs: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = fill
        e[n:] = xs[:-n]
    else:
        e[n:] = fill
        e[:n] = xs[-n:]
    return e


def nans(dims):
    return np.nan * np.ones(dims)


def column_vector(x):
    if isinstance(x, (pd.DataFrame, pd.Series)): x = x.values
    return np.reshape(x, (x.shape[0], -1))


def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    for i in range(0, x.shape[1]):
        ret = np.nancumsum(x[:, i])
        ret[n:] = ret[n:] - ret[:-n]
        x[:, i] = np.concatenate((nans(n - 1), ret[n - 1:]))
    return x


def _calc_kama(x, period, fast_span, slow_span):
    x = x.astype(np.float64)
    for i in range(0, x.shape[1]):
        nan_start = np.where(~np.isnan(x[:, i]))[0][0]
        x_s = x[:, i][nan_start:]
        if period >= len(x_s):
            raise ValueError('Wrong value for period. period parameter must be less than number of input observations')
        abs_diff = np.abs(x_s - shift(x_s, 1))
        er = np.abs(x_s - shift(x_s, period)) / rolling_sum(np.reshape(abs_diff, (len(abs_diff), -1)), period)[:, 0]
        sc = np.square((er * (2.0 / (fast_span + 1) - 2.0 / (slow_span + 1.0)) + 2 / (slow_span + 1.0)))
        ama = nans(sc.shape)

        # here ama_0 = x_0
        ama[period - 1] = x_s[period - 1]
        for n in range(period, len(ama)):
            ama[n] = ama[n - 1] + sc[n] * (x_s[n] - ama[n - 1])

        # drop 1-st kama value (just for compatibility with ta-lib)
        ama[period - 1] = np.nan

        x[:, i] = np.concatenate((nans(nan_start), ama))

    return x


def kama(x, period, fast_span=2, slow_span=30):
    x = column_vector(x)
    return _calc_kama(x, period, fast_span, slow_span)


def kama_indicator(price, period=10, period_fast=2, period_slow=30):
    #Efficiency Ratio
    change = abs(price-price.shift(period))
    volatility = (abs(price-price.shift())).rolling(period).sum()
    er = change/volatility
    # return er

    #Smoothing Constant
    sc_fatest = 2/(period_fast + 1)
    sc_slowest = 2/(period_slow + 1)
    sc= (er * (sc_fatest - sc_slowest) + sc_slowest)**2
    # return sc

    #KAMA
    kama = np.zeros_like(price)
    kama[period-1] = price[period-1]
    for i in range(period, len(price)):
        kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
    kama[kama==0]=np.nan

    return kama