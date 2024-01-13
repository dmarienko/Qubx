from datetime import datetime
from typing import List, Optional, Union
import numpy as np
import re

import pandas as pd

UNIX_T0 = np.datetime64('1970-01-01T00:00:00')


time_to_str = lambda t, u='us': np.datetime_as_string(t if isinstance(t, np.datetime64) else np.datetime64(t, u), unit=u)


def convert_tf_str_td64(c_tf: str) -> np.timedelta64:
    """
    Convert string timeframe to timedelta64

    '15Min' -> timedelta64(15, 'm') etc
    """
    _t = re.findall('(\d+)([A-Za-z]+)', c_tf)
    _dt = 0
    for g in _t:
        unit = g[1].lower()
        n = int(g[0])
        u1 = unit[0]
        u2 = unit[:2]
        unit = u1

        if u1 in ['d', 'w']:
            unit = u1.upper()

        if u1 in ['y']:
            n = 356 * n
            unit = 'D'

        if u2 in ['ms', 'ns', 'us', 'ps']:
            unit = u2

        _dt += np.timedelta64(n, unit)
    
    return _dt


def convert_seconds_to_str(seconds: int) -> str:
    """
    Convert seconds to string representation: 310 -> '5Min10S' etc
    """
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    r = ''
    if days > 0:
        r += '%dD' % days
    if hours > 0:
        r += '%dH' % hours
    if minutes > 0:
        r += '%dMin' % minutes
    if seconds > 0:
        r += '%dS' % seconds
    return r


def floor_t64(time: Union[np.datetime64, datetime], dt: Union[np.timedelta64, int, str]):
    """
    Floor timestamp by dt
    """
    if isinstance(dt, int):
        dt = np.timedelta64(dt, 's')

    if isinstance(dt, str):
        dt = convert_tf_str_td64(dt)

    if isinstance(time, datetime):
        time = np.datetime64(time)

    return time - (time - UNIX_T0) % dt


def infer_series_frequency(series: Union[List, pd.DataFrame, pd.Series, pd.DatetimeIndex]):
    """
    Infer frequency of given timeseries

    :param series: Series, DataFrame, DatetimeIndex or list of timestamps object
    :return: timedelta for found frequency
    """
    if isinstance(series, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
        times_index = (series if isinstance(series, pd.DatetimeIndex) else series.index).to_pydatetime()
    elif isinstance(series, (set, list, tuple)):
        times_index = np.array(series)
    else:
        raise ValueError("Can't recognize input data")

    if times_index.shape[0] < 2:
        raise ValueError("Series must have at least 2 points to determ frequency")

    values = np.array(sorted([(x if isinstance(x, np.timedelta64) else x.total_seconds()) for x in np.diff(times_index)]))
    diff = np.concatenate(([1], np.diff(values)))
    idx = np.concatenate((np.where(diff)[0], [len(values)]))
    freqs = dict(zip(values[idx[:-1]], np.diff(idx)))
    return np.timedelta64(max(freqs, key=freqs.get))