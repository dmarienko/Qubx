from typing import Callable, Dict, Optional, Union
from datetime import timedelta
import pandas as pd
import numpy as np

from numpy.lib.stride_tricks import as_strided as stride

from qubx.utils.misc import Struct


def drop_duplicated_indexes(df, keep='first'):
    """
    Drops duplicated indexes in dataframe/series
    Keeps either first or last occurence (parameter keep)
    """
    return df[~df.index.duplicated(keep=keep)]


def process_duplicated_indexes(data: Union[pd.DataFrame, pd.Series], ns=1) -> Union[pd.DataFrame, pd.Series]:
    """
    Finds duplicated indexes in frame/series and add shift (in nS) to every repeating one
    :param data: time indexed dataframe/series
    :param ns: shift constant in nanosec
    :return: return dataframe with all no duplicated rows (each duplicate has own unique index)
    """
    values = data.index.duplicated(keep='first').astype(float)
    values[values == 0] = np.NaN

    missings = np.isnan(values)
    cumsum = np.cumsum(~missings)
    diff = np.diff(np.concatenate(([0.], cumsum[missings])))
    values[missings] = -diff

    # set new index (1 ms)
    data.index = data.index.values + np.cumsum(values).astype(np.timedelta64) * ns
    return data


def scols(*xs, keys=None, names=None, keep='all') -> pd.DataFrame:
    """
    Concat dataframes/series from xs into single dataframe by axis 1
    :param keys: keys of new dataframe (see pd.concat's keys parameter)
    :param names: new column names or dict with replacements
    :return: combined dataframe
    
    Example
    -------
    >>>  scols(
            pd.DataFrame([1,2,3,4,-4], list('abcud')),
            pd.DataFrame([111,21,31,14], list('xyzu')), 
            pd.DataFrame([11,21,31,124], list('ertu')), 
            pd.DataFrame([11,21,31,14], list('WERT')), 
            names=['x', 'y', 'z', 'w'])
    """
    r = pd.concat((xs), axis=1, keys=keys)
    if names:
        if isinstance(names, (list, tuple)):
            if len(names) == len(r.columns):
                r.columns = names
            else:
                raise ValueError(
                    f"if 'names' contains new column names it must have same length as resulting df ({len(r.columns)})")
        elif isinstance(names, dict):
            r = r.rename(columns=names)
    return r


def srows(*xs, keep='all', sort=True) -> Union[pd.DataFrame, pd.Series]:
    """
    Concat dataframes/series from xs into single dataframe by axis 0
    :param sort: if true it sorts resulting dataframe by index (default)
    :param keep: how to deal with duplicated indexes. 
                 If set to 'all' it doesn't do anything (default). Otherwise keeps first or last occurences
    :return: combined dataframe
    
    Example
    -------
    >>>  srows(
            pd.DataFrame([1,2,3,4,-4], list('abcud')),
            pd.DataFrame([111,21,31,14], list('xyzu')), 
            pd.DataFrame([11,21,31,124], list('ertu')), 
            pd.DataFrame([11,21,31,14], list('WERT')), 
            sort=True, keep='last')
    """
    r = pd.concat((xs), axis=0)
    r = r.sort_index() if sort else r
    if keep != 'all':
        r = drop_duplicated_indexes(r, keep=keep)
    return r


def retain_columns_and_join(data: dict, columns) -> pd.DataFrame:
    """
    Retains given columns from every value of data dictionary and concatenate them into single data frame

    from qube.datasource import DataSource
    from qube.analysis.tools import retain_columns_and_join

    ds = DataSource('yahoo::daily')
    data = ds.load_data(['aapl', 'msft', 'spy'], '2000-01-01', 'now')

    closes = retain_columns_and_join(data, 'close')
    hi_lo = retain_columns_and_join(data, ['high', 'low'])

    :param data: dictionary with dataframes
    :param columns: columns names need to be retained
    :return: data frame
    """
    if not isinstance(data, dict):
        raise ValueError('Data must be passed as dictionary')

    return pd.concat([data[k][columns] for k in data.keys()], axis=1, keys=data.keys())


def continuous_periods(xs, cond) -> Struct:
    """
    Detect continues periods on series xs based on condition cond
    """
    df = scols(xs, cond, keys=['_XS_', 'sig'])
    df['block'] = (df.sig.shift(1) != df.sig).astype(int).cumsum()
    idx_col_name = xs.index.name

    blk = df[df.sig].reset_index().groupby('block')[idx_col_name].apply(np.array)
    starts = blk.apply(lambda x: x[0])
    ends = blk.apply(lambda x: x[-1])
    se_info = scols(starts, ends, keys=['start', 'end'])
    return Struct(blocks=blk.reset_index(drop=True), periods=se_info)


def roll(df: pd.DataFrame, w: int, **kwargs):
    """
    Rolling window on dataframe using multiple columns
    
    >>> roll(pd.DataFrame(np.random.randn(10,3), index=list('ABCDEFGHIJ')), 3).apply(print)
    
    or alternatively 
    
    >>> pd.DataFrame(np.random.randn(10,3), index=list('ABCDEFGHIJ')).pipe(roll, 3).apply(lambda x: print(x[2]))
    
    :param df: pandas DataFrame
    :param w: window size (only integers)
    :return: rolling window
    """
    if w > len(df):
        raise ValueError("Window size exceeds number of rows !")

    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides
    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))
    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)


def dict_to_frame(x: dict, index_type=None, orient='index', columns=None, column_types=dict()) -> pd.DataFrame:
    """
    Utility for convert dictionary to indexed DataFrame
    It's possible to pass columns names and type of index
    """
    y = pd.DataFrame().from_dict(x, orient=orient)
    if index_type:
        if index_type in ['ns', 'nano']:
            index_type = 'M8[ns]'
        y.index = y.index.astype(index_type)

    # rename if needed
    if columns:
        columns = [columns] if not isinstance(columns, (list, tuple, set)) else columns
        if len(columns) == len(y.columns):
            y.rename(columns=dict(zip(y.columns, columns)), inplace=True)
        else:
            raise ValueError('dict_to_frame> columns argument must contain %d elements' % len(y.columns))

    # if additional conversion is required
    if column_types:
        _existing_cols_conversion = {c: v for c, v in column_types.items() if c in y.columns}
        y = y.astype(_existing_cols_conversion)

    return y


def select_column_and_join(data: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """
    Select given column from every value of data dictionary and concatenate them into single data frame

    from qube.datasource import DataSource
    from qube.analysis.tools import retain_columns_and_join

    ds = DataSource('yahoo::daily')
    data = ds.load_data(['aapl', 'msft', 'spy'], '2000-01-01', 'now')

    closes = select_column_and_join(data, 'close')
    hi_lo = select_column_and_join(data, ['high', 'low'])

    :param data: dictionary with dataframes
    :param columns: column name need to be selected
    :return: pandas data frame
    """
    if not isinstance(data, dict):
        raise ValueError('Data must be passed as dictionary of pandas dataframes')

    return pd.concat([data[k][column] for k in data.keys()], axis=1, keys=data.keys())


def merge_columns_by_op(x: pd.DataFrame, y: pd.DataFrame, op):
    """
    Merge 2 dataframes into one and performing operation on intersected columns
    
    merge_columns_by_op(
        pd.DataFrame({'A': [1,2,3], 'B': [100,200,300]}), 
        pd.DataFrame({'B': [5,6,7], 'C': [10,20,30]}), 
        lambda x,y: x + y 
    )
    
        B 	    A   C
    0 	105 	1 	10
    1 	206 	2 	20
    2 	307 	3 	30
    
    """
    if x is None or x.empty: return y
    if y is None: return x
    r = []
    uc = set(x.columns & y.columns)
    for c in uc:
        r.append(op(x[c], y[c]))

    for c in set(x.columns) - uc:
        r.append(x[c])

    for c in set(y.columns) - uc:
        r.append(y[c])

    return scols(*r)


def bands_signals(
        prices: pd.DataFrame, 
        score: pd.DataFrame, 
        entry, exit, stop: Optional[float]=np.inf,
        entry_method='cross-out', # 'cross-in', 'revert-to-band'
        position_sizes_fn=lambda time, score, prices, side: np.zeros(len(p))
    ) -> pd.DataFrame:
    """
    Generate trading signals using score and entry / exit thresholds
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError('Prices must be a pandas DataFrame')

    _as_series = lambda xs, index, name: pd.Series(xs, index=index, name=name) if isscalar(xs) else xs

    # - entry function
    ent_fn: Callable[[float, float, float, float], int] = lambda t, s2, s1, s0: 0

    match entry_method:
        case 'cross-in':
            ent_fn = lambda t, s2, s1, s0:\
                +1 if (s2 < -t and s1 <= -t and s0 > -t) else\
                -1 if (s2 > +t and s1 >= +t and s0 < +t) else\
                 0

        case 'cross-out':
            ent_fn = lambda t, s2, s1, s0:\
                +1 if (s2 >= -t and s1 >= -t and s0 < -t) else\
                -1 if (s2 <= +t and s1 <= +t and s0 > +t) else\
                 0

        case 'revert-to-band':
            ent_fn = lambda t, s2, s1, s0:\
                +1 if (s1 <= -t and s0 < -t and s0 > s1) else\
                -1 if (s1 >= +t and s0 > +t and s0 < s1) else\
                 0

        case _:
            raise ValueError(f'Unknown entry_method: {entry_method}, supported methods are: cross-out, cross-in, revert-to-band')
    
    entry = abs(_as_series(entry, score.index, 'entry'))
    exit = _as_series(exit, score.index, 'exit')
    stop = abs(_as_series(stop, score.index, 'stop'))

    mx = scols(prices, scols(score.rename('score'), entry, exit, stop), keys=['price', 'service']).dropna()
    px:pd.DataFrame = mx['price']
    sx:pd.DataFrame = mx['service']
    P0 = np.zeros(px.shape[1]) # zero position

    signals = {}
    pos = 0
    s1, s2 = np.nan, np.nan

    for t, pi, (s0, en, ex, stp) in zip(px.index, px.values, sx.values):
        if pos == 0: # no positions yet
            # - only if there are at least 2 past scores available
            if not np.isnan(s1) and not np.isnan(s2):
                if (side := ent_fn(en, s2, s1, s0)) != 0:
                    signals[t] = position_sizes_fn(t, s0, pi, side)
                    pos = side  # - sell or buy spread 

        elif pos == -1:
            # - check for stop or exit
            if s0 >= +stp or s0 <= +ex:
                signals[t] = P0
                pos = 0

        elif pos == +1:
            # - check for stop or exit
            if s0 <= -stp or s0 >= -ex:
                signals[t] = P0
                pos = 0

        s1, s2 = s0, s1

    return pd.DataFrame.from_dict(signals, orient='index', columns=px.columns)