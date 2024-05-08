import re, os
from typing import Callable, List, Union, Optional, Iterable, Any
from os.path import exists, join
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv
import psycopg as pg
from functools import wraps

from qubx import logger
from qubx.core.series import TimeSeries, OHLCV, time_as_nsec, Quote, Trade
from qubx.utils.time import infer_series_frequency, handle_start_stop

_DT = lambda x: pd.Timedelta(x).to_numpy().item()
D1, H1 = _DT('1D'), _DT('1h')

DEFAULT_DAILY_SESSION = (_DT('00:00:00.100'), _DT('23:59:59.900'))
STOCK_DAILY_SESSION = (_DT('9:30:00.100'), _DT('15:59:59.900'))
CME_FUTURES_DAILY_SESSION = (_DT('8:30:00.100'), _DT('15:14:59.900'))


def _recognize_t(t: Union[int, str], defaultvalue, timeunit) -> int:
    if isinstance(t, (str, pd.Timestamp)):
        try:
            return np.datetime64(t, timeunit)
        except:
            pass
    return defaultvalue


def _find_column_index_in_list(xs, *args):
    xs = [x.lower() for x in xs]
    for a in args:
        ai = a.lower()
        if ai in xs:
            return xs.index(ai)
    raise IndexError(f"Can't find any from {args} in list: {xs}")


class DataTransformer:

    def __init__(self) -> None:
        self.buffer = []
        self._column_names = []

    def start_transform(self, name: str, column_names: List[str]):
        self._column_names = column_names
        self.buffer = []
    
    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None: 
            self.buffer.extend(rows_data)

    def collect(self) -> Any:
        return self.buffer


class DataReader:

    def get_names(self) -> List[str] :
        raise NotImplemented()

    def read(self, data_id: str, start: str | None=None, stop: str | None=None, 
             transform: DataTransformer = DataTransformer(), 
             chunksize=0, 
             **kwargs
            ) -> Iterable | List:
        raise NotImplemented()


class CsvStorageDataReader(DataReader):
    """
    Data reader for timeseries data stored as csv files in the specified directory
    """

    def __init__(self, path: str) -> None:
        if not exists(path):
            raise ValueError(f"Folder is not found at {path}")
        self.path = path

    def __find_time_idx(self, arr: pa.ChunkedArray, v) -> int:
        ix = arr.index(v).as_py()
        if ix < 0:
            for c in arr.iterchunks():
                a = c.to_numpy()
                ix = np.searchsorted(a, v, side='right')
                if ix > 0 and ix < len(c):
                    ix = arr.index(a[ix]).as_py() - 1
                    break
        return ix

    def __check_file_name(self, name: str) -> str | None:
        _f = join(self.path, name)
        for sfx in ['.csv', '.csv.gz', '']:
            if exists(p:=(_f + sfx)):
                return p 
        return None

    def read(self, data_id: str, start: str | None=None, stop: str | None=None, 
             transform: DataTransformer = DataTransformer(),
             chunksize=0,
             timestamp_formatters = None
            ) -> Iterable | Any:

        f_path = self.__check_file_name(data_id)
        if not f_path:
            ValueError(f"Can't find any csv data for {data_id} in {self.path} !")

        convert_options = None
        if timestamp_formatters is not None:
            convert_options=csv.ConvertOptions(timestamp_parsers=timestamp_formatters)

        table = csv.read_csv(
            f_path, 
            parse_options=csv.ParseOptions(ignore_empty_lines=True),
            convert_options=convert_options
        )
        fieldnames =  table.column_names  

        # - try to find range to load  
        start_idx, stop_idx = 0, table.num_rows
        try:
            _time_field_idx = _find_column_index_in_list(fieldnames, 'time', 'timestamp', 'datetime', 'date')
            _time_type = table.field(_time_field_idx).type
            _time_unit = _time_type.unit if hasattr(_time_type, 'unit') else 's'
            _time_data = table[_time_field_idx]

            # - check if need convert time to primitive types (i.e. Date32 -> timestamp[x])
            _time_cast_function = lambda xs: xs
            if _time_type != pa.timestamp(_time_unit):
                _time_cast_function = lambda xs: xs.cast(pa.timestamp(_time_unit)) 
                _time_data = _time_cast_function(_time_data)

            # - preprocessing start and stop
            t_0, t_1 = handle_start_stop(start, stop, convert=lambda x: _recognize_t(x, None, _time_unit))

            # - check requested range
            if t_0:
                start_idx = self.__find_time_idx(_time_data, t_0)
                if start_idx >= table.num_rows:
                    # no data for requested start date
                    return None

            if t_1:
                stop_idx = self.__find_time_idx(_time_data, t_1)
                if stop_idx < 0 or stop_idx < start_idx:
                    stop_idx = table.num_rows

        except Exception as exc:
            logger.warning(exc)
            logger.info('loading whole file')

        length = (stop_idx - start_idx + 1)
        selected_table = table.slice(start_idx, length)

        # - in this case we want to return iterable chunks of data
        if chunksize > 0:
            def _iter_chunks():
                for n in range(0, length // chunksize + 1):
                    transform.start_transform(data_id, fieldnames)
                    raw_data = selected_table[n*chunksize : min((n+1)*chunksize, length)].to_pandas().to_numpy()
                    transform.process_data(raw_data)
                    yield transform.collect()
            return _iter_chunks()

        transform.start_transform(data_id, fieldnames)
        raw_data = selected_table.to_pandas().to_numpy()
        transform.process_data(raw_data)
        return transform.collect()

    def get_names(self) -> List[str] :
        _n = []
        for s in os.listdir(self.path):
            if (m:=re.match(r'(.*)\.csv(.gz)?$', s)):
                _n.append(m.group(1))
        return _n


class AsPandasFrame(DataTransformer):
    """
    List of records to pandas dataframe transformer
    """

    def start_transform(self, name: str, column_names: List[str]):
        self._time_idx = _find_column_index_in_list(column_names, 'time', 'timestamp', 'datetime', 'date')
        self._column_names = column_names
        self._frame = pd.DataFrame()
    
    def process_data(self, rows_data: Iterable) -> Any:
        self._frame
        p = pd.DataFrame.from_records(rows_data, columns=self._column_names)
        p.set_index(self._column_names[self._time_idx], drop=True, inplace=True)
        p.sort_index(inplace=True)
        self._frame = pd.concat((self._frame, p), axis=0, sort=True)
        return p

    def collect(self) -> Any:
        return self._frame 


class AsOhlcvSeries(DataTransformer):

    def __init__(self, timeframe: str | None = None, timestamp_units='ns') -> None:
        super().__init__()
        self.timeframe = timeframe 
        self._series = None
        self._data_type = None
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: List[str]):
        self._time_idx = _find_column_index_in_list(column_names, 'time', 'timestamp', 'datetime', 'date')
        self._volume_idx = None
        self._b_volume_idx = None
        try:
            self._close_idx = _find_column_index_in_list(column_names, 'close')
            self._open_idx = _find_column_index_in_list(column_names, 'open')
            self._high_idx = _find_column_index_in_list(column_names, 'high')
            self._low_idx = _find_column_index_in_list(column_names, 'low')

            try:
                self._volume_idx = _find_column_index_in_list(column_names, 'quote_volume', 'volume', 'vol')
            except: pass

            try:
                self._b_volume_idx = _find_column_index_in_list(column_names, 'taker_buy_volume', 'taker_buy_quote_volume', 'buy_volume')
            except: pass

            self._data_type = 'ohlc'
        except: 
            try:
                self._ask_idx = _find_column_index_in_list(column_names, 'ask')
                self._bid_idx = _find_column_index_in_list(column_names, 'bid')
                self._data_type = 'quotes'
            except: 

                try:
                    self._price_idx = _find_column_index_in_list(column_names, 'price')
                    self._size_idx = _find_column_index_in_list(column_names, 'quote_qty', 'qty', 'size', 'amount', 'volume')
                    self._taker_idx = None
                    try:
                        self._taker_idx = _find_column_index_in_list(column_names, 'is_buyer_maker', 'side', 'aggressive', 'taker', 'is_taker')
                    except: pass

                    self._data_type = 'trades'
                except: 
                    raise ValueError(f"Can't recognize data for update from header: {column_names}")

        self._column_names = column_names
        self._name = name
        if self.timeframe:
            self._series = OHLCV(self._name, self.timeframe)

    def _time(self, t) -> int:
        if self.timestamp_units == 'ns':
            return np.datetime64(t, 'ns').item() 
        return np.datetime64(t, self.timestamp_units).astype('datetime64[ns]').item()

    def _proc_ohlc(self, rows_data: List[List]):
        for d in rows_data:
            self._series.update_by_bar(
                self._time(d[self._time_idx]),
                d[self._open_idx], d[self._high_idx], d[self._low_idx], d[self._close_idx], 
                d[self._volume_idx] if self._volume_idx else 0,
                d[self._b_volume_idx] if self._b_volume_idx else 0
            )

    def _proc_quotes(self, rows_data: List[List]):
        for d in rows_data:
            self._series.update(
                self._time(d[self._time_idx]),
                (d[self._ask_idx] + d[self._bid_idx])/2
            )

    def _proc_trades(self, rows_data: List[List]):
        for d in rows_data:
            a = d[self._taker_idx] if self._taker_idx else 0
            s = d[self._size_idx]
            b = s if a else 0 
            self._series.update(self._time(d[self._time_idx]), d[self._price_idx], s, b)

    def process_data(self, rows_data: List[List]) -> Any:
        if self._series is None:
            ts = [t[self._time_idx] for t in rows_data[:100]]
            self.timeframe = pd.Timedelta(infer_series_frequency(ts)).asm8.item()

            # - create instance after first data received if 
            self._series = OHLCV(self._name, self.timeframe)

        match self._data_type:
            case 'ohlc':
                self._proc_ohlc(rows_data)
            case 'quotes':
                self._proc_quotes(rows_data)
            case 'trades':
                self._proc_trades(rows_data)

        return None

    def collect(self) -> Any:
        return self._series


class AsQuotes(DataTransformer):

    def start_transform(self, name: str, column_names: List[str]):
        self.buffer = list()
        self._time_idx = _find_column_index_in_list(column_names, 'time', 'timestamp', 'datetime')
        self._bid_idx = _find_column_index_in_list(column_names, 'bid')
        self._ask_idx = _find_column_index_in_list(column_names, 'ask')
        self._bidvol_idx = _find_column_index_in_list(column_names, 'bidvol', 'bid_vol', 'bidsize', 'bid_size')
        self._askvol_idx = _find_column_index_in_list(column_names, 'askvol', 'ask_vol', 'asksize', 'ask_size')

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None: 
            for d in rows_data:
                t = d[self._time_idx] 
                b = d[self._bid_idx]
                a = d[self._ask_idx]
                bv = d[self._bidvol_idx]
                av = d[self._askvol_idx]
                self.buffer.append(Quote(t.as_unit('ns').asm8.item(), b, a, bv, av))


class RestoreTicksFromOHLC(DataTransformer):
    """
    Emulates quotes (and trades) from OHLC bars
    """

    def __init__(self, 
                 trades: bool=False,    # if we also wants 'trades'
                 default_bid_size=1e9,  # default bid/ask is big
                 default_ask_size=1e9,  # default bid/ask is big
                 daily_session_start_end=DEFAULT_DAILY_SESSION,
                 spread=0.0):
        super().__init__()
        self._trades = trades
        self._bid_size = default_bid_size
        self._ask_size = default_ask_size
        self._s2 = spread / 2.0
        self._d_session_start = daily_session_start_end[0]
        self._d_session_end = daily_session_start_end[1]

    def start_transform(self, name: str, column_names: List[str]):
        self.buffer = []
        # - it will fail if receive data doesn't look as ohlcv 
        self._time_idx = _find_column_index_in_list(column_names, 'time', 'timestamp', 'datetime', 'date')
        self._open_idx = _find_column_index_in_list(column_names, 'open')
        self._high_idx = _find_column_index_in_list(column_names, 'high')
        self._low_idx = _find_column_index_in_list(column_names, 'low')
        self._close_idx = _find_column_index_in_list(column_names, 'close')
        self._volume_idx = None
        self._freq = None
        try:
            self._volume_idx = _find_column_index_in_list(column_names, 'volume', 'vol')
        except: pass

        if self._volume_idx is None and self._trades:
            logger.warning("Input OHLC data doesn't contain volume information so trades can't be emulated !")
            self._trades = False

    def process_data(self, rows_data:List[List]) -> Any:
        if rows_data is None: 
            return

        s2 = self._s2

        if self._freq is None:
            ts = [t[self._time_idx] for t in rows_data[:100]]
            self._freq = infer_series_frequency(ts)

            # - timestamps when we emit simulated quotes
            dt = self._freq.astype('timedelta64[ns]').item()
            if dt < D1:
                self._t_start = dt // 10
                self._t_mid1 = dt // 2 - dt // 10
                self._t_mid2 = dt // 2 + dt // 10
                self._t_end = dt - dt // 10
            else:
                self._t_start = self._d_session_start
                self._t_mid1 = dt // 2 - H1
                self._t_mid2 = dt // 2 + H1
                self._t_end = self._d_session_end

        # - input data
        for data in rows_data:
            ti = pd.Timestamp(data[self._time_idx]).as_unit('ns').asm8.item()
            o = data[self._open_idx]
            h=  data[self._high_idx]
            l = data[self._low_idx]
            c = data[self._close_idx]
            rv = data[self._volume_idx] if self._volume_idx else 0

            # - opening quote
            self.buffer.append(Quote(ti + self._t_start, o - s2, o + s2, self._bid_size, self._ask_size))

            if c >= o:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o - s2, rv * (o - l))) # sell 1
                self.buffer.append(Quote(ti + self._t_mid1, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, l + s2, rv * (c - o)))  # buy 1
                self.buffer.append(Quote(ti + self._t_mid2, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, h - s2, rv * (h - c)))  # sell 2
            else:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o + s2, rv * (h - o))) # buy 1
                self.buffer.append(Quote(ti + self._t_mid1, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, h - s2, rv * (o - c))) # sell 1
                self.buffer.append(Quote(ti + self._t_mid2, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, l + s2, rv * (c - l))) # buy 2

            # - closing quote
            self.buffer.append(Quote(ti + self._t_end, c - s2, c + s2, self._bid_size, self._ask_size))


def _retry(fn):
    @wraps(fn)
    def wrapper(*args, **kw):
        cls = args[0]
        for x in range(cls._reconnect_tries):
            # print(x, cls._reconnect_tries)
            try:
                return fn(*args, **kw)
            except (pg.InterfaceError, pg.OperationalError) as e:
                logger.warning("Database Connection [InterfaceError or OperationalError]")
                # print ("Idle for %s seconds" % (cls._reconnect_idle))
                # time.sleep(cls._reconnect_idle)
                cls._connect()
    return wrapper


class QuestDBConnector(DataReader):
    """
    Very first version of QuestDB connector

    # Connect to an existing QuestDB instance
    >>> db = QuestDBConnector('user=admin password=quest host=localhost port=8812', OhlcvPandasDataProcessor())
    >>> db.read('BINANCEF.ETHUSDT', '2024-01-01')
    """
    _reconnect_tries = 5
    _reconnect_idle = 0.1  # wait seconds before retying

    def __init__(self, connection_url: str) -> None:
        self._connection = None
        self._cursor = None
        self.connection_url = connection_url
        self._connect()

    def _connect(self):
        logger.info("Connecting to QuestDB ...")
        self._connection = pg.connect(self.connection_url, autocommit=True)
        self._cursor = self._connection.cursor()

    @_retry
    def read(self, data_id: str, start: str|None=None, stop: str|None=None, 
             transform: DataTransformer = DataTransformer(),
             chunksize=0,  # TODO: use self._cursor.fetchmany in this case !!!!
             timeframe: str='1m') -> Any:
        start, end = handle_start_stop(start, stop)
        w0 = f"timestamp >= '{start}'" if start else ''
        w1 = f"timestamp <= '{end}'" if end else ''
        where = f'where {w0} and {w1}' if (w0 and w1) else f"where {(w0 or w1)}"

        # just a temp hack - actually we need to discuss symbology etc
        symbol = data_id#.split('.')[-1]

        self._cursor.execute(
            f"""
                select timestamp, 
                first(open) as open, 
                max(high) as high,
                min(low) as low,
                last(close) as close,
                sum(volume) as volume,
                sum(quote_volume) as quote_volume,
                sum(count) as count,
                sum(taker_buy_volume) as taker_buy_volume,
                sum(taker_buy_quote_volume) as taker_buy_quote_volume
                from "{symbol.upper()}" {where}
                SAMPLE by {timeframe};
            """ # type: ignore
        )
        records = self._cursor.fetchall() # TODO: for chunksize > 0 use fetchmany etc
        names = [d.name for d in self._cursor.description]

        transform.start_transform(data_id, names)
        
        # d = np.array(records)
        transform.process_data(records)
        return transform.collect()

    @_retry
    def get_names(self) -> List[str] :
        self._cursor.execute("select table_name from tables()")
        records = self._cursor.fetchall()
        return [r[0] for r in records]

    def __del__(self):
        for c in (self._cursor, self._connection):
            try:
                logger.info("Closing connection")
                c.close()
            except:
                pass

