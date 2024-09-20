import re, os
from typing import Callable, Dict, List, Union, Optional, Iterator, Iterable, Any
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
from psycopg.types.datetime import TimestampLoader

_DT = lambda x: pd.Timedelta(x).to_numpy().item()
D1, H1 = _DT("1D"), _DT("1h")
MS1 = 1_000_000

DEFAULT_DAILY_SESSION = (_DT("00:00:00.100"), _DT("23:59:59.900"))
STOCK_DAILY_SESSION = (_DT("9:30:00.100"), _DT("15:59:59.900"))
CME_FUTURES_DAILY_SESSION = (_DT("8:30:00.100"), _DT("15:14:59.900"))


class NpTimestampLoader(TimestampLoader):
    def load(self, data) -> np.datetime64:
        dt = super().load(data)
        return np.datetime64(dt)


def _recognize_t(t: Union[int, str], defaultvalue, timeunit) -> int:
    if isinstance(t, (str, pd.Timestamp)):
        try:
            return np.datetime64(t, timeunit)
        except:
            pass
    return defaultvalue


def _time(t, timestamp_units: str) -> int:
    t = int(t) if isinstance(t, float) else t
    if timestamp_units == "ns":
        return np.datetime64(t, "ns").item()
    return np.datetime64(t, timestamp_units).astype("datetime64[ns]").item()


def _find_column_index_in_list(xs, *args):
    xs = [x.lower() for x in xs]
    for a in args:
        ai = a.lower()
        if ai in xs:
            return xs.index(ai)
    raise IndexError(f"Can't find any from {args} in list: {xs}")


_FIND_TIME_COL_IDX = lambda column_names: _find_column_index_in_list(
    column_names, "time", "timestamp", "datetime", "date", "open_time", "ts"
)


class DataTransformer:

    def __init__(self) -> None:
        self.buffer = []
        self._column_names = []

    def start_transform(
        self,
        name: str,
        column_names: List[str],
        start: str | None = None,
        stop: str | None = None,
    ):
        self._column_names = column_names
        self.buffer = []

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None:
            self.buffer.extend(rows_data)

    def collect(self) -> Any:
        return self.buffer


class DataReader:

    def get_names(self, **kwargs) -> List[str]:
        raise NotImplemented()

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        **kwargs,
    ) -> Iterator | List:
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
                ix = np.searchsorted(a, v, side="right")
                if ix > 0 and ix < len(c):
                    ix = arr.index(a[ix]).as_py() - 1
                    break
        return ix

    def __check_file_name(self, name: str) -> str | None:
        _f = join(self.path, name.replace(":", os.sep))
        for sfx in [".csv", ".csv.gz", ""]:
            if exists(p := (_f + sfx)):
                return p
        return None

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timestamp_formatters=None,
        timeframe=None,
        **kwargs,
    ) -> Iterable | Any:

        f_path = self.__check_file_name(data_id)
        if not f_path:
            ValueError(f"Can't find any csv data for {data_id} in {self.path} !")

        convert_options = None
        if timestamp_formatters is not None:
            convert_options = csv.ConvertOptions(timestamp_parsers=timestamp_formatters)

        table = csv.read_csv(
            f_path,
            parse_options=csv.ParseOptions(ignore_empty_lines=True),
            convert_options=convert_options,
        )
        fieldnames = table.column_names

        # - try to find range to load
        start_idx, stop_idx = 0, table.num_rows
        try:
            _time_field_idx = _FIND_TIME_COL_IDX(fieldnames)
            _time_type = table.field(_time_field_idx).type
            _time_unit = _time_type.unit if hasattr(_time_type, "unit") else "ms"
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
            logger.info("loading whole file")

        length = stop_idx - start_idx + 1
        selected_table = table.slice(start_idx, length)

        # - in this case we want to return iterable chunks of data
        if chunksize > 0:

            def _iter_chunks():
                for n in range(0, length // chunksize + 1):
                    transform.start_transform(data_id, fieldnames, start=start, stop=stop)
                    raw_data = selected_table[n * chunksize : min((n + 1) * chunksize, length)].to_pandas().to_numpy()
                    transform.process_data(raw_data)
                    yield transform.collect()

            return _iter_chunks()

        transform.start_transform(data_id, fieldnames, start=start, stop=stop)
        raw_data = selected_table.to_pandas().to_numpy()
        transform.process_data(raw_data)
        return transform.collect()

    def get_names(self, **kwargs) -> List[str]:
        _n = []
        for root, _, files in os.walk(self.path):
            path = root.split(os.sep)
            for file in files:
                if m := re.match(r"(.*)\.csv(.gz)?$", file):
                    f = path[-1]
                    n = file.split(".")[0]
                    if f == self.path:
                        name = n
                    else:
                        name = f"{f}:{ n }" if f else n
                    _n.append(name)
        return _n


class InMemoryDataFrameReader(DataReader):
    """
    Data reader for pandas DataFrames
    """

    def __init__(self, data: Dict[str, pd.DataFrame], exchange: str | None = None) -> None:
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary of pandas DataFrames")
        self._data = data
        self.exchange = exchange

    def get_names(self, **kwargs) -> List[str]:
        keys = list(self._data.keys())
        if self.exchange:
            return [f"{self.exchange}:{k}" for k in keys]
        return keys

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        **kwargs,
    ) -> Iterable | List:
        """
        Read and transform data for a given data_id within a specified time range.

        Parameters:
        -----------
        data_id : str
            The identifier for the data to be read.
        start : str | None, optional
            The start time for the data range (inclusive). If None, start from the earliest available data.
        stop : str | None, optional
            The stop time for the data range (inclusive). If None, include data up to the latest available.
        transform : DataTransformer, optional
            An instance of DataTransformer to process the retrieved data. Defaults to DataTransformer().
        chunksize : int, optional
            The size of data chunks to process at a time. If 0, process all data at once. Defaults to 0.
        **kwargs : dict
            Additional keyword arguments for future extensions.

        Returns:
        --------
        Iterable | List
            The processed and transformed data, either as an iterable (if chunksize > 0) or as a list.

        Raises:
        -------
        ValueError
            If no data is found for the given data_id.
        """
        start, stop = handle_start_stop(start, stop)
        if data_id not in self._data:
            if data_id.startswith(self.exchange):
                data_id = data_id.split(":")[1]
        d = self._data.get(data_id)
        if d is None:
            raise ValueError(f"No data found for {data_id}")
        d2 = d.loc[start:stop].copy().reset_index()
        transform.start_transform(data_id, list(d2.columns), start=start, stop=stop)
        transform.process_data(d2.values)
        res = transform.collect()
        if chunksize > 0:

            def __iterable():
                yield res

            return __iterable()
        return res


class AsPandasFrame(DataTransformer):
    """
    List of records to pandas dataframe transformer
    """

    def __init__(self, timestamp_units=None) -> None:
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: List[str], **kwargs):
        self._time_idx = _FIND_TIME_COL_IDX(column_names)
        self._column_names = column_names
        self._frame = pd.DataFrame()

    def process_data(self, rows_data: Iterable) -> Any:
        self._frame
        p = pd.DataFrame.from_records(rows_data, columns=self._column_names)
        p.set_index(self._column_names[self._time_idx], drop=True, inplace=True)
        p.index = pd.to_datetime(p.index, unit=self.timestamp_units) if self.timestamp_units else p.index
        p.index.rename("timestamp", inplace=True)
        p.sort_index(inplace=True)
        self._frame = pd.concat((self._frame, p), axis=0, sort=True)
        return p

    def collect(self) -> Any:
        return self._frame


class AsOhlcvSeries(DataTransformer):
    """
    Convert incoming data into OHLCV series.

    Incoming data may have one of the following structures:

        ```
        ohlcv:        time,open,high,low,close,volume|quote_volume,(buy_volume)
        quotes:       time,bid,ask,bidsize,asksize
        trades (TAS): time,price,size,(is_taker)
        ```
    """

    def __init__(self, timeframe: str | None = None, timestamp_units="ns") -> None:
        super().__init__()
        self.timeframe = timeframe
        self._series = None
        self._data_type = None
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: List[str], **kwargs):
        self._time_idx = _FIND_TIME_COL_IDX(column_names)
        self._volume_idx = None
        self._b_volume_idx = None
        try:
            self._close_idx = _find_column_index_in_list(column_names, "close")
            self._open_idx = _find_column_index_in_list(column_names, "open")
            self._high_idx = _find_column_index_in_list(column_names, "high")
            self._low_idx = _find_column_index_in_list(column_names, "low")

            try:
                self._volume_idx = _find_column_index_in_list(column_names, "quote_volume", "volume", "vol")
            except:
                pass

            try:
                self._b_volume_idx = _find_column_index_in_list(
                    column_names,
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                    "buy_volume",
                )
            except:
                pass

            self._data_type = "ohlc"
        except:
            try:
                self._ask_idx = _find_column_index_in_list(column_names, "ask")
                self._bid_idx = _find_column_index_in_list(column_names, "bid")
                self._data_type = "quotes"
            except:

                try:
                    self._price_idx = _find_column_index_in_list(column_names, "price")
                    self._size_idx = _find_column_index_in_list(
                        column_names, "quote_qty", "qty", "size", "amount", "volume"
                    )
                    self._taker_idx = None
                    try:
                        self._taker_idx = _find_column_index_in_list(
                            column_names,
                            "is_buyer_maker",
                            "side",
                            "aggressive",
                            "taker",
                            "is_taker",
                        )
                    except:
                        pass

                    self._data_type = "trades"
                except:
                    raise ValueError(f"Can't recognize data for update from header: {column_names}")

        self._column_names = column_names
        self._name = name
        if self.timeframe:
            self._series = OHLCV(self._name, self.timeframe)

    def _proc_ohlc(self, rows_data: List[List]):
        for d in rows_data:
            self._series.update_by_bar(
                _time(d[self._time_idx], self.timestamp_units),
                d[self._open_idx],
                d[self._high_idx],
                d[self._low_idx],
                d[self._close_idx],
                d[self._volume_idx] if self._volume_idx else 0,
                d[self._b_volume_idx] if self._b_volume_idx else 0,
            )

    def _proc_quotes(self, rows_data: List[List]):
        for d in rows_data:
            self._series.update(
                _time(d[self._time_idx], self.timestamp_units),
                (d[self._ask_idx] + d[self._bid_idx]) / 2,
            )

    def _proc_trades(self, rows_data: List[List]):
        for d in rows_data:
            a = d[self._taker_idx] if self._taker_idx else 0
            s = d[self._size_idx]
            b = s if a else 0
            self._series.update(_time(d[self._time_idx], self.timestamp_units), d[self._price_idx], s, b)

    def process_data(self, rows_data: List[List]) -> Any:
        if self._series is None:
            ts = [t[self._time_idx] for t in rows_data[:100]]
            self.timeframe = pd.Timedelta(infer_series_frequency(ts)).asm8.item()

            # - create instance after first data received if
            self._series = OHLCV(self._name, self.timeframe)

        match self._data_type:
            case "ohlc":
                self._proc_ohlc(rows_data)
            case "quotes":
                self._proc_quotes(rows_data)
            case "trades":
                self._proc_trades(rows_data)

        return None

    def collect(self) -> Any:
        return self._series


class AsQuotes(DataTransformer):
    """
    Tries to convert incoming data to list of Quote's
    Data must have appropriate structure: bid, ask, bidsize, asksize and time
    """

    def start_transform(self, name: str, column_names: List[str], **kwargs):
        self.buffer = list()
        self._time_idx = _FIND_TIME_COL_IDX(column_names)
        self._bid_idx = _find_column_index_in_list(column_names, "bid")
        self._ask_idx = _find_column_index_in_list(column_names, "ask")
        self._bidvol_idx = _find_column_index_in_list(column_names, "bidvol", "bid_vol", "bidsize", "bid_size")
        self._askvol_idx = _find_column_index_in_list(column_names, "askvol", "ask_vol", "asksize", "ask_size")

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None:
            for d in rows_data:
                t = d[self._time_idx]
                b = d[self._bid_idx]
                a = d[self._ask_idx]
                bv = d[self._bidvol_idx]
                av = d[self._askvol_idx]
                self.buffer.append(Quote(_time(t, "ns"), b, a, bv, av))


class AsTrades(DataTransformer):
    """
    Tries to convert incoming data to list of Trades
    Data must have appropriate structure: price, size, market_maker (optional).
    Market maker column specifies if buyer is a maker or taker.
    """

    def start_transform(self, name: str, column_names: List[str], **kwargs):
        self.buffer: list[Trade] = list()
        self._time_idx = _FIND_TIME_COL_IDX(column_names)
        self._price_idx = _find_column_index_in_list(column_names, "price")
        self._size_idx = _find_column_index_in_list(column_names, "size")
        try:
            self._side_idx = _find_column_index_in_list(column_names, "market_maker")
        except:
            self._side_idx = None

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None:
            for d in rows_data:
                t = d[self._time_idx]
                price = d[self._price_idx]
                size = d[self._size_idx]
                side = d[self._side_idx] if self._side_idx else -1
                self.buffer.append(Trade(_time(t, "ns"), price, size, side))


class AsTimestampedRecords(DataTransformer):
    """
    Convert incoming data to list or dictionaries with preprocessed timestamps ('timestamp_ns' and 'timestamp')
    ```
    [
        {
            'open_time': 1711944240000.0,
            'open': 203.219,
            'high': 203.33,
            'low': 203.134,
            'close': 203.175,
            'volume': 10060.0,
            ....
            'timestamp_ns': 1711944240000000000,
            'timestamp': Timestamp('2024-04-01 04:04:00')
        },
        ...
    ] ```
    """

    def __init__(self, timestamp_units: str | None = None) -> None:
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: List[str], **kwargs):
        self.buffer = list()
        self._time_idx = _FIND_TIME_COL_IDX(column_names)
        self._column_names = column_names

    def process_data(self, rows_data: Iterable) -> Any:
        self.buffer.extend(rows_data)

    def collect(self) -> Any:
        res = []
        for r in self.buffer:
            t = r[self._time_idx]
            if self.timestamp_units:
                t = _time(t, self.timestamp_units)
            di = dict(zip(self._column_names, r)) | {
                "timestamp_ns": t,
                "timestamp": pd.Timestamp(t),
            }
            res.append(di)
        return res


class RestoreTicksFromOHLC(DataTransformer):
    """
    Emulates quotes (and trades) from OHLC bars
    """

    def __init__(
        self,
        trades: bool = False,  # if we also wants 'trades'
        default_bid_size=1e9,  # default bid/ask is big
        default_ask_size=1e9,  # default bid/ask is big
        daily_session_start_end=DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        spread=0.0,
    ):
        super().__init__()
        self._trades = trades
        self._bid_size = default_bid_size
        self._ask_size = default_ask_size
        self._s2 = spread / 2.0
        self._d_session_start = daily_session_start_end[0]
        self._d_session_end = daily_session_start_end[1]
        self._timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: List[str], **kwargs):
        self.buffer = []
        # - it will fail if receive data doesn't look as ohlcv
        self._time_idx = _FIND_TIME_COL_IDX(column_names)
        self._open_idx = _find_column_index_in_list(column_names, "open")
        self._high_idx = _find_column_index_in_list(column_names, "high")
        self._low_idx = _find_column_index_in_list(column_names, "low")
        self._close_idx = _find_column_index_in_list(column_names, "close")
        self._volume_idx = None
        self._freq = None
        try:
            self._volume_idx = _find_column_index_in_list(column_names, "volume", "vol")
        except:
            pass

        if self._volume_idx is None and self._trades:
            logger.warning("Input OHLC data doesn't contain volume information so trades can't be emulated !")
            self._trades = False

    def process_data(self, rows_data: List[List]) -> Any:
        if rows_data is None:
            return

        s2 = self._s2

        if self._freq is None:
            ts = [t[self._time_idx] for t in rows_data[:100]]
            try:
                self._freq = infer_series_frequency(ts)
            except ValueError:
                logger.warning("Can't determine frequency of incoming data")
                return

            # - timestamps when we emit simulated quotes
            dt = self._freq.astype("timedelta64[ns]").item()
            if dt < D1:
                self._t_start = MS1  # dt // 10
                self._t_mid1 = dt // 2 - dt // 10
                self._t_mid2 = dt // 2 + dt // 10
                self._t_end = dt - MS1  # dt - dt // 10
            else:
                self._t_start = self._d_session_start
                self._t_mid1 = dt // 2 - H1
                self._t_mid2 = dt // 2 + H1
                self._t_end = self._d_session_end

        # - input data
        for data in rows_data:
            # ti = pd.Timestamp(data[self._time_idx]).as_unit("ns").asm8.item()
            ti = _time(data[self._time_idx], self._timestamp_units)
            o = data[self._open_idx]
            h = data[self._high_idx]
            l = data[self._low_idx]
            c = data[self._close_idx]
            rv = data[self._volume_idx] if self._volume_idx else 0

            # - opening quote
            self.buffer.append(Quote(ti + self._t_start, o - s2, o + s2, self._bid_size, self._ask_size))

            if c >= o:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o - s2, rv * (o - l)))  # sell 1
                self.buffer.append(
                    Quote(
                        ti + self._t_mid1,
                        l - s2,
                        l + s2,
                        self._bid_size,
                        self._ask_size,
                    )
                )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, l + s2, rv * (c - o)))  # buy 1
                self.buffer.append(
                    Quote(
                        ti + self._t_mid2,
                        h - s2,
                        h + s2,
                        self._bid_size,
                        self._ask_size,
                    )
                )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, h - s2, rv * (h - c)))  # sell 2
            else:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o + s2, rv * (h - o)))  # buy 1
                self.buffer.append(
                    Quote(
                        ti + self._t_mid1,
                        h - s2,
                        h + s2,
                        self._bid_size,
                        self._ask_size,
                    )
                )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, h - s2, rv * (o - c)))  # sell 1
                self.buffer.append(
                    Quote(
                        ti + self._t_mid2,
                        l - s2,
                        l + s2,
                        self._bid_size,
                        self._ask_size,
                    )
                )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, l + s2, rv * (c - l)))  # buy 2

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
            except (pg.InterfaceError, pg.OperationalError, AttributeError) as e:
                logger.debug("Database Connection [InterfaceError or OperationalError]")
                # print ("Idle for %s seconds" % (cls._reconnect_idle))
                # time.sleep(cls._reconnect_idle)
                cls._connect()

    return wrapper


class QuestDBSqlBuilder:
    """
    Generic sql builder for QuestDB data
    """

    def get_table_name(self, data_id: str, sfx: str = "") -> str | None:
        pass

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str | None,
        data_type: str,
    ) -> str | None:
        pass

    def prepare_names_sql(self) -> str:
        return "select table_name from tables()"


class QuestDBSqlCandlesBuilder(QuestDBSqlBuilder):
    """
    Sql builder for candles data
    """

    def get_table_name(self, data_id: str, sfx: str = "") -> str:
        """
        Get table name for data_id
        data_id can have format <exchange>.<type>:<symbol>
        for example:
            BINANCE.UM:BTCUSDT or BINANCE:BTCUSDT for spot
        """
        _aliases = {"um": "umfutures", "cm": "cmfutures", "f": "futures"}
        sfx = sfx or "candles_1m"
        table_name = data_id
        _ss = data_id.split(":")
        if len(_ss) > 1:
            _exch, symb = _ss
            _mktype = "spot"
            _ss = _exch.split(".")
            if len(_ss) > 1:
                _exch = _ss[0]
                _mktype = _ss[1]
            _mktype = _mktype.lower()
            parts = [_exch.lower(), _aliases.get(_mktype, _mktype)]
            if "candles" not in sfx:
                parts.append(symb.lower())
            parts.append(sfx)
            table_name = ".".join(filter(lambda x: x, parts))
        return table_name

    def prepare_names_sql(self) -> str:
        return "select table_name from tables() where table_name like '%candles%'"

    @staticmethod
    def _convert_time_delta_to_qdb_resample_format(c_tf: str):
        if c_tf:
            _t = re.match(r"(\d+)(\w+)", c_tf)
            if _t and len(_t.groups()) > 1:
                c_tf = f"{_t[1]}{_t[2][0].lower()}"
        return c_tf

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str | None,
        data_type: str,
    ) -> str:
        _ss = data_id.split(":")
        if len(_ss) > 1:
            _exch, symb = _ss
        else:
            symb = data_id

        symb = symb.lower()

        where = f"where symbol = '{symb}'"
        w0 = f"timestamp >= '{start}'" if start else ""
        w1 = f"timestamp < '{end}'" if end else ""

        # - fix: when no data ranges are provided we must skip empy where keyword
        if w0 or w1:
            where = f"{where} and {w0} and {w1}" if (w0 and w1) else f"{where} and {(w0 or w1)}"

        # - filter out candles without any volume
        where = f"{where} and volume > 0"

        # - check resample format
        resample = (
            QuestDBSqlCandlesBuilder._convert_time_delta_to_qdb_resample_format(resample)
            if resample
            else "1m"  # if resample is empty let's use 1 minute timeframe
        )
        _rsmpl = f"SAMPLE by {resample} FILL(NONE)" if resample else ""

        table_name = self.get_table_name(data_id, data_type)
        return f"""
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
                from "{table_name}" {where} {_rsmpl};
            """


class QuestDBConnector(DataReader):
    """
    Very first version of QuestDB connector

    ### Connect to an existing QuestDB instance
    >>> db = QuestDBConnector()
    >>> db.read('BINANCE.UM:ETHUSDT', '2024-01-01', transform=AsPandasFrame())
    """

    _reconnect_tries = 5
    _reconnect_idle = 0.1  # wait seconds before retying
    _builder: QuestDBSqlBuilder

    def __init__(
        self,
        builder: QuestDBSqlBuilder = QuestDBSqlCandlesBuilder(),
        host="localhost",
        user="admin",
        password="quest",
        port=8812,
    ) -> None:
        self._connection = None
        self._host = host
        self._port = port
        self.connection_url = f"user={user} password={password} host={host} port={port}"
        self._builder = builder
        self._connect()

    def __getstate__(self):
        if self._connection:
            self._connection.close()
            self._connection = None
        state = self.__dict__.copy()
        return state

    def _connect(self):
        self._connection = pg.connect(self.connection_url, autocommit=True)
        logger.debug(f"Connected to QuestDB at {self._host}:{self._port}")

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timeframe: str | None = "1m",
        data_type="candles_1m",
    ) -> Any:
        return self._read(
            data_id,
            start,
            stop,
            transform,
            chunksize,
            timeframe,
            data_type,
            self._builder,
        )

    def get_symbols(self, exchange: str) -> list[str]:
        table_name = QuestDBSqlCandlesBuilder().get_table_name(f"{exchange}:BTCUSDT")
        query = f"""
        select distinct symbol
        from "{table_name}";
        """
        return self.execute(query)["symbol"].tolist()

    def get_candles(
        self,
        exchange: str,
        symbols: list[str],
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        assert len(symbols) > 0, "No symbols provided"
        quoted_symbols = [f"'{s.lower()}'" for s in symbols]
        where = f"where symbol in ({', '.join(quoted_symbols)}) and timestamp >= '{start}' and timestamp < '{stop}'"
        table_name = QuestDBSqlCandlesBuilder().get_table_name(f"{exchange}:{symbols[0]}")

        _rsmpl = f"sample by {timeframe}"

        query = f"""
        select timestamp, 
        symbol,
        first(open) as open, 
        max(high) as high,
        min(low) as low,
        last(close) as close,
        sum(volume) as volume,
        sum(quote_volume) as quote_volume,
        sum(count) as count,
        sum(taker_buy_volume) as taker_buy_volume,
        sum(taker_buy_quote_volume) as taker_buy_quote_volume
        from "{table_name}" {where} {_rsmpl};
        """
        return self.execute(query).set_index(["timestamp", "symbol"])

    def get_average_quote_volume(
        self,
        exchange: str,
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        timeframe: str = "1d",
    ) -> pd.Series:
        table_name = QuestDBSqlCandlesBuilder().get_table_name(f"{exchange}:BTCUSDT")
        query = f"""
        WITH sampled as (
            select timestamp, symbol, sum(quote_volume) as qvolume 
            from "{table_name}"
            where timestamp >= '{start}' and timestamp < '{stop}'
            SAMPLE BY {timeframe}
        )
        select symbol, avg(qvolume) as quote_volume from sampled
        group by symbol
        order by quote_volume desc;
        """
        vol_stats = self.execute(query)
        if vol_stats.empty:
            return pd.Series()
        return vol_stats.set_index("symbol")["quote_volume"]

    def get_fundamental_data(
        self, exchange: str, start: str | pd.Timestamp | None = None, stop: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        table_name = {"BINANCE.UM": "binance.umfutures.fundamental"}[exchange]
        query = f"select * from {table_name}"
        if start or stop:
            conditions = []
            if start:
                conditions.append(f"timestamp >= '{start}'")
            if stop:
                conditions.append(f"timestamp < '{stop}'")
            query += " where " + " and ".join(conditions)
        df = self.execute(query)
        if df.empty:
            return pd.DataFrame()
        return df.set_index(["timestamp", "symbol", "metric"]).value.unstack("metric")

    def get_names(self) -> List[str]:
        return self._get_names(self._builder)

    @_retry
    def execute(self, query: str) -> pd.DataFrame:
        _cursor = self._connection.cursor()  # type: ignore
        _cursor.execute(query)  # type: ignore
        names = [d.name for d in _cursor.description]  # type: ignore
        records = _cursor.fetchall()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records, columns=names)

    @_retry
    def _read(
        self,
        data_id: str,
        start: str | None,
        stop: str | None,
        transform: DataTransformer,
        chunksize: int,
        timeframe: str | None,
        data_type: str,
        builder: QuestDBSqlBuilder,
    ) -> Any:
        start, end = handle_start_stop(start, stop)
        _req = builder.prepare_data_sql(data_id, start, end, timeframe, data_type)

        _cursor = self._connection.cursor()  # type: ignore
        _cursor.execute(_req)  # type: ignore
        names = [d.name for d in _cursor.description]  # type: ignore

        if chunksize > 0:

            def _iter_chunks():
                while True:
                    records = _cursor.fetchmany(chunksize)
                    if not records:
                        _cursor.close()
                        break
                    transform.start_transform(data_id, names, start=start, stop=stop)
                    transform.process_data(records)
                    yield transform.collect()

            return _iter_chunks()

        try:
            records = _cursor.fetchall()
            if not records:
                return None
            transform.start_transform(data_id, names, start=start, stop=stop)
            transform.process_data(records)
            return transform.collect()
        finally:
            _cursor.close()

    @_retry
    def _get_names(self, builder: QuestDBSqlBuilder) -> List[str]:
        _cursor = None
        try:
            _cursor = self._connection.cursor()  # type: ignore
            _cursor.execute(builder.prepare_names_sql())  # type: ignore
            records = _cursor.fetchall()
        finally:
            if _cursor:
                _cursor.close()
        return [r[0] for r in records]

    def __del__(self):
        try:
            if self._connection is not None:
                logger.debug("Closing connection")
                self._connection.close()
        except:
            pass


class QuestDBSqlOrderBookBuilder(QuestDBSqlCandlesBuilder):
    """
    Sql builder for snapshot data
    """

    SNAPSHOT_DELTA = pd.Timedelta("1h")
    MIN_DELTA = pd.Timedelta("1s")

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str,
        data_type: str,
    ) -> str:
        if not start or not end:
            raise ValueError("Start and end dates must be provided for orderbook data!")
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
        delta = end_dt - start_dt

        raw_start_dt = start_dt.floor(self.SNAPSHOT_DELTA) - self.MIN_DELTA

        table_name = self.get_table_name(data_id, data_type)
        query = f"""
SELECT * FROM {table_name}
WHERE timestamp BETWEEN '{raw_start_dt}' AND '{end_dt}'
"""
        return query


class TradeSql(QuestDBSqlCandlesBuilder):

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str,
        data_type: str,
    ) -> str:
        table_name = self.get_table_name(data_id, data_type)
        where = ""
        w0 = f"timestamp >= '{start}'" if start else ""
        w1 = f"timestamp <= '{end}'" if end else ""

        # - fix: when no data ranges are provided we must skip empy where keyword
        if w0 or w1:
            where = f"where {w0} and {w1}" if (w0 and w1) else f"where {(w0 or w1)}"

        resample = (
            QuestDBSqlCandlesBuilder._convert_time_delta_to_qdb_resample_format(resample) if resample else resample
        )
        if resample:
            sql = f"""
                select timestamp, first(price) as open, max(price) as high, min(price) as low, last(price) as close, 
                sum(size) as volume from "{table_name}" {where} SAMPLE by {resample};"""
        else:
            sql = f"""select timestamp, price, size, market_maker from "{table_name}" {where};"""

        return sql


class MultiQdbConnector(QuestDBConnector):
    """
    Data connector for QuestDB which provides access to following data types:
      - candles
      - trades
      - orderbook snapshots
      - liquidations
      - funding rate

    Examples:
    1. Retrieving trades:
        qdb.read(
            "BINANCE.UM:BTCUSDT",
            "2023-01-01 00:00",
            "2023-01-01 10:00",
            timeframe="15Min",
            transform=AsPandasFrame(),
            data_type="trade"
        )
    """

    _TYPE_TO_BUILDER = {
        "candles_1m": QuestDBSqlCandlesBuilder(),
        "trade": TradeSql(),
        "agg_trade": TradeSql(),
        "orderbook": QuestDBSqlOrderBookBuilder(),
    }

    _TYPE_MAPPINGS = {
        "candles": "candles_1m",
        "ohlc": "candles_1m",
        "trades": "trade",
        "ob": "orderbook",
        "trd": "trade",
        "td": "trade",
        "aggTrade": "agg_trade",
        "agg_trades": "agg_trade",
        "aggTrades": "agg_trade",
    }

    def __init__(
        self,
        host="localhost",
        user="admin",
        password="quest",
        port=8812,
    ) -> None:
        self._connection = None
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._connect()

    @property
    def connection_url(self):
        return " ".join(
            [
                f"user={self._user}",
                f"password={self._password}",
                f"host={self._host}",
                f"port={self._port}",
            ]
        )

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize: int = 0,
        timeframe: str | None = None,
        data_type: str = "candles",
    ) -> Any:
        _mapped_data_type = self._TYPE_MAPPINGS.get(data_type, data_type)
        return self._read(
            data_id,
            start,
            stop,
            transform,
            chunksize,
            timeframe,
            _mapped_data_type,
            self._TYPE_TO_BUILDER[_mapped_data_type],
        )

    def get_names(self, data_type: str) -> List[str]:
        return self._get_names(self._TYPE_TO_BUILDER[self._TYPE_MAPPINGS.get(data_type, data_type)])
