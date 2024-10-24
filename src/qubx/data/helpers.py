from types import GeneratorType
from typing import Any, Dict, Iterable, List, Set, Type
from concurrent.futures import ThreadPoolExecutor

from joblib import delayed
import numpy as np
import pandas as pd
from collections import defaultdict

from qubx import logger
from qubx.core.basics import ITimeProvider
from qubx.core.series import TimeSeries
from qubx.data.readers import (
    AsPandasFrame,
    CsvStorageDataReader,
    DataReader,
    InMemoryDataFrameReader,
    MultiQdbConnector,
    DataTransformer,
    QuestDBConnector,
)
from qubx.pandaz.utils import OhlcDict, generate_equal_date_ranges, ohlc_resample, srows
from qubx.utils.misc import ProgressParallel
from qubx.utils.time import convert_seconds_to_str, handle_start_stop, infer_series_frequency


def _wrap_as_iterable(data: Any) -> Iterable:
    def __iterable():
        yield data

    return __iterable()


class InMemoryCachedReader(InMemoryDataFrameReader):
    """
    A class for caching and reading financial data from memory.

    This class extends InMemoryDataFrameReader to provide efficient data caching and retrieval
    for financial data from a specific exchange and timeframe.
    """

    exchange: str
    _data_timeframe: str
    _reader: DataReader
    _n_jobs: int
    _start: pd.Timestamp | None = None
    _stop: pd.Timestamp | None = None

    # - external data
    _external: Dict[str, pd.DataFrame | pd.Series]

    def __init__(
        self,
        exchange: str,
        reader: DataReader,
        base_timeframe: str,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        self._reader = reader
        self._n_jobs = n_jobs
        self._data_timeframe = base_timeframe
        self.exchange = exchange
        self._external = {}

        # - copy external data
        for k, v in kwargs.items():
            if isinstance(v, (pd.DataFrame, pd.Series)):
                self._external[k] = v

        super().__init__({}, exchange)

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        # timeframe: str | None = None,
        **kwargs,
    ) -> Iterable | List:
        _s_path = data_id
        if not data_id.startswith(self.exchange):
            _s_path = f"{self.exchange}:{data_id}"
        _, symb = _s_path.split(":")

        _start = str(self._start) if start is None else start
        _stop = str(self._stop) if stop is None else stop
        if _start is None or _stop is None:
            raise ValueError("Start and stop date must be provided")

        # - refresh symbol's data
        self._handle_symbols_data_from_to([symb], _start, _stop)

        # - we don't use chunksize from InMemoryDataFrameReader because it returns generator
        res = super().read(_s_path, start, stop, transform, chunksize=0, **kwargs)

        # - when it's asked to have chunks, it returns generator (single chunk)
        return _wrap_as_iterable(res) if chunksize > 0 else res

    def __getitem__(self, keys) -> Dict[str, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series:
        """
        This helper mostly for using in research notebooks
        """
        _start: str | None = None
        _stop: str | None = None
        _instruments: List[str] = []
        _as_dict = False

        if isinstance(keys, (tuple)):
            for k in keys:
                if isinstance(k, slice):
                    _start, _stop = k.start, k.stop
                if isinstance(k, (list, tuple, set)):
                    _instruments = list(k)
                    _as_dict = True
                if isinstance(k, str):
                    _instruments.append(k)
        else:
            if isinstance(keys, (list, tuple)):
                _instruments.extend(keys)
                _as_dict = True
            elif isinstance(keys, slice):
                _start, _stop = keys.start, keys.stop
            else:
                _instruments.append(keys)
        _as_dict |= len(_instruments) > 1

        if not _instruments:
            _instruments = list(self._data.keys())

        if not _instruments:
            raise ValueError("No symbols provided")

        if (_start is None and self._start is None) or (_stop is None and self._stop is None):
            raise ValueError("Start and stop date must be provided")

        _start = str(self._start) if _start is None else _start
        _stop = str(self._stop) if _stop is None else _stop

        _r = self._handle_symbols_data_from_to(_instruments, _start, _stop)
        if not _as_dict and len(_instruments) == 1:
            return _r.get(_instruments[0], pd.DataFrame())
        return _r

    def _load_candle_data(
        self, symbols: List[str], start: str | pd.Timestamp, stop: str | pd.Timestamp, timeframe: str
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        _ohlcs = defaultdict(list)
        _chunk_size_id_days = 30 * (4 if pd.Timedelta(timeframe) >= pd.Timedelta("1h") else 1)
        _ranges = list(generate_equal_date_ranges(str(start), str(stop), _chunk_size_id_days, "D"))

        # - for timeframes less than 1d generate_equal_date_ranges may skip days
        # so we need to fix intervals
        _es = list(zip(_ranges[:], _ranges[1:]))
        _es = [(start, end[0]) for (start, _), end in _es]
        _es.append((_ranges[-1][0], str(stop)))

        _results = ProgressParallel(n_jobs=self._n_jobs, silent=True, total=len(_ranges))(
            delayed(self._reader.get_aux_data)(
                "candles", exchange=self.exchange, symbols=symbols, start=s, stop=e, timeframe=timeframe
            )
            for s, e in _es
        )
        for (s, e), data in zip(_ranges, _results):
            assert isinstance(data, pd.DataFrame)
            try:
                data_symbols = data.index.get_level_values(1).unique()
                for smb in data_symbols:
                    _ohlcs[smb].append(data.loc[pd.IndexSlice[:, smb], :].droplevel(1))
            except Exception as exc:
                logger.error(f"> Failed to load data for {s} - {e} : {str(exc)}")

        ohlc = {smb.upper(): srows(*vs, keep="first") for smb, vs in _ohlcs.items() if len(vs) > 0}
        return ohlc

    def _handle_symbols_data_from_to(
        self, symbols: List[str], start: str, stop: str
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        _dtf = pd.Timedelta(self._data_timeframe)
        T = lambda x: pd.Timestamp(x)
        _start, _stop = map(T, handle_start_stop(start, stop))

        # - full interval
        _new_symbols = list(set([s for s in symbols if s not in self._data]))
        if _new_symbols:
            _s_req = min(_start, self._start if self._start else _start)
            _e_req = max(_stop, self._stop if self._stop else _stop)
            logger.debug(f"Loading all data {_s_req} - {_e_req} for { ','.join(_new_symbols)} ")
            _new_data = self._load_candle_data(_new_symbols, _s_req, _e_req + _dtf, self._data_timeframe)
            self._data |= _new_data

        # - part intervals
        if self._start and _start < self._start:
            _smbs = list(self._data.keys())
            logger.debug(f"Updating {len(_smbs)} symbols before interval {_start} : {self._start}")
            _before = self._load_candle_data(_smbs, _start, self._start + _dtf, self._data_timeframe)
            for k, c in _before.items():
                self._data[k] = srows(c, self._data[k], keep="first")

        # - part intervals
        if self._stop and _stop > self._stop:
            _smbs = list(self._data.keys())
            logger.debug(f"Updating {len(_smbs)} symbols after interval {self._stop} : {_stop}")
            _after = self._load_candle_data(_smbs, self._stop - _dtf, _stop, self._data_timeframe)
            for k, c in _after.items():
                self._data[k] = srows(self._data[k], c, keep="last")

        self._start = min(_start, self._start if self._start else _start)
        self._stop = max(_stop, self._stop if self._stop else _stop)
        return OhlcDict({s: self._data[s].loc[_start:_stop] for s in symbols if s in self._data})

    def get_aux_data_ids(self) -> Set[str]:
        return self._reader.get_aux_data_ids() | set(self._external.keys())

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        _exch = kwargs.pop("exchange") if "exchange" in kwargs else None
        if _exch and _exch != self.exchange:
            raise ValueError(f"Exchange mismatch: expected {self.exchange}, got {_exch}")

        match data_id:
            # - special case for candles - it builds them from loaded ohlc data
            case "candles":
                return self._get_candles(**kwargs)

            # - only symbols in cache
            case "symbols":
                return list(self._data.keys())

        if data_id not in self._external:
            self._external[data_id] = self._reader.get_aux_data(data_id, exchange=self.exchange)

        _ext_data = self._external.get(data_id)
        if _ext_data is not None:
            _s = kwargs.pop("start") if "start" in kwargs else None
            _e = kwargs.pop("stop") if "stop" in kwargs else None
            _ext_data = _ext_data[:_e] if _e else _ext_data
            _ext_data = _ext_data[_s:] if _s else _ext_data
        return _ext_data

    def _get_candles(
        self,
        symbols: List[str],
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        _xd: Dict[str, pd.DataFrame] = self[symbols, start:stop]
        _xd = ohlc_resample(_xd, timeframe) if timeframe else _xd
        _r = [x.assign(symbol=s.upper(), timestamp=x.index) for s, x in _xd.items()]
        return srows(*_r).set_index(["timestamp", "symbol"])

    def __str__(self) -> str:
        return f"InMemoryCachedReader(exchange={self.exchange},timeframe={self._data_timeframe})"


class TimeGuardedWrapper(DataReader):
    # - currently 'known' time, can be used for limiting data
    _time_guard_provider: ITimeProvider
    _reader: InMemoryCachedReader

    def __init__(
        self,
        reader: InMemoryCachedReader,
        time_guard: ITimeProvider | None = None,
    ) -> None:
        # - if no time provider is provided, use stub
        class _NoTimeGuard(ITimeProvider):
            def time(self) -> np.datetime64 | None:
                return None

        self._time_guard_provider = time_guard if time_guard is not None else _NoTimeGuard()
        self._reader = reader

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        # timeframe: str | None = None,
        **kwargs,
    ) -> Iterable | List:
        xs = self._time_guarded_data(
            self._reader.read(data_id, start=start, stop=stop, transform=transform, chunksize=0, **kwargs),  # type: ignore
            prev_bar=True,
        )
        return _wrap_as_iterable(xs) if chunksize > 0 else xs

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        return self._time_guarded_data(self._reader.get_aux_data(data_id, exchange=self._reader.exchange, **kwargs))

    def __getitem__(self, keys):
        return self._time_guarded_data(self._reader.__getitem__(keys), prev_bar=True)

    def _time_guarded_data(
        self, data: pd.DataFrame | pd.Series | Dict[str, pd.DataFrame | pd.Series] | List, prev_bar: bool = False
    ) -> pd.DataFrame | pd.Series | Dict[str, pd.DataFrame | pd.Series] | List:
        """
        This function is responsible for limiting the data based on a given time guard.

        Parameters:
        - data (pd.DataFrame | pd.Series | Dict[str, pd.DataFrame | pd.Series] | List): The data to be limited.
        - prev_bar (bool, optional): If True, the time guard is applied to the previous bar. Defaults to False.

        Returns:
        - pd.DataFrame | pd.Series | Dict[str, pd.DataFrame | pd.Series] | List: The limited data.
        """
        # - when no any limits - just returns it as is
        if (_c_time := self._time_guard_provider.time()) is None:
            return data

        _cut_dict = lambda xs, t: OhlcDict({s: v.loc[:t] for s, v in xs.items()})
        _cut_list_of_timestamped = lambda xs, t: list(filter(lambda x: x.time <= t, xs))
        _cut_list_raw = lambda xs, t: list(filter(lambda x: x[0] <= t, xs))
        _cut_time_series = lambda ts, t: ts.loc[: str(t)]

        if prev_bar:
            _c_time = _c_time - pd.Timedelta(self._reader._data_timeframe)

        # - input is Dict[str, pd.DataFrame]
        if isinstance(data, dict):
            return _cut_dict(data, _c_time)

        # - input is List[(time, *data)] or List[Quote | Trade | Bar]
        if isinstance(data, list):
            if isinstance(data[0], (list, tuple, np.ndarray)):
                return _cut_list_raw(data, _c_time)
            else:
                return _cut_list_of_timestamped(data, _c_time.asm8.item())

        # - input is TimeSeries
        if isinstance(data, TimeSeries):
            return _cut_time_series(data, _c_time)

        return data.loc[:_c_time]

    def __str__(self) -> str:
        return f"TimeGuarded @ {str(self._reader)}"


__KNOWN_READERS = {
    "mqdb": MultiQdbConnector,  # mqdb::xlydian-data
    "multi": MultiQdbConnector,
    "qdb": QuestDBConnector,  # questdb::localhost
    "questdb": MultiQdbConnector,  # questdb::localhost
    "csv": CsvStorageDataReader,  # csv::path_to_storage, csv::c:/ssss/
}


def loader(
    exchange: str, timeframe: str, *symbols: List[str], source: str = "mqdb::localhost", **kwargs
) -> InMemoryCachedReader:
    """
    Create and initialize an InMemoryCachedReader for a specific exchange and timeframe.

    This function sets up a cached reader for financial data, optionally pre-loading
    data for specified symbols from the beginning of time until now.

    Args:
        exchange (str): The name of the exchange to load data from.
        timeframe (str): The time interval for the data (e.g., '1d' for daily, '1h' for hourly).
        *symbols (List[str]): Variable number of symbol names to pre-load data for.
        reader (str): The data reader and it's parameter to use. Defaults to mqdb::localhost.

    Returns:
        InMemoryCachedReader: An initialized InMemoryCachedReader object, potentially pre-loaded with data.

    Examples:
    --------
    >>> ld = loader("BINANCE.UM", '1h', source="mqdb::xlydian-data")
        d = ld["BTCUSDT", "ETHUSDT", "SOLUSDT" , "2020-01-01":"2024-12-01"]
        d('1d').close.plot()
        print(d('1d'))
        d("4h").close.pct_change(fill_method=None).cov()
    """
    if not source:
        raise ValueError("Source parameter must be provided")

    _rcls_par = source.split("::")
    _c: Type[DataReader] | None = __KNOWN_READERS.get(_rcls_par[0])
    if _c is None:
        raise ValueError(
            f"Unsupported data reader type: {_rcls_par[0]}. Supported names: {', '.join(__KNOWN_READERS.keys())}."
        )

    reader_object: DataReader = _c(_rcls_par[1]) if len(_rcls_par) else _c()
    inmcr = InMemoryCachedReader(exchange, reader_object, timeframe, **kwargs)
    if symbols:
        # by default slicing from 1970-01-01 until now
        inmcr[list(symbols), slice("1970-01-01", str(pd.Timestamp("now")))]
    return inmcr
