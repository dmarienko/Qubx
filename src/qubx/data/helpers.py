from typing import Any, Dict, Iterable, List
from concurrent.futures import ThreadPoolExecutor

from joblib import delayed
import pandas as pd
from collections import defaultdict

from qubx import logger
from qubx.data.readers import (
    AsPandasFrame,
    DataReader,
    InMemoryDataFrameReader,
    MultiQdbConnector,
    DataTransformer,
)
from qubx.pandaz.utils import generate_equal_date_ranges, srows
from qubx.utils.misc import ProgressParallel
from qubx.utils.time import convert_seconds_to_str, handle_start_stop, infer_series_frequency


def load_data(
    qdb: MultiQdbConnector,
    exch: str,
    symbols: list | str,
    start: str,
    stop: str = "now",
    timeframe="1h",
    transform: DataTransformer = AsPandasFrame(),
    max_workers: int = 16,
) -> dict[str, Any]:
    if isinstance(symbols, str):
        symbols = [symbols]
    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(symbols)))
    data = {}
    res = {
        s: executor.submit(qdb.read, f"{exch}:{s}", start, stop, transform=transform, timeframe=timeframe)
        for s in symbols
    }
    data = {s: r.result() for s, r in res.items()}
    return data


class InMemoryCachedReader(InMemoryDataFrameReader):
    """
    TODO: ...
    """

    _data_timeframe: str
    _reader: DataReader
    _n_jobs: int
    exchange: str
    _start: pd.Timestamp | None = None
    _stop: pd.Timestamp | None = None

    # _fundamental: pd.DataFrame

    def __init__(
        self,
        exchange: str,
        reader: DataReader,
        base_timeframe: str,
        n_jobs: int = -1,
    ) -> None:
        self._reader = reader
        self._n_jobs = n_jobs
        self._data_timeframe = base_timeframe
        self.exchange = exchange
        super().__init__({}, exchange)

    # def get_fundamental_data(
    #     self, exchange: str, start: str | pd.Timestamp | None = None, stop: str | pd.Timestamp | None = None
    # ) -> pd.DataFrame:
    #     stop = (pd.Timestamp(stop) - pd.Timedelta("1d")) if stop else stop
    #     return self._fundamental.loc[slice(start, stop)]

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

        if symb not in self._data:
            logger.debug(f"Loading data for {_s_path}")
            self._data[symb] = self._reader.read(_s_path, transform=AsPandasFrame(), timeframe=self._data_timeframe)

        return super().read(_s_path, start, stop, transform, chunksize, **kwargs)

    # def get_candles(
    #     self,
    #     exchange: str,
    #     symbols: list[str],
    #     start: str | pd.Timestamp,
    #     stop: str | pd.Timestamp,
    #     timeframe: str = "1d",
    # ) -> pd.DataFrame:
    #     if exchange != self.exchange:
    #         raise ValueError(f"Exchange mismatch: {exchange}!= {self.exchange}")
    #     _r = []
    #     start, stop = handle_start_stop(start, stop)
    #     for s in sorted(symbols, reverse=True):
    #         if s not in self._data:
    #             logger.debug(f">>> LOADING DATA for {s} ...")
    #             try:
    #                 _d = self._reader.get_aux_data(
    #                     "candles", exchange, [s], start, stop, timeframe=self._data_timeframe
    #                 )
    #                 self._data[s] = _d.droplevel(1)
    #             except Exception as e:
    #                 continue

    #         _d = self._data[s]
    #         _d = _d[(_d.index >= start) & (_d.index < stop)].copy()
    #         _d = ohlc_resample(_d, timeframe) if timeframe else _d
    #         _r.append(_d.assign(symbol=s.upper(), timestamp=_d.index))
    #     return srows(*_r).set_index(["timestamp", "symbol"])

    def _load_data(
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

    def _get_smbs_at(self, symbols: List[str], start: str, stop: str) -> Dict[str, pd.DataFrame | pd.Series]:
        T = lambda x: pd.Timestamp(x)
        _start, _stop = map(T, handle_start_stop(start, stop))

        # - full interval
        _new_symbols = list(set([s for s in symbols if s not in self._data]))
        if _new_symbols:
            logger.debug(f"loading full interval {_new_symbols}")
            _new_data = self._load_data(_new_symbols, _start, _stop, self._data_timeframe)
            self._data |= _new_data

        # - part intervals
        if self._start and _start < self._start:
            logger.debug(f"Updating before interval {_start} : {self._start}")
            _before = self._load_data(
                list(self._data.keys()), _start, self._start + pd.Timedelta(self._data_timeframe), self._data_timeframe
            )
            for k, c in _before.items():
                self._data[k] = srows(c, self._data[k], keep="first")

        # - part intervals
        if self._stop and _stop > self._stop:
            logger.debug(f"Updating after interval {self._stop} : {_stop}")
            _after = self._load_data(
                list(self._data.keys()), self._stop - pd.Timedelta(self._data_timeframe), _stop, self._data_timeframe
            )
            for k, c in _after.items():
                self._data[k] = srows(self._data[k], c, keep="last")

        self._start = min(_start, self._start if self._start else _start)
        self._stop = max(_stop, self._stop if self._stop else _stop)
        return {s: self._data[s].loc[_start:_stop] for s in symbols if s in self._data}

    def __str__(self) -> str:
        return f"InMemoryCachedReader(exchange={self.exchange})"
