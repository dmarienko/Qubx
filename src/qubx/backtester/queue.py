import numpy as np
import pandas as pd
import heapq

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeAlias, Callable
from itertools import chain
from enum import Enum
from tqdm.auto import tqdm

from qubx import lookup, logger
from qubx.core.basics import (
    Instrument,
    dt_64,
)
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV
from qubx.backtester.ome import OrdersManagementEngine, OmeReport

from qubx.data.readers import (
    AsTrades,
    DataReader,
    DataTransformer,
    RestoreTicksFromOHLC,
    AsQuotes,
    AsTimestampedRecords,
    InMemoryDataFrameReader,
)
from qubx.pandaz.utils import scols


class DataLoader:
    def __init__(
        self,
        transformer: DataTransformer,
        reader: DataReader,
        instrument: Instrument,
        timeframe: str | None,
        preload_bars: int = 0,
        data_type: str = "candles",
        chunksize: int = 10_000,
    ) -> None:
        self._instrument = instrument
        self._spec = f"{instrument.exchange}:{instrument.symbol}"
        self._reader = reader
        self._transformer = transformer
        self._init_bars_required = preload_bars
        self._timeframe = timeframe
        self._data_type = data_type
        self._first_load = True
        self._chunksize = chunksize

    def load(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> Iterator:
        if self._first_load:
            if self._init_bars_required > 0 and self._timeframe:
                start = pd.Timestamp(start) - self._init_bars_required * pd.Timedelta(self._timeframe)
            self._first_load = False

        args = dict(
            data_id=self._spec,
            start=start,
            stop=end,
            transform=self._transformer,
            data_type=self._data_type,
            chunksize=self._chunksize,
        )

        if self._timeframe:
            args["timeframe"] = self._timeframe

        return self._reader.read(**args)  # type: ignore

    def get_historical_ohlc(self, timeframe: str, start_time: str, nbarsback: int) -> List[Bar]:
        start = pd.Timestamp(start_time)
        end = start - nbarsback * pd.Timedelta(timeframe)
        records = self._reader.read(
            data_id=self._spec, start=start, stop=end, transform=AsTimestampedRecords()  # type: ignore
        )
        return [
            Bar(np.datetime64(r["timestamp_ns"], "ns").item(), r["open"], r["high"], r["low"], r["close"], r["volume"])
            for r in records
        ]

    @property
    def instrument(self) -> Instrument:
        return self._instrument

    @property
    def symbol(self) -> str:
        return self._instrument.symbol

    @property
    def data_type(self) -> str:
        return self._data_type

    def __hash__(self) -> int:
        return hash((self._instrument.symbol, self._data_type))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataLoader):
            return False
        return self._instrument.symbol == other._instrument.symbol and self._data_type == other._data_type


class SimulatedDataQueue:
    _loaders: dict[str, list[DataLoader]]

    def __init__(self):
        self._loaders = defaultdict(list)
        self._start = None
        self._stop = None
        self._current_time = None
        self._index_to_loader: dict[int, DataLoader] = {}
        self._loader_to_index = {}
        self._latest_loader_index = -1
        self._removed_loader_indices = set()

    @property
    def is_running(self) -> bool:
        return self._current_time is not None

    def __add__(self, loader: DataLoader) -> "SimulatedDataQueue":
        self._latest_loader_index += 1
        new_loader_index = self._latest_loader_index
        self._loaders[loader.symbol].append(loader)
        self._index_to_loader[new_loader_index] = loader
        self._loader_to_index[loader] = new_loader_index
        if self.is_running:
            self._add_chunk_to_heap(new_loader_index)
        return self

    def __sub__(self, loader: DataLoader) -> "SimulatedDataQueue":
        loader_index = self._loader_to_index[loader]
        self._loaders[loader.symbol].remove(loader)
        del self._index_to_loader[loader_index]
        del self._loader_to_index[loader]
        del self._index_to_chunk_size[loader_index]
        del self._index_to_iterator[loader_index]
        self._removed_loader_indices.add(loader_index)
        return self

    def get_loader(self, symbol: str, data_type: str) -> DataLoader:
        loaders = self._loaders[symbol]
        for loader in loaders:
            if loader.data_type == data_type:
                return loader
        raise ValueError(f"Loader for {symbol} and {data_type} not found")

    def create_iterator(self, start: str | pd.Timestamp, stop: str | pd.Timestamp) -> Iterator:
        self._start = start
        self._stop = stop
        return iter(self)

    def __iter__(self) -> Iterator:
        logger.info("Initializing chunks for each loader")
        self._current_time = self._start
        self._index_to_chunk_size = {}
        self._index_to_iterator = {}
        self._event_heap = []
        for loader_index in self._index_to_loader.keys():
            self._add_chunk_to_heap(loader_index)
        return self

    def __next__(self) -> tuple[str, Any]:
        if not self._event_heap:
            raise StopIteration

        loader_index = None

        # get the next event from the heap
        # if the loader_index is in the removed_loader_indices, skip it (optimization to avoid unnecessary heap operations)
        while self._event_heap and (loader_index is None or loader_index in self._removed_loader_indices):
            dt, loader_index, chunk_index, event = heapq.heappop(self._event_heap)

        if loader_index is None or loader_index in self._removed_loader_indices:
            raise StopIteration

        self._current_time = dt
        chunk_size = self._index_to_chunk_size[loader_index]
        if chunk_index + 1 == chunk_size:
            self._add_chunk_to_heap(loader_index)

        s = self._index_to_loader[loader_index].symbol
        return s, event

    def _add_chunk_to_heap(self, loader_index: int):
        chunk = self._next_chunk(loader_index)
        self._index_to_chunk_size[loader_index] = len(chunk)
        for chunk_index, event in enumerate(chunk):
            dt = event.time  # type: ignore
            heapq.heappush(self._event_heap, (dt, loader_index, chunk_index, event))

    def _next_chunk(self, index: int) -> list[Any]:
        if index not in self._index_to_iterator:
            self._index_to_iterator[index] = self._index_to_loader[index].load(self._current_time, self._stop)
        iterator = self._index_to_iterator[index]
        try:
            return next(iterator)
        except StopIteration:
            return []
