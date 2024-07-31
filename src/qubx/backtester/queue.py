import pandas as pd
import heapq

from collections import defaultdict
from typing import Any, Iterator

from qubx import logger
from qubx.core.basics import Instrument, dt_64
from qubx.data.readers import DataReader, DataTransformer
from qubx.utils.misc import Stopwatch


_SW = Stopwatch()


class DataLoader:
    def __init__(
        self,
        transformer: DataTransformer,
        reader: DataReader,
        instrument: Instrument,
        timeframe: str | None,
        preload_bars: int = 0,
        data_type: str = "ohlc",
        chunksize: int = 5_000,
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

    def create_iterable(self, start: str | pd.Timestamp, stop: str | pd.Timestamp) -> Iterator:
        self._start = start
        self._stop = stop
        self._current_time = None
        return self

    def __iter__(self) -> Iterator:
        logger.info("Initializing chunks for each loader")
        self._current_time = self._start
        self._index_to_chunk_size = {}
        self._index_to_iterator = {}
        self._event_heap = []
        for loader_index in self._index_to_loader.keys():
            self._add_chunk_to_heap(loader_index)
        return self

    @_SW.watch("DataQueue")
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

        loader = self._index_to_loader[loader_index]
        return loader.symbol, loader.data_type, event

    @_SW.watch("DataQueue")
    def _add_chunk_to_heap(self, loader_index: int):
        chunk = self._next_chunk(loader_index)
        self._index_to_chunk_size[loader_index] = len(chunk)
        for chunk_index, event in enumerate(chunk):
            dt = event.time  # type: ignore
            heapq.heappush(self._event_heap, (dt, loader_index, chunk_index, event))

    @_SW.watch("DataQueue")
    def _next_chunk(self, index: int) -> list[Any]:
        if index not in self._index_to_iterator:
            self._index_to_iterator[index] = self._index_to_loader[index].load(self._current_time, self._stop)
        iterator = self._index_to_iterator[index]
        try:
            return next(iterator)
        except StopIteration:
            return []


class EventBatcher:
    _BATCH_SETTINGS = {
        "trade": "1Sec",
        "agg_trade": "1Sec",
        "orderbook": "1Sec",
    }

    def __init__(self, source_iterator: Iterator | list, passthrough: bool = False, **kwargs):
        self.source_iterator = iter(source_iterator) if isinstance(source_iterator, list) else source_iterator
        self._passthrough = passthrough
        self._batch_settings = {**self._BATCH_SETTINGS, **kwargs}
        self._batch_settings = {k: pd.Timedelta(v) for k, v in self._batch_settings.items()}
        self._event_buffers = defaultdict(lambda: defaultdict(list))

    def __iter__(self):
        if self._passthrough:
            yield from self.source_iterator
        for symbol, data_type, event in self.source_iterator:
            time: dt_64 = event.time  # type: ignore
            yield from self._process_buffers(time)

            if data_type not in self._batch_settings:
                yield symbol, data_type, event
                continue

            symbol_buffers = self._event_buffers[symbol]
            buffer = symbol_buffers[data_type]
            buffer.append(event)
            delta = pd.Timedelta(time - buffer[0].time)

            if delta >= self._batch_settings[data_type]:
                symbol_buffers[data_type] = []
                yield symbol, data_type, buffer

        yield from self._cleanup_buffers()

    def _process_buffers(self, time: dt_64):
        """
        Yield all buffers that are older than the batch settings.
        """
        yield_buffers = []
        for symbol, buffers in self._event_buffers.items():
            for data_type, buffer in buffers.items():
                if buffer and pd.Timedelta(time - buffer[0].time) >= self._batch_settings[data_type]:
                    yield_buffers.append((symbol, data_type, buffer))
                    buffers[data_type] = []
        yield_buffers.sort(key=lambda x: x[-1][0].time)
        for symbol, data_type, buffer in yield_buffers:
            yield symbol, data_type, buffer

    def _cleanup_buffers(self):
        yield_buffers = []
        for symbol, buffers in self._event_buffers.items():
            for data_type, buffer in buffers.items():
                if buffer:
                    yield_buffers.append((symbol, data_type, buffer))
                    buffers[data_type] = []
        yield_buffers.sort(key=lambda x: x[-1][0].time)
        for symbol, data_type, buffer in yield_buffers:
            yield symbol, data_type, buffer
