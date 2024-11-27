import pandas as pd

from collections import defaultdict, deque
from typing import Any, Iterator, Iterable

# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

from qubx import logger
from qubx.core.basics import Instrument, Subtype, dt_64, BatchEvent
from qubx.data.readers import AsTimestampedRecords, DataReader, DataTransformer, RestoreTicksFromOHLC
from qubx.utils.misc import Stopwatch


class BiDirectionIndexedObjects:
    _obj_to_index: dict[object, int]
    _index_to_obj: dict[int, object]
    _last_attached_idx: int
    _removed_obj_indices: set[int]

    def __init__(self):
        self._obj_to_index = {}
        self._index_to_obj = {}
        self._last_attached_idx = 0
        self._removed_obj_indices = set()

    def contains(self, obj: object) -> bool:
        return obj in self._obj_to_index

    def add_value(self, obj: object) -> int:
        if not self.contains(obj):
            self._last_attached_idx += 1
            self._index_to_obj[self._last_attached_idx] = obj
            self._obj_to_index[obj] = self._last_attached_idx
        else:
            return -1
        return self._last_attached_idx

    def remove_value(self, obj: object) -> int:
        idx_to_remove = -1
        if self.contains(obj):
            self._index_to_obj.pop(idx_to_remove := self._obj_to_index.pop(obj))
            self._removed_obj_indices.add(idx_to_remove)
        return idx_to_remove

    def get_value_by_index(self, idx: int) -> object:
        return self._index_to_obj.get(idx)

    def get_index_of_value(self, obj: object) -> object:
        return self._obj_to_index.get(obj, -1)

    def items(self) -> Iterator[tuple[int, object]]:
        return ((idx, obj) for idx, obj in self._index_to_obj.items() if idx not in self._removed_obj_indices)

    def values(self) -> list[object]:
        return list(self._index_to_obj.values())

    def indices(self) -> list[object]:
        return list(self._index_to_obj.keys())

    def is_removed(self, idx: int) -> bool:
        return idx in self._removed_obj_indices

    def __str__(self) -> str:
        _r = ""
        for i, o in self.items():
            _r += f"[{i}]: {str(o)}\n"
        if self._removed_obj_indices:
            _r += "removed: " + ",".join([f"{-i}" for i in self._removed_obj_indices])
        return _r

    def __repr__(self) -> str:
        return str(self)


class IteratorsTimeSlicer(Iterator):
    _iterators: dict[int, Iterator]
    _datas: dict[int, list[tuple[int, int, Any]]]

    _keys: deque[int]
    _r_keys: deque[int]
    _init_k_maxes: list[int]
    _init_k_idxs: list[int]
    _k_max: int
    _iterating: bool

    def __init__(self):
        self._datas = defaultdict(list)
        self._iterators = {}
        self._keys = deque()
        self._r_keys = deque()
        self._init_k_maxes = []
        self._init_k_idxs = []
        self._k_max = 0
        self._iterating = False

    def put(self, data: dict[int, Iterator]):
        _rebuild = False
        for k, vi in data.items():
            if k not in self._keys:
                self._iterators[k] = vi
                self._datas[k] = self._fetch_next_chunk(k)  # do initial chunk fetching
                self._keys.append(k)
                _rebuild = True

        # - rebuild strategy
        if _rebuild and self._iterating:
            self._build_initial_iteration_strategy()

    def __add__(self, data: dict[int, Iterator]) -> "IteratorsTimeSlicer":
        self.put(data)
        return self

    def remove(self, keys: list[int] | int):
        """
        Remove data iterator and associated keys from the queue.
        If the key is not found, it does nothing.
        """
        _keys = keys if isinstance(keys, list) else [keys]
        _rebuild = False
        for i in _keys:
            if i in self._datas:
                self._datas.pop(i)
                self._iterators.pop(i)
                self._keys.remove(i)
                _rebuild = True

        # - rebuild strategy
        if _rebuild and self._iterating:
            self._build_initial_iteration_strategy()

    def __iter__(self) -> Iterator:
        # - for more than 1 iterator we need to build initial iteration strategy
        self._build_initial_iteration_strategy()
        self._iterating = True
        return self

    def _build_initial_iteration_strategy(self):
        self._k_max = 0
        self._init_k_idxs = []
        self._init_k_maxes = []
        self._r_keys = deque(self._keys)

        if len(self._datas) > 1:
            self._r_keys = deque()

            _init_seq = {k: self._datas[k][-1].time for k in self._keys}
            _init_seq = dict(sorted(_init_seq.items(), key=lambda item: item[1]))

            self._init_k_maxes = list(_init_seq.values())[1:]
            self._init_k_idxs = list(_init_seq.keys())

            self._k_max = self._init_k_maxes.pop(0)
            self._r_keys.append(self._init_k_idxs.pop(0))

    def _fetch_next_chunk(self, index: int) -> list[tuple[int, int, Any]]:
        return list(reversed(next(self._iterators[index])))

    def __next__(self) -> tuple[int, int, Any]:
        if not self._r_keys:
            self._iterating = False
            raise StopIteration

        k = self._r_keys[0]
        d = self._datas[k]

        if not d:
            try:
                d.extend(self._fetch_next_chunk(k))
            except StopIteration:
                print(f" > Iterator[{k}] is empty")
                self.remove(k)
                return ()
        _last = d.pop()
        r = (k, _t := _last.time, _last)

        if self._init_k_idxs:
            if _t >= self._k_max:
                self._r_keys.append(self._init_k_idxs.pop(0))

                if self._init_k_maxes:
                    self._k_max = self._init_k_maxes.pop(0)

            self._r_keys.rotate(-1)
        else:
            if _t >= self._k_max:
                self._r_keys.rotate(-1)  # - switch to the next iterated data
                self._k_max = _t
        return r


class SimulationDataLoader:
    def __init__(
        self,
        reader: DataReader,
        subscription: str,
        instruments: list[Instrument],
        warmup_period: str | None = None,
        chunksize: int = 5000,
    ):
        self._instruments = BiDirectionIndexedObjects()
        self._reader = reader
        self._subscription = subscription
        self._subtype, self._subparams = Subtype.from_str(subscription)
        self._warmup_period = warmup_period
        self._warmed = {}
        self._timeframe = None

        match self._subtype:
            case Subtype.OHLC:
                # - making ticks out of OHLC
                self._transformer = RestoreTicksFromOHLC()
                self._timeframe = self._subparams.get("timeframe")
                self._data_type = "ohlc"
                _id = self._data_type + str(self._timeframe)
            case Subtype.TRADE:
                self._transformer = AsTimestampedRecords()
                self._data_type = "agg_trades"
                _id = self._data_type
            case Subtype.QUOTE:
                self._transformer = AsTimestampedRecords()
                self._data_type = "orderbook"
                _id = self._data_type
            case _:
                raise ValueError(f"Unsupported subscription type: {self._subtype}")

        for i in instruments:
            self.attach_instrument(i)

        self._id = hash(_id)
        self._chunksize = chunksize  # TODO:

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SimulationDataLoader):
            return False
        return self._data_type == other._data_type and self._timeframe == other._timeframe

    def instrument(self, idx: int) -> Instrument:
        return self._instruments.get_value_by_index(idx)

    def attach_instrument(self, instrument: Instrument) -> int:
        if not self._instruments.contains(instrument):
            self._warmed |= {f"{instrument.exchange}:{instrument.symbol}": False}
            return self._instruments.add_value(instrument)
        return -1

    def remove_instrument(self, instrument: Instrument) -> int:
        _ix = -1
        if self._instruments.contains(instrument):
            _ix = self._instruments.remove_value(instrument)
            self._warmed.pop(f"{instrument.exchange}:{instrument.symbol}")
        return _ix

    def shutdown(self):
        for s in self._instruments.values():
            self.remove_instrument(s)

    def is_instrument_removed(self, idx: int) -> bool:
        return self._instruments.is_removed(idx)

    def load(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp, indices: list[int] | None
    ) -> dict[int, Iterator]:
        # - iterate over all instruments if no indices specified
        _indices = self._instruments.indices() if not indices else indices
        _r_iters = {}
        for ix in _indices:
            if ix == -1:
                continue

            if _i := self._instruments.get_value_by_index(ix):
                _s = f"{_i.exchange}:{_i.symbol}"
                _start = pd.Timestamp(start)
                if self._warmup_period and not self._warmed.get(_s):
                    _start -= pd.Timedelta(self._warmup_period)
                    self._warmed[_s] = True

                _args = dict(
                    data_id=_s,
                    start=_start,
                    stop=end,
                    transform=self._transformer,
                    data_type=self._data_type,
                    chunksize=self._chunksize,
                )

                if self._timeframe:
                    _args["timeframe"] = self._timeframe

                _r_iters[ix] = self._reader.read(**_args)  # type: ignore
            else:
                raise IndexError(f"No instrument found for index {ix}")

        return _r_iters
