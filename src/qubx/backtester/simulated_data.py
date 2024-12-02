import pandas as pd

from collections import defaultdict, deque
from typing import Any, Iterator, TypeAlias

from qubx import logger
from qubx.core.basics import Instrument, Subtype, dt_64
from qubx.core.series import Quote, Trade, Bar, OrderBook
from qubx.data.readers import (
    AsTimestampedRecords,
    DataReader,
    DataTransformer,
    RestoreTicksFromOHLC,
    RestoredBarsFromOHLC,
)

_T: TypeAlias = Quote | Trade | Bar | OrderBook
_D: TypeAlias = tuple[str, int, _T] | tuple


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


class IteratedDataStreamsSlicer(Iterator[_D]):
    """
    Slicer for iterated data streams.
    """

    _iterators: dict[str, Iterator[list[_T]]]
    _buffers: dict[str, list[_T]]

    _keys: deque[str]
    _r_keys: deque[str]
    _init_k_maxes: list[int]
    _init_k_idxs: list[str]
    _k_max: int
    _iterating: bool

    def __init__(self):
        self._buffers = defaultdict(list)
        self._iterators = {}
        self._keys = deque()
        self._r_keys = deque()
        self._init_k_maxes = []
        self._init_k_idxs = []
        self._k_max = 0
        self._iterating = False

    def put(self, data: dict[str, Iterator[list[_T]]]):
        _rebuild = False
        for k, vi in data.items():
            if k not in self._keys:
                self._iterators[k] = vi
                self._buffers[k] = self._get_next_chunk_to_buffer(k)  # do initial chunk fetching
                self._keys.append(k)
                _rebuild = True

        # - rebuild strategy
        if _rebuild and self._iterating:
            self._build_initial_iteration_strategy()

    def __add__(self, data: dict[str, Iterator]) -> "IteratedDataStreamsSlicer":
        self.put(data)
        return self

    def remove(self, keys: list[str] | str):
        """
        Remove data iterator and associated keys from the queue.
        If the key is not found, it does nothing.
        """
        _keys = keys if isinstance(keys, list) else [keys]
        _rebuild = False
        for i in _keys:
            if i in self._buffers:
                self._buffers.pop(i)
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

        if len(self._buffers) > 1:
            self._r_keys = deque()

            _init_seq = {k: self._buffers[k][-1].time for k in self._keys}
            _init_seq = dict(sorted(_init_seq.items(), key=lambda item: item[1]))

            self._init_k_maxes = list(_init_seq.values())[1:]
            self._init_k_idxs = list(_init_seq.keys())

            self._k_max = self._init_k_maxes.pop(0)
            self._r_keys.append(self._init_k_idxs.pop(0))

    def _get_next_chunk_to_buffer(self, index: str) -> list[_T]:
        return list(reversed(next(self._iterators[index])))

    def __next__(self) -> _D:
        if not self._r_keys:
            self._iterating = False
            raise StopIteration

        k = self._r_keys[0]
        data = self._buffers[k]

        if not data:
            try:
                # - get next chunk of data
                data.extend(self._get_next_chunk_to_buffer(k))
            except StopIteration:
                print(f" > Iterator[{k}] is empty")

                # - remove iterable data
                self._buffers.pop(k)
                self._iterators.pop(k)
                self._keys.remove(k)
                self._r_keys.remove(k)

                return ()

        _last = data.pop()
        value = (k, _t := _last.time, _last)
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
        return value


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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
                self._id = hash(self._data_type + str(self._timeframe))
            case Subtype.TRADE:
                self._transformer = AsTimestampedRecords()
                self._data_type = "agg_trades"
                self._id = hash(self._data_type)
            case Subtype.QUOTE:
                self._transformer = AsTimestampedRecords()
                self._data_type = "orderbook"
                self._id = hash(self._data_type)
            case _:
                raise ValueError(f"Unsupported subscription type: {self._subtype}")

        for i in instruments:
            self.attach_instrument(i)

        self._chunksize = chunksize  # TODO:

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class DataFetcher:
    _fetcher_id: str
    _requested_data_type: str
    _producing_data_type: str
    _params: dict[str, object]
    _specs: list[str]

    _transformer: DataTransformer
    _timeframe: str | None = None
    _warmup_period: pd.Timedelta | None = None
    _chunksize: int = 5000

    def __init__(
        self,
        fetcher_id: str,
        subtype: str,
        params: dict[str, Any],
        warmup_period: pd.Timedelta | None = None,
        chunksize: int = 5000,
    ) -> None:
        self._fetcher_id = fetcher_id
        self._params = params

        match subtype:
            case Subtype.OHLC_TICKS:
                self._transformer = RestoreTicksFromOHLC()
                self._requested_data_type = "ohlc"
                self._producing_data_type = "quote"
                if "timeframe" in params:
                    self._timeframe = params.get("timeframe", "1Min")

            case Subtype.OHLC:
                # TODO: open/close shift may depends on simulation
                self._transformer = RestoredBarsFromOHLC(open_close_time_shift_secs=1)
                self._requested_data_type = "ohlc"
                self._producing_data_type = "ohlc"
                if "timeframe" in params:
                    self._timeframe = params.get("timeframe", "1Min")

            case Subtype.TRADE:
                self._requested_data_type = "trade"
                self._producing_data_type = "trade"
                self._transformer = AsTimestampedRecords()

            case Subtype.QUOTE:
                self._requested_data_type = "orderbook"
                self._producing_data_type = "quote"
                self._transformer = AsTimestampedRecords()

            case _:
                raise ValueError(f"Unsupported subscription type: {subtype}")

        self._warmup_period = warmup_period
        self._warmed = {}
        self._specs = []
        self._chunksize = chunksize

    @staticmethod
    def _make_request_id(instrument: Instrument) -> str:
        return f"{instrument.exchange}:{instrument.symbol}"

    def attach_instrument(self, instrument: Instrument) -> str:
        _data_id = self._make_request_id(instrument)

        if _data_id not in self._specs:
            self._specs.append(_data_id)
            self._warmed[_data_id] = False

        return self._fetcher_id + "." + _data_id

    def remove_instrument(self, instrument: Instrument) -> str:
        _data_id = self._make_request_id(instrument)

        if _data_id in self._specs:
            self._specs.remove(_data_id)
            del self._warmed[_data_id]

        return self._fetcher_id + "." + _data_id

    def has_instrument(self, instrument: Instrument) -> bool:
        return self._make_request_id(instrument) in self._specs

    def load(
        self, reader: DataReader, start: str | pd.Timestamp, end: str | pd.Timestamp, to_load: list[Instrument] | None
    ) -> dict[str, Iterator]:
        # - iterate over all instruments if no indices specified
        _requests = self._specs if not to_load else set(self._make_request_id(i) for i in to_load)
        _r_iters = {}

        logger.debug(f"{self._fetcher_id} loading {_requests}")

        for _r in _requests:  # - TODO: replace this loop with multi-instrument request after DataReader refactoring
            if _r in self._specs:
                _start = pd.Timestamp(start)
                if self._warmup_period and not self._warmed.get(_r):
                    _start -= self._warmup_period
                    self._warmed[_r] = True

                _args = dict(
                    data_id=_r,
                    start=_start,
                    stop=end,
                    transform=self._transformer,
                    data_type=self._requested_data_type,
                    chunksize=self._chunksize,
                )

                if self._timeframe:
                    _args["timeframe"] = self._timeframe

                _r_iters[self._fetcher_id + "." + _r] = reader.read(**_args)  # type: ignore
            else:
                raise IndexError(
                    f"Instrument {_r} is not subscribed for this data {self._requested_data_type} in {self._fetcher_id} !"
                )

        return _r_iters

    def __repr__(self) -> str:
        return f"{self._requested_data_type}({self._params}) (-{self._warmup_period if self._warmup_period else '--'}) [{','.join(self._specs)}] :-> {self._transformer.__class__.__name__}"


class IterableSimulatorData(Iterator):
    _reader: DataReader
    _slicer_ctrl: IteratedDataStreamsSlicer | None
    _subt_to_fetcher: dict[str, DataFetcher]
    _warmups: dict[str, pd.Timedelta]
    _instruments: dict[str, tuple[Instrument, DataFetcher]]

    _start: pd.Timestamp | None
    _stop: pd.Timestamp | None
    _current_time: int | None

    def __init__(self, reader: DataReader):
        self._reader = reader
        self._instruments = {}
        self._subt_to_fetcher = {}
        self._warmups = {}

        self._slicer_ctrl = None
        self._slicing_iterator = None
        self._start = None
        self._stop = None
        self._current_time = None

    def set_warmup_period(self, subscription: str, warmup_period: str | None = None):
        if warmup_period:
            _access_key, _, _ = self._parse_subscription_spec(subscription)
            self._warmups[_access_key] = pd.Timedelta(warmup_period)

    def _parse_subscription_spec(self, subscription: str) -> tuple[str, str, dict[str, object]]:
        _subtype, _params = Subtype.from_str(subscription)
        match _subtype:
            case Subtype.OHLC | Subtype.OHLC_TICKS:
                _timeframe = _params.get("timeframe", "1Min")
                _access_key = f"{_subtype}.{_timeframe}"
            case Subtype.TRADE | Subtype.QUOTE:
                _access_key = f"{_subtype}"
            case _:
                raise ValueError(f"Unsupported subscription type: {_subtype}")
        return _access_key, _subtype, _params

    def add_instruments_for_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        instruments = instruments if isinstance(instruments, list) else [instruments]
        _subt_key, _data_type, _params = self._parse_subscription_spec(subscription)

        fetcher = self._subt_to_fetcher.get(_subt_key)
        if not fetcher:
            self._subt_to_fetcher[_subt_key] = (
                fetcher := DataFetcher(_subt_key, _data_type, _params, warmup_period=self._warmups.get(_subt_key))
            )

        _instrs_to_preload = []
        for i in instruments:
            if not fetcher.has_instrument(i):
                idx = fetcher.attach_instrument(i)
                self._instruments[idx] = (i, fetcher)  # type: ignore
                _instrs_to_preload.append(i)

        if self.is_running and _instrs_to_preload:
            self._slicer_ctrl += fetcher.load(
                self._reader,
                pd.Timestamp(self._current_time, unit="ns"),  # type: ignore
                self._stop,  # type: ignore
                _instrs_to_preload,
            )

    def remove_instruments_from_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        def _remove_from_fetcher(_subt_key: str, instruments: list[Instrument]):
            fetcher = self._subt_to_fetcher.get(_subt_key)
            if not fetcher:
                logger.warning(f"No configured data fetcher for '{_subt_key}' subscription !")
                return

            _keys_to_remove = []
            for i in instruments:
                # - try to remove from data fetcher
                if idx := fetcher.remove_instrument(i):
                    if idx in self._instruments:
                        self._instruments.pop(idx)
                        _keys_to_remove.append(idx)

            print("REMOVING FROM:", _keys_to_remove)
            if self.is_running and _keys_to_remove:
                self._slicer_ctrl.remove(_keys_to_remove)  # type: ignore

        instruments = instruments if isinstance(instruments, list) else [instruments]

        # - if we want to remove instruments from all subscriptions
        if subscription == Subtype.ALL:
            _f_keys = list(self._subt_to_fetcher.keys())
            for s in _f_keys:
                _remove_from_fetcher(s, instruments)
            return

        _subt_key, _, _ = self._parse_subscription_spec(subscription)
        _remove_from_fetcher(_subt_key, instruments)

    @property
    def is_running(self) -> bool:
        return self._current_time is not None

    def create_iterable(self, start: str | pd.Timestamp, stop: str | pd.Timestamp) -> Iterator:
        self._start = pd.Timestamp(start)
        self._stop = pd.Timestamp(stop)
        self._current_time = None
        self._slicer_ctrl = IteratedDataStreamsSlicer()
        return self

    def __iter__(self) -> Iterator:
        logger.debug("Preloading initial data for each fetcher ...")
        assert self._start is not None
        self._current_time = int(pd.Timestamp(self._start).timestamp() * 1e9)
        _ct_timestap = pd.Timestamp(self._current_time, unit="ns")

        for f in self._subt_to_fetcher.values():
            self._slicer_ctrl += f.load(self._reader, _ct_timestap, self._stop, None)  # type: ignore

        self._slicing_iterator = iter(self._slicer_ctrl)
        return self

    def __next__(self) -> tuple[Instrument, str, Any]:
        try:
            while data := next(self._slicing_iterator):  # type: ignore
                k, t, v = data
                instr, fetcher = self._instruments[k]
                data_type = fetcher._producing_data_type
                if t < self._current_time:  # type: ignore
                    data_type = f"hist_{data_type}"

                else:
                    # only update the current time if the event is not historical
                    self._current_time = t

                return instr, data_type, v
        except StopIteration as e:
            raise StopIteration
