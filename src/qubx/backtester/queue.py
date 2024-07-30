from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Callable
from itertools import chain
import numpy as np
import pandas as pd
from enum import Enum
from tqdm.auto import tqdm

from qubx import lookup, logger
from qubx.core.helpers import BasicScheduler
from qubx.core.loggers import InMemoryLogsWriter
from qubx.core.series import Quote
from qubx.core.account import AccountProcessor
from qubx.core.basics import (
    Instrument,
    Deal,
    Order,
    Signal,
    SimulatedCtrlChannel,
    Position,
    TradingSessionResult,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV
from qubx.core.strategy import (
    IStrategy,
    IBrokerServiceProvider,
    ITradingServiceProvider,
    PositionsTracker,
    StrategyContext,
    TriggerEvent,
)
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
    ) -> None:
        self._instrument = instrument
        self._spec = f"{instrument.exchange}:{instrument.symbol}"
        self._reader = reader
        self._transformer = transformer
        self._init_bars_required = preload_bars
        self._timeframe = timeframe
        self._data_type = data_type
        self._first_load = True

    def load(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> List[Any]:
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


class SimulatedDataQueue:
    _loaders: dict[str, list[DataLoader]]

    def __init__(self, start: str, stop: str):
        self._loaders = defaultdict(list)
        self.start = start
        self.stop = stop

    def add_loader(self, loader: DataLoader):
        self._loaders[loader.symbol].append(loader)

    def __iter__(self):
        _index_to_loader = dict(enumerate(chain.from_iterable(self._loaders.values())))
        _index_to_events = {
            index: loader.load(self.start, self.stop) for index, loader in _index_to_loader.items()
        }
        events = sorted(chain.from_iterable(_index_to_events.values()), key=lambda e: e.time)
        yield from events
