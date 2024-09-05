import numpy as np
from typing import Any, Tuple

import pandas as pd

class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    def __init__(self, time, open, high, low, close, volume, bought_volume=0): ...

class Quote:
    time: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def __init__(self, time, bid, ask, bid_size, ask_size): ...
    def mid_price(self) -> float: ...

class Trade:
    time: int
    price: float
    size: float
    taker: int
    trade_id: int
    def __init__(self, time, price, size, taker=-1, trade_id=0): ...

class Locator:
    def __getitem__(self, idx): ...
    def find(self, t: str) -> Tuple[np.datetime64, Any]: ...

class Indexed:
    def __getitem__(self, idx): ...
    def lookup_idx(self, value, method: str) -> int: ...

class TimeSeries:
    name: str
    loc: Locator
    timeframe: int
    max_series_length: int
    times: Indexed
    values: Indexed
    def __init__(self, name, timeframe, max_series_length, process_every_update=True) -> None: ...
    def __getitem__(self, idx): ...
    def update(self, time: int, value: float) -> bool: ...
    def copy(self, start: int, stop: int) -> "TimeSeries": ...
    def shift(self, period: int) -> TimeSeries: ...
    def get_indicators(self) -> dict: ...
    def plot(self, *args, **kwargs): ...
    def to_records(self) -> dict: ...
    def to_series(self) -> pd.Series: ...
    def pd(self) -> pd.Series: ...

class OHLCV(TimeSeries):
    open: TimeSeries
    high: TimeSeries
    low: TimeSeries
    close: TimeSeries
    volume: TimeSeries
    bvolume: TimeSeries

    def __init__(self, name, timeframe, max_series_length) -> None: ...
    def __len__(self) -> int: ...
    def update(self, time: int, price: float, volume: float = 0.0, bvolume: float = 0.0) -> bool: ...
    def update_by_bar(
        self,
        time: int,
        open: float,
        high: float,
        low: float,
        close: float,
        vol_incr: float = 0.0,
        b_vol_incr: float = 0.0,
    ) -> bool: ...
    def to_records(self) -> dict: ...
    def pd(self) -> pd.DataFrame: ...

class Indicator(TimeSeries):
    name: str
    series: TimeSeries

    def update(self, time: int, value, new_item_started: bool) -> object: ...

class IndicatorOHLC(Indicator):
    series: OHLCV

def time_as_nsec(time: Any) -> np.datetime64: ...

class RollingSum:
    def __init__(self, period: int) -> None: ...
    def update(self, value: float, new_item_started: bool) -> float: ...
