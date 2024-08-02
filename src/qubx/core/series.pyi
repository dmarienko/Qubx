class Bar:
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

class TimeSeries:
    def __init__(self, name, timeframe, max_series_length, process_every_update=True) -> None: ...
    def __getitem__(self, idx): ...

class OHLCV(TimeSeries):
    def __init__(self, name, timeframe, max_series_length) -> None: ...

class Indicator(TimeSeries): ...
