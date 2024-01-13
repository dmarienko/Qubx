from dataclasses import dataclass
from datetime import datetime

from typing import Any, List, Optional, Union

import numpy as np

from qube.utils import convert_tf_str_td64, convert_seconds_to_str, time_to_str, infer_series_frequency
from qube.utils.math import nans
from qube.utils.time import floor_t64


# define convinient types shortcuts
Time = Union[np.timedelta64, str]
dt64_t = np.datetime64
td64_t = np.timedelta64


def recognize_time(time: Time):
    return np.datetime64(time) if isinstance(time, str) else np.datetime64(time, 's')


@dataclass
class TimeItem:
    time: Union[datetime, dt64_t]

    def value(self) -> float:
        pass


@dataclass
class Bar(TimeItem):
    open: float
    high: float
    low: float
    close: float
    volume: float

    def value(self) -> float:
        return self.close

    def __repr__(self):
        return "[%s]\t{o:%f | h:%f | l:%f | c:%f | v:%f}" % (
            time_to_str(self.time, 's'), self.open, self.high, self.low, self.close, self.volume
        )


@dataclass
class Trade(TimeItem):
    """
    Trade (TAS) representing data class.
    """
    price: float
    size: float
    taker: int = -1

    def value(self) -> float:
        return self.price

    def __repr__(self):
        return "[%s]\t%.5f (%.1f) <%s>" % (
            time_to_str(self.time, 's'), self.price, self.size, 
            'take' if self.taker == 1 else 'make' if self.taker == 0 else '???'
        )


@dataclass
class Quote(TimeItem):
    """
    Quote representing data class.
    """
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def value(self) -> float:
        return self.midprice()

    def midprice(self):
        """
        Midpoint price
        
        :return: midpoint price
        """
        return 0.5 * (self.ask + self.bid)

    def vmpt(self):
        """
        Volume weighted midprice for this quote. It holds midprice if summary size is zero.

        :return: volume weighted midprice
        """
        _e_size = self.ask_size + self.bid_size
        if _e_size == 0.0:
            return self.midprice()

        return (self.bid * self.ask_size + self.ask * self.bid_size) / _e_size

    def __repr__(self):
        return "[%s]\t%.5f (%.1f) | %.5f (%.1f)" % (
            time_to_str(self.time, 's'), self.bid, self.bid_size, self.ask, self.ask_size
        )


@dataclass
class Float(TimeItem):
    """
    Just value representing data class.
    """
    v: float
    info: str = None  # we can attach some additional info if need

    def value(self) -> float:
        return self.v

    def __repr__(self):
        return f"[{time_to_str(self.time, 's')}]\t{self.v}{ f' ({str(self.info)})' if self.info else '' }"


class Indicator:
    _values: 'TimeSeries'
    _timeframe: dt64_t

    def _initialize(self, timeframe: dt64_t):
        self._timeframe = timeframe
        self._init_internal_data()
    
    def _init_internal_data(self):
        pass

    def update(self, input: TimeItem, last_item: bool) -> TimeItem:
        iv = self.calculate(input, last_item)
        if last_item:
            if len(self._values) > 0:
                self._values._update_last_item(iv)
            else:
                self._values._add_new_item(iv)
        else:
            self._values._add_new_item(iv)
        return iv

    def calculate(self, input: TimeItem, last_item: bool) -> TimeItem:
        pass


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TimeSeries experimental
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class TimeSeries:
    name: str
    _items: List[TimeItem]
    _indicators: List[Indicator]

    def __init__(self, 
                name: str,
                timeframe: Optional[Union[int, str, np.timedelta64]] = None,
                max_series_length=np.inf  # TODO: - pruning or ?
                ) -> None:
        self._timeframe = 0
        self.name = name
        self.max_series_length = max_series_length

        # here we set this flag to true when new record is just formed
        self._is_new_item = False
        
        if isinstance(timeframe, str):
            self._timeframe = convert_tf_str_td64(timeframe) 

        elif isinstance(timeframe, (int, float)) and timeframe > 0:
            self._timeframe = np.timedelta64(int(timeframe), 's')

        elif isinstance(timeframe, np.timedelta64):
            self._timeframe = timeframe

        else:
            raise ValueError('Unknown timeframe type !')

        # - initialize container
        self._items = []
        self._indicators = []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            return [self._item_at(self._get_index(i)) for i in range(*idx.indices(len(self._items)))]
        return self._item_at(self._get_index(idx))

    def _get_index(self, idx: int) -> int:
        n_len = len(self)
        if n_len == 0 or (idx > 0 and idx > (n_len - 1)) or (idx < 0 and abs(idx) > n_len):
            raise IndexError(f"Can't find record at index {idx}")
        return (n_len - idx - 1) if idx >= 0 else abs(1 + idx)

    def _item_at(self, idx: int) -> Union[Bar, None]:
        # TODO: may be just to drop this method ?
        return self._items[idx]

    def update(self, item: TimeItem) -> bool:
        if not self._items:
            self._add_new_item(item)

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False
        elif item.time - self._items[-1].time >= self._timeframe:
            # first we update indicators by current last bar
            self._update_indicators(self._item_at(-1), False)

            # then add new bar
            self._add_new_item(item)
        else:
            self._update_last_item(item)

        # update indicators by new data
        self._update_indicators(self._item_at(-1), True)

        return self._is_new_item

    def _add_new_item(self, item: TimeItem):
        item_start_time = floor_t64(item.time, self._timeframe)
        if len(self._items) >= self.max_series_length:
            self._items.pop(0)

        self._items.append(self._convert(item_start_time, item))
        self._is_new_item = True

    def _update_last_item(self, item: TimeItem):
        item_start_time = floor_t64(item.time, self._timeframe)
        self._items[-1] = self._convert(item_start_time, item)
        self._is_new_item = False

    def _convert(self, time: dt64_t, item: TimeItem) -> TimeItem:
        raise ValueError("NOT IMPLEMENTED")

    def _update_indicators(self, item: TimeItem, is_last_item: bool):
        for i in self._indicators:
            i.update(item, is_last_item)

    def attach(self, indicator: Indicator) -> 'TimeSeries':
        if indicator is not None and isinstance(indicator, Indicator):
            # - initialize indicator
            indicator._initialize(self._timeframe)

            self._indicators.append(indicator)

            # and we already have some data in this series
            if len(self) > 0:
                # - push all items as new excepting last one
                [indicator.update(v, False) for v in self._items[:-1]]

                # - finally push last item
                indicator.update(self._items[-1], True)
        else:
            raise ValueError("Can't attach empty indicator or non-Indicator object")

        return self


    def __str__(self):
        nl = len(self)
        r = f"{self.name}[{str(self._timeframe)}] | {nl} records\n"
        hd, tl = 3, 3 
        if nl < hd + tl:
            hd, tl = nl, 0
        
        for n in range(hd):
            r += "  " + str(self[n]) + "\n"
        
        if tl > 0:
            r += "   .......... \n"
            for n in range(-tl, 0):
                r += "  " + str(self[n]) + "\n"

        return r


class FloatSeries(TimeSeries):
    def __init__(self, 
                name: str,
                timeframe: Optional[Union[int, str, np.timedelta64]] = None,
                max_series_length=np.inf) -> None:
        super().__init__(name, timeframe, max_series_length)

    def _convert(self, time: dt64_t, item: TimeItem) -> TimeItem:
        return Float(time, item.value())


class OHLCSeries(TimeSeries):
    
    def __init__(self, 
                timeframe: Optional[Union[int, str, np.timedelta64]] = None,
                max_series_length=np.inf 
                ) -> None:
        pass


class Sma(Indicator):
    """
    Simple moving average
    """
    def __init__(self, period):
        self.period = period
        self.__s = nans(period)
        self.__i = 0

    def _init_internal_data(self):
        self._values = FloatSeries(f'sma({self.period})', self._timeframe)

    def calculate(self, x: TimeItem, last_item: bool):
        _x = x.value() / self.period
        self.__s[self.__i] = _x

        if not last_item:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0

        return Float(x.time, np.sum(self.__s))
