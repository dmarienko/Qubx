from dataclasses import dataclass
from datetime import datetime

from typing import Any, List, Optional, Union

import numpy as np
from qube.core.series import Bar, TimeItem

from qube.utils import convert_tf_str_td64, convert_seconds_to_str, time_to_str, infer_series_frequency
from qube.utils.time import floor_t64


# define convinient types shortcuts
Time = Union[np.timedelta64, str]
dt64_t = np.datetime64
td64_t = np.timedelta64


def recognize_time(time: Time):
    return np.datetime64(time) if isinstance(time, str) else np.datetime64(time, 's')


class Indicator:
    pass


class TemporalData:
    # temporal series data storage
    _times: List[Time]
    _timeframe: td64_t

    def __init__(
            self, 
            timeframe: Optional[Union[int, str, np.timedelta64]] = None,
            time: Optional[List[Time]] = None, 
            max_series_length=np.inf  # TODO: - pruning or ?
    ):
        self._timeframe = 0
        self.max_series_length = max_series_length

        # here we set this flag to true when new record is just formed
        self.is_new_item = False

        # todo: consistency check
        self._times = [recognize_time(t) for t in time] if time else []
        
        if isinstance(timeframe, str):
            self._timeframe = convert_tf_str_td64(timeframe) 

        elif isinstance(timeframe, (int, float)) and timeframe > 0:
            self._timeframe = np.timedelta64(int(timeframe), 's')

        else:
            self._try_detect_timeframe()

        # indicators
        self.indicators: List[Indicator] = []

    def _try_detect_timeframe(self):
        if self._times:
            self._timeframe = infer_series_frequency(self._times)
        else:
            raise ValueError("Can't detect timeframe on empty data")
    
    def __len__(self) -> int:
        return len(self._times)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            return [self._item_at(self._get_index(i)) for i in range(*idx.indices(len(self._times)))]
        return self._item_at(self._get_index(idx))
    
    def _get_index(self, idx: int) -> int:
        n_len = len(self)
        if n_len == 0 or (idx > 0 and idx > (n_len - 1)) or (idx < 0 and abs(idx) > n_len):
            raise IndexError(f"Can't find record at index {idx}")
        return (n_len - idx - 1) if idx >= 0 else abs(1 + idx)

    def update(self, item: TimeItem) -> bool:
        if not self._times:
            self._add_new_item(item)

            # Here we disable first notification because first item may be incomplete
            self.is_new_item = False
        elif item.time - self._times[-1] >= self._timeframe:
            # first we update indicators by current last bar
            self._update_all_indicators(self._item_at(-1), True)

            # then add new bar
            self._add_new_item(item)
        else:
            self._update_last_item(item)

        # update indicators by new data
        self._update_all_indicators(self._item_at(-1), False)

        return self.is_new_item

    def _add_new_item(self, item: TimeItem):
        pass

    def _update_last_item(self, item: TimeItem):
        pass

    def _update_all_indicators(self, item: TimeItem, is_new_item: bool):
        pass

    def __str__(self):
        nl = len(self)
        r = f"Series[{str(self._timeframe)}] | {nl} records\n"
        fn, ln = 3, 3 
        if nl < fn + ln:
            fn, ln = nl, 0
        
        for n in range(fn):
            r += "  " + str(self[nl - n -1]) + "\n"
        
        if ln > 0:
            r += "   .......... \n"
            for n in range(-ln, 0):
                r += "  " + str(self[n]) + "\n"

        return r


class BarSeries(TemporalData):
    _opens: List[float]
    _highs: List[float]
    _lows: List[float]
    _closes: List[float]
    _volumes: List[float]

    def __init__(
            self, 
            timeframe: Optional[Union[int, str, np.timedelta64]] = None,
            time: Optional[List[Time]] = None, 
            data: Optional[List[Any]] = None,
            max_series_length=np.inf  # TODO: - pruning or ?
    ):
        super().__init__(timeframe, time, max_series_length)
        self._opens = []
        self._highs = []
        self._lows = []
        self._closes = []
        self._volumes = []

        if data:
            if len(self._times) != len(data):
                raise ValueError(f"data and time index must have equal sizes !")
            for r in data:
                self._opens.append(r[0])
                self._highs.append(r[1])
                self._lows.append(r[2])
                self._closes.append(r[3])
                self._volumes.append(r[4] if len(r) > 4 else 0.0)

    def _item_at(self, idx: int) -> Union[Bar, None]:
        """
        Get bar at specified index. Indexing order is reversed for convenient accessing to previous bars :
         - _item_at(0) returns most recent (current) record
         - _item_at(1) returns previous record
         - _item_at(-1) returns first record in the series

        :param idx: bar index
        :return: bar at index specified or none if bar not exists
        """
        return Bar(time=self._times[idx], 
                   open=self._opens[idx], high=self._highs[idx],
                   low=self._lows[idx], close=self._closes[idx], volume=self._volumes[idx])

    def _add_new_item(self, item: TimeItem):
        pass

    def _add_new_item(self, item: TimeItem):
        # get bar's start time for this time
        bar_start_time = floor_t64(item.time, self._timeframe)

        if len(self._times) >= self.max_series_length:
            self._times.pop(0)
            self._opens.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)
            self._volumes.pop(0)

        self._times.append(bar_start_time)

    def _update_last_item(self, item: TimeItem):
        pass
