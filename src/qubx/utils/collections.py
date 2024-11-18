import numpy as np
import pandas as pd

from collections import deque
from typing import Any


class TimeLimitedDeque(deque):
    """
    A deque that removes elements older than a given time limit.
    Assumes that elements are inserted in increasing order of time.
    """

    def __init__(self, time_limit: str, time_key=lambda x: x[0], unit="ns", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_limit = pd.Timedelta(time_limit).to_timedelta64()
        self.unit = unit
        self.time_key = lambda x: self._to_datetime64(time_key(x))

    def append(self, item):
        super().append(item)
        self._remove_old_elements()

    def __getitem__(self, idx) -> list[Any]:
        if isinstance(idx, slice) and (isinstance(idx.start, str) or isinstance(idx.stop, str)):
            start_loc, end_loc = 0, len(self)
            if idx.start is not None:
                start = self._to_datetime64(idx.start)
                while start_loc < len(self) and self.time_key(self[start_loc]) < start:
                    start_loc += 1
            if idx.stop is not None:
                stop = self._to_datetime64(idx.stop)
                while end_loc > 0 and self.time_key(self[end_loc - 1]) > stop:
                    end_loc -= 1
            return list(self)[start_loc:end_loc]
        else:
            return super().__getitem__(idx)

    def appendleft(self, item):
        raise NotImplementedError("appendleft is not supported for TimeLimitedDeque")

    def extendleft(self, items):
        raise NotImplementedError("extendleft is not supported for TimeLimitedDeque")

    def _remove_old_elements(self):
        if not self:
            return
        current_time = self.time_key(self[-1])
        while self and (current_time - self.time_key(self[0])) > self.time_limit:
            self.popleft()

    def _to_datetime64(self, time):
        return np.datetime64(time, self.unit)
