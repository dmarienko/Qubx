import numpy as np
cimport numpy as np
from cpython.datetime cimport datetime
from qube.utils import convert_tf_str_td64#, convert_seconds_to_str, time_to_str#, floor_t64

cpdef recognize_time(time):
    return np.datetime64(time, 'ns') if isinstance(time, str) else np.datetime64(time, 'ms')

NS = 1_000_000_000 

cdef extern from "math.h":
    float INFINITY


cpdef str time_to_str(long long t, str units):
    return str(np.datetime64(t, units)) #.isoformat()


cdef str time_delta_to_str(long long d):
    """
    Convert timedelta object to pretty print format

    :param d:
    :return:
    """
    days, seconds = divmod(d, 86400*NS)
    hours, seconds = divmod(seconds, 3600*NS)
    minutes, seconds = divmod(seconds, 60*NS)
    seconds, rem  = divmod(seconds, NS)
    r = ''
    if days > 0:
        r += '%dD' % days
    if hours > 0:
        r += '%dH' % hours
    if minutes > 0:
        r += '%dMin' % minutes
    if seconds > 0:
        r += '%dS' % seconds
    if rem > 0:
        r += '%dmS' % (rem // 1000000)
    return r


cdef nans(dims):
    """
    nans((M,N,P,...)) is an M-by-N-by-P-by-... array of NaNs.
    
    :param dims: dimensions tuple 
    :return: nans matrix 
    """
    return np.nan * np.ones(dims)

UNIX_T0 = np.datetime64('1970-01-01T00:00:00').astype(np.int64)

cdef floor_t64(time, dt):
    """
    Floor timestamp by dt
    """
    # if isinstance(dt, int):
    #     dt = np.timedelta64(dt, 's')

    # if isinstance(dt, str):
    #     dt = convert_tf_str_td64(dt)

    # if isinstance(time, datetime):
    #     time = np.datetime64(time)

    # return time - (time - UNIX_T0) % dt
    # return time - (time  % dt)
    return time - time % dt


cdef class TimeItem:
    cdef public long long time

    def __init__(self, time) -> None:
        self.time = time.astype(np.int64) if isinstance(time, np.datetime64) else time


cdef class Float(TimeItem):
    cdef public float v
    cdef public str info

    def __init__(self, time, v, info=None) -> None:
        super().__init__(time)
        self.v = v
        self.info = info

    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}]\t{self.v}{ f' ({str(self.info)})' if self.info else '' }"


cdef class Indicator:
    cdef public values
    cdef long long timeframe

    def _initialize(self, timeframe):
        self.timeframe = timeframe
        self._init_internal_data()
    
    def _init_internal_data(self):
        pass

    def update(self, input: TimeItem, last_item: bool) -> TimeItem:
        iv = self.calculate(input, last_item)
        if last_item:
            if len(self.values) > 0:
                self.values._update_last_item(iv)
            else:
                self.values._add_new_item(iv)
        else:
            self.values._add_new_item(iv)
        return iv

    def calculate(self, input: TimeItem, last_item: bool) -> TimeItem:
        pass



cdef class TimeSeries:
    cdef public str name
    cdef list items
    cdef list indicators
    cdef public long long timeframe
    cdef unsigned short _is_new_item
    cdef public float max_series_length

    def __init__(self, str name, timeframe, max_series_length=INFINITY) -> None:
        self.items = []
        self.indicators = []
        self.name = name
        self.max_series_length = max_series_length

        if isinstance(timeframe, str):
            self.timeframe = np.int64(convert_tf_str_td64(timeframe).item().total_seconds() * NS)

        elif isinstance(timeframe, (int, float)) and timeframe > 0:
            self.timeframe = timeframe

        elif isinstance(timeframe, np.timedelta64):
            self.timeframe = np.int64(timeframe.item().total_seconds() * NS) 

        else:
            raise ValueError('Unknown timeframe type !')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.items[self._get_index(i)] for i in range(*idx.indices(len(self.items)))]
        return self.items[self._get_index(idx)]

    def _get_index(self, idx: int) -> int:
        n_len = len(self)
        if n_len == 0 or (idx > 0 and idx > (n_len - 1)) or (idx < 0 and abs(idx) > n_len):
            raise IndexError(f"Can't find record at index {idx}")
        return (n_len - idx - 1) if idx >= 0 else abs(1 + idx)

    def update(self, item: TimeItem) -> short:
        if not self.items:
            self._add_new_item(item)

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False
        elif item.time - self.items[-1].time >= self.timeframe:
            # first we update indicators by current last bar
            self._update_indicators(self.items[-1], False)

            # then add new bar
            self._add_new_item(item)
        else:
            self._update_last_item(item)

        # update indicators by new data
        self._update_indicators(self.items[-1], True)

        return self._is_new_item

    def _add_new_item(self, item: TimeItem):
        item_start_time = floor_t64(item.time, self.timeframe)
        if len(self.items) >= self.max_series_length:
            self.items.pop(0)

        self.items.append(self._convert(item_start_time, item))
        self._is_new_item = True

    def _update_last_item(self, item: TimeItem):
        # item_start_time = floor_t64(item.time, self.timeframe)
        item_start_time = self.items[-1].time
        self.items[-1] = self._convert(item_start_time, item)
        self._is_new_item = False

    def _convert(self, time, item: TimeItem) -> TimeItem:
        raise ValueError("NOT IMPLEMENTED")

    def attach(self, indicator: Indicator) -> 'TimeSeries':
        if indicator is not None and isinstance(indicator, Indicator):
            # - initialize indicator
            indicator._initialize(self.timeframe)

            self.indicators.append(indicator)

            # and we already have some data in this series
            if len(self) > 0:
                # - push all items as new excepting last one
                [indicator.update(v, False) for v in self.items[:-1]]

                # - finally push last item
                indicator.update(self.items[-1], True)
        else:
            raise ValueError("Can't attach empty indicator or non-Indicator object")

        return self

    cdef _update_indicators(self, item: TimeItem, is_last_item: bool):
        for i in self.indicators:
            i.update(item, is_last_item)

    def __str__(self):
        nl = len(self)
        r = f"{self.name}[{time_delta_to_str(self.timeframe)}] | {nl} records\n"
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


cdef class FloatSeries(TimeSeries):
    def __init__(self, 
                name: str,
                timeframe,
                max_series_length=INFINITY) -> None:
        super().__init__(name, timeframe, max_series_length)

    def _convert(self, time, item: TimeItem) -> TimeItem:
        return Float(time, item.v)


cdef class Sma(Indicator):
    cdef int period
    cdef np.ndarray __s
    cdef int __i

    """
    Simple moving average
    """
    def __init__(self, period):
        self.period = period
        self.__s = nans(period)
        self.__i = 0

    def _init_internal_data(self):
        self.values = FloatSeries(f'sma({self.period})', self.timeframe)

    cpdef calculate(self, x: TimeItem, last_item: int):
        cdef float _x = x.v / self.period
        self.__s[self.__i] = _x

        if not last_item:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0

        return Float(x.time, np.sum(self.__s))


cdef class Ema(Indicator):
    """
    Exponential moving average
    """
    cdef int period
    cdef np.ndarray __s
    cdef int __i
    cdef float alpha
    cdef float alpha_1
    cdef unsigned short init_mean 

    def __init__(self, period, init_mean=True):
        self.period = period

        # when it's required to initialize this ema by mean on first period
        self.init_mean = init_mean
        if init_mean:
            self.__s = nans(period)
            self.__i = 0

        self.alpha = 2.0 / (1.0 + period)
        self.alpha_1 = (1 - self.alpha)

    def _init_internal_data(self):
        self.values = FloatSeries(f'ema({self.period})', self.timeframe)

    def calculate(self, x: TimeItem, last_item: int):
        # when we need to initialize ema by average from initial period
        if self.init_mean and self.__i < self.period:
            # we skip any nans on initial period (tema, dema, ...)
            if np.isnan(x.v):
                return Float(x.time, np.nan)

            self.__s[self.__i] = x.v / self.period
            if last_item > 0:
                self.__i += 1

                if self.__i >= self.period:
                    self.init_mean = False
                    return Float(x.time, np.sum(self.__s))

            return Float(x.time, np.nan)

        if len(self.values) == 0:
            return Float(x.time, x.v)

        return Float(x.time, self.alpha * x.v + self.alpha_1 * self.values[0].v)
