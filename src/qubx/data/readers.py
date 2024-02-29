from typing import List, Union, Optional, Iterable, Any
from os.path import exists
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv

from qubx import logger
from qubx.core.series import TimeSeries, OHLCV, time_as_nsec, Quote, Trade
from qubx.utils.time import infer_series_frequency, handle_start_stop

_DT = lambda x: pd.Timedelta(x).to_numpy().item()
D1, H1 = _DT('1D'), _DT('1H')

DEFAULT_DAILY_SESSION = (_DT('00:00:00.100'), _DT('23:59:59.900'))
STOCK_DAILY_SESSION = (_DT('9:30:00.100'), _DT('15:59:59.900'))
CME_FUTURES_DAILY_SESSION = (_DT('8:30:00.100'), _DT('15:14:59.900'))


def _recognize_t(t: Union[int, str], defaultvalue, timeunit) -> int:
    if isinstance(t, (str, pd.Timestamp)):
        try:
            return np.datetime64(t, timeunit)
        except:
            pass
    return defaultvalue


def _find_column_index_in_list(xs, *args):
    xs = [x.lower() for x in xs]
    for a in args:
        ai = a.lower()
        if ai in xs:
            return xs.index(ai)
    raise IndexError(f"Can't find any from {args} in list: {xs}")


class DataProcessor:
    """
    Common interface for data processor with default aggregating into list implementation
    """
    def __init__(self) -> None:
        self.buffer = {}
        self._column_names = []

    def start_processing(self, column_names: List[str]):
        self._column_names = column_names
        self.buffer = {c: [] for c in column_names}

    def process_data(self, columns_data: list) -> Optional[Iterable]:
        for i, c in enumerate(columns_data):
            self.buffer[self._column_names[i]].append(c)
        return None

    def get_result(self) -> Any:
        return self.buffer


class DataReader:
    """
    Common interface for data reader
    """
    _processor: DataProcessor

    def __init__(self, processor=None) -> None:
        self._processor = DataProcessor() if processor is None else processor

    def read(self, start: Optional[str]=None, stop: Optional[str]=None) -> Any:
        pass

    
class QuotesDataProcessor(DataProcessor):
    """
    Process quotes data and collect them as list
    """
    def start_processing(self, fieldnames: List[str]):
        self.buffer = list()
        self._time_idx = _find_column_index_in_list(fieldnames, 'time', 'timestamp', 'datetime')
        self._bid_idx = _find_column_index_in_list(fieldnames, 'bid')
        self._ask_idx = _find_column_index_in_list(fieldnames, 'ask')
        self._bidvol_idx = _find_column_index_in_list(fieldnames, 'bidvol', 'bid_vol', 'bidsize', 'bid_size')
        self._askvol_idx = _find_column_index_in_list(fieldnames, 'askvol', 'ask_vol', 'asksize', 'ask_size')

    def process_data(self, columns_data: list) -> Optional[Iterable]:
        tms = columns_data[self._time_idx] 
        bids = columns_data[self._bid_idx]
        asks = columns_data[self._ask_idx]
        bidvol = columns_data[self._bidvol_idx]
        askvol = columns_data[self._askvol_idx]
        for i in range(len(tms)):
            self.buffer.append(Quote(tms[i], bids[i], asks[i], bidvol[i], askvol[i]))
        return None


class QuotesFromOHLCVDataProcessor(DataProcessor):
    """
    Process OHLC and generate Quotes (+ Trades) from it
    """
    def __init__(self, trades: bool=False, 
                 default_bid_size=1e9,  # default bid/ask is big
                 default_ask_size=1e9,  # default bid/ask is big
                 daily_session_start_end=DEFAULT_DAILY_SESSION,
                 spread=0.0,
                ) -> None:
        super().__init__()
        self._trades = trades
        self._bid_size = default_bid_size
        self._ask_size = default_ask_size
        self._s2 = spread / 2.0
        self._d_session_start = daily_session_start_end[0]
        self._d_session_end = daily_session_start_end[1]

    def start_processing(self, fieldnames: List[str]):
        self._time_idx = _find_column_index_in_list(fieldnames, 'time', 'timestamp', 'datetime', 'date')
        self._open_idx = _find_column_index_in_list(fieldnames, 'open')
        self._high_idx = _find_column_index_in_list(fieldnames, 'high')
        self._low_idx = _find_column_index_in_list(fieldnames, 'low')
        self._close_idx = _find_column_index_in_list(fieldnames, 'close')
        self._volume_idx = None
        self._timeframe = None

        try:
            self._volume_idx = _find_column_index_in_list(fieldnames, 'volume', 'vol')
        except:
            pass

        self.buffer = []

    def process_data(self, data: list) -> Optional[Iterable]:
        s2 = self._s2
        if self._timeframe is None:
            _freq = infer_series_frequency(data[self._time_idx])
            self._timeframe = _freq.astype('timedelta64[s]')

            # - timestamps when we emit simulated quotes
            dt = _freq.astype('timedelta64[ns]').item()
            if dt < D1:
                self._t_start = dt // 10
                self._t_mid1 = dt // 2 - dt // 10
                self._t_mid2 = dt // 2 + dt // 10
                self._t_end = dt - dt // 10
            else:
                self._t_start = self._d_session_start
                self._t_mid1 = dt // 2 - H1
                self._t_mid2 = dt // 2 + H1
                self._t_end = self._d_session_end

        # - input data
        times = data[self._time_idx]
        opens = data[self._open_idx]
        highs = data[self._high_idx]
        lows = data[self._low_idx]
        closes = data[self._close_idx]
        volumes = data[self._volume_idx] if self._volume_idx else None
        if volumes is None and self._trades:
            logger.warning("Input OHLC data doesn't contain volume information so trades can't be emulated !")
            self._trades = False

        for i in range(len(times)):
            ti, o, h, l, c = times[i].astype('datetime64[ns]'), opens[i], highs[i], lows[i], closes[i]

            if self._trades:
                rv = volumes[i] / (h - l) 

            # - opening quote
            self.buffer.append(Quote(ti + self._t_start, o - s2, o + s2, self._bid_size, self._ask_size))

            if c >= o:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o - s2, rv * (o - l))) # sell 1
                self.buffer.append(Quote(ti + self._t_mid1, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, l + s2, rv * (c - o)))  # buy 1
                self.buffer.append(Quote(ti + self._t_mid2, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, h - s2, rv * (h - c)))  # sell 2
            else:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o + s2, rv * (h - o))) # buy 1
                self.buffer.append(Quote(ti + self._t_mid1, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, h - s2, rv * (o - c))) # sell 1
                self.buffer.append(Quote(ti + self._t_mid2, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, l + s2, rv * (c - l))) # buy 2

            # - closing quote
            self.buffer.append(Quote(ti + self._t_end, c - s2, c + s2, self._bid_size, self._ask_size))

        return None

    def get_result(self) -> Any:
        return self.buffer


class OhlcvDataProcessor(DataProcessor):
    """
    Process data and convert it to Qube OHLCV timeseries 
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def start_processing(self, fieldnames: List[str]):
        self._time_idx = _find_column_index_in_list(fieldnames, 'time', 'timestamp', 'datetime', 'date')
        self._open_idx = _find_column_index_in_list(fieldnames, 'open')
        self._high_idx = _find_column_index_in_list(fieldnames, 'high')
        self._low_idx = _find_column_index_in_list(fieldnames, 'low')
        self._close_idx = _find_column_index_in_list(fieldnames, 'close')
        self._volume_idx = None
        self._timeframe = None

        try:
            self._volume_idx = _find_column_index_in_list(fieldnames, 'volume', 'vol')
        except:
            pass

        self.ohlc = None

    def process_data(self, data: list) -> Optional[Iterable]:
        if self._timeframe is None:
            self._timeframe = infer_series_frequency(data[self._time_idx]).astype('timedelta64[s]')

            # - create instance after first data received
            self.ohlc = OHLCV(self._name, self._timeframe)

        self.ohlc.append_data(
            data[self._time_idx],
            data[self._open_idx], data[self._high_idx], data[self._low_idx], data[self._close_idx], 
            data[self._volume_idx] if self._volume_idx else []
        )
        return None

    def get_result(self) -> Any:
        return self.ohlc


class OhlcvPandasDataProcessor(DataProcessor):
    """
    Process data and convert it to pandas OHLCV dataframes 
    """
    def __init__(self) -> None:
        super().__init__()

    def start_processing(self, fieldnames: List[str]):
        self._time_idx = _find_column_index_in_list(fieldnames, 'time', 'timestamp', 'datetime', 'date')
        self._open_idx = _find_column_index_in_list(fieldnames, 'open')
        self._high_idx = _find_column_index_in_list(fieldnames, 'high')
        self._low_idx = _find_column_index_in_list(fieldnames, 'low')
        self._close_idx = _find_column_index_in_list(fieldnames, 'close')
        self._volume_idx = None
        self._timeframe = None

        try:
            self._volume_idx = _find_column_index_in_list(fieldnames, 'volume', 'vol')
        except:
            pass

        # self.ohlc = pd.DataFrame()

        self._time = np.array([], dtype=np.datetime64)
        self._open = np.array([])
        self._high = np.array([])
        self._low = np.array([])
        self._close = np.array([])
        self._volume = np.array([])

    def process_data(self, data: list) -> Optional[Iterable]:
        # p = pd.DataFrame({
        #     'open': data[self._open_idx], 
        #     'high': data[self._high_idx], 
        #     'low': data[self._low_idx], 
        #     'close': data[self._close_idx], 
        #     'volume': data[self._volume_idx] if self._volume_idx else []},
        #     index = data[self._time_idx]
        # )
        # self.ohlc = pd.concat((self.ohlc, p), axis=0, sort=True, copy=True)
        self._time = np.concatenate((self._time, data[self._time_idx]))
        self._open = np.concatenate((self._open, data[self._open_idx]))
        self._high = np.concatenate((self._high, data[self._high_idx]))
        self._low = np.concatenate((self._low, data[self._low_idx]))
        self._close = np.concatenate((self._close, data[self._close_idx]))
        if self._volume_idx:
            self._volume = np.concatenate((self._volume, data[self._volume_idx]))

        return None

    def get_result(self) -> Any:
        # self.ohlc.index.name = 'time'
        # return self.ohlc

        return pd.DataFrame(
            {
                'open': self._open, 
                'high': self._high, 
                'low': self._low, 
                'close': self._close, 
                'volume': self._volume if self._volume_idx else []
            },
            index = self._time
        ).sort_index()
 

class CsvDataReader(DataReader):
    """
    CSV data file reader
    """

    def __init__(self, path: str, processor: DataProcessor=None, timestamp_parsers=None) -> None:
        if not exists(path):
            raise ValueError(f"CSV file not found at {path}")
        super().__init__(processor)
        self.time_parsers = timestamp_parsers
        self.path = path

    def __find_time_idx(self, arr: pa.ChunkedArray, v) -> int:
        ix = arr.index(v).as_py()
        if ix < 0:
            for c in arr.iterchunks():
                a = c.to_numpy()
                ix = np.searchsorted(a, v, side='right')
                if ix > 0 and ix < len(c):
                    ix = arr.index(a[ix]).as_py() - 1
                    break
        return ix

    def read(self, start: Optional[str]=None, stop: Optional[str]=None) -> Any:
        convert_options = None
        if self.time_parsers:
            convert_options=csv.ConvertOptions(timestamp_parsers=self.time_parsers)

        table = csv.read_csv(
            self.path, 
            parse_options=csv.ParseOptions(ignore_empty_lines=True),
            convert_options=convert_options
        )
        fieldnames =  table.column_names  

        # - try to find range to load  
        start_idx, stop_idx = 0, table.num_rows
        try:
            _time_field_idx = _find_column_index_in_list(fieldnames, 'time', 'timestamp', 'datetime', 'date')
            _time_type = table.field(_time_field_idx).type
            _time_unit = _time_type.unit if hasattr(_time_type, 'unit') else 's'
            _time_data = table[_time_field_idx]

            # - check if need convert time to primitive types (i.e. Date32 -> timestamp[x])
            _time_cast_function = lambda xs: xs
            if _time_type != pa.timestamp(_time_unit):
                _time_cast_function = lambda xs: xs.cast(pa.timestamp(_time_unit)) 
                _time_data = _time_cast_function(_time_data)

            # - preprocessing start and stop
            t_0, t_1 = handle_start_stop(start, stop, convert=lambda x: _recognize_t(x, None, _time_unit))

            # - check requested range
            if t_0:
                start_idx = self.__find_time_idx(_time_data, t_0)
                if start_idx >= table.num_rows:
                    # no data for requested start date
                    return None

            if t_1:
                stop_idx = self.__find_time_idx(_time_data, t_1)
                if stop_idx < 0 or stop_idx < start_idx:
                    stop_idx = table.num_rows

        except Exception as exc:
            logger.warning(exc)
            logger.info('loading whole file')

        length = (stop_idx - start_idx + 1)
        self._processor.start_processing(fieldnames)
        selected_table = table.slice(start_idx, length)
        n_chunks = selected_table[table.column_names[0]].num_chunks
        for n in range(n_chunks):
            data = [
                # - in some cases we need to convert time index to primitive type
                _time_cast_function(selected_table[k].chunk(n)).to_numpy() if k == _time_field_idx else selected_table[k].chunk(n).to_numpy()
                for k in range(selected_table.num_columns)]
            self._processor.process_data(data)
        return self._processor.get_result()
            