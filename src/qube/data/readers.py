from typing import List, Union, Optional, Iterable, Any
from os.path import exists
import numpy as np
import pyarrow as pa
from pyarrow import csv

from qube.core.series import TimeSeries, OHLCV, time_as_nsec, Quote, Trade
from qube.utils.time import infer_series_frequency


def _recognize_t(t: Union[int, str], defaultvalue, timeunit) -> int:
    if isinstance(t, str):
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
    Common interface for data processor and default aggregator implementation
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
            self.buffer.append(
                Quote(tms[i], bids[i], asks[i], bidvol[i], askvol[i])
            )
        return None


class OhlcvToQuotesDataProcessor(DataProcessor):
    """
    Process OHLC and restore Quotes
    """
    pass


class OhlcvDataProcessor(DataProcessor):
    """
    Process data and convert it to TimeSeries
    """
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
            self._timeframe = infer_series_frequency(data[self._time_idx])

            # TODO: ---- name ------ !
            self.ohlc = OHLCV('Test1', self._timeframe)

        self.ohlc.append_data(
            data[self._time_idx],
            data[self._open_idx], data[self._high_idx], 
            data[self._low_idx], data[self._close_idx], 
            data[self._volume_idx] 
        )
        return None

    def get_result(self) -> Any:
        return self.ohlc


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
            time_unit = table.field(_time_field_idx).type.unit
            time_data = table[_time_field_idx]

            t_0 = _recognize_t(start, None, time_unit)
            t_1 = _recognize_t(stop, None, time_unit)

            # - what range is requested
            if t_0:
                start_idx = self.__find_time_idx(time_data, t_0)

            if t_1:
                stop_idx = self.__find_time_idx(time_data, t_1)
                if stop_idx < 0:
                    stop_idx = table.num_rows
        except:
            pass

        length = (stop_idx - start_idx + 1)
        self._processor.start_processing(fieldnames)
        selected_table = table.slice(start_idx, length)
        n_chunks = selected_table[table.column_names[0]].num_chunks
        for n in range(n_chunks):
            data = [selected_table[k].chunk(n).to_numpy() for k in range(selected_table.num_columns)]
            self._processor.process_data(data)
        return self._processor.get_result()
            