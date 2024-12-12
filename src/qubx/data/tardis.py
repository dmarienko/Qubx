from dataclasses import field
from os.path import exists, expanduser
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pyarrow import csv

from qubx.utils.time import handle_start_stop, infer_series_frequency

from .readers import CsvStorageDataReader, DataReader, DataTransformer, _recognize_t

TARDIS_EXCHANGE_MAPPERS = {
    "bitfinex.f": "bitfinex-derivatives",
    "binance.um": "binance-futures",
}


class TardisCsvDataReader(DataReader):
    def __init__(self, path: str | Path) -> None:
        _path = expanduser(path)
        if not exists(_path):
            raise ValueError(f"Folder is not found at {path}")
        self.path = Path(_path)

    def get_names(self, exchange: str | None = None, data_type: str | None = None) -> list[str]:
        symbols = []
        exchanges = [exchange] if exchange else self.get_exchanges()
        for exchange in exchanges:
            exchange_path = Path(self.path) / exchange
            if not exists(exchange_path):
                raise ValueError(f"Exchange is not found at {exchange_path}")
            data_types = [data_type] if data_type else self.get_data_types(exchange)
            for data_type in data_types:
                data_type_path = exchange_path / data_type
                if not exists(data_type_path):
                    return []
                symbols += self._get_symbols(data_type_path)
        return symbols

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timeframe=None,
        data_type="trades",
    ) -> Iterable | Any:
        if chunksize > 0:
            raise NotImplementedError("Chunksize is not supported for TardisCsvDataReader")
        exchange, symbol = data_id.split(":")
        _exchange = exchange.lower()
        _exchange = TARDIS_EXCHANGE_MAPPERS.get(_exchange, _exchange)
        t_0, t_1 = handle_start_stop(start, stop, lambda x: pd.Timestamp(x).date().isoformat())
        _path = self.path / _exchange / data_type
        if not _path.exists():
            raise ValueError(f"Data type is not found at {_path}")
        _files = sorted(_path.glob(f"*_{symbol}.csv.gz"))
        if not _files:
            return None
        _dates = [file.stem.split("_")[0] for file in _files]
        if t_0 is None:
            t_0 = _dates[0]
        if t_1 is None:
            t_1 = _dates[-1]
        _filt_files = [file for file in _files if t_0 <= file.stem.split("_")[0] <= t_1]

        tables = []
        fieldnames = None
        for f_path in _filt_files:
            table = csv.read_csv(
                f_path,
                parse_options=csv.ParseOptions(ignore_empty_lines=True),
            )
            if not fieldnames:
                fieldnames = table.column_names
            tables.append(table.to_pandas())

        transform.start_transform(data_id, fieldnames, start=start, stop=stop)
        raw_data = pd.concat(tables).to_numpy()
        transform.process_data(raw_data)

        return transform.collect()

    def get_exchanges(self) -> list[str]:
        return [exchange.name for exchange in self.path.iterdir() if exchange.is_dir()]

    def get_data_types(self, exchange: str) -> list[str]:
        exchange_path = Path(self.path) / exchange
        return [data_type.name for data_type in exchange_path.iterdir() if data_type.is_dir()]

    def _get_symbols(self, data_type_path: Path) -> list[str]:
        symbols = set()
        for file in data_type_path.glob("*.gz"):
            parts = file.stem.replace(".csv", "").split("_")
            if len(parts) == 2:
                symbols.add(parts[1])
        return list(symbols)
