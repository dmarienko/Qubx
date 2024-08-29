from typing import Any
from concurrent.futures import ThreadPoolExecutor
from qubx.data.readers import (
    AsPandasFrame,
    MultiQdbConnector,
    DataTransformer,
)


def load_data(
    qdb: MultiQdbConnector,
    exch: str,
    symbols: list | str,
    start: str,
    stop: str = "now",
    timeframe="1h",
    transform: DataTransformer = AsPandasFrame(),
    max_workers: int = 16,
) -> dict[str, Any]:
    if isinstance(symbols, str):
        symbols = [symbols]
    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(symbols)))
    data = {}
    res = {
        s: executor.submit(qdb.read, f"{exch}:{s}", start, stop, transform=transform, timeframe=timeframe)
        for s in symbols
    }
    data = {s: r.result() for s, r in res.items()}
    return data
