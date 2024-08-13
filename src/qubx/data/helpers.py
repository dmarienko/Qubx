import pandas as pd

from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from qubx import logger
from qubx.data.readers import (
    AsPandasFrame,
    AsTimestampedRecords,
    MultiQdbConnector,
    DataTransformer,
    AsOhlcvSeries,
)


def load_data(
    qdb: MultiQdbConnector,
    exch: str,
    symbols: list | str,
    start: str,
    stop: str = "now",
    timeframe="1h",
    transform: DataTransformer = AsPandasFrame(),
) -> dict[str, Any]:
    if isinstance(symbols, str):
        symbols = [symbols]
    data = {}
    for s in symbols:
        d = qdb.read(f"{exch}:{s}", start, stop, transform=transform, timeframe=timeframe)
        data[s] = d
    return data


def get_symbols_from_questdb(exchange: str, reader: MultiQdbConnector) -> list[str]:
    exch, market = exchange.lower().split(".")
    market = {"um": "umfutures", "cm": "cmfutures", "f": "Any"}.get(market)
    if market is None:
        raise ValueError(f"Unknown market {market}")

    db_exchange = f"{exch}.{market}"
    names = reader.get_names("candles")
    symbols = []
    for name in names:
        if name.startswith(db_exchange):
            symbols.append(name.split(".")[-2])
    return symbols


def get_candles_df_from_questdb(
    exchange: str,
    symbols: list[str],
    qdb: MultiQdbConnector,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    timeframe: str = "1d",
) -> pd.DataFrame:
    _stop = pd.Timestamp(stop) - pd.Timedelta(minutes=1)

    def __fetch_candles(symbol: str):
        candles = qdb.read(
            f"{exchange}:{symbol}", start=start, stop=_stop, transform=AsPandasFrame(), timeframe=timeframe  # type: ignore
        )
        return symbol, candles if candles is not None else None

    symbol_to_candles = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(__fetch_candles, symbol): symbol for symbol in symbols}

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result[1] is not None:
                    symbol_to_candles[result[0]] = result[1]
            except Exception as e:
                logger.error(f"Error fetching data for symbol {symbol}: {e}")

    return pd.concat(symbol_to_candles, axis=1)


def get_prices_qdb(
    exchange: str,
    symbols: list[str],
    qdb: MultiQdbConnector,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    timeframe: str = "1d",
) -> pd.DataFrame:
    pass
