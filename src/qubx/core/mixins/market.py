import pandas as pd

from qubx import lookup
from qubx.core.interfaces import IMarketDataProvider, IBrokerServiceProvider, IUniverseManager
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.series import Quote, OHLCV
from qubx.core.basics import Instrument
from qubx.data.readers import DataReader


class MarketDataProvider(IMarketDataProvider):
    __cache: CachedMarketDataHolder
    __broker: IBrokerServiceProvider
    __aux_data_provider: DataReader | None

    def __init__(
        self,
        cache: CachedMarketDataHolder,
        broker: IBrokerServiceProvider,
        universe_manager: IUniverseManager,
        aux_data_provider: DataReader | None = None,
    ):
        self.__cache = cache
        self.__broker = broker
        self.__universe_manager = universe_manager
        self.__aux_data_provider = aux_data_provider

    def ohlc(self, instrument: Instrument, timeframe: str) -> OHLCV:
        return self.__cache.get_ohlcv(instrument, timeframe)

    def quote(self, instrument: Instrument) -> Quote | None:
        return self.__broker.get_quote(instrument)

    def get_historical_ohlcs(self, instrument: Instrument, timeframe: str, length: int) -> OHLCV:
        rc = self.ohlc(instrument, timeframe)
        if len(rc) >= length:
            return rc
        # - send request for historical data
        bars = self.__broker.get_historical_ohlcs(instrument, timeframe, length)
        return self.__cache.update_by_bars(instrument, timeframe, bars)

    def get_aux_data(self, data_id: str, **parameters) -> pd.DataFrame | None:
        return self.__aux_data_provider.get_aux_data(data_id, **parameters) if self.__aux_data_provider else None

    def get_instruments(self) -> list[Instrument]:
        return self.__universe_manager.instruments

    def get_instrument(self, symbol: str, exchange: str) -> Instrument | None:
        return lookup.find_symbol(exchange, symbol)
