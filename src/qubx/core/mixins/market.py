import pandas as pd
from typing import List, Any

from qubx import lookup
from qubx.core.interfaces import (
    IMarketDataProvider,
    IBrokerServiceProvider,
    IUniverseManager,
)
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.series import Quote, OHLCV
from qubx.core.basics import Instrument
from qubx.data.readers import DataReader
from qubx.utils import convert_seconds_to_str


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

    def ohlc(
        self,
        instrument: Instrument,
        timeframe: str | None = None,
        length: int | None = None,
    ) -> OHLCV:
        timeframe = timeframe or convert_seconds_to_str(
            int(pd.Timedelta(self.__cache.default_timeframe).total_seconds())
        )
        rc = self.__cache.get_ohlcv(instrument, timeframe)

        # - check if we need to fetch more data
        _need_history_request = False
        if (_l_rc := len(rc)) > 0:
            _last_bar_time = rc[0].time
            _timeframe_ns = pd.Timedelta(timeframe).asm8.item()

            # - check if we need to fetch more data
            if (_last_bar_time + _timeframe_ns <= self.__broker.time().item()) or (
                length and _l_rc < length
            ):
                _need_history_request = True

        else:
            _need_history_request = True

        # - send request for historical data
        if _need_history_request and length is not None:
            bars = self.__broker.get_historical_ohlcs(instrument, timeframe, length)
            rc = self.__cache.update_by_bars(instrument, timeframe, bars)
        return rc

    def quote(self, instrument: Instrument) -> Quote | None:
        return self.__broker.get_quote(instrument)

    def get_data(self, instrument: Instrument, sub_type: str) -> List[Any]:
        return self.__cache.get_data(instrument, sub_type)

    def get_aux_data(self, data_id: str, **parameters) -> pd.DataFrame | None:
        return (
            self.__aux_data_provider.get_aux_data(data_id, **parameters)
            if self.__aux_data_provider
            else None
        )

    def get_instruments(self) -> list[Instrument]:
        return self.__universe_manager.instruments

    def get_instrument(self, symbol: str, exchange: str) -> Instrument | None:
        return lookup.find_symbol(exchange, symbol)
