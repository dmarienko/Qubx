from typing import Any, List

import pandas as pd

from qubx import lookup
from qubx.core.basics import Instrument
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.interfaces import (
    IDataProvider,
    IMarketManager,
    IUniverseManager,
)
from qubx.core.series import OHLCV, Quote
from qubx.data.readers import DataReader
from qubx.utils import convert_seconds_to_str


class MarketDataProvider(IMarketManager):
    _cache: CachedMarketDataHolder
    _broker: IDataProvider
    _universe_manager: IUniverseManager
    _aux_data_provider: DataReader | None

    def __init__(
        self,
        cache: CachedMarketDataHolder,
        broker: IDataProvider,
        universe_manager: IUniverseManager,
        aux_data_provider: DataReader | None = None,
    ):
        self._cache = cache
        self._broker = broker
        self._universe_manager = universe_manager
        self._aux_data_provider = aux_data_provider

    def ohlc(
        self,
        instrument: Instrument,
        timeframe: str | None = None,
        length: int | None = None,
    ) -> OHLCV:
        timeframe = timeframe or convert_seconds_to_str(
            int(pd.Timedelta(self._cache.default_timeframe).total_seconds())
        )
        rc = self._cache.get_ohlcv(instrument, timeframe)

        # - check if we need to fetch more data
        _need_history_request = False
        if (_l_rc := len(rc)) > 0:
            _last_bar_time = rc[0].time
            _timeframe_ns = pd.Timedelta(timeframe).asm8.item()

            # - check if we need to fetch more data
            if (_last_bar_time + _timeframe_ns <= self._broker.time_provider.time().item()) or (
                length and _l_rc < length
            ):
                _need_history_request = True

        else:
            _need_history_request = True

        # - send request for historical data
        if _need_history_request and length is not None:
            bars = self._broker.get_ohlc(instrument, timeframe, length)
            rc = self._cache.update_by_bars(instrument, timeframe, bars)
        return rc

    def quote(self, instrument: Instrument) -> Quote | None:
        return self._broker.get_quote(instrument)

    def get_data(self, instrument: Instrument, sub_type: str) -> List[Any]:
        return self._cache.get_data(instrument, sub_type)

    def get_aux_data(self, data_id: str, **parameters) -> pd.DataFrame | None:
        return self._aux_data_provider.get_aux_data(data_id, **parameters) if self._aux_data_provider else None

    def get_instruments(self) -> list[Instrument]:
        return self._universe_manager.instruments

    def get_instrument(self, symbol: str, exchange: str) -> Instrument | None:
        return lookup.find_symbol(exchange, symbol)
