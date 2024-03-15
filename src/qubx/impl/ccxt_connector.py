from typing import List
import ccxt.pro as cxp
from ccxt.base.decimal_to_precision import ROUND_UP
from ccxt.base.exchange import Exchange


from qubx.core.basics import Instrument, dt_64
from qubx.core.strategy import IDataProvider

_aliases = {
    'binance.um': 'binanceusdm',
    'binance.cm': 'binancecoinm',
    'kraken.f': 'kreakenfutures'
}

class CCXT_connector(IDataProvider):
    exch: Exchange

    def __init__(self, exchange: str):
        exch = _aliases.get(exchange.lower(), exchange.lower())
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange} is not supported by CCXT!")
        self.exch = getattr(cxp, exch)()

    def request_historical_data(self, 
                                instruments: List[Instrument], 
                                timeframe: str, 
                                start: str | int | dt_64, 
                                stop: str | int | dt_64):
        data = self.exch.fetch_ohlcv('PF_ETHUSD', '1m', since=since, limit=limit)
