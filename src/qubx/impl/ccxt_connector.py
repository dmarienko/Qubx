from typing import List
import ccxt.pro as cxp
from ccxt.base.decimal_to_precision import ROUND_UP
from ccxt.base.exchange import Exchange

from threading import Thread, Event, Lock
from queue import Queue
# from multiprocessing import Queue #as Queue


from qubx.core.basics import Instrument, dt_64
from qubx.core.strategy import IDataProvider


_aliases = {
    'binance.um': 'binanceusdm',
    'binance.cm': 'binancecoinm',
    'kraken.f': 'kreakenfutures'
}


class CommChannel:
    control: Event
    queue: Queue
    name: str
    lock: Lock

    def __init__(self, name: str):
        self.name = name
        self.control = Event()
        self.queue = Queue()
        self.lock = Lock()

    def stop(self):
        if self.control.is_set():
            self.control.clear()

    def start(self):
        self.control.set()


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
