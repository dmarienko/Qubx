from typing import Any, Dict, List, Optional
import ccxt.pro as cxp
from ccxt.base.decimal_to_precision import ROUND_UP
from ccxt.base.exchange import Exchange
from ccxt.async_support.base.ws.client import Client
from ccxt.async_support.base.ws.cache import ArrayCacheByTimestamp
import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, dt_64
from qubx.core.strategy import IDataProvider, CtrlChannel
from qubx.core.series import TimeSeries, Bar


_aliases = {
    'binance': 'binanceqv',
    'binance.um': 'binanceusdm',
    'binance.cm': 'binancecoinm',
    'kraken.f': 'kreakenfutures'
}


class BinanceQV(cxp.binance):
    """
    Extended binance exchange to provide quote asset volumes support
    """
    def parse_ohlcv(self, ohlcv, market=None):
        """
        [
            1499040000000,      // Kline open time                   0
            "0.01634790",       // Open price                        1
            "0.80000000",       // High price                        2
            "0.01575800",       // Low price                         3
            "0.01577100",       // Close price                       4
            "148976.11427815",  // Volume                            5
            1499644799999,      // Kline Close time                  6
            "2434.19055334",    // Quote asset volume                7
            308,                // Number of trades                  8
            "1756.87402397",    // Taker buy base asset volume       9
            "28.46694368",      // Taker buy quote asset volume     10
            "0"                 // Unused field, ignore.
        ]
        """
        return [
            self.safe_integer(ohlcv, 0),
            self.safe_number(ohlcv, 1),
            self.safe_number(ohlcv, 2),
            self.safe_number(ohlcv, 3),
            self.safe_number(ohlcv, 4),
            self.safe_number(ohlcv, 5),
            self.safe_number(ohlcv, 7),   # Quote asset volume
            self.safe_number(ohlcv, 10),  # Taker buy quote asset volume
        ]

    def handle_ohlcv(self, client: Client, message):
        event = self.safe_string(message, 'e')
        eventMap = {
            'indexPrice_kline': 'indexPriceKline',
            'markPrice_kline': 'markPriceKline',
        }
        event = self.safe_string(eventMap, event, event)
        kline = self.safe_value(message, 'k')
        marketId = self.safe_string_2(kline, 's', 'ps')
        if event == 'indexPriceKline':
            # indexPriceKline doesn't have the _PERP suffix
            marketId = self.safe_string(message, 'ps')
        lowercaseMarketId = marketId.lower()
        interval = self.safe_string(kline, 'i')
        # use a reverse lookup in a static map instead
        timeframe = self.find_timeframe(interval)
        messageHash = lowercaseMarketId + '@' + event + '_' + interval
        parsed = [
            self.safe_integer(kline, 't'),
            self.safe_float(kline, 'o'),
            self.safe_float(kline, 'h'),
            self.safe_float(kline, 'l'),
            self.safe_float(kline, 'c'),
            self.safe_float(kline, 'v'),
            # - additional fields
            self.safe_float(kline, 'q'), # - quote asset volume
            self.safe_float(kline, 'Q'), # - taker buy quote asset volume
        ]
        isSpot = ((client.url.find('/stream') > -1) or (client.url.find('/testnet.binance') > -1))
        marketType = 'spot' if (isSpot) else 'contract'
        symbol = self.safe_symbol(marketId, None, None, marketType)
        self.ohlcvs[symbol] = self.safe_value(self.ohlcvs, symbol, {})
        stored = self.safe_value(self.ohlcvs[symbol], timeframe)
        if stored is None:
            limit = self.safe_integer(self.options, 'OHLCVLimit', 2)
            stored = ArrayCacheByTimestamp(limit)
            # self.ohlcvs[symbol][timeframe] = stored
        stored.append(parsed)
        client.resolve(stored, messageHash)

# - register custom wrappers
cxp.binanceqv = BinanceQV
cxp.exchanges.append('binanceqv')


import asyncio
from threading import Thread

class AsyncioThread(Thread):
    def __init__(self, channel: CtrlChannel):
        self.result = None
        self.channel = channel
        self.loops = []
        super().__init__()

    def add(self, func, *args, **kwargs) -> 'AsyncioThread':
        self.loops.append(func(self.channel, *args, **kwargs))
        return self

    async def run_loop(self):
        self.result = await asyncio.gather(*self.loops)

    def run(self):
        self.channel.control.set()
        asyncio.run(self.run_loop())

    def stop(self):
        self.channel.control.clear()
        self.channel.queue.put((None, None)) # send sentinel


class CCXTConnector(IDataProvider):
    exch: Exchange
    subsriptions: Dict[str, AsyncioThread]

    def __init__(self, exchange: str):
        exchange = exchange.lower()
        exch = _aliases.get(exchange, exchange)
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange} -> {exch} is not supported by CCXT!")
        self.exch = getattr(cxp, exch)()
        self.subsriptions = {}

    def subscribe(self, subscription_type: str, symbols: List[str], 
                  timeframe:Optional[str]=None, nback:int=0) -> CtrlChannel:
        self._check_subscription(subscription_type.lower())

        match sbscr := subscription_type.lower():
            case 'ohlc':
                if timeframe is None:
                    raise ValueError("timeframe must not be None for OHLC subscription")

                r = AsyncioThread(comm:=CtrlChannel('OHLC'))
                for s in symbols:
                    r.add(self.listen_to_ohlcv, s, timeframe, nback)

                self.subsriptions[sbscr] = r
                r.start()
                return comm

            case 'trades':
                pass
            case 'quotes':
                pass
        return None

    def _check_subscription(self, subscription_type):
        if subscription_type in self.subsriptions:
            self.subsriptions[subscription_type].stop()
            del self.subsriptions[subscription_type]

    def prefetch_ohlcs(self, symbol: str, timeframe: str, nbarsback: int):
        assert nbarsback > 1
        start = ((pd.Timestamp('now', tz='UTC') - nbarsback * pd.Timedelta(timeframe)).asm8.item()//1000000) if nbarsback > 1 else None 
        ohlcv = self.exch.fetch_ohlcv(symbol, timeframe, since=start, limit=nbarsback + 1)
        return ohlcv

    async def listen_to_ohlcv(self, channel: CtrlChannel, symbol: str, timeframe: str, nbarsback: int):
        # - check if we need to load initial 'snapshot'
        if nbarsback > 1:
            ohlcv = await self.prefetch_ohlcs(symbol, timeframe, nbarsback)
            for oh in ohlcv:
                channel.queue.put((symbol, Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

        while channel.control.is_set():
            try:
                ohlcv = await self.exch.watch_ohlcv(symbol, timeframe)
                for oh in ohlcv:
                    channel.queue.put((symbol, Bar(oh[0] * 1000000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

            except Exception as e:
                # print(type(e).__name__, str(e), flush=True)
                logger.error(str(e))
                await self.exch.close()
                raise e
