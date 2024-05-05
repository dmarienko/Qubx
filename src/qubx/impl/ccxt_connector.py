from threading import Thread
from typing import Any, Dict, List, Optional

import asyncio
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from collections import defaultdict
import stackprinter

import ccxt.pro as cxp
from ccxt.base.exchange import Exchange
from ccxt import NetworkError, ExchangeClosedByUser

import re
import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, Position, dt_64, Deal
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.impl.ccxt_utils import DATA_PROVIDERS_ALIASES

from .ccxt_trading import CCXTSyncTradingConnector

# - register custom wrappers
from .ccxt_customizations import BinanceQV
cxp.binanceqv = BinanceQV            # type: ignore
cxp.exchanges.append('binanceqv')


class CCXTConnector(IDataProvider, CCXTSyncTradingConnector):
    _exchange: Exchange
    _subsriptions: Dict[str, List[str]]

    _ch_market_data: CtrlChannel
    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop 
    _thread_event_loop: Thread

    def __init__(self, exchange_id: str, account_id: str, base_currency: str | None = None, commissions: str|None = None, **exchange_auth):
        super().__init__(exchange_id, account_id, base_currency, commissions, **exchange_auth)
        
        exchange_id = exchange_id.lower()
        exch = DATA_PROVIDERS_ALIASES.get(exchange_id, exchange_id)
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange_id} -> {exch} is not supported by CCXT.pro !")

        # - create new even loop
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # - create exchange's instance
        self._exchange = getattr(cxp, exch)(exchange_auth | {'asyncio_loop': self._loop})
        self._ch_market_data = CtrlChannel(exch + '.marketdata', sentinel=(None, None, None))
        self._last_quotes = defaultdict(lambda: None)
        self._subsriptions: Dict[str, List[str]] = defaultdict(list)

        logger.info(f"{self.get_name().upper()} initialized - current time {self.time()}")

    def subscribe(self, subscription_type: str, symbols: List[str], timeframe:Optional[str]=None, nback:int=0) -> bool:
        to_process = self._check_existing_subscription(subscription_type.lower(), symbols)
        if not to_process:
            logger.info(f"Symbols {symbols} already subscribed on {subscription_type} data")
            return False

        # - subscribe to market data updates
        match sbscr := subscription_type.lower():
            case 'ohlc':
                if timeframe is None:
                    raise ValueError("timeframe must not be None for OHLC data subscription")

                # convert to exchange format
                tframe = self._get_exch_timeframe(timeframe)
                for s in to_process:
                    # self._task_a(self._listen_to_ohlcv(self.get_communication_channel(), s, tframe, nback))
                    asyncio.run_coroutine_threadsafe(self._listen_to_ohlcv(self.get_communication_channel(), s, tframe, nback), self._loop)
                    self._subsriptions[sbscr].append(s.lower())
                logger.info(f'Subscribed on {sbscr} updates for {len(to_process)} symbols: \n\t\t{to_process}')

            case 'trades':
                raise ValueError("TODO")

            case 'quotes':
                raise ValueError("TODO")

            case _:
                raise ValueError("TODO")

        # - subscibe to executions reports
        for s in to_process:
            asyncio.run_coroutine_threadsafe(self._listen_to_execution_reports(self.get_communication_channel(), s), self._loop)

        self._thread_event_loop = Thread(target=self._loop.run_forever, args=(), daemon=True)
        self._thread_event_loop.start()

        return True

    def get_communication_channel(self) -> CtrlChannel:
        return self._ch_market_data

    def _check_existing_subscription(self, subscription_type, symbols: List[str]) -> List[str]:
        subscribed = self._subsriptions[subscription_type]
        to_subscribe = []
        for s in symbols: 
            if s not in subscribed:
                to_subscribe.append(s)
        return to_subscribe

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int) -> int:
        return (self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> List[Bar]:
        assert nbarsback >= 1
        since = self._time_msec_nbars_back(timeframe, nbarsback)

        # - get this from sync 
        # - TODO: check if nbarsback > max_limit (1000) we need to do more requests
        # - TODO: how to get quoted volumes ?
        res = self.sync.fetch_ohlcv(symbol, self._get_exch_timeframe(timeframe), since=since, limit=1000)
        _arr = []  
        for oh in res: # type: ignore
            _arr.append(
                Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) 
                if len(oh) > 6 else 
                Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[5]) 
            )
        return _arr

    async def _listen_to_execution_reports(self, channel: CtrlChannel, symbol: str):
        while channel.control.is_set():
            try:
                exec = await self._exchange.watch_orders(symbol)        # type: ignore
                _msg = f"\nexecs_{symbol} = [\n"
                for report in exec:
                    _msg += '\t' + str(report) + ',\n'
                    order, deals = self._process_execution_report(symbol, report)
                    # - send update to client 
                    channel.queue.put((symbol, 'order', order))
                    if deals: 
                        channel.queue.put((symbol, 'deals', deals))
                logger.debug(_msg + "]\n")

            except NetworkError as e:
                logger.error(f"(CCXTConnector) NetworkError in _listen_to_execution_reports : {e}")
                await asyncio.sleep(1)
                continue

            except ExchangeClosedByUser:
                # - we closed connection so just stop it 
                logger.info(f"(CCXTConnector) {symbol} execution reports listening has been stopped")
                break

            except Exception as err:
                logger.error(f"(CCXTConnector) exception in _listen_to_execution_reports : {err}")
                logger.error(stackprinter.format(err))

    async def _listen_to_ohlcv(self, channel: CtrlChannel, symbol: str, timeframe: str, nbarsback: int):
        # - check if we need to load initial 'snapshot'
        if nbarsback >= 1:
            start = self._time_msec_nbars_back(timeframe, nbarsback)
            ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, since=start, limit=nbarsback + 1)        # type: ignore
            # - just send data as the list
            channel.queue.put((
                symbol, 
                'hist_bars', 
                [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv]
            ))
            logger.info(f"{symbol}: loaded {len(ohlcv)} {timeframe} bars")

        while channel.control.is_set():
            try:
                ohlcv = await self._exchange.watch_ohlcv(symbol, timeframe)        # type: ignore

                # - update positions by actual close price
                self.update_position_price(symbol, ohlcv[-1][4])

                for oh in ohlcv:
                    channel.queue.put((symbol, 'bar', Bar(oh[0] * 1000000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

            except NetworkError as e:
                logger.error(f"(CCXTConnector) NetworkError in _listen_to_ohlcv : {e}")
                await asyncio.sleep(1)
                continue

            except ExchangeClosedByUser:
                # - we closed connection so just stop it 
                logger.info(f"(CCXTConnector) {symbol} OHLCV listening has been stopped")
                break

            except Exception as e:
                # logger.error(str(e))
                logger.error(f"(CCXTConnector) exception in _listen_to_ohlcv : {e}")
                logger.error(stackprinter.format(e))
                await self._exchange.close()        # type: ignore
                raise e

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quotes[symbol]

    def _get_exch_timeframe(self, timeframe: str):
        if timeframe is not None:
            _t = re.match(r'(\d+)(\w+)', timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self._exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self.get_name()}")

        return tframe

    def close(self):
        try:
            self._loop.run_until_complete(self._exchange.close()) # type: ignore
        except Exception as e:
            logger.error(e)
