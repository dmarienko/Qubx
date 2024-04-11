from typing import Any, Dict, List, Optional

import asyncio
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from collections import defaultdict
import stackprinter

import ccxt.pro as cxp
from ccxt.base.exchange import Exchange
from ccxt import NetworkError

import re
import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, Position, dt_64, Deal
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.impl.utils import ALIASES

from .ccxt_trading import CCXTSyncTradingConnector

# - register custom wrappers
from .exchange_customizations import BinanceQV
cxp.binanceqv = BinanceQV            # type: ignore
cxp.exchanges.append('binanceqv')


class CCXTConnector(IDataProvider, CCXTSyncTradingConnector):
    exchange: Exchange
    subsriptions: Dict[str, List[str]]

    _ch_market_data: CtrlChannel
    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop 

    def __init__(self, exchange_id: str, base_currency: str | None = None, commissions: str|None = None, **exchange_auth):
        super().__init__(exchange_id, base_currency, commissions, **exchange_auth)
        
        exchange_id = exchange_id.lower()
        exch = ALIASES.get(exchange_id, exchange_id)
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange_id} -> {exch} is not supported by CCXT.pro !")

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.info("started new event loop")
            # exchange_auth |= {'asyncio_loop':self._loop}

        self.exchange = getattr(cxp, exch)(exchange_auth)
        self.subsriptions: Dict[str, List[str]] = defaultdict(list)
        self._ch_market_data = CtrlChannel(exch + '.marketdata')
        self._last_quotes = defaultdict(lambda: None)

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
                    self._task_a(self._listen_to_ohlcv(self.get_communication_channel(), s, tframe, nback))
                    self.subsriptions[sbscr].append(s.lower())
                logger.info(f'Subscribed on {sbscr} updates for {len(to_process)} symbols: \n\t\t{to_process}')

            case 'trades':
                raise ValueError("TODO")

            case 'quotes':
                raise ValueError("TODO")

            case _:
                raise ValueError("TODO")

        # - subscibe to executions reports
        for s in to_process:
            self._task_a(self._listen_to_execution_reports(self.get_communication_channel(), s))

        return True

    def get_communication_channel(self) -> CtrlChannel:
        return self._ch_market_data

    def _check_existing_subscription(self, subscription_type, symbols: List[str]) -> List[str]:
        subscribed = self.subsriptions[subscription_type]
        to_subscribe = []
        for s in symbols: 
            if s not in subscribed:
                to_subscribe.append(s)
        return to_subscribe

    async def _fetch_ohlcs_a(self, symbol: str, timeframe: str, nbarsback: int):
        assert nbarsback > 1
        start = ((self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item()//1000000) if nbarsback > 1 else None 
        return await self.exchange.fetch_ohlcv(symbol, timeframe, since=start, limit=nbarsback + 1)        # type: ignore

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> Optional[List[Bar]]:
        assert nbarsback > 1
        # we want to wait until initial snapshot is arrived so run it in sync mode
        r = self._task_s(self._fetch_ohlcs_a(symbol, self._get_exch_timeframe(timeframe), nbarsback))
        if len(r) > 0:
            return [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in r]

    async def _listen_to_execution_reports(self, channel: CtrlChannel, symbol: str):
        while channel.control.is_set():
            try:
                exec = await self.exchange.watch_orders(symbol)        # type: ignore
                _msg = f"\nexecs_{symbol} = [\n"
                for report in exec:
                    _msg += '\t' + str(report) + ',\n'
                    order = self._process_execution_report(symbol, report)
                    # - send update to client 
                    channel.queue.put((symbol, order))
                logger.info(_msg + "]\n")
            except NetworkError as e:
                logger.error(f"(CCXTConnector) NetworkError in _listen_to_execution_reports : {e}")
                await asyncio.sleep(1)
                continue

            except Exception as err:
                logger.error(f"(CCXTConnector) exception in _listen_to_execution_reports : {err}")
                logger.error(stackprinter.format(err))

    async def _listen_to_ohlcv(self, channel: CtrlChannel, symbol: str, timeframe: str, nbarsback: int):
        # - check if we need to load initial 'snapshot'
        if nbarsback > 1:
            # ohlcv = asyncio.run(self._fetch_ohlcs_a(symbol, timeframe, nbarsback))
            ohlcv = self._task_s(self._fetch_ohlcs_a(symbol, timeframe, nbarsback))
            for oh in ohlcv:
                channel.queue.put((symbol, Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))
            logger.info(f"{symbol}: loaded {len(ohlcv)} {timeframe} bars")

        while channel.control.is_set():
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe)        # type: ignore

                # - update positions by actual close price
                self.update_position_price(symbol, ohlcv[-1][4])

                for oh in ohlcv:
                    channel.queue.put((symbol, Bar(oh[0] * 1000000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

            except NetworkError as e:
                logger.error(f"(CCXTConnector) NetworkError in _listen_to_ohlcv : {e}")
                await asyncio.sleep(1)
                continue

            except Exception as e:
                # logger.error(str(e))
                logger.error(f"(CCXTConnector) exception in _listen_to_ohlcv : {e}")
                logger.error(stackprinter.format(e))
                await self.exchange.close()        # type: ignore
                raise e

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quotes[symbol]

    def _get_exch_timeframe(self, timeframe: str):
        if timeframe is not None:
            _t = re.match(r'(\d+)(\w+)', timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self.exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self.get_name()}")

        return tframe

    def _task_a(self, coro) -> Task:
        return self._loop.create_task(coro)

    def _task_s(self, coro) -> Any:
        return self._loop.run_until_complete(coro)

    def close(self):
        try:
            self._task_s(self.exchange.close()) # type: ignore
        except Exception as e:
            logger.error(e)