from typing import Any, Dict, List, Optional

import asyncio
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from collections import defaultdict

import ccxt.pro as cxp
from ccxt.base.decimal_to_precision import ROUND_UP
from ccxt.base.exchange import Exchange
from ccxt import NetworkError
import re
import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, Position, dt_64
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.core.basics import Trade, Quote
from qubx.core.series import TimeSeries, Bar


_aliases = {
    'binance': 'binanceqv',
    'binance.um': 'binanceusdm',
    'binance.cm': 'binancecoinm',
    'kraken.f': 'kreakenfutures'
}

# - register custom wrappers
from .exchange_customizations import BinanceQV
cxp.binanceqv = BinanceQV            # type: ignore
cxp.exchanges.append('binanceqv')


class CCXTConnector(IDataProvider, IExchangeServiceProvider):
    exchange: Exchange
    subsriptions: Dict[str, List[str]]
    _ch_market_data: CtrlChannel
    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop 

    def __init__(self, exchange_id: str, **exchange_auth):
        super().__init__()
        exchange_id = exchange_id.lower()
        exch = _aliases.get(exchange_id, exchange_id)
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange_id} -> {exch} is not supported by CCXT!")

        self.exchange = getattr(cxp, exch)(exchange_auth)
        self.subsriptions: Dict[str, List[str]] = defaultdict(list)
        self._ch_market_data = CtrlChannel(exch + '.marketdata')
        self._last_quotes = defaultdict(lambda: None)
        self._loop = asyncio.get_running_loop()

    def subscribe(self, subscription_type: str, symbols: List[str], 
                  timeframe:Optional[str]=None, 
                  nback:int=0
    ) -> bool:
        to_process = self._check_existing_subscription(subscription_type.lower(), symbols)
        if not to_process:
            logger.info(f"Symbols {symbols} already subscribed on {subscription_type} data")
            return False

        match sbscr := subscription_type.lower():
            case 'ohlc':
                if timeframe is None:
                    raise ValueError("timeframe must not be None for OHLC data subscription")

                # convert to exchange format
                tframe = self._get_exch_timeframe(timeframe)
                for s in to_process:
                    self._run_async_task(self._listen_to_ohlcv(self.get_communication_channel(), s, tframe, nback))
                    self.subsriptions[sbscr].append(s.lower())
                logger.info(f'Subscribed on {sbscr} updates for {len(to_process)} symbols: \n\t\t{to_process}')
                return True

            case 'trades':
                raise ValueError("TODO")

            case 'quotes':
                raise ValueError("TODO")

            case _:
                raise ValueError("TODO")

        return False

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
        start = ((pd.Timestamp('now', tz='UTC') - nbarsback * pd.Timedelta(timeframe)).asm8.item()//1000000) if nbarsback > 1 else None 
        return await self.exchange.fetch_ohlcv(symbol, timeframe, since=start, limit=nbarsback + 1)        # type: ignore

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> Optional[List[Bar]]:
        assert nbarsback > 1
        r = self._run_async_task(self._fetch_ohlcs_a(symbol, self._get_exch_timeframe(timeframe), nbarsback), wait_result=True)
        if len(r) > 0:
            return [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in r]

    async def _listen_to_ohlcv(self, channel: CtrlChannel, symbol: str, timeframe: str, nbarsback: int):
        # - check if we need to load initial 'snapshot'
        if nbarsback > 1:
            ohlcv = asyncio.run(self._fetch_ohlcs_a(symbol, timeframe, nbarsback))
            for oh in ohlcv:
                channel.queue.put((symbol, Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

        while channel.control.is_set():
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe)        # type: ignore
                for oh in ohlcv:
                    channel.queue.put((symbol, Bar(oh[0] * 1000000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

            except NetworkError as e:
                logger.error(str(e))
                await asyncio.sleep(1)
                continue

            except Exception as e:
                logger.error(str(e))
                await self.exchange.close()        # type: ignore
                raise e

    async def _init_account(self):
        # TODO: 
        # free_bal = await self.exchange.fetch_free_balance()
        pass

    def sync_position(self, position: Position) -> Position:
        # TODO: read positions from exchange !!!
        b = self._run_async_task(self.exchange.fetch_balance(position.instrument.symbol), wait_result=True)
        r = self._run_async_task(self.exchange.fetch_position(position.instrument.symbol), wait_result=True)
        print(r)
        return position

    def time(self) -> dt_64:
        """
        Returns current time in nanoseconds
        """
        return np.datetime64(self.exchange.microseconds() * 1000, 'ns')

    def get_name(self) -> str:
        return self.exchange.name 

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quotes[symbol]

    def _get_exch_timeframe(self, timeframe: str):
        if timeframe is not None:
            _t = re.match('(\d+)(\w+)', timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self.exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self.get_name()}")

        return tframe

    def _run_async_task(self, coro, wait_result: bool = False) -> Task | Any:
        task = self._loop.create_task(coro)
        if wait_result:
            while not task.done():
                ...
                # await asyncio.sleep(0.01)
            return task.result()
        return task