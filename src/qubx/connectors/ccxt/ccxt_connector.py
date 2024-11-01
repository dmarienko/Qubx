from threading import Thread
import threading
from typing import Any, Dict, List, Optional, Callable, Awaitable

import asyncio
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from collections import defaultdict

import ccxt.pro as cxp
from ccxt.base.exchange import Exchange
from ccxt import NetworkError, ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable

import re
import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, Position, dt_64, Deal, CtrlChannel
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IBrokerServiceProvider, ITradingServiceProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.utils.ntp import time_now
from .ccxt_utils import DATA_PROVIDERS_ALIASES, ccxt_convert_trade, ccxt_convert_orderbook

# - register custom wrappers
from .ccxt_customizations import BinanceQV, BinanceQVUSDM

cxp.binanceqv = BinanceQV  # type: ignore
cxp.binanceqv_usdm = BinanceQVUSDM  # type: ignore
cxp.exchanges.append("binanceqv")
cxp.exchanges.append("binanceqv_usdm")


class CCXTExchangesConnector(IBrokerServiceProvider):
    # TODO: implement multi symbol subscription from one channel

    _exchange: Exchange
    _subscriptions: Dict[str, List[Instrument]]
    _scheduler: BasicScheduler | None = None

    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop
    _thread_event_loop: Thread

    def __init__(
        self,
        exchange_id: str,
        trading_service: ITradingServiceProvider,
        read_only: bool = False,
        loop: AbstractEventLoop | None = None,
        max_ws_retries: int = 10,
        **exchange_auth,
    ):
        super().__init__(exchange_id, trading_service)
        self.trading_service = trading_service
        self.read_only = read_only
        self.max_ws_retries = max_ws_retries
        exchange_id = exchange_id.lower()

        # - setup communication bus
        self.set_communication_channel(bus := CtrlChannel("databus", sentinel=(None, None, None)))
        self.trading_service.set_communication_channel(bus)

        # - init CCXT stuff
        exch = DATA_PROVIDERS_ALIASES.get(exchange_id, exchange_id)
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange_id} -> {exch} is not supported by CCXT.pro !")

        # - create new even loop
        self._loop = asyncio.new_event_loop() if loop is None else loop
        asyncio.set_event_loop(self._loop)

        # - create exchange's instance
        self._exchange = getattr(cxp, exch)(exchange_auth | {"asyncio_loop": self._loop})
        self._last_quotes = defaultdict(lambda: None)
        self._subscriptions = defaultdict(list)

        logger.info(f"{exchange_id} initialized - current time {self.trading_service.time()}")

    @property
    def is_simulated_trading(self) -> bool:
        return False

    def get_scheduler(self) -> BasicScheduler:
        # - standard scheduler
        if self._scheduler is None:
            self._scheduler = BasicScheduler(self.get_communication_channel(), lambda: self.time().item())
        return self._scheduler

    def subscribe(
        self, subscription_type: str, instruments: List[Instrument], timeframe: Optional[str] = None, nback: int = 0
    ) -> bool:
        to_process = self._check_existing_subscription(subscription_type.lower(), instruments)
        if not to_process:
            logger.info(f"Symbols {to_process} already subscribed on {subscription_type} data")
            return False

        # - start event loop if not already running
        if not self._loop.is_running():
            self._thread_event_loop = Thread(target=self._loop.run_forever, args=(), daemon=True)
            self._thread_event_loop.start()

        to_process_symbols = [s.symbol for s in to_process]

        # - subscribe to market data updates
        match sbscr := subscription_type.lower():
            case "ohlc":
                if timeframe is None:
                    raise ValueError("timeframe must not be None for OHLC data subscription")

                # convert to exchange format
                tframe = self._get_exch_timeframe(timeframe)
                for s in to_process:
                    # self._task_a(self._listen_to_ohlcv(self.get_communication_channel(), s, tframe, nback))
                    asyncio.run_coroutine_threadsafe(
                        self._listen_to_ohlcv(self.get_communication_channel(), s, tframe, nback), self._loop
                    )
                    self._subscriptions[sbscr].append(s)
                logger.info(f"Subscribed on {sbscr} updates for {len(to_process)} symbols: \n\t\t{to_process_symbols}")

            case "trade":
                if timeframe is None:
                    timeframe = "1Min"
                tframe = self._get_exch_timeframe(timeframe)
                for s in to_process:
                    asyncio.run_coroutine_threadsafe(
                        self._listen_to_trades(self.get_communication_channel(), s, tframe, nback), self._loop
                    )
                    self._subscriptions[sbscr].append(s)
                logger.info(f"Subscribed on {sbscr} updates for {len(to_process)} symbols: \n\t\t{to_process_symbols}")

            case "orderbook":
                if timeframe is None:
                    timeframe = "1Min"
                tframe = self._get_exch_timeframe(timeframe)
                for s in to_process:
                    asyncio.run_coroutine_threadsafe(
                        self._listen_to_orderbook(self.get_communication_channel(), s, timeframe, nback), self._loop
                    )
                    self._subscriptions[sbscr].append(s)
                logger.info(f"Subscribed on {sbscr} updates for {len(to_process)} symbols: \n\t\t{to_process_symbols}")

            case "quote":
                raise ValueError("TODO")

            case _:
                raise ValueError("TODO")

        # - subscibe to executions reports
        if not self.read_only:
            for s in to_process:
                asyncio.run_coroutine_threadsafe(
                    self._listen_to_execution_reports(self.get_communication_channel(), s), self._loop
                )

        return True

    def has_subscription(self, subscription_type: str, instrument: Instrument) -> bool:
        sub = subscription_type.lower()
        return sub in self._subscriptions and instrument in self._subscriptions[sub]

    def _check_existing_subscription(self, subscription_type, instruments: List[Instrument]) -> List[Instrument]:
        subscribed = self._subscriptions[subscription_type]
        to_subscribe = []
        for s in instruments:
            if s not in subscribed:
                to_subscribe.append(s)
        return to_subscribe

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int) -> int:
        return (self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def get_historical_ohlcs(self, instrument: Instrument, timeframe: str, nbarsback: int) -> List[Bar]:
        assert nbarsback >= 1
        symbol = instrument.symbol
        since = self._time_msec_nbars_back(timeframe, nbarsback)

        # - retrieve OHLC data
        # - TODO: check if nbarsback > max_limit (1000) we need to do more requests
        # - TODO: how to get quoted volumes ?
        async def _get():
            return await self._exchange.fetch_ohlcv(symbol, self._get_exch_timeframe(timeframe), since=since, limit=nbarsback + 1)  # type: ignore

        fut = asyncio.run_coroutine_threadsafe(_get(), self._loop)
        res = fut.result(60)

        _arr = []
        for oh in res:  # type: ignore
            _arr.append(
                Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])
                if len(oh) > 6
                else Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[5])
            )
        return _arr

    async def _listen_to_execution_reports(self, channel: CtrlChannel, instrument: Instrument):
        async def _watch_executions():
            exec = await self._exchange.watch_orders(instrument.symbol)  # type: ignore
            _msg = f"\nexecs_{instrument.symbol} = [\n"
            for report in exec:
                _msg += "\t" + str(report) + ",\n"
                order, deals = self.trading_service.process_execution_report(instrument, report)
                # - send update to client
                channel.send((instrument, "order", order))
                if deals:
                    channel.send((instrument, "deals", deals))
            logger.debug(_msg + "]\n")

        await self._listen_to_stream(
            _watch_executions, self._exchange, channel, f"{instrument.symbol} execution reports"
        )

    async def _listen_to_ohlcv(self, channel: CtrlChannel, instrument: Instrument, timeframe: str, nbarsback: int):
        symbol = instrument.symbol

        async def watch_ohlcv():
            ohlcv = await self._exchange.watch_ohlcv(symbol, timeframe)  # type: ignore
            # - update positions by actual close price
            last_close = ohlcv[-1][4]
            # - there is no single method to get OHLC update's event time for every broker
            # - for instance it's possible to do for Binance but for example Bitmex doesn't provide such info
            # - so we will use ntp adjusted time here
            self.trading_service.update_position_price(instrument, self.time(), last_close)
            for oh in ohlcv:
                channel.send((symbol, "bar", Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

        await self._stream_hist_bars(channel, timeframe, instrument.symbol, nbarsback)
        await self._listen_to_stream(watch_ohlcv, self._exchange, channel, f"{symbol} {timeframe} OHLCV")

    async def _listen_to_trades(self, channel: CtrlChannel, instrument: Instrument, timeframe: str, nbarsback: int):
        symbol = instrument.symbol

        async def _watch_trades():
            trades = await self._exchange.watch_trades(symbol)
            # - update positions by actual close price
            last_trade = ccxt_convert_trade(trades[-1])
            self.trading_service.update_position_price(instrument, self.time(), last_trade)
            for trade in trades:
                channel.send((symbol, "trade", ccxt_convert_trade(trade)))

        # TODO: stream historical trades for some period
        await self._stream_hist_bars(channel, timeframe, symbol, nbarsback)
        await self._listen_to_stream(_watch_trades, self._exchange, channel, f"{symbol} trades")

    async def _listen_to_orderbook(self, channel: CtrlChannel, instrument: Instrument, timeframe: str, nbarsback: int):
        symbol = instrument.symbol

        async def _watch_orderbook():
            ccxt_ob = await self._exchange.watch_order_book(symbol)
            ob = ccxt_convert_orderbook(ccxt_ob, instrument)
            self.trading_service.update_position_price(instrument, self.time(), ob.to_quote())
            channel.send((symbol, "orderbook", ob))

        # TODO: stream historical orderbooks for some period
        await self._stream_hist_bars(channel, timeframe, symbol, nbarsback)
        await self._listen_to_stream(_watch_orderbook, self._exchange, channel, f"{symbol} orderbook")

    async def _listen_to_stream(
        self, subscriber: Callable[[], Awaitable[None]], exchange: Exchange, channel: CtrlChannel, name: str
    ):
        logger.debug(f"Listening to {name}")
        n_retry = 0
        while channel.control.is_set():
            try:
                await subscriber()
                n_retry = 0
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                logger.error(f"Error in {name} : {e}")
                await asyncio.sleep(1)
                continue
            except ExchangeClosedByUser:
                # - we closed connection so just stop it
                logger.info(f"{name} listening has been stopped")
                break
            except Exception as e:
                logger.error(f"exception in {name} : {e}")
                logger.exception(e)
                n_retry += 1
                if n_retry >= self.max_ws_retries:
                    logger.error(f"Max retries reached for {name}. Closing connection.")
                    await exchange.close()  # type: ignore
                    break
                await asyncio.sleep(min(2**n_retry, 60))  # Exponential backoff with a cap at 60 seconds

    async def _stream_hist_bars(self, channel: CtrlChannel, timeframe: str, symbol: str, nbarsback: int) -> None:
        if nbarsback < 1:
            return
        start = self._time_msec_nbars_back(timeframe, nbarsback)
        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, since=start, limit=nbarsback + 1)  # type: ignore
        # - just send data as the list
        channel.send(
            (
                symbol,
                "hist_bars",
                [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv],
            )
        )
        logger.info(f"{symbol}: loaded {len(ohlcv)} {timeframe} bars")

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument.symbol]

    def _get_exch_timeframe(self, timeframe: str) -> str:
        if timeframe is not None:
            _t = re.match(r"(\d+)(\w+)", timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self._exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self._exchange.name}")

        return tframe

    def close(self):
        try:
            # self._loop.run_until_complete(self._exchange.close())  # type: ignore
            asyncio.run_coroutine_threadsafe(self._exchange.close(), self._loop)
        except Exception as e:
            logger.error(e)

    def time(self) -> dt_64:
        """
        Returns current time as dt64
        """
        return time_now()
