import asyncio
import re
import numpy as np
import pandas as pd
import ccxt.pro as cxp

from threading import Thread
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple, Set
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from collections import defaultdict
from types import FunctionType
from ccxt import NetworkError, ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable
from ccxt.base.exchange import Exchange

from qubx import logger
from qubx.core.basics import Instrument, Position, dt_64, Deal, CtrlChannel
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IBrokerServiceProvider, ITradingServiceProvider, SubscriptionType
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.utils.ntp import start_ntp_thread, time_now
from .ccxt_utils import DATA_PROVIDERS_ALIASES, ccxt_convert_trade, ccxt_convert_orderbook

# - register custom wrappers
from .ccxt_customizations import BinanceQV, BinanceQVUSDM

cxp.binanceqv = BinanceQV  # type: ignore
cxp.binanceqv_usdm = BinanceQVUSDM  # type: ignore
cxp.exchanges.append("binanceqv")
cxp.exchanges.append("binanceqv_usdm")


EXCH_SYMBOL_PATTERN = re.compile(r"(?P<base>[^/]+)/(?P<quote>[^:]+)(?::(?P<margin>.+))?")


class CCXTExchangesConnector(IBrokerServiceProvider):

    _exchange: Exchange
    _scheduler: BasicScheduler | None = None

    # - subscriptions
    _subscriptions: Dict[str, Set[Instrument]]
    _pending_stream_subscriptions: Dict[Tuple[str, Tuple], Set[Instrument]]
    _pending_stream_unsubscriptions: Dict[str, Set[Instrument]]
    _pending_instrument_unsubscriptions: Set[Instrument]
    _sub_to_coro: Dict[str, asyncio.Future]

    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop
    _thread_event_loop: Thread
    _subscribers: Dict[str, Callable]

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

        # - start NTP thread
        start_ntp_thread()

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
        self._subscriptions = defaultdict(set)
        self._pending_stream_subscriptions = defaultdict(set)
        self._pending_stream_unsubscriptions = defaultdict(set)
        self._pending_instrument_unsubscriptions = set()
        self._sub_to_coro = {}

        self._subscribers = {
            n.split("_subscribe_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) == FunctionType and n.startswith("_subscribe_")
        }

        logger.info(f"{exchange_id} initialized - current time {self.trading_service.time()}")

    @property
    def is_simulated_trading(self) -> bool:
        return False

    def get_scheduler(self) -> BasicScheduler:
        if self._scheduler is None:
            self._scheduler = BasicScheduler(self.get_communication_channel(), lambda: self.time().item())
        return self._scheduler

    def time(self) -> dt_64:
        return time_now()

    def subscribe(
        self,
        instruments: List[Instrument],
        subscription_type: str,
        **kwargs,
    ) -> None:
        self._pending_stream_subscriptions[
            self._parse_subscription_type(subscription_type), tuple(kwargs.items())
        ].update(instruments)

    def unsubscribe(self, instruments: List[Instrument], subscription_type: str | None = None) -> None:
        if subscription_type is None:
            self._pending_instrument_unsubscriptions.update(instruments)
        else:
            self._pending_stream_unsubscriptions[self._parse_subscription_type(subscription_type)].update(instruments)

    def has_subscription(self, subscription_type: str, instrument: Instrument) -> bool:
        sub = subscription_type.lower()
        return sub in self._subscriptions and instrument in self._subscriptions[sub]

    def commit(self) -> None:
        _has_something_changed = False
        for (sub, sub_args), instrs in self._pending_stream_subscriptions.items():
            _has_something_changed |= self._subscribe(instrs, sub, **dict(sub_args))

        if _has_something_changed:
            if not self.read_only:
                # - subscribe to executions reports
                self._resubscribe_stream("executions", self.get_communication_channel(), self.subscribed_instruments)

        # - clean up pending subs and unsubs
        self._pending_stream_subscriptions.clear()
        self._pending_instrument_unsubscriptions.clear()
        self._pending_stream_unsubscriptions.clear()

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument.symbol]

    def get_historical_ohlcs(self, instrument: Instrument, timeframe: str, nbarsback: int) -> List[Bar]:
        assert nbarsback >= 1
        symbol = instrument.symbol
        since = self._time_msec_nbars_back(timeframe, nbarsback)

        # - retrieve OHLC data
        # - TODO: check if nbarsback > max_limit (1000) we need to do more requests
        # - TODO: how to get quoted volumes ?
        async def _get():
            return await self._exchange.fetch_ohlcv(
                symbol, self._get_exch_timeframe(timeframe), since=since, limit=nbarsback + 1
            )  # type: ignore

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

    def close(self):
        try:
            if hasattr(self._exchange, "close"):
                self._exchange.close()  # type: ignore
            else:
                del self._exchange
        except Exception as e:
            logger.error(e)

    @property
    def subscribed_instruments(self) -> Set[Instrument]:
        return set.union(*self._subscriptions.values())

    def _get_pending_unsubscriptions(self, sub: str) -> Set[Instrument]:
        return self._pending_stream_unsubscriptions.get(sub, set()).union(self._pending_instrument_unsubscriptions)

    def _subscribe(
        self,
        instruments: Set[Instrument],
        sub_type: str,
        **kwargs,
    ) -> bool:
        _subscriber = self._subscribers.get(sub_type)
        if _subscriber is None:
            raise ValueError(f"Subscription type {sub_type} is not supported")

        self._check_event_loop_is_running()

        _current_instruments = self._subscriptions[sub_type]
        _removed_instruments = self._get_pending_unsubscriptions(sub_type)
        _updated_instruments = instruments.union(_current_instruments).difference(_removed_instruments)
        _added_instruments = _updated_instruments.difference(_current_instruments)

        if _current_instruments == _updated_instruments:
            return False

        _channel = self.get_communication_channel()

        # - fetch historical ohlc for added instruments
        if (ohlc_warmup_period := kwargs.get("ohlc_warmup_period")) is not None and (
            timeframe := kwargs.get("timeframe")
        ) is not None:
            self._run_ohlc_warmup(_channel, _added_instruments, timeframe, ohlc_warmup_period)

        self._resubscribe_stream(sub_type, _channel, _updated_instruments, **kwargs)

        return True

    def _resubscribe_stream(self, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **kwargs):
        if sub_type in self._sub_to_coro:
            logger.debug(f"Canceling previous subscription {sub_type}")
            future = self._sub_to_coro[sub_type]
            future.cancel()

        _subscriber = self._subscribers[sub_type]
        _subscriber_params = set(_subscriber.__code__.co_varnames)
        # - get only parameters that are needed for subscriber
        kwargs = {k: v for k, v in kwargs.items() if k in _subscriber_params}
        self._sub_to_coro[sub_type] = self._submit_coro(_subscriber(self, channel, instruments, **kwargs))

    def _check_event_loop_is_running(self) -> None:
        if self._loop.is_running():
            return
        self._thread_event_loop = Thread(target=self._loop.run_forever, args=(), daemon=True)
        self._thread_event_loop.start()

    def _parse_subscription_type(self, subscription_type: str) -> str:
        return subscription_type.lower()

    def _submit_coro(self, coro: Awaitable[None]) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int) -> int:
        return (self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def _run_ohlc_warmup(
        self, channel: CtrlChannel, instruments: Set[Instrument], timeframe: str, warmup_period: str
    ) -> None:
        logger.debug(f"Running OHLC warmup for {instruments} with period {warmup_period}")
        _warmup_futures = []
        for instr in instruments:
            _warmup_futures.append(
                self._submit_coro(
                    self._stream_historical_ohlc(channel, instr, timeframe, warmup_period),
                )
            )
        # - wait for all warmup futures
        asyncio.gather(*_warmup_futures)

    async def _listen_to_stream(
        self, subscriber: Callable[[], Awaitable[None]], exchange: Exchange, channel: CtrlChannel, name: str
    ):
        logger.debug(f"Listening to {name}")
        n_retry = 0
        while channel.control.is_set():
            try:
                await subscriber()
                n_retry = 0
            except ExchangeClosedByUser:
                # - we closed connection so just stop it
                logger.info(f"{name} listening has been stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                logger.error(f"Error in {name} : {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"exception in {name} : {e}")
                logger.exception(e)
                n_retry += 1
                if n_retry >= self.max_ws_retries:
                    logger.error(f"Max retries reached for {name}. Closing connection.")
                    del exchange
                    break
                await asyncio.sleep(min(2**n_retry, 60))  # Exponential backoff with a cap at 60 seconds

    async def _stream_historical_ohlc(
        self, channel: CtrlChannel, instrument: Instrument, timeframe: str, warmup_period: str
    ) -> None:
        nbarsback = pd.Timedelta(warmup_period) // pd.Timedelta(timeframe)
        exch_timeframe = self._get_exch_timeframe(timeframe)
        start = self._time_msec_nbars_back(timeframe, nbarsback)
        ohlcv = await self._exchange.fetch_ohlcv(instrument.symbol, exch_timeframe, since=start, limit=nbarsback + 1)
        # - just send data as the list
        channel.send(
            (
                instrument,
                "hist_bars",
                [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv],
            )
        )
        logger.info(f"{instrument}: loaded {len(ohlcv)} {timeframe} bars")

    def _get_exch_timeframe(self, timeframe: str) -> str:
        if timeframe is not None:
            _t = re.match(r"(\d+)(\w+)", timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self._exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self._exchange.name}")

        return tframe

    def _get_exch_symbol(self, instrument: Instrument) -> str:
        return f"{instrument.base}/{instrument.quote}:{instrument.margin_symbol}"

    def _find_instrument_for_exch_symbol(
        self, exch_symbol: str, symbol_to_instrument: Dict[str, Instrument]
    ) -> Instrument:
        match = EXCH_SYMBOL_PATTERN.match(exch_symbol)
        if not match:
            raise ValueError(f"Invalid exchange symbol {exch_symbol}")
        base = match.group("base")
        quote = match.group("quote")
        symbol = f"{base}{quote}"
        if symbol not in symbol_to_instrument:
            raise ValueError(f"Unknown symbol {symbol}")
        return symbol_to_instrument[symbol]

    #############################
    # - Subscription methods
    #############################
    async def _subscribe_ohlc(
        self,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        warmup_period: str | None = None,
        timeframe: str = "1m",
    ):
        symbols = [i.symbol for i in instruments]
        _exchange_timeframe = self._get_exch_timeframe(timeframe)
        _symbol_timeframe_pairs = [[symbol, _exchange_timeframe] for symbol in symbols]
        _symbol_to_instrument = {i.symbol: i for i in instruments}

        async def watch_ohlcv():
            ohlcv = await self._exchange.watch_ohlcv_for_symbols(_symbol_timeframe_pairs)
            _time = self.time()
            # - ohlcv is symbol -> timeframe -> list[timestamp, open, high, low, close, volume]
            for exch_symbol, _data in ohlcv.items():
                try:
                    instrument = self._find_instrument_for_exch_symbol(exch_symbol, _symbol_to_instrument)
                    for _, ohlcvs in _data.items():
                        for oh in ohlcvs:
                            channel.send(
                                (instrument, "bar", Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]))
                            )
                        last_close = ohlcvs[-1][4]
                        self.trading_service.update_position_price(
                            instrument,
                            _time,
                            last_close,
                        )
                except Exception as e:
                    logger.error(f"Error in watch_ohlcv : {e}")

        if warmup_period is not None:
            await self._run_ohlc_warmup(channel, instruments, timeframe, warmup_period)

        await self._listen_to_stream(watch_ohlcv, self._exchange, channel, f"{','.join(symbols)} {timeframe} OHLCV")

    async def _subscribe_executions(self, channel: CtrlChannel, instruments: List[Instrument]):
        # TODO: continue here
        symbols = [i.symbol for i in instruments]

        async def _watch_executions():
            exec = await self._exchange.watch_orders_for_symbols(symbols)
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
        await self._stream_historical_ohlc(channel, timeframe, symbol, nbarsback)
        await self._listen_to_stream(_watch_trades, self._exchange, channel, f"{symbol} trades")

    async def _listen_to_orderbook(self, channel: CtrlChannel, instrument: Instrument, timeframe: str, nbarsback: int):
        symbol = instrument.symbol

        async def _watch_orderbook():
            ccxt_ob = await self._exchange.watch_order_book(symbol)
            ob = ccxt_convert_orderbook(ccxt_ob, instrument)
            self.trading_service.update_position_price(instrument, self.time(), ob.to_quote())
            channel.send((symbol, "orderbook", ob))

        # TODO: stream historical orderbooks for some period
        await self._stream_historical_ohlc(channel, timeframe, symbol, nbarsback)
        await self._listen_to_stream(_watch_orderbook, self._exchange, channel, f"{symbol} orderbook")
