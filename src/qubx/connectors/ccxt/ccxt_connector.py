import asyncio
import re
import numpy as np
import pandas as pd
import ccxt.pro as cxp
import concurrent.futures

from threading import Thread
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple, Set
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from asyncio.exceptions import CancelledError
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
from .ccxt_exceptions import CcxtSymbolNotRecognized

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
    _subscription_to_params: Dict[str, Dict[str, Any]]
    _pending_stream_subscriptions: Dict[Tuple[str, Tuple], Set[Instrument]]
    _pending_stream_unsubscriptions: Dict[str, Set[Instrument]]
    _pending_instrument_unsubscriptions: Set[Instrument]
    _sub_to_coro: Dict[str, concurrent.futures.Future]

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
        self._subscription_to_params = defaultdict(dict)
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

    def get_subscriptions(self, instrument: Instrument) -> Dict[str, Dict[str, Any]]:
        _sub_to_params = {}
        for sub, instrs in self._subscriptions.items():
            if instrument in instrs:
                _sub_to_params[sub] = self._subscription_to_params[sub]
        return _sub_to_params

    def has_subscription(self, subscription_type: str, instrument: Instrument) -> bool:
        sub = subscription_type.lower()
        return sub in self._subscriptions and instrument in self._subscriptions[sub]

    def commit(self) -> None:
        _sub_to_params = self._get_updated_sub_to_params()

        _has_something_changed = False
        for sub, params in _sub_to_params.items():
            _current_instruments = self._subscriptions[sub]
            _removed_instruments = self._get_pending_unsub_instr(sub)
            _added_instruments = self._get_pending_sub_instr(sub)
            _updated_instruments = _current_instruments.union(_added_instruments).difference(_removed_instruments)
            if _updated_instruments != _current_instruments:
                _has_something_changed = True
                self._subscribe(_updated_instruments, sub, **params)

        if _has_something_changed and not self.read_only:
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
                future = self._submit_coro(self._exchange.close())  # type: ignore
                # - wait for 5 seconds for connection to close
                future.result(5)
            else:
                del self._exchange
        except Exception as e:
            logger.error(e)

    @property
    def subscribed_instruments(self) -> Set[Instrument]:
        return set.union(*self._subscriptions.values())

    def _get_pending_unsub_instr(self, sub: str) -> Set[Instrument]:
        return self._pending_stream_unsubscriptions.get(sub, set()).union(self._pending_instrument_unsubscriptions)

    def _get_pending_sub_instr(self, sub: str) -> Set[Instrument]:
        for (sub_type, _), instrs in self._pending_stream_subscriptions.items():
            if sub_type == sub:
                return instrs
        return set()

    def _subscribe(
        self,
        instruments: Set[Instrument],
        sub_type: str,
        **kwargs,
    ) -> None:
        _subscriber = self._subscribers.get(sub_type)
        if _subscriber is None:
            raise ValueError(f"Subscription type {sub_type} is not supported")

        self._check_event_loop_is_running()

        _current_instruments = self._subscriptions[sub_type]
        _added_instruments = instruments.difference(_current_instruments)

        _channel = self.get_communication_channel()

        # - fetch historical ohlc for added instruments
        if (ohlc_warmup_period := kwargs.get("ohlc_warmup_period")) is not None and (
            timeframe := kwargs.get("timeframe")
        ) is not None:
            self._run_ohlc_warmup_sync(_channel, _added_instruments, timeframe, ohlc_warmup_period, timeout=60 * 10)

        self._resubscribe_stream(sub_type, _channel, instruments, **kwargs)
        self._subscriptions[sub_type] = instruments
        self._subscription_to_params[sub_type] = kwargs

    def _resubscribe_stream(self, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **kwargs) -> None:
        if sub_type in self._sub_to_coro:
            logger.debug(f"Canceling existing {sub_type} subscription for {self._subscriptions[sub_type]}")
            future = self._sub_to_coro[sub_type]
            future.cancel()

        if not instruments:
            del self._sub_to_coro[sub_type]
            return

        _subscriber = self._subscribers[sub_type]
        _subscriber_params = set(_subscriber.__code__.co_varnames[: _subscriber.__code__.co_argcount])
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

    def _get_updated_sub_to_params(self) -> Dict[str, Dict[str, Any]]:
        _update_subs = set(self._pending_stream_unsubscriptions.keys())
        if self._pending_instrument_unsubscriptions:
            _update_subs |= set(self._subscriptions.keys())
        _update_subs.update(sub for sub, _ in self._pending_stream_subscriptions)
        _sub_to_params = {
            **self._subscription_to_params,
            **{sub: sub_args for sub, sub_args in self._pending_stream_subscriptions.keys()},
        }
        _sub_to_params = {sub: self._subscription_to_params.get(sub, {}) for sub in _update_subs}
        return _sub_to_params

    def _submit_coro(self, coro: Awaitable[None]) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int) -> int:
        return (self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def _run_ohlc_warmup_sync(
        self, channel: CtrlChannel, instruments: Set[Instrument], timeframe: str, warmup_period: str, timeout: int
    ) -> None:
        future = self._submit_coro(self._run_ohlc_warmup(channel, instruments, timeframe, warmup_period))
        future.result(timeout)

    async def _run_ohlc_warmup(
        self, channel: CtrlChannel, instruments: Set[Instrument], timeframe: str, warmup_period: str
    ) -> None:
        logger.debug(f"Running OHLC warmup for {instruments} with period {warmup_period}")
        await asyncio.gather(
            *[self._stream_historical_ohlc(channel, instr, timeframe, warmup_period) for instr in instruments]
        )

    async def _listen_to_stream(
        self,
        subscriber: Callable[[], Awaitable[None]],
        exchange: Exchange,
        channel: CtrlChannel,
        name: str,
        unsubscriber: Callable[[], Awaitable[None]] | None = None,
    ):
        logger.debug(f"Listening to {name}")
        n_retry = 0
        while channel.control.is_set():
            try:
                await subscriber()
                n_retry = 0
            except CancelledError:
                if unsubscriber is not None:
                    try:
                        # - unsubscribe from stream, but ignore if there is an error
                        # because we anyway close the connection
                        await unsubscriber()
                    except:
                        pass
                break
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
            raise CcxtSymbolNotRecognized(f"Invalid exchange symbol {exch_symbol}")
        base = match.group("base")
        quote = match.group("quote")
        symbol = f"{base}{quote}"
        if symbol not in symbol_to_instrument:
            raise CcxtSymbolNotRecognized(f"Unknown symbol {symbol}")
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
                                (
                                    instrument,
                                    "bar",
                                    Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]),
                                )
                            )
                        last_close = ohlcvs[-1][4]
                        self.trading_service.update_position_price(
                            instrument,
                            _time,
                            last_close,
                        )
                except CcxtSymbolNotRecognized as e:
                    logger.warning(e)

        async def un_watch_ohlcv():
            unwatch = getattr(self._exchange, "un_watch_ohlcv_for_symbols", lambda _: None)(_symbol_timeframe_pairs)
            if unwatch is not None:
                await unwatch

        if warmup_period is not None:
            await self._run_ohlc_warmup(channel, instruments, timeframe, warmup_period)

        await self._listen_to_stream(
            watch_ohlcv, self._exchange, channel, f"{','.join(symbols)} {timeframe} OHLCV", unsubscriber=un_watch_ohlcv
        )

    async def _subscribe_trade(
        self,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        warmup_period: str | None = None,
    ):
        symbols = [i.symbol for i in instruments]
        _symbol_to_instrument = {i.symbol: i for i in instruments}

        async def watch_trades():
            trades = await self._exchange.watch_trades_for_symbols(symbols)
            symbol = trades[0]["symbol"]
            instrument = self._find_instrument_for_exch_symbol(symbol, _symbol_to_instrument)
            last_trade = ccxt_convert_trade(trades[-1])
            self.trading_service.update_position_price(instrument, self.time(), last_trade)
            for trade in trades:
                channel.send((instrument, "trade", ccxt_convert_trade(trade)))

        async def un_watch_trades():
            unwatch = getattr(self._exchange, "un_watch_trades_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

        # TODO: add historical trade fetching
        await self._listen_to_stream(
            watch_trades,
            self._exchange,
            channel,
            f"{','.join(symbols)} trades",
            unsubscriber=un_watch_trades,
        )

    async def _subscribe_orderbook(
        self,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        warmup_period: str | None = None,
    ):
        symbols = [i.symbol for i in instruments]
        _symbol_to_instrument = {i.symbol: i for i in instruments}

        async def watch_orderbook():
            ccxt_ob = await self._exchange.watch_order_book_for_symbols(symbols)
            exch_symbol = ccxt_ob["symbol"]
            instrument = self._find_instrument_for_exch_symbol(exch_symbol, _symbol_to_instrument)
            ob = ccxt_convert_orderbook(ccxt_ob, instrument)
            if ob is None:
                return
            quote = ob.to_quote()
            self._last_quotes[instrument.symbol] = quote
            self.trading_service.update_position_price(instrument, self.time(), quote)
            channel.send((instrument, "orderbook", ob))

        async def un_watch_orderbook():
            unwatch = getattr(self._exchange, "un_watch_order_book_for_symbols", lambda _: None)(symbols)
            logger.debug(f"Unwatching orderbook for {symbols}")
            if unwatch is not None:
                await unwatch

        # TODO: add historical orderbook fetching
        await self._listen_to_stream(
            watch_orderbook,
            self._exchange,
            channel,
            f"{','.join(symbols)} orderbook",
            unsubscriber=un_watch_orderbook,
        )

    async def _subscribe_executions(self, channel: CtrlChannel, instruments: List[Instrument]):
        # unfortunately binance and probably some others don't support watching orders for multiple symbols
        # so we need to watch orders for each symbol separately
        async def _watch_executions(instrument: Instrument):
            exec = await self._exchange.watch_orders(instrument.symbol)
            _msg = f"\nexecs_{instrument.symbol} = [\n"
            for report in exec:
                _msg += "\t" + str(report) + ",\n"
                order, deals = self.trading_service.process_execution_report(instrument, report)
                # - send update to client
                channel.send((instrument, "order", order))
                if deals:
                    channel.send((instrument, "deals", deals))
            # logger.debug(_msg + "]\n")

        # also I didn't find any unsubscribe method for watching orders
        tasks = [
            self._listen_to_stream(
                lambda i=instrument: _watch_executions(i),
                self._exchange,
                channel,
                f"{instrument.symbol} execution reports",
            )
            for instrument in instruments
        ]
        await asyncio.gather(*tasks)
