import asyncio
import re
import numpy as np
import pandas as pd
import ccxt.pro as cxp
import concurrent.futures

from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple, Set
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from asyncio.exceptions import CancelledError
from collections import defaultdict
from types import FunctionType
from ccxt import NetworkError, ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable
from ccxt.base.exchange import Exchange

from qubx import logger
from qubx.core.basics import Instrument, Position, dt_64, Deal, CtrlChannel, SubscriptionType
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IBrokerServiceProvider, ITradingServiceProvider, ITimeProvider
from qubx.utils.threading import synchronized
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


class ProxyCtrlChannel(CtrlChannel):
    """
    ProxyCtrlChannel is a wrapper around CtrlChannel that tracks the time of the last message sent.
    """

    control: Event
    name: str
    lock: Lock

    _channel: CtrlChannel
    _time_provider: ITimeProvider
    _sub_instr_to_time: Dict[Tuple[str, Instrument], dt_64]

    def __init__(
        self, channel: CtrlChannel, time_provider: ITimeProvider, sub_instr_to_time: Dict[Tuple[str, Instrument], dt_64]
    ):
        self.name = channel.name
        self.control = channel.control
        self.lock = channel.lock
        self._channel = channel
        self._time_provider = time_provider
        self._sub_instr_to_time = sub_instr_to_time

    def send(self, msg: Tuple):
        self._channel.send(msg)
        if len(msg) != 3:
            raise ValueError(f"Invalid message {msg}")
        instr, sub_type, _ = msg
        self._sub_instr_to_time[(sub_type, instr)] = self._time_provider.time()


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
    _sub_to_name: Dict[str, str]
    _is_sub_name_enabled: Dict[str, bool]

    _sub_instr_to_time: Dict[Tuple[str, Instrument], dt_64]
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
        use_testnet: bool = False,
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
        self._check_event_loop_is_running()

        # - create exchange's instance
        self._exchange = getattr(cxp, exch)(exchange_auth | {"asyncio_loop": self._loop})
        if use_testnet:
            self._exchange.set_sandbox_mode(True)

        self._last_quotes = defaultdict(lambda: None)
        self._subscriptions = defaultdict(set)
        self._subscription_to_params = defaultdict(dict)
        self._pending_stream_subscriptions = defaultdict(set)
        self._pending_stream_unsubscriptions = defaultdict(set)
        self._pending_instrument_unsubscriptions = set()
        self._sub_to_coro = {}
        self._sub_to_name = {}
        self._is_sub_name_enabled = defaultdict(lambda: False)
        self._sub_instr_to_time = {}

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

    @synchronized
    def commit(self) -> None:
        _current_instruments = self.subscribed_instruments
        _sub_to_params = self._get_updated_sub_to_params()
        _has_something_changed = False
        for sub, params in _sub_to_params.items():
            _current_sub_instruments = self._subscriptions[sub]
            _removed_instruments = self._get_pending_unsub_instr(sub)
            _added_instruments = self._get_pending_sub_instr(sub)
            _updated_instruments = _current_sub_instruments.union(_added_instruments).difference(_removed_instruments)
            if _updated_instruments != _current_sub_instruments:
                _has_something_changed = True
                self._subscribe(_updated_instruments, sub, **params)

        if _has_something_changed and not self.read_only:
            _new_instruments = self.subscribed_instruments
            _removed_instruments = _current_instruments.difference(_new_instruments)
            _added_instruments = _new_instruments.difference(_current_instruments)
            self._update_execution_subscriptions(_added_instruments, _removed_instruments)

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
        if not self._subscriptions:
            return set()
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

        _current_instruments = self._subscriptions[sub_type]
        _added_instruments = instruments.difference(_current_instruments)

        _channel = self._get_proxy_channel()

        # - fetch historical ohlc for added instruments
        if (ohlc_warmup_period := kwargs.get("ohlc_warmup_period")) is not None and (
            timeframe := kwargs.get("timeframe")
        ) is not None:
            self._run_ohlc_warmup_sync(_channel, _added_instruments, timeframe, ohlc_warmup_period, timeout=60 * 10)

        self._resubscribe_stream(sub_type, _channel, instruments, **kwargs)
        self._subscriptions[sub_type] = instruments
        self._subscription_to_params[sub_type] = kwargs

    def _stop_subscriber(self, sub_type: str) -> None:
        if sub_type not in self._sub_to_coro:
            return
        sub_name = self._sub_to_name[sub_type]
        self._is_sub_name_enabled[sub_name] = False  # stop the subscriber
        future = self._sub_to_coro[sub_type]
        future.result(10)  # wait for 10 seconds for the future to finish
        if future.running():
            future.cancel()
        del self._sub_to_coro[sub_type]
        del self._sub_to_name[sub_type]
        del self._is_sub_name_enabled[sub_name]

    def _get_proxy_channel(self) -> ProxyCtrlChannel:
        return ProxyCtrlChannel(self.get_communication_channel(), self, self._sub_instr_to_time)

    def _resubscribe_stream(self, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **kwargs) -> None:
        if sub_type in self._sub_to_coro:
            logger.debug(f"Canceling existing {sub_type} subscription for {self._subscriptions[sub_type]}")
            self._stop_subscriber(sub_type)

        if not instruments:
            return

        _subscriber = self._subscribers[sub_type]
        _subscriber_params = set(_subscriber.__code__.co_varnames[: _subscriber.__code__.co_argcount])
        # - get only parameters that are needed for subscriber
        kwargs = {k: v for k, v in kwargs.items() if k in _subscriber_params}
        self._sub_to_name[sub_type] = (name := self._get_subscription_name(instruments, sub_type, **kwargs))
        self._sub_to_coro[sub_type] = self._submit_coro(
            _subscriber(self, name, sub_type, channel, instruments, **kwargs)
        )

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
            **{sub: dict(sub_args) for sub, sub_args in self._pending_stream_subscriptions.keys()},
        }
        _sub_to_params = {sub: _sub_to_params.get(sub, {}) for sub in _update_subs}
        return _sub_to_params

    def _submit_coro(self, coro: Awaitable[None]) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int = 1) -> int:
        return (self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def _run_ohlc_warmup_sync(
        self, channel: CtrlChannel, instruments: Set[Instrument], timeframe: str, warmup_period: str, timeout: int
    ) -> None:
        future = self._submit_coro(self._run_ohlc_warmup(channel, instruments, timeframe, warmup_period))
        future.result(timeout)

    async def _run_ohlc_warmup(
        self, channel: CtrlChannel, instruments: Set[Instrument], timeframe: str, warmup_period: str
    ) -> None:
        if not instruments:
            return
        logger.debug(f"Running OHLC warmup for {instruments} with period {warmup_period}")
        await asyncio.gather(
            *[self._fetch_and_send_ohlc(channel, instr, timeframe, warmup_period) for instr in instruments]
        )

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

    def _get_subscription_name(
        self, instruments: List[Instrument] | Set[Instrument] | Instrument, subscription: str, **kwargs
    ) -> str:
        if isinstance(instruments, Instrument):
            instruments = [instruments]
        _symbols = [i.symbol for i in instruments]
        _name = f"{','.join(_symbols)} {subscription}"
        if kwargs:
            kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            _name += f" ({kwargs_str})"
        return _name

    def _get_hist_type(self, sub_type: str) -> str:
        return f"hist_{sub_type}"

    def _update_execution_subscriptions(self, added_instruments: Set[Instrument], removed_instruments: Set[Instrument]):
        channel = self._get_proxy_channel()
        _get_name = lambda instr: self._get_subscription_name(instr, "executions")
        # - subscribe to added instruments
        for instr in added_instruments:
            _name = _get_name(instr)
            future = self._sub_to_coro.get(_name)
            if future is not None and future.running():
                logger.warning(f"Execution subscription for {instr} is already running")
                continue
            self._sub_to_coro[_name] = self._submit_coro(
                self._subscribe_executions(_name, "executions", channel, instr)
            )
        # - unsubscribe from removed instruments
        for instr in removed_instruments:
            _name = _get_name(instr)
            self._is_sub_name_enabled[_name] = False
            future = self._sub_to_coro[_name]
            try:
                # usually this will raise a timeout error because there will be no order updates
                # so just cancel
                future.result(0.1)
            except TimeoutError:
                future.cancel()
            finally:
                del self._sub_to_coro[_name]

    def _get_latest_instruments(self, sub_type: str, warmup_period: str) -> Set[Instrument]:
        """
        Get the latest instruments that have relatively fresh data that does not exceed 1/4 of the warmup period.
        """
        _oldest_allowed_time = (pd.Timestamp(self.time()) - pd.Timedelta(warmup_period) / 4).asm8
        return {
            instr
            for (_sub_type, instr), _time in self._sub_instr_to_time.items()
            if _sub_type == sub_type and _time > _oldest_allowed_time
        }

    async def _listen_to_stream(
        self,
        subscriber: Callable[[], Awaitable[None]],
        exchange: Exchange,
        channel: CtrlChannel,
        name: str,
        unsubscriber: Callable[[], Awaitable[None]] | None = None,
    ):
        logger.debug(f"Listening to {name}")
        self._is_sub_name_enabled[name] = True
        n_retry = 0
        while channel.control.is_set():
            try:
                await subscriber()
                n_retry = 0
                if not self._is_sub_name_enabled[name]:
                    if unsubscriber is not None:
                        await unsubscriber()
                    break
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

    async def _fetch_and_send_ohlc(
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
                self._get_hist_type(SubscriptionType.OHLC),
                [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv],
            )
        )
        logger.info(f"{instrument}: loaded {len(ohlcv)} {timeframe} bars")

    #############################
    # - Subscription methods
    #############################
    async def _subscribe_ohlc(
        self,
        name: str,
        sub_type: str,
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
                instrument = self._find_instrument_for_exch_symbol(exch_symbol, _symbol_to_instrument)
                for _, ohlcvs in _data.items():
                    for oh in ohlcvs:
                        channel.send(
                            (
                                instrument,
                                sub_type,
                                Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]),
                            )
                        )
                    last_close = ohlcvs[-1][4]
                    self.trading_service.update_position_price(
                        instrument,
                        _time,
                        last_close,
                    )

        async def un_watch_ohlcv():
            unwatch = getattr(self._exchange, "un_watch_ohlcv_for_symbols", lambda _: None)(_symbol_timeframe_pairs)
            if unwatch is not None:
                await unwatch

        if warmup_period is not None:
            _new_instruments = instruments.difference(self._get_latest_instruments(sub_type, warmup_period))
            await self._run_ohlc_warmup(channel, _new_instruments, timeframe, warmup_period)

        await self._listen_to_stream(
            subscriber=watch_ohlcv,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_ohlcv,
        )

    async def _subscribe_trade(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        warmup_period: str | None = None,
    ):
        symbols = [i.symbol for i in instruments]
        _symbol_to_instrument = {i.symbol: i for i in instruments}

        async def watch_trades():
            trades = await self._exchange.watch_trades_for_symbols(symbols)
            symbol = trades[0]["symbol"]
            _time = self.time()
            instrument = self._find_instrument_for_exch_symbol(symbol, _symbol_to_instrument)
            last_trade = ccxt_convert_trade(trades[-1])
            self.trading_service.update_position_price(instrument, _time, last_trade)
            for trade in trades:
                channel.send((instrument, sub_type, ccxt_convert_trade(trade)))

        async def un_watch_trades():
            unwatch = getattr(self._exchange, "un_watch_trades_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

        if warmup_period is not None:
            _new_instruments = instruments.difference(self._get_latest_instruments(sub_type, warmup_period))
            if _new_instruments:
                logger.debug(f"New instruments for trade warmup: {_new_instruments}")

                async def fetch_and_send_trades(instr: Instrument):
                    trades = await self._exchange.fetch_trades(
                        instr.symbol, since=self._time_msec_nbars_back(warmup_period)
                    )
                    logger.debug(f"Loaded {len(trades)} trades for {instr}")
                    channel.send(
                        (
                            instr,
                            self._get_hist_type(sub_type),
                            [ccxt_convert_trade(trade) for trade in trades],
                        )
                    )

                await asyncio.gather(*(fetch_and_send_trades(instr) for instr in _new_instruments))

        await self._listen_to_stream(
            subscriber=watch_trades,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_trades,
        )

    async def _subscribe_orderbook(
        self,
        name: str,
        sub_type: str,
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
            channel.send((instrument, sub_type, ob))

        async def un_watch_orderbook():
            unwatch = getattr(self._exchange, "un_watch_order_book_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

        # TODO: add historical orderbook fetching

        await self._listen_to_stream(
            subscriber=watch_orderbook,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_orderbook,
        )

    async def _subscribe_executions(self, name: str, sub_type: str, channel: CtrlChannel, instrument: Instrument):
        # unfortunately binance and probably some others don't support watching orders for multiple symbols
        # so we need to watch orders for each symbol separately
        async def _watch_executions():
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

        await self._listen_to_stream(
            subscriber=_watch_executions,
            exchange=self._exchange,
            channel=channel,
            name=name,
        )
