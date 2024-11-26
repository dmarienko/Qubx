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
from ccxt import NetworkError, ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, BadSymbol
from ccxt.base.exchange import Exchange

from qubx import logger
from qubx.core.basics import Instrument, Position, dt_64, Deal, CtrlChannel, Subtype
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IBrokerServiceProvider, ITradingServiceProvider, ITimeProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.utils.ntp import start_ntp_thread, time_now
from .exceptions import CcxtSymbolNotRecognized, CcxtLiquidationParsingError
from .utils import (
    ccxt_convert_trade,
    ccxt_convert_orderbook,
    ccxt_convert_liquidation,
    ccxt_convert_funding_rate,
    ccxt_symbol_info_to_instrument,
)


EXCH_SYMBOL_PATTERN = re.compile(r"(?P<base>[^/]+)/(?P<quote>[^:]+)(?::(?P<margin>.+))?")


class CcxtBrokerServiceProvider(IBrokerServiceProvider):
    _exchange: Exchange
    _scheduler: BasicScheduler | None = None

    # - subscriptions
    _subscriptions: Dict[str, Set[Instrument]]
    _sub_to_coro: Dict[str, concurrent.futures.Future]
    _sub_to_name: Dict[str, str]
    _is_sub_name_enabled: Dict[str, bool]

    _sub_instr_to_time: Dict[Tuple[str, Instrument], dt_64]
    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop
    _thread_event_loop: Thread
    _warmup_timeout: int

    _subscribers: Dict[str, Callable]
    _warmupers: Dict[str, Callable]

    def __init__(
        self,
        exchange: cxp.Exchange,
        trading_service: ITradingServiceProvider,
        max_ws_retries: int = 10,
        warmup_timeout: int = 120,
    ):
        super().__init__(str(exchange.name), trading_service)
        self.trading_service = trading_service
        self.max_ws_retries = max_ws_retries
        self._warmup_timeout = warmup_timeout

        # - start NTP thread
        start_ntp_thread()

        # - setup communication bus
        self.set_communication_channel(bus := CtrlChannel("databus", sentinel=(None, None, None)))
        self.trading_service.set_communication_channel(bus)

        # - create new even loop
        self._exchange = exchange
        self._loop = self._exchange.asyncio_loop

        self._last_quotes = defaultdict(lambda: None)
        self._subscriptions = defaultdict(set)
        self._sub_to_coro = {}
        self._sub_to_name = {}
        self._is_sub_name_enabled = defaultdict(lambda: False)
        self._symbol_to_instrument = {}

        self._subscribers = {
            n.split("_subscribe_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) == FunctionType and n.startswith("_subscribe_")
        }
        self._warmupers = {
            n.split("_warmup_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) == FunctionType and n.startswith("_warmup_")
        }

        if not self.is_read_only:
            self._subscribe_stream("executions", self.get_communication_channel())

        logger.info(f"Initialized {self._exchange_id}")

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
        subscription_type: str,
        instruments: List[Instrument],
        reset: bool = False,
    ) -> None:
        _updated_instruments = set(instruments)

        # - update symbol to instrument mapping
        self._symbol_to_instrument.update({i.symbol: i for i in instruments})

        # - add existing subscription instruments if reset is False
        if not reset:
            _current_instruments = self.get_subscribed_instruments(subscription_type)
            _updated_instruments = _updated_instruments.union(_current_instruments)

        # - update subscriptions
        self._subscribe(_updated_instruments, subscription_type)

    def unsubscribe(self, subscription_type: str, instruments: List[Instrument]) -> None:
        _current_instruments = self.get_subscribed_instruments(subscription_type)
        _updated_instruments = set(_current_instruments).difference(instruments)
        self._subscribe(_updated_instruments, subscription_type)

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        return (
            [sub for sub, instrs in self._subscriptions.items() if instrument in instrs]
            if instrument is not None
            else list(self._subscriptions.keys())
        )

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> List[Instrument]:
        return (
            list(self._subscriptions.get(subscription_type, set()))
            if subscription_type is not None
            else list(self.subscribed_instruments)
        )

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        sub = subscription_type.lower()
        return sub in self._subscriptions and instrument in self._subscriptions[sub]

    def warmup(self, warmups: Dict[Tuple[str, Instrument], str]) -> None:
        _coros = []

        for (sub_type, instrument), period in warmups.items():
            _sub_type, _params = Subtype.from_str(sub_type)
            _warmuper = self._warmupers.get(_sub_type)
            if _warmuper is None:
                logger.warning(f"Warmup for {sub_type} is not supported")
                continue
            _coros.append(
                _warmuper(
                    self,
                    channel=self.get_communication_channel(),
                    instrument=instrument,
                    warmup_period=period,
                    **_params,
                )
            )

        async def gather_coros():
            return await asyncio.gather(*_coros)

        if _coros:
            asyncio.run_coroutine_threadsafe(gather_coros(), self._loop).result(self._warmup_timeout)

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
                future = self._task_s(self._exchange.close())  # type: ignore
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

    @property
    def is_read_only(self) -> bool:
        _key = self._exchange.apiKey
        return _key is None or _key == ""

    def _subscribe(
        self,
        instruments: Set[Instrument],
        sub_type: str,
    ) -> None:
        _sub_type, _params = Subtype.from_str(sub_type)
        _subscriber = self._subscribers.get(_sub_type)
        if _subscriber is None:
            raise ValueError(f"Subscription type {sub_type} is not supported")
        _channel = self.get_communication_channel()
        self._resubscribe_stream(_sub_type, _channel, instruments, **_params)
        self._subscriptions[sub_type] = instruments

    def _resubscribe_stream(
        self, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument] | None = None, **kwargs
    ) -> None:
        if sub_type in self._sub_to_coro:
            logger.debug(f"Canceling existing {sub_type} subscription for {self._subscriptions[sub_type]}")
            self._stop_subscriber(sub_type)
        if instruments is not None and len(instruments) == 0:
            return
        self._subscribe_stream(sub_type, channel, instruments=instruments, **kwargs)

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

    def _subscribe_stream(self, sub_type: str, channel: CtrlChannel, **kwargs) -> None:
        _subscriber = self._subscribers[sub_type]
        _subscriber_params = set(_subscriber.__code__.co_varnames[: _subscriber.__code__.co_argcount])
        # - get only parameters that are needed for subscriber
        kwargs = {k: v for k, v in kwargs.items() if k in _subscriber_params}
        self._sub_to_name[sub_type] = (name := self._get_subscription_name(sub_type, **kwargs))
        self._sub_to_coro[sub_type] = self._task_s(_subscriber(self, name, sub_type, channel, **kwargs))

    def _task_s(self, coro: Awaitable[None]) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int = 1) -> int:
        return (self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

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

    def _get_instrument(self, symbol: str, symbol_to_instrument: Dict[str, Instrument] | None = None) -> Instrument:
        instrument = self._symbol_to_instrument.get(symbol)
        if instrument is None and symbol_to_instrument is not None:
            try:
                instrument = self._find_instrument_for_exch_symbol(symbol, symbol_to_instrument)
            except CcxtSymbolNotRecognized:
                pass
        if instrument is None:
            try:
                symbol_info = self._exchange.market(symbol)
            except BadSymbol:
                raise CcxtSymbolNotRecognized(f"Unknown symbol {symbol}")
            instrument = ccxt_symbol_info_to_instrument(self._exchange_id, symbol_info)
        if symbol not in self._symbol_to_instrument:
            self._symbol_to_instrument[symbol] = instrument
        return instrument

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
        self, subscription: str, instruments: List[Instrument] | Set[Instrument] | Instrument | None = None, **kwargs
    ) -> str:
        if isinstance(instruments, Instrument):
            instruments = [instruments]
        _symbols = [i.symbol for i in instruments] if instruments is not None else []
        _name = f"{','.join(_symbols)} {subscription}" if _symbols else subscription
        if kwargs:
            kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            _name += f" ({kwargs_str})"
        return _name

    def _get_hist_type(self, sub_type: str) -> str:
        return f"hist_{sub_type}"

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
            except CcxtSymbolNotRecognized as e:
                continue
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

    #############################
    # - Warmup methods
    #############################
    async def _warmup_ohlc(
        self, channel: CtrlChannel, instrument: Instrument, warmup_period: str, timeframe: str
    ) -> None:
        nbarsback = pd.Timedelta(warmup_period) // pd.Timedelta(timeframe)
        exch_timeframe = self._get_exch_timeframe(timeframe)
        start = self._time_msec_nbars_back(timeframe, nbarsback)
        ohlcv = await self._exchange.fetch_ohlcv(instrument.symbol, exch_timeframe, since=start, limit=nbarsback + 1)
        logger.debug(f"{instrument}: loaded {len(ohlcv)} {timeframe} bars")
        channel.send(
            (
                instrument,
                self._get_hist_type(Subtype.OHLC[timeframe]),
                [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv],
            )
        )

    async def _warmup_trade(self, channel: CtrlChannel, instrument: Instrument, warmup_period: str):
        trades = await self._exchange.fetch_trades(instrument.symbol, since=self._time_msec_nbars_back(warmup_period))
        logger.debug(f"Loaded {len(trades)} trades for {instrument}")
        channel.send(
            (
                instrument,
                self._get_hist_type(Subtype.TRADE),
                [ccxt_convert_trade(trade) for trade in trades],
            )
        )

    #############################
    # - Subscription methods
    #############################
    async def _subscribe_executions(self, name: str, sub_type: str, channel: CtrlChannel):
        async def _watch_executions():
            exec = await self._exchange.watch_orders()
            for report in exec:
                instrument = self._get_instrument(report["symbol"], self._symbol_to_instrument)
                order, deals = self.trading_service.process_execution_report(instrument, report)
                channel.send((instrument, "order", order))
                if deals:
                    channel.send((instrument, "deals", deals))

        await self._listen_to_stream(
            subscriber=_watch_executions,
            exchange=self._exchange,
            channel=channel,
            name=name,
        )

    async def _subscribe_ohlc(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
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
                instrument = self._get_instrument(exch_symbol, _symbol_to_instrument)
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
                    # TODO: move trading server update to context impl
                    self.trading_service.update_position_price(
                        instrument,
                        _time,
                        last_close,
                    )

        async def un_watch_ohlcv():
            unwatch = getattr(self._exchange, "un_watch_ohlcv_for_symbols", lambda _: None)(_symbol_timeframe_pairs)
            if unwatch is not None:
                await unwatch

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
    ):
        symbols = [i.symbol for i in instruments]
        _symbol_to_instrument = {i.symbol: i for i in instruments}

        async def watch_trades():
            trades = await self._exchange.watch_trades_for_symbols(symbols)
            symbol = trades[0]["symbol"]
            _time = self.time()
            instrument = self._get_instrument(symbol, _symbol_to_instrument)
            last_trade = ccxt_convert_trade(trades[-1])
            # TODO: move trading server update to context impl
            self.trading_service.update_position_price(instrument, _time, last_trade)
            for trade in trades:
                channel.send((instrument, sub_type, ccxt_convert_trade(trade)))

        async def un_watch_trades():
            unwatch = getattr(self._exchange, "un_watch_trades_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

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
    ):
        symbols = [i.symbol for i in instruments]
        _symbol_to_instrument = {i.symbol: i for i in instruments}

        async def watch_orderbook():
            ccxt_ob = await self._exchange.watch_order_book_for_symbols(symbols)
            exch_symbol = ccxt_ob["symbol"]
            instrument = self._get_instrument(exch_symbol, _symbol_to_instrument)
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

        # - fetching of orderbooks for warmup is not supported by ccxt
        await self._listen_to_stream(
            subscriber=watch_orderbook,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_orderbook,
        )

    async def _subscribe_liquidation(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        symbols = [i.symbol for i in instruments]
        _symbol_to_instrument = {i.symbol: i for i in instruments}

        async def watch_liquidation():
            liquidations = await self._exchange.watch_liquidations_for_symbols(symbols)
            for liquidation in liquidations:
                try:
                    instrument = self._get_instrument(liquidation["symbol"], _symbol_to_instrument)
                    channel.send((instrument, sub_type, ccxt_convert_liquidation(liquidation)))
                except CcxtLiquidationParsingError:
                    logger.debug(f"Could not parse liquidation {liquidation}")
                    continue

        async def un_watch_liquidation():
            unwatch = getattr(self._exchange, "un_watch_liquidations_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

        # - fetching of liquidations for warmup is not supported by ccxt
        await self._listen_to_stream(
            subscriber=watch_liquidation,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_liquidation,
        )

    async def _subscribe_funding_rate(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
    ):
        # it is expected that we can retrieve funding rates for all instruments
        async def watch_funding_rates():
            funding_rates = await self._exchange.watch_funding_rates()  # type: ignore
            instrument_to_funding_rate = {}
            for symbol, info in funding_rates.items():
                try:
                    instrument = self._get_instrument(symbol)
                    instrument_to_funding_rate[instrument] = ccxt_convert_funding_rate(info)
                except CcxtSymbolNotRecognized as e:
                    continue
            channel.send((None, sub_type, instrument_to_funding_rate))

        async def un_watch_funding_rates():
            unwatch = getattr(self._exchange, "un_watch_funding_rates", lambda: None)()
            if unwatch is not None:
                await unwatch

        await self._listen_to_stream(
            subscriber=watch_funding_rates,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_funding_rates,
        )
