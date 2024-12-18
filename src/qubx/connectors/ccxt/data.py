import asyncio
import concurrent.futures
import re
from asyncio.exceptions import CancelledError
from collections import defaultdict
from threading import Thread
from types import FunctionType
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

import ccxt.pro as cxp
from ccxt import (
    ExchangeClosedByUser,
    ExchangeError,
    ExchangeNotAvailable,
    NetworkError,
)
from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider, dt_64
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IDataProvider
from qubx.core.series import Bar, Quote
from qubx.utils.misc import AsyncThreadLoop

from .exceptions import CcxtLiquidationParsingError, CcxtSymbolNotRecognized
from .utils import (
    ccxt_convert_funding_rate,
    ccxt_convert_liquidation,
    ccxt_convert_orderbook,
    ccxt_convert_ticker,
    ccxt_convert_trade,
    ccxt_find_instrument,
    instrument_to_ccxt_symbol,
)


class CcxtDataProvider(IDataProvider):
    time_provider: ITimeProvider
    _exchange: Exchange
    _scheduler: BasicScheduler | None = None

    # - subscriptions
    _subscriptions: Dict[str, Set[Instrument]]
    _sub_to_coro: Dict[str, concurrent.futures.Future]
    _sub_to_name: Dict[str, str]
    _sub_to_unsubscribe: Dict[str, Callable[[], Awaitable[None]]]
    _is_sub_name_enabled: Dict[str, bool]

    _sub_instr_to_time: Dict[Tuple[str, Instrument], dt_64]
    _last_quotes: Dict[Instrument, Optional[Quote]]
    _loop: AsyncThreadLoop
    _thread_event_loop: Thread
    _warmup_timeout: int

    _subscribers: Dict[str, Callable]
    _warmupers: Dict[str, Callable]

    def __init__(
        self,
        exchange: cxp.Exchange,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
        max_ws_retries: int = 10,
        warmup_timeout: int = 120,
    ):
        self._exchange_id = str(exchange.name)
        self.time_provider = time_provider
        self.channel = channel
        self.max_ws_retries = max_ws_retries
        self._warmup_timeout = warmup_timeout

        # - create new even loop
        self._exchange = exchange
        self._loop = AsyncThreadLoop(self._exchange.asyncio_loop)

        self._last_quotes = defaultdict(lambda: None)
        self._subscriptions = defaultdict(set)
        self._sub_to_coro = {}
        self._sub_to_name = {}
        self._sub_to_unsubscribe = {}
        self._is_sub_name_enabled = defaultdict(lambda: False)
        self._symbol_to_instrument = {}
        self._subscribers = {
            n.split("_subscribe_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) is FunctionType and n.startswith("_subscribe_")
        }
        self._warmupers = {
            n.split("_warmup_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) is FunctionType and n.startswith("_warmup_")
        }
        logger.info(f"Initialized {self._exchange_id}")

    @property
    def is_simulation(self) -> bool:
        return False

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
        # _current_instruments = self.get_subscribed_instruments(subscription_type)
        # _updated_instruments = set(_current_instruments).difference(instruments)
        # self._subscribe(_updated_instruments, subscription_type)
        # unsubscribe functionality is handled for ccxt via subscribe with reset=True
        pass

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        if instrument is not None:
            return [sub for sub, instrs in self._subscriptions.items() if instrument in instrs]
        return [sub for sub, instruments in self._subscriptions.items() if instruments]

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        if not subscription_type:
            return list(self.subscribed_instruments)
        return list(self._subscriptions[subscription_type]) if subscription_type in self._subscriptions else []

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        sub = subscription_type.lower()
        return sub in self._subscriptions and instrument in self._subscriptions[sub]

    def warmup(self, warmups: Dict[Tuple[str, Instrument], str]) -> None:
        _coros = []

        for (sub_type, instrument), period in warmups.items():
            _sub_type, _params = DataType.from_str(sub_type)
            _warmuper = self._warmupers.get(_sub_type)
            if _warmuper is None:
                logger.warning(f"Warmup for {sub_type} is not supported")
                continue
            _coros.append(
                _warmuper(
                    self,
                    channel=self.channel,
                    instrument=instrument,
                    warmup_period=period,
                    **_params,
                )
            )

        async def gather_coros():
            return await asyncio.gather(*_coros)

        if _coros:
            self._loop.submit(gather_coros()).result(self._warmup_timeout)

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument]

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> List[Bar]:
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

        res = self._loop.submit(_get()).result(60)

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
                future = self._loop.submit(self._exchange.close())  # type: ignore
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
        _sub_type, _params = DataType.from_str(sub_type)
        _subscriber = self._subscribers.get(_sub_type)
        if _subscriber is None:
            raise ValueError(f"Subscription type {sub_type} is not supported")

        if sub_type in self._sub_to_coro:
            logger.debug(f"Canceling existing {sub_type} subscription for {self._subscriptions[sub_type]}")
            self._loop.submit(self._stop_subscriber(sub_type, self._sub_to_name[sub_type]))
            del self._sub_to_coro[sub_type]
            del self._sub_to_name[sub_type]

        if instruments is not None and len(instruments) == 0:
            return

        kwargs = {"instruments": instruments, **_params}
        _subscriber = self._subscribers[_sub_type]
        _subscriber_params = set(_subscriber.__code__.co_varnames[: _subscriber.__code__.co_argcount])
        # - get only parameters that are needed for subscriber
        kwargs = {k: v for k, v in kwargs.items() if k in _subscriber_params}
        self._sub_to_name[sub_type] = (name := self._get_subscription_name(_sub_type, **kwargs))
        self._sub_to_coro[sub_type] = self._loop.submit(_subscriber(self, name, _sub_type, self.channel, **kwargs))

        self._subscriptions[sub_type] = instruments

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int = 1) -> int:
        return (self.time_provider.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def _get_exch_timeframe(self, timeframe: str) -> str:
        if timeframe is not None:
            _t = re.match(r"(\d+)(\w+)", timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self._exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self._exchange.name}")

        return tframe

    def _get_exch_symbol(self, instrument: Instrument) -> str:
        return f"{instrument.base}/{instrument.quote}:{instrument.settle}"

    def _get_subscription_name(
        self, subscription: str, instruments: List[Instrument] | Set[Instrument] | Instrument | None = None, **kwargs
    ) -> str:
        if isinstance(instruments, Instrument):
            instruments = [instruments]
        _symbols = [instrument_to_ccxt_symbol(i) for i in instruments] if instruments is not None else []
        _name = f"{','.join(_symbols)} {subscription}" if _symbols else subscription
        if kwargs:
            kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            _name += f" ({kwargs_str})"
        return _name

    async def _stop_subscriber(self, sub_type: str, sub_name: str) -> None:
        try:
            self._is_sub_name_enabled[sub_name] = False  # stop the subscriber
            future = self._sub_to_coro[sub_type]
            total_sleep_time = 0.0
            while future.running():
                await asyncio.sleep(1.0)
                total_sleep_time += 1.0
                if total_sleep_time >= 20.0:
                    break

            if future.running():
                logger.warning(f"Subscriber {sub_name} is still running. Cancelling it.")
                future.cancel()
            else:
                logger.debug(f"Subscriber {sub_name} has been stopped")

            if sub_name in self._sub_to_unsubscribe:
                logger.debug(f"Unsubscribing from {sub_name}")
                await self._sub_to_unsubscribe[sub_name]()
                del self._sub_to_unsubscribe[sub_name]

            del self._is_sub_name_enabled[sub_name]
            logger.debug(f"Unsubscribed from {sub_name}")
        except Exception as e:
            logger.error(f"Error stopping {sub_name}")
            logger.exception(e)

    async def _listen_to_stream(
        self,
        subscriber: Callable[[], Awaitable[None]],
        exchange: Exchange,
        channel: CtrlChannel,
        name: str,
        unsubscriber: Callable[[], Awaitable[None]] | None = None,
    ):
        logger.info(f"Listening to {name}")
        if unsubscriber is not None:
            self._sub_to_unsubscribe[name] = unsubscriber

        self._is_sub_name_enabled[name] = True
        n_retry = 0
        while channel.control.is_set() and self._is_sub_name_enabled[name]:
            try:
                await subscriber()
                n_retry = 0
                if not self._is_sub_name_enabled[name]:
                    break
            except CcxtSymbolNotRecognized:
                continue
            except CancelledError:
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
                if not channel.control.is_set() or not self._is_sub_name_enabled[name]:
                    # If the channel is closed, then ignore all exceptions and exit
                    break
                logger.error(f"Exception in {name}")
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
                DataType.OHLC[timeframe],
                [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv],
                True,
            )
        )

    async def _warmup_trade(self, channel: CtrlChannel, instrument: Instrument, warmup_period: str):
        trades = await self._exchange.fetch_trades(instrument.symbol, since=self._time_msec_nbars_back(warmup_period))
        logger.debug(f"Loaded {len(trades)} trades for {instrument}")
        channel.send(
            (
                instrument,
                DataType.TRADE,
                [ccxt_convert_trade(trade) for trade in trades],
                True,
            )
        )

    def _call_by_market_type(
        self, subscriber: Callable[[list[Instrument]], Awaitable[None]], instruments: set[Instrument]
    ) -> Any:
        """Call subscriber for each market type"""
        _instr_by_type: dict[str, list[Instrument]] = defaultdict(list)
        for instr in instruments:
            _instr_by_type[instr.market_type].append(instr)

        # sort instruments by symbol
        for instrs in _instr_by_type.values():
            instrs.sort(key=lambda i: i.symbol)

        async def _call_subscriber():
            await asyncio.gather(*[subscriber(instrs) for instrs in _instr_by_type.values()])

        return _call_subscriber

    #############################
    # - Subscription methods
    #############################
    async def _subscribe_ohlc(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        timeframe: str = "1m",
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _exchange_timeframe = self._get_exch_timeframe(timeframe)
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_ohlcv(instruments: list[Instrument]):
            _symbol_timeframe_pairs = [[_instr_to_ccxt_symbol[i], _exchange_timeframe] for i in instruments]
            ohlcv = await self._exchange.watch_ohlcv_for_symbols(_symbol_timeframe_pairs)
            # - ohlcv is symbol -> timeframe -> list[timestamp, open, high, low, close, volume]
            for exch_symbol, _data in ohlcv.items():
                instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                for _, ohlcvs in _data.items():
                    for oh in ohlcvs:
                        channel.send(
                            (
                                instrument,
                                sub_type,
                                Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]),
                                False,  # not historical bar
                            )
                        )

        # ohlc subscription reuses the same connection always, unsubscriptions don't work properly
        # but it's likely not very needed
        # async def un_watch_ohlcv(instruments: list[Instrument]):
        #     _symbol_timeframe_pairs = [[_instr_to_ccxt_symbol[i], _exchange_timeframe] for i in instruments]
        #     await self._exchange.un_watch_ohlcv_for_symbols(_symbol_timeframe_pairs)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_ohlcv, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            # unsubscriber=self._call_by_market_type(un_watch_ohlcv, instruments),
        )

    async def _subscribe_trade(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_trades(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            trades = await self._exchange.watch_trades_for_symbols(symbols)
            exch_symbol = trades[0]["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
            for trade in trades:
                channel.send((instrument, sub_type, ccxt_convert_trade(trade), False))

        async def un_watch_trades(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            await self._exchange.un_watch_trades_for_symbols(symbols)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_trades, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_trades, instruments),
        )

    async def _subscribe_orderbook(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_orderbook(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            ccxt_ob = await self._exchange.watch_order_book_for_symbols(symbols)
            exch_symbol = ccxt_ob["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
            ob = ccxt_convert_orderbook(ccxt_ob, instrument)
            if ob is None:
                return
            quote = ob.to_quote()
            self._last_quotes[instrument] = quote
            channel.send((instrument, sub_type, ob, False))

        async def un_watch_orderbook(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            await self._exchange.un_watch_order_book_for_symbols(symbols)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_orderbook, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_orderbook, instruments),
        )

    async def _subscribe_quote(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_quote(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            ccxt_tickers: dict[str, dict] = await self._exchange.watch_tickers(symbols)
            for exch_symbol, ccxt_ticker in ccxt_tickers.items():
                instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                quote = ccxt_convert_ticker(ccxt_ticker, instrument)
                self._last_quotes[instrument] = quote
                channel.send((instrument, sub_type, quote, False))

        async def un_watch_quote(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            await self._exchange.un_watch_tickers(symbols)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_quote, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_quote, instruments),
        )

    async def _subscribe_liquidation(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_liquidation(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            liquidations = await self._exchange.watch_liquidations_for_symbols(symbols)
            for liquidation in liquidations:
                try:
                    instrument = ccxt_find_instrument(liquidation["symbol"], self._exchange, _symbol_to_instrument)
                    channel.send((instrument, sub_type, ccxt_convert_liquidation(liquidation), False))
                except CcxtLiquidationParsingError:
                    logger.debug(f"Could not parse liquidation {liquidation}")
                    continue

        async def un_watch_liquidation(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            unwatch = getattr(self._exchange, "un_watch_liquidations_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

        # - fetching of liquidations for warmup is not supported by ccxt
        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_liquidation, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_liquidation, instruments),
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
                    instrument = ccxt_find_instrument(symbol, self._exchange)
                    instrument_to_funding_rate[instrument] = ccxt_convert_funding_rate(info)
                except CcxtSymbolNotRecognized:
                    continue
            channel.send((None, sub_type, instrument_to_funding_rate, False))

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
