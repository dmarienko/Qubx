import asyncio
import concurrent.futures
from asyncio.exceptions import CancelledError
from collections import defaultdict
from typing import Awaitable, Callable

import numpy as np
import pandas as pd

import ccxt.pro as cxp
from ccxt import ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError
from qubx import logger
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    AssetBalance,
    CtrlChannel,
    Deal,
    Instrument,
    Order,
    Position,
    Subtype,
    dt_64,
)
from qubx.core.interfaces import ISubscriptionManager
from qubx.utils.marketdata.ccxt import ccxt_symbol_to_instrument
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.ntp import time_now

from .utils import (
    ccxt_convert_balance,
    ccxt_convert_positions,
    ccxt_convert_ticker,
    instrument_to_ccxt_symbol,
)


class CcxtAccountProcessor(BasicAccountProcessor):
    """
    Subscribes to account information from the exchange.
    """

    exchange: cxp.Exchange
    base_currency: str
    balance_interval: str
    position_interval: str
    subscription_interval: str
    max_retries: int

    _loop: AsyncThreadLoop
    _polling_tasks: dict[str, concurrent.futures.Future]
    _subscription_manager: ISubscriptionManager | None
    _polling_to_init: dict[str, bool]
    _required_instruments: set[Instrument]
    _latest_instruments: set[Instrument]

    _free_capital: float = np.nan
    _total_capital: float = np.nan
    _instrument_to_last_price: dict[Instrument, tuple[dt_64, float]]

    def __init__(
        self,
        account_id: str,
        exchange: cxp.Exchange,
        base_currency: str,
        balance_interval: str = "10Sec",
        position_interval: str = "10Sec",
        subscription_interval: str = "2Sec",
        max_retries: int = 10,
    ):
        super().__init__(account_id, base_currency, initial_capital=0)
        self.exchange = exchange
        self.max_retries = max_retries
        self.balance_interval = balance_interval
        self.position_interval = position_interval
        self.subscription_interval = subscription_interval
        self._loop = AsyncThreadLoop(exchange.asyncio_loop)
        self._is_running = False
        self._polling_tasks = {}
        self._polling_to_init = defaultdict(bool)
        self._instrument_to_last_price = {}
        self._required_instruments = set()
        self._latest_instruments = set()
        self._subscription_manager = None

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        self._subscription_manager = manager

    def start(self):
        """Start the balance and position polling tasks"""
        channel = self.get_communication_channel()
        if channel is None or not channel.control.is_set():
            return
        if self._subscription_manager is None:
            return
        if self._is_running:
            logger.debug("Account polling is already running")
            return
        self._is_running = True
        self._polling_tasks["balance"] = self._loop.submit(
            self._poller("balance", self._update_balance, self.balance_interval)
        )
        self._polling_tasks["position"] = self._loop.submit(
            self._poller("position", self._update_positions, self.position_interval)
        )
        self._polling_tasks["subscription"] = self._loop.submit(
            self._poller("subscription", self._update_subscriptions, self.subscription_interval)
        )
        logger.debug("Waiting for account polling tasks to be initialized")
        _waiter = self._loop.submit(self._wait_for_init())
        _waiter.result()

    def close(self):
        """Stop all polling tasks"""
        for task in self._polling_tasks.values():
            if not task.done():
                task.cancel()
        self._polling_tasks.clear()
        self._is_running = False

    def time(self) -> dt_64:
        return time_now()

    def update_position_price(self, time: dt_64, instrument: Instrument, price: float) -> None:
        self._instrument_to_last_price[instrument] = (time, price)
        super().update_position_price(time, instrument, price)

    def get_total_capital(self) -> float:
        # sum of balances + market value of all positions on non spot/margin
        _currency_to_value = {c: self._get_currency_value(b.total, c) for c, b in self._balances.items()}
        _positions_value = sum([p.market_value_funds for p in self._positions.values() if p.instrument.is_futures()])
        return sum(_currency_to_value.values()) + _positions_value

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        # do nothing on deal updates, because we are updating balances directly from exchange
        pass

    def process_order(self, order: Order) -> None:
        # don't update locked value, because we are updating balances directly from exchange
        super().process_order(order, update_locked_value=False)

    def _get_instrument_for_currency(self, currency: str) -> Instrument:
        symbol = f"{currency}/{self.base_currency}"
        market = self.exchange.market(symbol)
        exchange_name = self.exchange.name
        assert exchange_name is not None
        return ccxt_symbol_to_instrument(exchange_name, market)

    def _get_currency_value(self, amount: float, currency: str) -> float:
        if not amount:
            return 0.0
        if currency == self.base_currency:
            return amount
        instr = self._get_instrument_for_currency(currency)
        _dt, _price = self._instrument_to_last_price.get(instr, (None, None))
        if not _dt or not _price:
            logger.warning(f"Price for {instr} not available. Using 0.")
            return 0.0
        return amount * _price

    async def _poller(
        self,
        name: str,
        coroutine: Callable[[], Awaitable],
        interval: str,
    ):
        channel = self.get_communication_channel()
        sleep_time = pd.Timedelta(interval).total_seconds()
        retries = 0

        while channel.control.is_set():
            try:
                await coroutine()

                if not self._polling_to_init[name]:
                    logger.debug(f"{name} polling task has been initialized")
                    self._polling_to_init[name] = True

                retries = 0  # Reset retry counter on success
            except CancelledError:
                logger.info(f"{name} listening has been cancelled")
                break
            except ExchangeClosedByUser:
                logger.info(f"{name} listening has been stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                logger.error(f"Error polling account data: {e}")
                retries += 1
                if retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached. Stopping poller.")
                    break
            except Exception as e:
                logger.error(f"Unexpected error during account polling: {e}")
                logger.exception(e)
                retries += 1
                if retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached. Stopping poller.")
                    break
            finally:
                if not channel.control.is_set():
                    break
                await asyncio.sleep(min(sleep_time * (2 ** (retries)), 60))  # Exponential backoff capped at 60s

        logger.debug(f"{name} polling task has been stopped")

    async def _wait_for_init(self):
        while not all(self._polling_to_init.values()):
            await asyncio.sleep(0.1)

    async def _update_subscriptions(self) -> None:
        """Subscribe to required instruments"""
        assert self._subscription_manager is not None
        await asyncio.sleep(pd.Timedelta(self.subscription_interval).total_seconds())

        # if required instruments have changed, subscribe to them
        if not self._latest_instruments.issuperset(self._required_instruments):
            await self._subscribe_instruments(list(self._required_instruments))
            self._latest_instruments.update(self._required_instruments)

    async def _update_balance(self) -> None:
        """Fetch and update balances from exchange"""
        logger.debug("Updating account balances")
        await self.exchange.load_markets()
        balances_raw = await self.exchange.fetch_balance()
        balances = ccxt_convert_balance(balances_raw)
        current_balances = self.get_balances()

        # remove balances that are not there anymore
        _removed_currencies = set(current_balances.keys()) - set(balances.keys())
        for currency in _removed_currencies:
            self.update_balance(currency, 0, 0)

        # update current balances
        for currency, data in balances.items():
            self.update_balance(currency=currency, total=data.total, locked=data.locked)

        # update required instruments that we need to subscribe to
        currencies = list(self.get_balances().keys())
        instruments = [
            self._get_instrument_for_currency(c) for c in currencies if c.upper() != self.base_currency.upper()
        ]
        self._required_instruments.update(instruments)

        # fetch tickers for instruments that don't have recent price updates
        await self._fetch_missing_tickers(instruments)

    async def _update_positions(self) -> None:
        # fetch and update positions from exchange
        ccxt_positions = await self.exchange.fetch_positions()
        positions = ccxt_convert_positions(ccxt_positions, self.exchange.name, self.exchange.markets)
        self.attach_positions(*positions)
        # update required instruments that we need to subscribe to
        self._required_instruments.update([p.instrument for p in positions])

    async def _subscribe_instruments(self, instruments: list[Instrument]) -> None:
        assert self._subscription_manager is not None

        # find missing subscriptions
        _base_sub = self._subscription_manager.get_base_subscription()
        _subscribed_instruments = self._subscription_manager.get_subscribed_instruments(_base_sub)
        _add_instruments = list(set(instruments) - set(_subscribed_instruments))

        if _add_instruments:
            # subscribe to instruments
            self._subscription_manager.subscribe(_base_sub, _add_instruments)
            self._subscription_manager.commit()

    async def _fetch_missing_tickers(self, instruments: list[Instrument]) -> None:
        _current_time = self.time()
        _fetch_instruments: list[Instrument] = []
        for instr in instruments:
            _dt, _ = self._instrument_to_last_price.get(instr, (None, None))
            if _dt is None or pd.Timedelta(_current_time - _dt) > pd.Timedelta(self.balance_interval):
                _fetch_instruments.append(instr)

        _symbol_to_instrument = {instr.symbol: instr for instr in instruments}
        if _fetch_instruments:
            logger.debug(f"Fetching missing tickers for {_fetch_instruments}")
            _fetch_symbols = [instrument_to_ccxt_symbol(instr) for instr in _fetch_instruments]
            tickers: dict[str, dict] = await self.exchange.fetch_tickers(_fetch_symbols)
            for symbol, ticker in tickers.items():
                instr = _symbol_to_instrument.get(symbol)
                if instr is not None:
                    quote = ccxt_convert_ticker(ticker)
                    self.update_position_price(_current_time, instr, quote.mid_price())
