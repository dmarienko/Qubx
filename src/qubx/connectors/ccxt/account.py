import asyncio
import pandas as pd
import numpy as np

import ccxt.pro as cxp
import concurrent.futures
from ccxt import NetworkError, ExchangeError, ExchangeNotAvailable, ExchangeClosedByUser
from typing import Awaitable
from asyncio.exceptions import CancelledError
from qubx import logger
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import Instrument, Position, dt_64, Deal, Order, AssetBalance
from qubx.utils.misc import AsyncThreadLoop
from .utils import ccxt_convert_balance, ccxt_convert_positions


class CcxtAccountProcessor(BasicAccountProcessor):
    """
    Subscribes to account information from the exchange.
    """

    exchange: cxp.Exchange
    base_currency: str
    balance_interval: str
    position_interval: str
    max_retries: int

    _loop: AsyncThreadLoop
    _polling_tasks: dict[str, concurrent.futures.Future]

    _free_capital: float = np.nan
    _total_capital: float = np.nan

    def __init__(
        self,
        exchange: cxp.Exchange,
        base_currency: str,
        balance_interval: str = "10Sec",
        position_interval: str = "10Sec",
        max_retries: int = 10,
    ):
        self.exchange = exchange
        self.base_currency = base_currency
        self.max_retries = max_retries
        self.balance_interval = balance_interval
        self.position_interval = position_interval
        self._loop = AsyncThreadLoop(exchange.asyncio_loop)
        self._polling_tasks = {}
        self.start()

    def get_capital(self) -> float:
        pass

    def get_total_capital(self) -> float:
        pass

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        # do nothing on deal updates, because we are updating balances directly from exchange
        pass

    def process_order(self, order: Order) -> None:
        # don't update locked value, because we are updating balances directly from exchange
        super().process_order(order, update_locked_value=False)

    def start(self):
        """Start the balance and position polling tasks"""
        self._polling_tasks["balance"] = self._loop.submit(
            self._poller("balance", self._update_balance(), self.balance_interval)
        )
        self._polling_tasks["position"] = self._loop.submit(
            self._poller("position", self._update_positions(), self.position_interval)
        )

    def close(self):
        """Stop all polling tasks"""
        for task in self._polling_tasks.values():
            if not task.done():
                task.cancel()
        self._polling_tasks.clear()

    async def _poller(
        self,
        name: str,
        coroutine: Awaitable,
        interval: str,
    ):

        channel = self.get_communication_channel()
        sleep_time = pd.Timedelta(interval).total_seconds()
        retries = 0

        while channel.control.is_set():
            try:
                await coroutine
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

    async def _update_balance(self) -> None:
        """Fetch and update balances from exchange"""
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

    async def _update_positions(self) -> None:
        """Fetch and update positions from exchange"""
        positions = await self.exchange.fetch_positions()
        # TODO: implement
        if not positions:
            return

        for pos in positions:
            if not pos or pos.get("contracts") == 0:
                continue

            symbol = pos["symbol"]
            instrument = self._get_instrument(symbol)

            self.update_position(
                Position(
                    instrument=instrument,
                    quantity=float(pos["contracts"]),
                    price=float(pos.get("entryPrice", 0)),
                    liquidation_price=float(pos.get("liquidationPrice", 0)),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                )
            )

    def _get_instrument(self, symbol: str) -> Instrument:
        """Helper method to get instrument from symbol"""
        # You'll need to implement this based on your instrument mapping logic
        # This could be part of a shared utility or injected dependency
        raise NotImplementedError("Implement _get_instrument method")
