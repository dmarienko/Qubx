import stackprinter
import traceback
import ccxt
import ccxt.pro as cxp
import numpy as np
import pandas as pd
import asyncio
import concurrent.futures

from ccxt.base.errors import ExchangeError, NotSupported
from collections import defaultdict
from typing import Any, Coroutine, Dict, List, Optional, Callable, Awaitable, Tuple, Set
from qubx import logger, lookup
from qubx.core.account import AccountProcessor
from qubx.core.basics import Instrument, Position, Order, TransactionCostsCalculator, dt_64, Deal, CtrlChannel
from qubx.core.interfaces import IBrokerServiceProvider, ITradingServiceProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.utils.ntp import time_now
from .exceptions import CcxtPositionRestoreError
from .utils import (
    ccxt_convert_order_info,
    ccxt_convert_deal_info,
    ccxt_extract_deals_from_exec,
    ccxt_restore_position_from_deals,
    ccxt_restore_positions_from_info,
)


ORDERS_HISTORY_LOOKBACK_DAYS = 30


class CcxtTradingConnector(ITradingServiceProvider):
    exchange: cxp.Exchange

    _fees_calculator: TransactionCostsCalculator | None = None
    _positions: Dict[Instrument, Position]

    def __init__(
        self,
        exchange: cxp.Exchange,
        account_id: str,
        commissions: str | None = None,
    ):
        self.exchange = exchange
        self.exchange_id = str(exchange.name)
        self.account_id = account_id
        self.commissions = commissions
        self._loop = exchange.asyncio_loop

    def set_account(self, acc: AccountProcessor):
        super().set_account(acc)
        self._task_s(self._sync_account_info(self.commissions)).result()
        self._positions = self.acc._positions
        self._log_reserved()

    def get_position(self, instrument: Instrument) -> Position:
        if instrument not in self._positions:
            position: Position = self._task_s(self._sync_position_and_orders(instrument)).result()
            self.acc.attach_positions(position)
        return self._positions[instrument]

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
    ) -> Order:
        params = {}

        if order_type == "limit":
            params["timeInForce"] = time_in_force.upper()
            if price is None:
                raise ValueError("Price must be specified for limit order")

        if client_id:
            params["newClientOrderId"] = client_id

        params["type"] = "swap" if instrument.is_futures else "spot"

        r: Dict[str, Any] | None = None
        try:
            r = self._task_s(
                self.exchange.create_order(
                    symbol=instrument.symbol, type=order_type, side=order_side, amount=amount, price=price, params=params  # type: ignore
                )
            ).result()
        except ccxt.BadRequest as exc:
            logger.error(
                f"(::send_order) BAD REQUEST for {order_side} {amount} {order_type} for {instrument.symbol} : {exc}"
            )
            raise exc
        except Exception as err:
            logger.error(f"(::send_order) {order_side} {amount} {order_type} for {instrument.symbol} exception : {err}")
            logger.error(traceback.format_exc())
            raise err

        if r is None:
            logger.error(f"(::send_order) No response from exchange")
            raise ExchangeError("(::send_order) No response from exchange")

        order = ccxt_convert_order_info(instrument, r)
        logger.info(f"New order {order}")
        return order

    def cancel_order(self, order_id: str) -> Order | None:
        order = None
        if order_id in self.acc._active_orders:
            order = self.acc._active_orders[order_id]
            try:
                logger.info(f"Canceling order {order_id} ...")
                r = self._task_s(self.exchange.cancel_order(order_id, symbol=order.instrument.symbol)).result()
            except Exception as err:
                logger.error(f"Canceling [{order}] exception : {err}")
                logger.error(traceback.format_exc())
                raise err
        return order

    def time(self) -> dt_64:
        """
        Returns current time as dt64
        """
        return time_now()

    def get_name(self) -> str:
        return self.exchange.name  # type: ignore

    def get_account_id(self) -> str:
        return self.account_id

    def get_orders(self, instrument: Instrument | None = None) -> list[Order]:
        return self.acc.get_orders(instrument)

    def get_base_currency(self) -> str:
        return self.acc.base_currency

    def process_execution_report(self, instrument: Instrument, report: dict[str, Any]) -> Tuple[Order, List[Deal]]:
        order = ccxt_convert_order_info(instrument, report)
        deals = ccxt_extract_deals_from_exec(report)
        self._fill_missing_fee_info(instrument, deals)
        self.acc.process_deals(instrument, deals)
        self.acc.process_order(order)
        return order, deals

    def _task_s(self, coro: Coroutine) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _fill_missing_fee_info(self, instrument: Instrument, deals: List[Deal]) -> None:
        for d in deals:
            if d.fee_amount is None:
                assert self._fees_calculator is not None
                d.fee_amount = self._fees_calculator.get_execution_fees(
                    instrument=instrument, exec_price=d.price, amount=d.amount, crossed_market=d.aggressive
                )
                # this is only true for linear contracts
                d.fee_currency = instrument.quote

    def _get_free_balance_for_asset(self, asset: str, balance: dict) -> float:
        # - try parse it from top level but it is not always there
        _free = balance["free"].get(asset)
        if _free is not None:
            return _free
        # - on portfolio margin account on binance it's inside of info -> asset -> crossMarginFree
        _info = balance.get("info")
        if isinstance(_info, list):
            for a in _info:
                if a["asset"] == asset:
                    _free = a.get("crossMarginFree")
                    if _free is not None:
                        return float(_free)
        return 0.0

    def _log_reserved(self):
        for s, v in self.acc.reserved.items():
            logger.info(f" > {v} of {s} is reserved from trading")

    async def _init_exchange(self):
        await self.exchange.load_markets()

    async def _sync_account_info(self, default_commissions: str | None):
        await self.exchange.load_markets()
        logger.info(f"Loading account data for {self.get_name()}")
        self._balance = await self.exchange.fetch_balance()
        _info = self._balance.get("info")

        # - check what we have on balance
        for k, vol in self._balance["total"].items():  # type: ignore
            if vol != 0.0:  # - get all non zero balances
                _locked = vol - self._get_free_balance_for_asset(k, self._balance)
                self.acc.update_balance(k, vol, _locked)

        # - try to get account's commissions calculator or set default one
        if _info:
            _fees = _info.get("commissionRates") if isinstance(_info, dict) else None
            if _fees:
                self._fees_calculator = TransactionCostsCalculator(
                    "account", 100 * float(_fees["maker"]), 100 * float(_fees["taker"])
                )

        if self._fees_calculator is None:
            self._fees_calculator = lookup.fees.find(self.exchange_id.lower(), default_commissions)

        _future_restored = await self._try_restore_futures_positions()
        _spot_restored = await self._try_restore_spot_positions()
        # if not _restored:
        # _restored = await self._try_restore_spot_positions()
        if not _future_restored and not _spot_restored:
            raise CcxtPositionRestoreError("Could not restore positions from exchange")

    async def _try_restore_futures_positions(self) -> bool:
        try:
            infos = await self.exchange.fetch_positions()
            positions = ccxt_restore_positions_from_info(infos, self.exchange_id.upper())

            async def get_orders_for_position(position: Position):
                return (position, await self._get_open_orders_from_exchange(position.instrument))

            positions_with_orders = await asyncio.gather(*[get_orders_for_position(position) for position in positions])

            for p, orders in positions_with_orders:
                self.acc.attach_positions(p)
                self.acc.add_active_orders(orders)

            return True

        except NotSupported:
            logger.debug(f"Exchange {self.get_name()} doesn't support positions fetching")

        return False

    async def _try_restore_spot_positions(self) -> bool:
        _balances = self.acc.get_balances()
        _currencies = set(_balances.keys())
        _quote_asset = self.acc.base_currency
        _currencies.discard(_quote_asset)

        _instruments = []
        for currency in _currencies:
            _instr = lookup.find_instrument(self.exchange_id, base=currency, quote=_quote_asset)
            if _instr:
                _instruments.append(_instr)
            else:
                logger.warning(f"No instrument found for {currency}/{_quote_asset}")

        positions = await asyncio.gather(*[self._sync_position_and_orders(instrument) for instrument in _instruments])
        for p in positions:
            self.acc.attach_positions(p)

        return True

    async def _get_open_orders_from_exchange(self, instrument: Instrument, days_before: int = 60) -> Dict[str, Order]:
        """
        We need only open orders to restore list of active ones in connector
        method returns open orders sorted by creation time in ascending order
        """
        t_orders_start_ms = (self.time() - days_before * pd.Timedelta("1d")).asm8.item() // 1000000
        orders_data = await self.exchange.fetch_open_orders(instrument.symbol, since=t_orders_start_ms)
        orders: Dict[str, Order] = {}
        for o in orders_data:
            order = ccxt_convert_order_info(instrument, o)
            orders[order.id] = order
        if orders:
            logger.info(f"{instrument.symbol} - loaded {len(orders)} open orders")
        return dict(sorted(orders.items(), key=lambda x: x[1].time, reverse=False))

    async def _get_deals_from_exchange(self, symbol: str, days_before: int = 60) -> List[Deal]:
        """
        Load trades for given symbol
        method returns account's trades sorted by creation time in reversed order (latest - first)
        """
        t_orders_start_ms = (self.time() - days_before * pd.Timedelta("1d")).asm8.item() // 1000000
        deals_data = await self.exchange.fetch_my_trades(symbol, since=t_orders_start_ms)
        deals: List[Deal] = [ccxt_convert_deal_info(o) for o in deals_data]  # type: ignore
        if deals:
            return list(sorted(deals, key=lambda x: x.time, reverse=False))
        return list()

    async def _sync_position_and_orders(self, instrument: Instrument) -> Position:
        position = Position(instrument)
        asset = instrument.base
        symbol = instrument.symbol
        total_amnts = self._balance["total"]
        vol_from_exch = total_amnts.get(asset, total_amnts.get(symbol, 0))

        # - get orders from exchange
        orders = await self._get_open_orders_from_exchange(position.instrument, ORDERS_HISTORY_LOOKBACK_DAYS)
        self.acc.add_active_orders(orders)

        # - get deals from exchange if position is not zero
        if vol_from_exch != 0:
            deals = await self._get_deals_from_exchange(symbol)

            # - actualize position
            position = ccxt_restore_position_from_deals(
                position, vol_from_exch, deals, reserved_amount=self.acc.get_reserved(position.instrument)
            )

        if position.quantity == 0 and vol_from_exch != 0:
            position.quantity = vol_from_exch
            position.change_position_by(self.time(), vol_from_exch, 0)

        return position
