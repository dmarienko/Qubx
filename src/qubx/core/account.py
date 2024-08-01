from typing import Any, Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np

from qubx import lookup, logger
from qubx.core.basics import Instrument, Position, TransactionCostsCalculator, dt_64, Deal, Order


class AccountProcessor:
    """
    Account processor class
    """

    account_id: str
    base_currency: str
    reserved: Dict[str, float]  # how much asset is reserved against the trading
    _balances: Dict[str, Tuple[float, float]]
    _active_orders: Dict[str | int, Order]  # active orders
    _processed_trades: Dict[str | int, List[str | int]]
    _positions: Dict[str, Position]
    _locked_capital_by_order: Dict[str | int, float]

    def __init__(
        self,
        account_id: str,
        base_currency: str,
        reserves: Dict[str, float] | None,
        initial_capital: float = 0,
    ) -> None:
        self.account_id = account_id
        self.base_currency = base_currency.upper()
        self.reserved = dict() if reserves is None else reserves
        self._processed_trades = defaultdict(list)
        self._active_orders = dict()
        self._positions = {}
        self._locked_capital_by_order = dict()
        self._balances = dict()
        self._balances[self.base_currency] = (initial_capital, 0)

    def update_balance(self, currency: str, total_capital: float, locked_capital: float):
        if currency == self.base_currency:
            current_balance = self._balances[currency][0]
            if current_balance > 0 and current_balance < total_capital:
                logger.info(
                    f"Restricting {self.base_currency} capital from {total_capital} "
                    f"down to {current_balance} due to initial capital constraint."
                )
                total_capital = current_balance
        self._balances[currency] = (total_capital, locked_capital)

    def get_balances(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._balances)

    def attach_positions(self, *position: Position) -> "AccountProcessor":
        for p in position:
            self._positions[p.instrument.symbol] = p
        return self

    def update_position_price(self, time: dt_64, symbol: str, price: float):
        p = self._positions[symbol]
        p.update_market_price(time, price, 1)

    def get_free_capital(self) -> float:
        """
        Get free capital in base currency.
        """
        # TODO: separate cash and margin accounts in the future
        return self.base_currency_balance - self.base_currency_locked_balance

    def get_total_capital(self) -> float:
        """
        Get margin capital in base currency. It includes capital locked in positions.
        """
        positions_value_in_base = sum([p.market_value_funds for p in self._positions.values()])
        return self.base_currency_balance + positions_value_in_base - self.get_reserved_capital()

    def get_leverage(self, inst: str | Instrument) -> float:
        inst = inst if isinstance(inst, str) else inst.symbol
        pos = self._positions.get(inst)
        if pos is not None:
            return pos.market_value_funds / self.get_total_capital()
        return 0.0

    def get_leverages(self) -> dict[str, float]:
        return {s: self.get_leverage(s) for s in self._positions.keys()}

    def get_net_leverage(self) -> float:
        return sum(self.get_leverages().values())

    def get_gross_leverage(self) -> float:
        return sum(map(abs, self.get_leverages().values()))

    def get_reserved(self, instrument: Instrument) -> float:
        """
        Check how much were reserved for this instrument
        """
        return self.reserved.get(instrument.symbol, self.reserved.get(instrument.base, 0))

    def get_reserved_capital(self) -> float:
        """
        Get total amount of capital reserved from trading.
        """
        if not self.reserved:
            return 0.0
        reserved_capital = 0.0
        # TODO: fix this to work with reserved currency keys instead of symbols
        for symbol, amount in self.reserved.items():
            reserved_capital += amount * self._positions[symbol].last_update_price
        return reserved_capital

    def process_deals(self, symbol: str, deals: List[Deal]):
        pos = self._positions.get(symbol)

        if pos is not None:
            conversion_rate = 1
            instr = pos.instrument
            traded_amnt, realized_pnl, deal_cost = 0, 0, 0

            # - check if we need conversion rate for this instrument
            # - TODO - need test on it !
            if instr._aux_instrument is not None:
                aux_pos = self._positions.get(instr._aux_instrument.symbol)
                if aux_pos:
                    conversion_rate = aux_pos.last_update_price
                else:
                    logger.error(
                        f"Can't find additional instrument {instr._aux_instrument} for estimating {symbol} position value !!!"
                    )

            # - process deals
            for d in deals:
                if d.id not in self._processed_trades[d.order_id]:
                    self._processed_trades[d.order_id].append(d.id)
                    r_pnl, fee_in_base = pos.update_position_by_deal(d, conversion_rate)
                    realized_pnl += r_pnl
                    deal_cost += d.amount * d.price / conversion_rate
                    traded_amnt += d.amount
                    logger.debug(f"  ::  traded {d.amount} for {symbol} @ {d.price} -> {realized_pnl:.2f}")
                    self.base_currency_balance -= deal_cost + fee_in_base

    def _lock_limit_order_value(self, order: Order) -> float:
        pos = self._positions.get(order.symbol)
        excess = 0.0
        # - we handle only instruments it;s subscribed to
        if pos:
            sgn = -1 if order.side == "SELL" else +1
            pos_change = sgn * order.quantity
            direction = np.sign(pos_change)
            prev_direction = np.sign(pos.quantity)
            # how many shares are closed/open
            qty_closing = min(abs(pos.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
            qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing
            excess = abs(qty_opening) * order.price

            if excess > 0:
                self.base_currency_locked_balance += excess
                self._locked_capital_by_order[order.id] = excess

        return excess

    def _unlock_limit_order_value(self, order: Order):
        if order.id in self._locked_capital_by_order:
            excess = self._locked_capital_by_order.pop(order.id)
            self.base_currency_locked_balance -= excess

    def process_order(self, order: Order):
        _new = order.status == "NEW"
        _open = order.status == "OPEN"
        _closed = order.status == "CLOSED"
        _cancel = order.status == "CANCELED"

        if _open or _new:
            self._active_orders[order.id] = order

            # - calculate amount locked by this order
            if order.type == "LIMIT":
                self._lock_limit_order_value(order)

        if _closed or _cancel:
            if order.id in self._processed_trades:
                self._processed_trades.pop(order.id)

            if order.id in self._active_orders:
                self._active_orders.pop(order.id)

        # - calculate amount to unlock after canceling
        if _cancel and order.type == "LIMIT":
            self._unlock_limit_order_value(order)

        logger.debug(f"Order {order.id} {order.type} {order.side} {order.quantity} of {order.symbol} -> {order.status}")

    def add_active_orders(self, orders: Dict[str, Order]):
        for oid, od in orders.items():
            self._active_orders[oid] = od

    def get_orders(self, symbol: str | None = None) -> List[Order]:
        ols = list(self._active_orders.values())
        if symbol is not None:
            ols = list(filter(lambda x: x.symbol == symbol, ols))
        return ols

    def positions_report(self) -> dict:
        rep = {}
        for p in self._positions.values():
            rep[p.instrument.symbol] = {
                "Qty": p.quantity,
                "Price": p.position_avg_price_funds,
                "PnL": p.pnl,
                "MktValue": p.market_value_funds,
                "Leverage": self.get_leverage(p.instrument.symbol),
            }
        return rep

    @property
    def base_currency_balance(self) -> float:
        return self._balances[self.base_currency][0]

    @property
    def base_currency_locked_balance(self) -> float:
        return self._balances[self.base_currency][1]

    @base_currency_balance.setter
    def base_currency_balance(self, value: float):
        self._balances[self.base_currency] = (value, self._balances[self.base_currency][1])

    @base_currency_locked_balance.setter
    def base_currency_locked_balance(self, value: float):
        self._balances[self.base_currency] = (self._balances[self.base_currency][0], value)
