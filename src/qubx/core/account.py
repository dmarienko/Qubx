import numpy as np

from collections import defaultdict
from qubx import logger
from qubx.core.interfaces import IAccountProcessor
from qubx.core.basics import Instrument, Position, dt_64, Deal, Order, AssetBalance


class BasicAccountProcessor(IAccountProcessor):
    account_id: str
    base_currency: str
    _balances: dict[str, AssetBalance]
    _active_orders: dict[str, Order]
    _processed_trades: dict[str, list[str | int]]
    _positions: dict[Instrument, Position]
    _locked_capital_by_order: dict[str, float]

    def __init__(
        self,
        account_id: str,
        base_currency: str,
        initial_capital: float = 100_000,
    ) -> None:
        self.account_id = account_id
        self.base_currency = base_currency.upper()
        self._processed_trades = defaultdict(list)
        self._active_orders = dict()
        self._positions = {}
        self._locked_capital_by_order = dict()
        self._balances = defaultdict(lambda: AssetBalance())
        self._balances[self.base_currency] += initial_capital

    def get_base_currency(self) -> str:
        return self.base_currency

    ########################################################
    # Balance and position information
    ########################################################
    def get_capital(self) -> float:
        return self.get_available_margin()

    def get_total_capital(self) -> float:
        # sum of cash + market value of all positions
        _cash_amount = self._balances[self.base_currency].total
        _positions_value = sum([p.market_value_funds for p in self._positions.values()])
        return _cash_amount + _positions_value

    def get_balances(self) -> dict[str, AssetBalance]:
        return self._balances

    def get_positions(self) -> dict[Instrument, Position]:
        return self._positions

    def get_orders(self, instrument: Instrument | None = None) -> list[Order]:
        ols = list(self._active_orders.values())
        if instrument is not None:
            ols = list(filter(lambda x: x.instrument == instrument, ols))
        return ols

    def get_reserved(self, instrument: Instrument) -> float:
        return 0.0

    def get_reserved_capital(self) -> float:
        return 0.0

    def position_report(self) -> dict:
        rep = {}
        for p in self._positions.values():
            rep[p.instrument.symbol] = {
                "Qty": p.quantity,
                "Price": p.position_avg_price_funds,
                "PnL": p.pnl,
                "MktValue": p.market_value_funds,
                "Leverage": self.get_leverage(p.instrument),
            }
        return rep

    ########################################################
    # Leverage information
    ########################################################
    def get_leverage(self, instrument: Instrument) -> float:
        pos = self._positions.get(instrument)
        if pos is not None:
            return pos.market_value_funds / self.get_total_capital()
        return 0.0

    def get_leverages(self) -> dict[Instrument, float]:
        return {s: self.get_leverage(s) for s in self._positions.keys()}

    def get_net_leverage(self) -> float:
        return sum(self.get_leverages().values())

    def get_gross_leverage(self) -> float:
        return sum(map(abs, self.get_leverages().values()))

    ########################################################
    # Margin information
    # Used for margin, swap, futures, options trading
    ########################################################
    def get_total_required_margin(self) -> float:
        # sum of margin required for all positions
        return sum([p.maint_margin for p in self._positions.values()])

    def get_available_margin(self) -> float:
        # total capital - total required margin
        return self.get_total_capital() - self.get_total_required_margin()

    def get_margin_ratio(self) -> float:
        # total capital / total required margin
        return self.get_total_capital() / self.get_total_required_margin()

    ########################################################
    # Order and trade processing
    ########################################################
    def update_balance(self, currency: str, total: float, locked: float):
        # create new asset balance if doesn't exist, otherwise update existing
        if currency not in self._balances:
            self._balances[currency] = AssetBalance(free=total, locked=locked, total=total)
        else:
            self._balances[currency].free = total - locked
            self._balances[currency].locked = locked
            self._balances[currency].total = total

    def attach_positions(self, *position: Position) -> IAccountProcessor:
        for p in position:
            self._positions[p.instrument] = p
        return self

    def add_active_orders(self, orders: dict[str, Order]):
        for oid, od in orders.items():
            self._active_orders[oid] = od

    def update_position_price(self, time: dt_64, instrument: Instrument, price: float) -> None:
        p = self._positions[instrument]
        p.update_market_price(time, price, 1)

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        pos = self._positions.get(instrument)

        if pos is not None:
            conversion_rate = 1
            instr = pos.instrument
            traded_amnt, realized_pnl, deal_cost = 0, 0, 0

            # - process deals
            for d in deals:
                if d.id not in self._processed_trades[d.order_id]:
                    self._processed_trades[d.order_id].append(d.id)
                    r_pnl, fee_in_base = pos.update_position_by_deal(d, conversion_rate)
                    realized_pnl += r_pnl
                    deal_cost += d.amount * d.price / conversion_rate
                    traded_amnt += d.amount
                    total_cost = deal_cost + fee_in_base
                    logger.debug(f"  ::  traded {d.amount} for {instrument} @ {d.price} -> {realized_pnl:.2f}")
                    if not instrument.is_futures():
                        self._balances[self.base_currency] -= total_cost
                        self._balances[instrument.base] += d.amount
                    else:
                        self._balances[self.base_currency] -= fee_in_base
                        self._balances[instrument.settle] += realized_pnl

    def process_order(self, order: Order) -> None:
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

        logger.debug(
            f"Order {order.id} {order.type} {order.side} {order.quantity} of {order.instrument} -> {order.status}"
        )

    def _lock_limit_order_value(self, order: Order) -> float:
        pos = self._positions.get(order.instrument)
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

            # TODO: locking likely doesn't work correctly for spot accounts
            # Example: if we have 1 BTC at price 100k and set a limit order for 0.1 BTC at 110k
            # it will not lock 0.1 BTC
            if excess > 0:
                self._balances[self.base_currency].lock(excess)
                self._locked_capital_by_order[order.id] = excess

        return excess

    def _unlock_limit_order_value(self, order: Order):
        if order.id in self._locked_capital_by_order:
            excess = self._locked_capital_by_order.pop(order.id)
            self._balances[self.base_currency].lock(-excess)
