from typing import List, Dict
from dataclasses import dataclass
from operator import neg

import numpy as np
from sortedcontainers import SortedDict

from qubx import logger
from qubx.core.basics import Deal, Instrument, Order, Position, Signal, TransactionCostsCalculator, dt_64, ITimeProvider
from qubx.core.series import Quote, Trade
from qubx.core.exceptions import (
    ExchangeError,
    InvalidOrder,
)


@dataclass
class OmeReport:
    timestamp: dt_64
    order: Order
    exec: Deal | None


class OrdersManagementEngine:
    instrument: Instrument
    time_service: ITimeProvider
    active_orders: Dict[str, Order]
    asks: SortedDict[float, List[str]]
    bids: SortedDict[float, List[str]]
    bbo: Quote | None  # current best bid/ask order book (simplest impl)
    __order_id: int
    __trade_id: int

    def __init__(
        self, instrument: Instrument, time_provider: ITimeProvider, tcc: TransactionCostsCalculator, debug: bool = True
    ) -> None:
        self.instrument = instrument
        self.time_service = time_provider
        self.tcc = tcc
        self.asks = SortedDict()
        self.bids = SortedDict(neg)
        self.active_orders = dict()
        self.bbo = None
        self.__order_id = 100000
        self.__trade_id = 100000
        if not debug:
            self._dbg = lambda message, **kwargs: None

    def _generate_order_id(self) -> str:
        self.__order_id += 1
        return "SIM-ORDER-" + self.instrument.symbol + "-" + str(self.__order_id)

    def _generate_trade_id(self) -> str:
        self.__trade_id += 1
        return "SIM-EXEC-" + self.instrument.symbol + "-" + str(self.__trade_id)

    def get_quote(self) -> Quote:
        return self.bbo

    def get_open_orders(self) -> List[Order]:
        return list(self.active_orders.values())

    def update_bbo(self, quote: Quote) -> List[OmeReport]:
        timestamp = self.time_service.time()
        rep = []

        if self.bbo is not None:
            if quote.bid >= self.bbo.ask:
                for level in self.asks.irange(0, quote.bid):
                    for order_id in self.asks[level]:
                        order = self.active_orders.pop(order_id)
                        rep.append(self._execute_order(timestamp, order.price, order, False))
                    self.asks.pop(level)

            if quote.ask <= self.bbo.bid:
                for level in self.bids.irange(np.inf, quote.ask):
                    for order_id in self.bids[level]:
                        order = self.active_orders.pop(order_id)
                        rep.append(self._execute_order(timestamp, order.price, order, False))
                    self.bids.pop(level)

        self.bbo = quote
        return rep

    def place_order(
        self,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        fill_at_price: bool = False,
    ) -> OmeReport:

        if self.bbo is None:
            raise ExchangeError(
                f"Simulator is not ready for order management - no any quote for {self.instrument.symbol}"
            )

        # - validate order parameters
        self._validate_order(order_side, order_type, amount, price, time_in_force)

        timestamp = self.time_service.time()
        order = Order(
            self._generate_order_id(),
            order_type,
            self.instrument.symbol,
            timestamp,
            amount,
            price if price is not None else 0,
            order_side,
            "NEW",
            time_in_force,
            client_id,
        )

        return self._process_order(timestamp, order, fill_at_price=fill_at_price)

    def _dbg(self, message, **kwargs) -> None:
        logger.debug(f"[OMS] {self.instrument.symbol} - {message}", **kwargs)

    def _process_order(self, timestamp: dt_64, order: Order, fill_at_price: bool = False) -> OmeReport:
        if order.status in ["CLOSED", "CANCELED"]:
            raise InvalidOrder(f"Order {order.id} is already closed or canceled.")

        buy_side = order.side == "BUY"
        c_ask = self.bbo.ask
        c_bid = self.bbo.bid

        # - check if order can be "executed" immediately
        exec_price = None
        if fill_at_price and order.price:
            exec_price = order.price

        elif order.type == "MARKET":
            exec_price = c_ask if buy_side else c_bid

        elif order.type == "LIMIT":
            if (buy_side and order.price >= c_ask) or (not buy_side and order.price <= c_bid):
                exec_price = c_ask if buy_side else c_bid

        # - if order must be "executed" immediately
        if exec_price is not None:
            return self._execute_order(timestamp, exec_price, order, True)

        # - processing limit orders
        if buy_side:
            self.bids.setdefault(order.price, list()).append(order.id)
        else:
            self.asks.setdefault(order.price, list()).append(order.id)
        order.status = "OPEN"
        self._dbg(f"registered {order.id} {order.type} {order.side} {order.quantity} {order.price}")
        self.active_orders[order.id] = order
        return OmeReport(timestamp, order, None)

    def _execute_order(self, timestamp: dt_64, exec_price: float, order: Order, taker: bool) -> OmeReport:
        order.status = "CLOSED"
        self._dbg(f"{order.id} {order.type} {order.side} {order.quantity} executed at {exec_price}")
        return OmeReport(
            timestamp,
            order,
            Deal(
                id=self._generate_trade_id(),
                order_id=order.id,
                time=timestamp,
                amount=order.quantity if order.side == "BUY" else -order.quantity,
                price=exec_price,
                aggressive=taker,
                fee_amount=self.tcc.get_execution_fees(
                    instrument=self.instrument, exec_price=exec_price, amount=order.quantity, crossed_market=taker
                ),
                fee_currency=self.instrument.quote,
            ),
        )

    def _validate_order(
        self, order_side: str, order_type: str, amount: float, price: float | None, time_in_force: str
    ) -> None:
        if order_side.upper() not in ["BUY", "SELL"]:
            raise InvalidOrder("Invalid order side. Only BUY or SELL is allowed.")

        if order_type.upper() not in ["LIMIT", "MARKET"]:
            raise InvalidOrder("Invalid order type. Only LIMIT or MARKET is supported.")

        if amount <= 0:
            raise InvalidOrder("Invalid order amount. Amount must be positive.")

        if order_type.upper() == "LIMIT" and (price is None or price <= 0):
            raise InvalidOrder("Invalid order price. Price must be positively defined for LIMIT orders.")

        if time_in_force.upper() not in ["GTC", "IOC"]:
            raise InvalidOrder("Invalid time in force. Only GTC or IOC is supported for now.")

    def cancel_order(self, order_id: str) -> OmeReport:
        if order_id not in self.active_orders:
            raise InvalidOrder(f"Order {order_id} not found for {self.instrument.symbol}")

        timestamp = self.time_service.time()
        order = self.active_orders.pop(order_id)
        if order.side == "BUY":
            oids = self.bids[order.price]
            oids.remove(order_id)
            if not oids:
                self.bids.pop(order.price)
        else:
            oids = self.asks[order.price]
            oids.remove(order_id)
            if not oids:
                self.asks.pop(order.price)

        order.status = "CANCELED"
        self._dbg(f"{order.id} {order.type} {order.side} {order.quantity} canceled")
        return OmeReport(timestamp, order, None)

    def __str__(self) -> str:
        _a, _b = True, True

        timestamp = self.time_service.time()
        _s = f"= = ({np.datetime64(timestamp, 'ns')}) = =\n"
        for k, v in reversed(self.asks.items()):
            _sizes = ",".join([f"{self.active_orders[o].quantity}" for o in v])
            _s += f"  {k} : [{ _sizes }]\n"
            if k == self.bbo.ask:
                _a = False

        if _a:
            _s += f"  {self.bbo.ask} : \n"
        _s += "- - - - - - - - - - - - - - - - - - - -\n"

        _s1 = ""
        for k, v in self.bids.items():
            _sizes = ",".join([f"{self.active_orders[o].quantity}" for o in v])
            _s1 += f"  {k} : [{ _sizes }]\n"
            if k == self.bbo.bid:
                _b = False
        _s1 += "= = = = = = = = = = = = = = = = = = = =\n"

        _s1 = f"  {self.bbo.bid} : \n" + _s1 if _b else _s1

        return _s + _s1
