from typing import List, Dict
from dataclasses import dataclass

import numpy as np
from sortedcontainers import SortedDict

from qubx.core.basics import Deal, Instrument, Order, Position, Signal, Quote
from qubx.core.series import Quote, Trade, time_as_nsec


class BaseError(Exception):
    pass


class ExchangeError(BaseError):
    pass


class BadRequest(ExchangeError):
    pass


class InvalidOrder(ExchangeError):
    pass


class OrderNotFound(InvalidOrder):
    pass


class NotSupported(ExchangeError):
    pass


class ExecReport:
    pass


class OrderManagementEngine:
    instrument: Instrument
    active_orders: Dict[str, Order]
    asks: SortedDict[float, List[Order]]
    bids: SortedDict[float, List[Order]]
    bbo: Quote | None  # current best bid/ask order book (simplest impl)
    __order_id: int

    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument
        self.asks = SortedDict()
        self.bids = SortedDict()
        self.bbo = None
        self.__order_id = 0

    def _generate_order_id(self) -> str:
        self.__order_id += 1
        return self.instrument.symbol + "_" + str(self.__order_id)

    def get_quote(self) -> Quote:
        return self.bbo

    def get_open_orders(self) -> List[Order]:
        return list(self.active_orders.values())

    def update_bbo(self, quote: Quote) -> None:
        self.bbo = quote
        self._process_active_orders()

    def place_order(
        self,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
    ) -> Order:
        if self.bbo is None:
            raise ExchangeError(
                f"Simulator is not ready for order management - no any quote for {self.instrument.symbol}"
            )

        # - validate order parameters
        self._validate_order(order_side, order_type, amount, price, time_in_force)

        order = Order(
            self._generate_order_id(),
            order_type,
            self.instrument.symbol,
            self.bbo.time,
            amount,
            price if price is not None else 0,
            order_side,
            "NEW",
            time_in_force,
            client_id,
        )

        # todo ...
        return order

    def _process_order(self, order: Order) -> ExecReport | None:
        return None

    def _process_active_orders(self) -> List[ExecReport]:
        return []

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

    def cancel_order(self, order_id: str) -> None:
        if order_id not in self.active_orders:
            raise InvalidOrder(f"Order {order_id} not found for {self.instrument.symbol}")
        # todo ...
