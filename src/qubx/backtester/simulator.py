from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from qubx import lookup, logger
from qubx.core.series import Quote
from qubx.core.account import AccountProcessor
from qubx.core.basics import Instrument, Deal, Order, Position, TransactionCostsCalculator, dt_64
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.backtester.ome import OrdersManagementEngine, OmeReport


class SimulatedExchangeService(IExchangeServiceProvider):
    _current_time: dt_64
    _name: str
    _ome: Dict[str, OrdersManagementEngine]
    _fees_calculator: TransactionCostsCalculator | None
    _channel: CtrlChannel
    _order_to_symbol: Dict[str, str]

    def __init__(self, name: str, capital: float, commissions: str, base_currency: str) -> None:
        self._current_time = np.datetime64(0, "ns")
        self._name = name
        self._ome = {}
        self.acc = AccountProcessor("Simulated0", base_currency, None, capital, 0)
        self._fees_calculator = lookup.fees.find(name.lower(), commissions)

        self._channel = CtrlChannel(name + ".simulator", sentinel=(None, None, None))

        self._order_to_symbol = {}
        if self._fees_calculator is None:
            raise ValueError(
                f"SimulatedExchangeService :: Fees configuration '{commissions}' is not found for '{name}' !"
            )

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
    ) -> Order | None:
        ome = self._ome.get(instrument.symbol)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument.symbol}'!")

        # - try to place order in OME
        report = ome.place_order(order_side.upper(), order_type.upper(), amount, price, client_id, time_in_force)
        order = report.order
        self._order_to_symbol[order.id] = instrument.symbol

        if report.exec is not None:
            self.process_execution_report(instrument.symbol, {"order": order, "deals": [report.exec]})
        else:
            self.acc.add_active_orders({order.id: order})

        # - send reports to channel
        self._channel.queue.put((instrument.symbol, "order", report.order))
        if report.exec is not None:
            self._channel.queue.put((instrument.symbol, "deals", [report.exec]))

        return report.order

    def cancel_order(self, order_id: str) -> Order | None:
        symb = self._order_to_symbol.get(order_id)
        if symb is None:
            raise ValueError(f"ExchangeService:cancel_order :: can't find order with id = '{order_id}'!")

        ome = self._ome.get(symb)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{symb}'!")

        # - cancel order in OME and remove from the map to free memory
        self._order_to_symbol.pop(order_id)
        order_update = ome.cancel_order(order_id).order
        self.acc.process_order(order_update)
        return order_update

    def get_orders(self, symbol: str | None = None) -> List[Order]:
        if symbol is not None:
            ome = self._ome.get(symbol)
            if ome is None:
                raise ValueError(f"ExchangeService:get_orders :: No OME configured for '{symbol}'!")
            return ome.get_open_orders()

        return [o for ome in self._ome.values() for o in ome.get_open_orders()]

    def get_position(self, instrument: Instrument) -> Position:
        symbol = instrument.symbol

        if symbol not in self.acc._positions:
            # - initiolize OME for this instrument
            self._ome[instrument.symbol] = OrdersManagementEngine(instrument, self)  # type: ignore

            # - initiolize empty position
            position = Position(instrument, self._fees_calculator)  # type: ignore
            self.acc.attach_positions(position)

        return self.acc._positions[symbol]

    def time(self) -> dt_64:
        return self._current_time

    def get_base_currency(self) -> str:
        return self.acc.base_currency

    def get_name(self) -> str:
        return self._name

    def process_execution_report(self, symbol: str, report: Dict[str, Any]) -> Tuple[Order, List[Deal]]:
        order = report["order"]
        deals = report.get("deals", [])
        self.acc.process_deals(symbol, deals)
        self.acc.process_order(order)
        return order, deals

    def update_position_price(self, symbol: str, price: float):
        super().update_position_price(symbol, price)

    def _update(self, symbol: str, data: Quote) -> None:
        ome = self._ome.get(symbol)
        if ome is None:
            logger.warning("ExchangeService:update :: No OME configured for '{symbol}' yet !")
            return
        self._current_time = data.time
        for r in ome.update_bbo(data):
            if r.exec is not None:
                self._order_to_symbol.pop(r.order.id)
                self.process_execution_report(symbol, {"order": r.order, "deals": [r.exec]})

    def _get_ohlcv_data_sync(self, symbol: str, timeframe: str, since: int, limit: int) -> List:
        return []
