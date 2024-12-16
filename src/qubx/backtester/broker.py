from qubx.backtester.ome import OmeReport, OrdersManagementEngine
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    Order,
    dt_64,
)
from qubx.core.interfaces import IBroker

from .account import SimulatedAccountProcessor


class SimulatedBroker(IBroker):
    channel: CtrlChannel
    _account: SimulatedAccountProcessor

    _current_time: dt_64
    _name: str
    _ome: dict[Instrument, OrdersManagementEngine]

    def __init__(
        self,
        channel: CtrlChannel,
        account: SimulatedAccountProcessor,
    ) -> None:
        self.channel = channel
        self._account = account

    @property
    def is_simulated_trading(self) -> bool:
        return True

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> Order:
        ome = self._ome.get(instrument)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument.symbol}'!")

        # - try to place order in OME
        report = ome.place_order(
            order_side.upper(),  # type: ignore
            order_type.upper(),  # type: ignore
            amount,
            price,
            client_id,
            time_in_force,
            **options,
        )

        order = report.order
        self._account.order_to_instrument[order.id] = instrument
        self.channel.send((instrument, "order", order))
        if report.exec is not None:
            self.channel.send((instrument, "deals", [report.exec]))

        # - send reports to channel
        self._send_exec_report(instrument, report)

        return report.order

    def cancel_order(self, order_id: str) -> Order | None:
        instrument = self._account.order_to_instrument.get(order_id)
        if instrument is None:
            raise ValueError(f"ExchangeService:cancel_order :: can't find order with id = '{order_id}'!")

        ome = self._ome.get(instrument)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument}'!")

        # - cancel order in OME and remove from the map to free memory
        self._account.order_to_instrument.pop(order_id)
        order_update = ome.cancel_order(order_id)
        self._account.process_order(order_update.order)

        # - notify channel about order cancellation
        self._send_exec_report(instrument, order_update)

        return order_update.order

    def cancel_orders(self, instrument: Instrument) -> None:
        raise NotImplementedError("Not implemented yet")

    def update_order(self, order_id: str, price: float | None = None, amount: float | None = None) -> Order:
        raise NotImplementedError("Not implemented yet")

    def _send_exec_report(self, instrument: Instrument, report: OmeReport):
        self.channel.send((instrument, "order", report.order))
        if report.exec is not None:
            self.channel.send((instrument, "deals", [report.exec]))
