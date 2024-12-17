from qubx import logger
from qubx.backtester.ome import OrdersManagementEngine
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    ZERO_COSTS,
    CtrlChannel,
    Instrument,
    Order,
    Position,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.interfaces import ITimeProvider
from qubx.core.series import Bar, Quote, Trade


class SimulatedAccountProcessor(BasicAccountProcessor):
    ome: dict[Instrument, OrdersManagementEngine]
    order_to_instrument: dict[str, Instrument]

    _channel: CtrlChannel
    _fill_stop_order_at_price: bool
    _half_tick_size: dict[Instrument, float]

    def __init__(
        self,
        account_id: str,
        channel: CtrlChannel,
        base_currency: str,
        initial_capital: float,
        time_provider: ITimeProvider,
        tcc: TransactionCostsCalculator = ZERO_COSTS,
        accurate_stop_orders_execution: bool = False,
    ) -> None:
        super().__init__(
            account_id=account_id,
            time_provider=time_provider,
            base_currency=base_currency,
            tcc=tcc,
            initial_capital=initial_capital,
        )
        self.ome = {}
        self.order_to_instrument = {}
        self._channel = channel
        self._half_tick_size = {}
        self._fill_stop_order_at_price = accurate_stop_orders_execution
        if self._fill_stop_order_at_price:
            logger.info(f"{self.__class__.__name__} emulates stop orders executions at exact price")

    def get_orders(self, instrument: Instrument | None = None) -> list[Order]:
        if instrument is not None:
            ome = self.ome.get(instrument)
            if ome is None:
                raise ValueError(f"ExchangeService:get_orders :: No OME configured for '{instrument}'!")
            return ome.get_open_orders()

        return [o for ome in self.ome.values() for o in ome.get_open_orders()]

    def get_position(self, instrument: Instrument) -> Position:
        if instrument in self.positions:
            return self.positions[instrument]

        # - initiolize OME for this instrument
        self.ome[instrument] = OrdersManagementEngine(
            instrument=instrument,
            time_provider=self.time_provider,
            tcc=self._tcc,  # type: ignore
            fill_stop_order_at_price=self._fill_stop_order_at_price,
        )

        # - initiolize empty position
        position = Position(instrument)  # type: ignore
        self._half_tick_size[instrument] = instrument.tick_size / 2  # type: ignore
        self.attach_positions(position)
        return self.positions[instrument]

    def update_position_price(self, time: dt_64, instrument: Instrument, price: float) -> None:
        super().update_position_price(time, instrument, price)

        # - first we need to update OME with new quote.
        # - if update is not a quote we need 'emulate' it.
        # - actually if SimulatedExchangeService is used in backtesting mode it will recieve only quotes
        # - case when we need that - SimulatedExchangeService is used for paper trading and data provider configured to listen to OHLC or TAS.
        # - probably we need to subscribe to quotes in real data provider in any case and then this emulation won't be needed.
        quote = price if isinstance(price, Quote) else self.emulate_quote_from_data(instrument, time, price)
        if quote is None:
            return

        # - process new quote
        self._process_new_quote(instrument, quote)

    def process_order(self, order: Order, update_locked_value: bool = True) -> None:
        _new = order.status == "NEW"
        _open = order.status == "OPEN"
        _cancel = order.status == "CANCELED"
        _closed = order.status == "CLOSED"
        if _new or _open:
            self.order_to_instrument[order.id] = order.instrument
        if (_cancel or _closed) and order.id in self.order_to_instrument:
            self.order_to_instrument.pop(order.id)
        return super().process_order(order, update_locked_value)

    def emulate_quote_from_data(
        self, instrument: Instrument, timestamp: dt_64, data: float | Trade | Bar
    ) -> Quote | None:
        if instrument not in self._half_tick_size:
            _ = self.get_position(instrument)

        _ts2 = self._half_tick_size[instrument]
        if isinstance(data, Quote):
            return data
        elif isinstance(data, Trade):
            if data.taker:  # type: ignore
                return Quote(timestamp, data.price - _ts2 * 2, data.price, 0, 0)  # type: ignore
            else:
                return Quote(timestamp, data.price, data.price + _ts2 * 2, 0, 0)  # type: ignore
        elif isinstance(data, Bar):
            return Quote(timestamp, data.close - _ts2, data.close + _ts2, 0, 0)  # type: ignore
        elif isinstance(data, float):
            return Quote(timestamp, data - _ts2, data + _ts2, 0, 0)
        else:
            return None

    def _process_new_quote(self, instrument: Instrument, data: Quote) -> None:
        ome = self.ome.get(instrument)
        if ome is None:
            logger.warning("ExchangeService:update :: No OME configured for '{symbol}' yet !")
            return
        for r in ome.update_bbo(data):
            if r.exec is not None:
                self.order_to_instrument.pop(r.order.id)
                # - process methods will be called from stg context
                self._channel.send((instrument, "order", r.order, False))
                self._channel.send((instrument, "deals", [r.exec], False))
