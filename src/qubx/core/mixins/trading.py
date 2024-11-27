from qubx import logger
from qubx.core.basics import Instrument, Order
from qubx.core.interfaces import ITradingServiceProvider, ITradingManager, ITimeProvider, ITradingServiceProvider


class TradingManager(ITradingManager):
    _time_provider: ITimeProvider
    _trading_service: ITradingServiceProvider
    _strategy_name: str

    _order_id: int | None = None

    def __init__(
        self, time_provider: ITimeProvider, trading_service: ITradingServiceProvider, strategy_name: str
    ) -> None:
        self._time_provider = time_provider
        self._trading_service = trading_service
        self._strategy_name = strategy_name

    def trade(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **options,
    ) -> Order:
        # - adjust size
        size_adj = instrument.round_size_down(abs(amount))
        if size_adj < instrument.min_size:
            raise ValueError(f"Attempt to trade size {abs(amount)} less than minimal allowed {instrument.min_size} !")

        side = "buy" if amount > 0 else "sell"
        type = "market"
        if price is not None:
            price = instrument.round_price_down(price) if amount > 0 else instrument.round_price_up(price)
            type = "limit"
            if (stp_type := options.get("stop_type")) is not None:
                type = f"stop_{stp_type}"

        logger.debug(
            f"(StrategyContext) sending {type} {side} for {size_adj} of <green>{instrument.symbol}</green> @ {price} ..."
        )
        client_id = self._generate_order_client_id(instrument.symbol)

        order = self._trading_service.send_order(
            instrument=instrument,
            order_side=side,
            order_type=type,
            amount=size_adj,
            price=price,
            time_in_force=time_in_force,
            client_id=client_id,
            **options,
        )

        return order

    def cancel(self, instrument: Instrument) -> None:
        for o in self._trading_service.get_orders(instrument):
            self._trading_service.cancel_order(o.id)

    def cancel_order(self, order_id: str) -> None:
        if not order_id:
            return
        self._trading_service.cancel_order(order_id)

    def _generate_order_client_id(self, symbol: str) -> str:
        if self._order_id is None:
            self._order_id = self._time_provider.time().item() // 100_000_000
        assert self._order_id is not None
        self._order_id += 1
        return "_".join([self._strategy_name, symbol, str(self._order_id)])
