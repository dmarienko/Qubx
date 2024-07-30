from qubx import logger
from qubx.core.basics import Deal, Instrument
from qubx.core.strategy import IPositionGathering, StrategyContext


class SimplePositionGatherer(IPositionGathering):
    """
    Default implementation of positions gathering by single orders through strategy context
    """

    def alter_position_size(
        self, ctx: StrategyContext, instrument: Instrument, new_size: float, at_price: float | None = None
    ) -> float:
        current_position = ctx.positions[instrument.symbol].quantity
        to_trade = new_size - current_position
        if abs(to_trade) < instrument.min_size:
            logger.warning(
                f"Can't change position size for {instrument}. Current position: {current_position}, requested size: {new_size}"
            )
        else:
            # - here is default inplementation:
            #   just trade it through the strategy context by using market (or limit) orders.
            # - but in general it may have complex logic for position adjustment
            r = ctx.trade(instrument, to_trade, at_price)
            logger.info(f"{instrument.symbol} >>> Adjusting position from {current_position} to {new_size} : {r}")

            current_position = new_size
            # - TODO: need to check how fast position is being updated on live
            # current_position = ctx.positions[instrument.symbol].quantity

        return current_position

    def update_by_deal_data(self, instrument: Instrument, deal: Deal): ...


class SplittedOrdersPositionGatherer(IPositionGathering):
    """
    Gather position by splitting order into smaller parts randomly
    """

    pass
