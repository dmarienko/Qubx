from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.strategy import IPositionAdjuster, StrategyContext


class DefaultPoaitionAdjuster(IPositionAdjuster):
    """
    Default implementation of signals executor
    """

    def adjust_position_size(
        self, ctx: StrategyContext, instrument: Instrument, new_size: float, at_price: float | None = None
    ) -> float:
        current_position = ctx.positions[instrument.symbol].quantity
        to_trade = new_size - current_position
        if to_trade < instrument.min_size:
            logger.warning(
                "Can't change position size for {instrument}. Current position: {current_position}, requested size: {new_size}"
            )
        else:
            # - here is default inplementation:
            #   just trade it through the strategy context by using market (or limit) orders.
            # - but in general it may have complex logic for position adjustment
            r = ctx.trade(instrument, to_trade, at_price)
            logger.info("{instrument} >>> Adjusting position from {current_position} to {new_size} : {r}")

            current_position = new_size
            # - TODO: need to check how fast position is being updated on live
            # current_position = ctx.positions[instrument.symbol].quantity

        return current_position
