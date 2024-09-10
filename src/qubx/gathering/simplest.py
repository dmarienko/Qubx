from qubx import logger
from qubx.core.basics import Deal, Instrument, TargetPosition
from qubx.core.strategy import IPositionGathering, StrategyContext


class SimplePositionGatherer(IPositionGathering):
    """
    Default implementation of positions gathering by single orders through strategy context
    """

    entry_order_id: str | None = None

    def _cncl_order(self, ctx: StrategyContext, instrument: Instrument) -> None:
        if self.entry_order_id:
            logger.debug(
                f"<green>{instrument.symbol}</green>: Cancelling previous entry order <red>{self.entry_order_id}</red>"
            )
            try:
                ctx.cancel_order(self.entry_order_id)
            except Exception as e:
                logger.error(f"Cancelling entry order failed: {str(e)}")
            self.entry_order_id = None

    def alter_position_size(self, ctx: StrategyContext, target: TargetPosition) -> float:
        #  Here is default inplementation:
        #  just trade it through the strategy context by using market (or limit) orders.
        #  but in general it may have complex logic for position adjustment
        instrument, new_size, at_price = target.instrument, target.target_position_size, target.price
        current_position = ctx.positions[instrument.symbol].quantity
        to_trade = new_size - current_position

        # - first cancel previous entry order if exists
        self._cncl_order(ctx, instrument)

        if abs(to_trade) < instrument.min_size:
            if current_position != 0:
                logger.debug(
                    f"{instrument.exchange}:{instrument.symbol}: Unable change position from {current_position} to {new_size} : too small difference"
                )
        else:

            # - check how it should be traded: market or limit or stop order
            opts = {}
            _is_stop_or_limit = False
            if at_price:
                # - we already havbe position but it's requested to change at a specific price
                if abs(current_position) > instrument.min_size:
                    logger.debug(
                        f"<green>{instrument.symbol}</green>: Attempt to change current position {current_position} to {new_size} at {at_price} !"
                    )

                quote = ctx.quote(instrument.symbol)
                if (to_trade > 0 and at_price > quote.ask) or (to_trade < 0 and at_price < quote.bid):
                    opts["stop_type"] = "market"
                    _is_stop_or_limit = True

                if (to_trade > 0 and at_price <= quote.bid) or (to_trade < 0 and at_price >= quote.ask):
                    _is_stop_or_limit = True

            r = ctx.trade(instrument, to_trade, at_price, **opts)
            if _is_stop_or_limit:
                self.entry_order_id = r.id
                logger.debug(
                    f"<green>{instrument.symbol}</green>: Position may be adjusted from {current_position} to {new_size} at {at_price} : {r}"
                )
            else:
                self.entry_order_id = None
                logger.debug(
                    f"<green>{instrument.symbol}</green>: Adjusting position from {current_position} to {new_size} : {r}"
                )

            current_position = new_size
            # - TODO: need to check how fast position is being updated on live
            # current_position = ctx.positions[instrument.symbol].quantity

        return current_position

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        if deal.order_id == self.entry_order_id:
            self.entry_order_id = None


class SplittedOrdersPositionGatherer(IPositionGathering):
    """
    Gather position by splitting order into smaller parts randomly
    """

    pass
