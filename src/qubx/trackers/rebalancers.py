from typing import Iterable, List, Set, Tuple, Union, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import Position, Signal
from qubx.core.strategy import IPositionGathering, StrategyContext, PositionsTracker
from qubx.trackers.sizers import WeightedPortfolioSizer


@dataclass
class Capital:
    capital: float
    released_amount: float
    symbols_to_close: List[str] | None = None


class PortfolioRebalancerTracker(PositionsTracker):
    """
    Simple portfolio rebalancer based on fixed weights
    """

    capital_invested: float
    tolerance: float

    def __init__(self, capital_invested: float, tolerance: float, positions_sizer=WeightedPortfolioSizer()) -> None:
        self.capital_invested = capital_invested
        self.tolerance = tolerance
        self._positions_sizer = positions_sizer

    def calculate_released_capital(
        self, ctx: StrategyContext, symbols_to_close: List[str] | None = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate capital that would be released if close positions for provided symbols_to_close list
        """
        released_capital_after_close = 0.0
        closed_symbols = []
        if symbols_to_close is not None:
            for symbol in symbols_to_close:
                p = ctx.positions.get(symbol)
                if p is not None and p.quantity != 0:
                    released_capital_after_close += p.get_amount_released_funds_after_closing(
                        to_remain=ctx.get_reserved(p.instrument)
                    )
                    closed_symbols.append(symbol)
        return released_capital_after_close, closed_symbols

    def estimate_capital_to_trade(self, ctx: StrategyContext, symbols_to_close: List[str] | None = None) -> Capital:
        released_capital = 0.0
        closed_positions = None

        if symbols_to_close is not None:
            released_capital, closed_positions = self.calculate_released_capital(ctx, symbols_to_close)

        cap_to_invest = ctx.get_capital() + released_capital
        if self.capital_invested > 0:
            cap_to_invest = min(self.capital_invested, cap_to_invest)

        return Capital(cap_to_invest, released_capital, closed_positions)

    def process_signals(self, ctx: StrategyContext, gathering: IPositionGathering, signals: List[Signal]):
        """
        Portfolio rebalancer - makes rebalancing portfolio based on provided signals.
        It checks how much funds can be released first and then reallocate it into positions need to be opened.
        """

        # - close positions first - we need to release capital
        to_close = [s.instrument.symbol for s in signals if s.signal == 0]
        for s in to_close:
            if pos := ctx.positions.get(s):
                reserved = ctx.get_reserved(pos.instrument)
                to_close = self._how_much_can_be_closed(pos.quantity, reserved)
                logger.info(
                    f"(PortfolioRebalancerTracker) {s} - closing {to_close} from {pos.quantity} amount (reserved: {reserved})"
                )
                try:
                    gathering.alter_position_size(ctx, pos.instrument, reserved)
                except Exception as err:
                    logger.error(f"(PortfolioRebalancerTracker) {s} Error processing closing order: {str(err)}")
            else:
                logger.error(
                    f"(PortfolioRebalancerTracker) Position for {s} is required to be closed but can't be found in context !"
                )

        # - alter or open new positions
        openers = self._positions_sizer.calculate_position_sizes(ctx, [s for s in signals if s.signal != 0])
        to_open = {s.instrument.symbol: s.processed_position_size for s in openers}

        for s, n in to_open.items():
            if pos := ctx.positions.get(s):
                trade_size = n - pos.quantity
                trade_size_change_pct = abs(trade_size / pos.quantity) if pos.quantity != 0 else 1
                if 100 * trade_size_change_pct > self.tolerance:
                    logger.info(
                        f"(PortfolioRebalancerTracker) {s} - change position {pos.quantity} -> {n} (tolerance: {self.tolerance}%)"
                    )
                    try:
                        gathering.alter_position_size(ctx, pos.instrument, n)
                    except Exception as err:
                        logger.error(f"(PortfolioRebalancerTracker) {s} Error processing opening order: {str(err)}")
                else:
                    logger.info(
                        f"(PortfolioRebalancerTracker) {s} - position change ({pos.quantity} -> {n}) is smaller than tolerance {self.tolerance}%"
                    )

            else:
                logger.error(
                    f"(PortfolioRebalancerTracker) Position for {s} is required to be changed but can't be found in context !"
                )

    def close_all(self, ctx: StrategyContext) -> None:
        """
        Emergency close all positions
        """
        for s, pos in ctx.positions.items():
            if pos.quantity != 0:
                reserved = ctx.get_reserved(pos.instrument)
                to_close = self._how_much_can_be_closed(pos.quantity, reserved)
                if to_close != 0:
                    try:
                        logger.info(
                            f"(PortfolioRebalancerTracker) {s} - closing {to_close} from {pos.quantity} amount (reserved: {reserved})"
                        )
                        ctx.trade(s, -pos.quantity)
                        # gathering.alter_position_size(ctx, pos.instrument, reserved)
                    except Exception as err:
                        logger.error(f"(PortfolioRebalancerTracker) {s} Error processing closing order: {str(err)}")

    def _how_much_can_be_closed(self, position: float, to_remain: float) -> float:
        d = np.sign(position)
        qty_to_close = position
        if to_remain != 0 and position != 0 and np.sign(to_remain) == d:
            qty_to_close = max(position - to_remain, 0) if d > 0 else min(position - to_remain, 0)
        return qty_to_close
