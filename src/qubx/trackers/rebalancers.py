from typing import Iterable, List, Set, Tuple, Union, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import Position, Signal, TargetPosition
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

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        """
        Portfolio rebalancer - makes rebalancing portfolio based on provided signals.
        It checks how much funds can be released first and then reallocate it into positions need to be opened.
        """
        targets = self._positions_sizer.calculate_target_positions(ctx, signals)

        # - find positions where exposure will be decreased
        _close_first = []
        _then_open = []
        for t in targets:
            pos = ctx.positions.get(t.instrument.symbol)
            if pos is None:
                logger.error(f"(PortfolioRebalancerTracker) No position for {t.instrument.symbol} instrument !")
                continue

            _pa, _ta = abs(pos.quantity), abs(t.target_position_size)

            # - ones which decreases exposure
            if _ta < _pa:
                reserved = ctx.get_reserved(pos.instrument)
                # - when we have some reserved amount we should check target position size
                t.target_position_size = self._correct_target_position(pos.quantity, t.target_position_size, reserved)
                _close_first.append(t)
                logger.info(
                    f"(PortfolioRebalancerTracker) Decreasing exposure for {t.instrument.symbol} from {pos.quantity} -> {t.target_position_size} (reserved: {reserved})"
                )

            # - ones which increases exposure
            elif _ta > _pa:
                _then_open.append(t)
                logger.info(
                    f"(PortfolioRebalancerTracker) Increasing exposure for {t.instrument.symbol} from {pos.quantity} -> {t.target_position_size})"
                )

        return _close_first + _then_open

    def close_all(self, ctx: StrategyContext) -> None:
        """
        Emergency close all positions
        """
        for s, pos in ctx.positions.items():
            if pos.quantity != 0:
                reserved = ctx.get_reserved(pos.instrument)
                to_close = self._max_size_can_be_closed(pos.quantity, reserved)
                if to_close != 0:
                    try:
                        logger.info(
                            f"(PortfolioRebalancerTracker) {s} - closing {to_close} from {pos.quantity} amount (reserved: {reserved})"
                        )
                        ctx.trade(s, -pos.quantity)
                    except Exception as err:
                        logger.error(f"(PortfolioRebalancerTracker) {s} Error processing closing order: {str(err)}")

    def _correct_target_position(self, start_position: float, new_position: float, reserved: float) -> float:
        """
        Calcluate target position size considering reserved quantity.
        """
        d = np.sign(start_position)
        qty_to_close = start_position

        if reserved != 0 and start_position != 0 and np.sign(reserved) == d:
            qty_to_close = max(start_position - reserved, 0) if d > 0 else min(start_position - reserved, 0)

        # - what's max value allowed to close taking in account reserved quantity
        max_to_close = -d * qty_to_close
        pos_change = new_position - start_position
        direction = np.sign(pos_change)
        prev_direction = np.sign(start_position)

        # - how many shares are closed/open
        qty_closing = min(abs(start_position), abs(pos_change)) * direction if prev_direction != direction else 0
        # qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing
        # print(qty_closing, qty_opening, max_to_close)

        if abs(qty_closing) > abs(max_to_close):
            return reserved

        return new_position
