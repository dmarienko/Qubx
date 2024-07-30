from typing import List
import numpy as np

from qubx import logger
from qubx.core.basics import Position, Signal
from qubx.core.strategy import IPositionSizer, StrategyContext
from qubx.utils.misc import round_down_at_min_qty


class FixedSizer(IPositionSizer):
    """
    Simplest fixed sizer class. It uses same fixed size for all signals.
    We use it for quick backtesting of generated signals in most cases.
    """

    def __init__(self, fixed_size):
        self.fixed_size = abs(fixed_size)

    def calculate_position_sizes(self, ctx: StrategyContext, signals: List[Signal]) -> List[Signal]:
        for s in signals:
            s.processed_position_size = s.signal * self.fixed_size
        return signals


class FixedRiskSizer(IPositionSizer):
    def __init__(self, max_cap_in_risk: float, max_allowed_position=np.inf):
        """
        Create fixed risk sizer calculator instance.
        :param max_cap_in_risk: maximal risked capital (in percentage)
        :param max_allowed_position: limitation for max position size
        """
        self.max_cap_in_risk = max_cap_in_risk
        self.max_allowed_position = max_allowed_position

    def calculate_position_sizes(self, ctx: StrategyContext, signals: List[Signal]) -> List[Signal]:
        for signal in signals:
            if signal.signal != 0:
                if signal.stop and signal.stop > 0:
                    _pos = ctx.positions[signal.instrument.symbol]
                    _q = ctx.quote(signal.instrument.symbol)

                    _direction = np.sign(signal.signal)
                    _cap = ctx.get_capital() + max(_pos.total_pnl(), 0)
                    _entry = _q.ask if _direction > 0 else _q.bid

                    signal.processed_position_size = _direction * min(
                        round((_cap * self.max_cap_in_risk / 100) / abs(signal.stop / _entry - 1)),
                        self.max_allowed_position,
                    )
                else:
                    logger.warning(" >>> FixedRiskSizer: stop is not specified - can't calculate position !")

        return signals


class WeightedPortfolioSizer(IPositionSizer):

    def __init__(self, cap_used_pct: float = 1.0):
        """
        Weighted portfolio sizer. Signals are cosidered as weigths.
        :param cap_used_pct: percentage of total capital to be used for sizing
        """
        self.cap_used_pct = min(abs(cap_used_pct), 1.0)

    def calculate_position_sizes(self, ctx: StrategyContext, signals: List[Signal]) -> List[Signal]:
        sw = np.sum([max(s.signal, 0) for s in signals])
        cap = self.cap_used_pct * ctx.get_capital()

        for signal in signals:
            # _pos = ctx.positions[signal.instrument.symbol]
            _q = ctx.quote(signal.instrument.symbol)
            if _q is not None:
                signal.processed_position_size = round_down_at_min_qty(
                    cap * max(signal.signal, 0) / sw / _q.mid_price(), signal.instrument.min_size_step
                )

        return signals
