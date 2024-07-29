import numpy as np

from qubx import logger
from qubx.core.basics import Position, Signal
from qubx.core.strategy import IPositionSizer, StrategyContext


class FixedSizer(IPositionSizer):
    """
    Simplest fixed sizer class. It uses same fixed size for all signals.
    We use it for quick backtesting of generated signals in most cases.
    """

    def __init__(self, fixed_size):
        self.fixed_size = abs(fixed_size)

    def get_position_size(self, ctx: StrategyContext, signal: Signal) -> float:
        return signal.signal * self.fixed_size


class FixedRiskSizer(IPositionSizer):
    def __init__(self, max_cap_in_risk: float, max_allowed_position=np.inf):
        """
        Create fixed risk sizer calculator instance.
        :param max_cap_in_risk: maximal risked capital (in percentage)
        :param max_allowed_position: limitation for max position size
        """
        self.max_cap_in_risk = max_cap_in_risk
        self.max_allowed_position = max_allowed_position

    def get_position_size(self, ctx: StrategyContext, signal: Signal) -> float:
        if signal.signal != 0:
            if signal.stop and signal.stop > 0:
                _pos = ctx.positions[signal.instrument.symbol]
                _q = ctx.quote(signal.instrument.symbol)

                _direction = np.sign(signal.signal)
                _cap = ctx.get_capital() + max(_pos.total_pnl(), 0)
                _entry = _q.ask if _direction > 0 else _q.bid

                pos_size = _direction * min(
                    round((_cap * self.max_cap_in_risk / 100) / abs(signal.stop / _entry - 1)),
                    self.max_allowed_position,
                )

                return pos_size
            else:
                logger.warning(" >>> FixedRiskSizer: stop is not specified - can't calculate position !")

        return 0
