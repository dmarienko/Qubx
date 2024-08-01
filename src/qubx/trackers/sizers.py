from typing import List
import numpy as np

from qubx import logger
from qubx.core.basics import Position, Signal, TargetPosition
from qubx.core.strategy import IPositionSizer, StrategyContext
from qubx.utils.misc import round_down_at_min_qty


class FixedSizer(IPositionSizer):
    """
    Simplest fixed sizer class. It uses same fixed size for all signals.
    We use it for quick backtesting of generated signals in most cases.
    """

    def __init__(self, fixed_size):
        self.fixed_size = abs(fixed_size)

    def calculate_target_positions(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        return [TargetPosition(s, s.signal * self.fixed_size) for s in signals]


class FixedRiskSizer(IPositionSizer):
    def __init__(self, max_cap_in_risk: float, max_allowed_position=np.inf, reinvest_profit=True):
        """
        Create fixed risk sizer calculator instance.
        :param max_cap_in_risk: maximal risked capital (in percentage)
        :param max_allowed_position: limitation for max position size in quoted currency (i.e. max 5000 in USDT)
        :param reinvest_profit: if true use profit to reinvest
        """
        self.max_cap_in_risk = max_cap_in_risk / 100
        self.max_allowed_position_quoted = max_allowed_position
        self.reinvest_profit = reinvest_profit

    def calculate_target_positions(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        t_pos = []
        for signal in signals:
            target_position_size = 0
            if signal.signal != 0:
                if signal.stop and signal.stop > 0:
                    _pos = ctx.positions[signal.instrument.symbol]
                    _q = ctx.quote(signal.instrument.symbol)

                    _direction = np.sign(signal.signal)
                    # - hey, we can't trade using negative balance ;)
                    _cap = max(
                        ctx.acc.get_total_capital() + (max(_pos.total_pnl(), 0) if self.reinvest_profit else 0), 0
                    )
                    _entry = _q.ask if _direction > 0 else _q.bid
                    # fmt: off
                    target_position_size = (  
                        _direction
                        *min((_cap * self.max_cap_in_risk) / abs(signal.stop / _entry - 1), self.max_allowed_position_quoted) / _entry
                    )  
                    # fmt: on

                else:
                    logger.warning(
                        f" >>> {self.__class__.__name__}: stop is not specified for {str(signal)} - can't calculate position !"
                    )
                    continue

            t_pos.append(TargetPosition(signal, target_position_size))

        return t_pos


class WeightedPortfolioSizer(IPositionSizer):

    def __init__(self, capital_using: float = 1.0):
        """
        Weighted portfolio sizer. Signals are cosidered as weigths.

        :param cap_used_pct: how much of total capital to be used for sizing
        """
        self.capital_using = min(abs(capital_using), 1.0)

    def calculate_target_positions(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        sw = np.sum([max(s.signal, 0) for s in signals])

        # - in case if all positions need to be closed
        sw = 1 if sw == 0 else sw

        cap = self.capital_using * ctx.get_capital()
        t_pos = []

        for signal in signals:
            # _pos = ctx.positions[signal.instrument.symbol]
            _q = ctx.quote(signal.instrument.symbol)
            if _q is not None:
                t_pos.append(
                    TargetPosition(
                        signal,
                        round_down_at_min_qty(
                            cap * max(signal.signal, 0) / sw / _q.mid_price(), signal.instrument.min_size_step
                        ),
                    )
                )
            else:
                logger.error(
                    f"{self.__class__.__name__}: Can't get actual market quote for {signal.instrument.symbol} !"
                )

        return t_pos
