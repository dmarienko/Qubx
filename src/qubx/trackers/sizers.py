from typing import List
import numpy as np

from qubx import logger
from qubx.core.basics import Position, Signal, TargetPosition
from qubx.core.strategy import IPositionSizer, StrategyContext


class FixedSizer(IPositionSizer):
    """
    Simplest fixed sizer class. It uses same fixed size for all signals.
    We use it for quick backtesting of generated signals in most cases.
    """

    def __init__(self, fixed_size: float, amount_in_quote: bool = True):
        self.amount_in_quote = amount_in_quote
        self.fixed_size = abs(fixed_size)

    def calculate_target_positions(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        if not self.amount_in_quote:
            return [TargetPosition.create(ctx, s, s.signal * self.fixed_size) for s in signals]
        positions = []
        for signal in signals:
            q = ctx.quote(signal.instrument.symbol)
            if q is None:
                logger.error(
                    f"{self.__class__.__name__}: Can't get actual market quote for {signal.instrument.symbol} !"
                )
                continue
            positions.append(TargetPosition.create(ctx, signal, signal.signal * self.fixed_size / q.mid_price()))
        return positions


class FixedLeverageSizer(IPositionSizer):
    """
    Defines the leverage per each unit of signal. If leverage is 1.0, then
    the position leverage will be equal to the signal value.
    """

    def __init__(self, leverage: float):
        """
        Args:
            leverage (float): leverage value per a unit of signal.
            split_by_symbols (bool): Should the calculated leverage by divided
            by the number of symbols in the universe.
        """
        self.leverage = leverage

    def calculate_target_positions(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        total_capital = ctx.acc.get_total_capital()
        positions = []
        for signal in signals:
            q = ctx.quote(signal.instrument.symbol)
            if q is None:
                logger.error(
                    f"{self.__class__.__name__}: Can't get actual market quote for {signal.instrument.symbol} !"
                )
                continue
            size = signal.signal * self.leverage * total_capital / q.mid_price() / len(ctx.instruments)
            positions.append(TargetPosition.create(ctx, signal, size))
        return positions


class FixedRiskSizer(IPositionSizer):
    def __init__(
        self,
        max_cap_in_risk: float,
        max_allowed_position=np.inf,
        reinvest_profit: bool = True,
        divide_by_symbols: bool = True,
    ):
        """
        Create fixed risk sizer calculator instance.
        :param max_cap_in_risk: maximal risked capital (in percentage)
        :param max_allowed_position: limitation for max position size in quoted currency (i.e. max 5000 in USDT)
        :param reinvest_profit: if true use profit to reinvest
        """
        self.max_cap_in_risk = max_cap_in_risk / 100
        self.max_allowed_position_quoted = max_allowed_position
        self.reinvest_profit = reinvest_profit
        self.divide_by_symbols = divide_by_symbols

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
                    _cap = max(ctx.acc.get_total_capital() if self.reinvest_profit else ctx.acc.get_free_capital(), 0)
                    _entry = _q.ask if _direction > 0 else _q.bid
                    # fmt: off
                    target_position_size = (  
                        _direction
                        *min((_cap * self.max_cap_in_risk) / abs(signal.stop / _entry - 1), self.max_allowed_position_quoted) / _entry
                        / (len(ctx.instruments) if self.divide_by_symbols else 1)
                    )  
                    # fmt: on

                else:
                    logger.warning(
                        f" >>> {self.__class__.__name__}: stop is not specified for {str(signal)} - can't calculate position !"
                    )
                    continue

            t_pos.append(TargetPosition.create(ctx, signal, target_position_size))

        return t_pos


class LongShortRatioPortfolioSizer(IPositionSizer):
    """
    Weighted portfolio sizer. Signals are cosidered as weigths.
    It's supposed to split capital in the given ratio between longs and shorts positions.
    For example if ratio is 1 capital invested in long and short positions should be the same.

    So if we S_l = sum all long signals, S_s = abs sum all short signals, r (longs_shorts_ratio) given ratio

        k_s * S_s + k_l * S_l = 1
        k_l * S_l / k_s * S_s = r

    then

        k_s = 1 / S_s * (1 + r) or 0 if S_s == 0 (no short signals)
        k_l = r / S_l * (1 + r) or 0 if S_l == 0 (no long signals)

    and final positions:
        P_i = S_i * available_capital * capital_using * (k_l if S_i > 0 else k_s)
    """

    _r: float

    def __init__(self, capital_using: float = 1.0, longs_to_shorts_ratio: float = 1):
        """
        Create weighted portfolio sizer.

        :param capital_using: how much of total capital to be used for positions
        :param longs_shorts_ratio: ratio of longs to shorts positions
        """
        assert 0 < capital_using <= 1, f"Capital using factor must be between 0 and 1, got {capital_using}"
        assert 0 < longs_to_shorts_ratio, f"Longs/shorts ratio must be greater 0, got {longs_to_shorts_ratio}"
        self.capital_using = capital_using
        self._r = longs_to_shorts_ratio

    def calculate_target_positions(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        """
        Calculates target positions for each signal using weighted portfolio approach.

        Parameters:
        ctx (StrategyContext): The strategy context containing information about the current state of the strategy.
        signals (List[Signal]): A list of signals generated by the strategy.

        Returns:
        List[TargetPosition]: A list of target positions for each signal, representing the desired size of the position
        in the corresponding instrument.
        """
        total_capital = ctx.acc.get_total_capital()
        cap = self.capital_using * total_capital

        _S_l, _S_s = 0, 0
        for s in signals:
            _S_l += s.signal if s.signal > 0 else 0
            _S_s += abs(s.signal) if s.signal < 0 else 0
        k_s = 1 / (_S_s * (1 + self._r)) if _S_s > 0 else 0
        k_l = self._r / (_S_l * (1 + self._r)) if _S_l > 0 else 0

        t_pos = []
        for signal in signals:
            _q = ctx.quote(signal.instrument.symbol)
            if _q is not None:
                _p_q = cap / _q.mid_price()
                # _t_p = (_c_p / _S_l) if signal.signal > 0 else (_c_p / _S_s) if signal.signal < 0 else 0
                _p = k_l * signal.signal if signal.signal > 0 else k_s * signal.signal
                t_pos.append(TargetPosition.create(ctx, signal, _p * _p_q))
            else:
                logger.error(
                    f"{self.__class__.__name__}: Can't get actual market quote for {signal.instrument.symbol} !"
                )

        return t_pos
