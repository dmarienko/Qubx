from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np

from qubx import logger
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition
from qubx.core.series import Bar, Quote, Trade
from qubx.core.strategy import IPositionSizer, PositionsTracker, StrategyContext
from qubx.trackers.sizers import FixedRiskSizer, FixedSizer

from qubx.ta.indicators import atr

State = Literal[
    "NEW",
    "OPEN",
    "RISK-TRIGGERED",
    "DONE",
]


@dataclass
class SgnCtrl:
    signal: Signal
    target: TargetPosition
    status: State = "NEW"


class AtrRiskTracker(PositionsTracker):
    """
    ATR based risk management
    Take at entry +/- ATR[1] * take_target
    Stop at entry -/+ ATR[1] * stop_risk
    """

    _signals: Dict[Instrument, SgnCtrl]

    def __init__(
        self,
        take_target: float,
        stop_risk: float,
        atr_timeframe: str,
        atr_period: int,
        atr_smoother="sma",
        sizer: IPositionSizer = FixedSizer(1.0),
    ) -> None:
        self.take_target = take_target
        self.stop_risk = stop_risk
        self.atr_timeframe = atr_timeframe
        self.atr_period = atr_period
        self.atr_smoother = atr_smoother
        self._signals = dict()
        super().__init__(sizer)

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        targets = []
        for s in signals:
            volatility = atr(
                ctx.ohlc(s.instrument, self.atr_timeframe),
                self.atr_period,
                smoother=self.atr_smoother,
                percentage=False,
            )
            last_volatility = volatility[1]
            quote = ctx.quote(s.instrument.symbol)
            if last_volatility is None or not np.isfinite(last_volatility) or quote is None:
                continue

            if s.signal > 0:
                entry = s.price if s.price else quote.ask
                s.stop = entry - self.stop_risk * last_volatility
                s.take = entry + self.take_target * last_volatility

            elif s.signal < 0:
                entry = s.price if s.price else quote.bid
                s.stop = entry + self.stop_risk * last_volatility
                s.take = entry - self.take_target * last_volatility

            target = self.get_position_sizer().calculate_target_positions(ctx, [s])[0]
            targets.append(target)
            self._signals[s.instrument] = SgnCtrl(s, target, "NEW")
            logger.debug(
                f"\t ::: <yellow>Start tracking {target}</yellow> of {s.instrument.symbol} take: {s.take} stop: {s.stop}"
            )

        return targets

    @staticmethod
    def _get_price(update: float | Quote | Trade | Bar, direction: int) -> float:
        if isinstance(update, float):
            return update
        elif isinstance(update, Quote):
            return update.ask if direction > 0 else update.bid  # type: ignore
        elif isinstance(update, Trade):
            return update.price  # type: ignore
        elif isinstance(update, Bar):
            return update.close  # type: ignore
        else:
            raise ValueError(f"Unknown update type: {type(update)}")

    def update(self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar) -> List[TargetPosition]:
        c = self._signals.get(instrument)
        if c is None:
            return []

        match c.status:
            case "NEW":
                # - nothing to do just waiting for position to be open
                pass

            case "TRISK-TRIGGERED":
                # - nothing to do just waiting for position to be closed
                pass

            case "OPEN":
                pos = ctx.positions[instrument.symbol].quantity
                if c.signal.stop:
                    if (
                        pos > 0
                        and self._get_price(update, +1) <= c.signal.stop
                        or (pos < 0 and self._get_price(update, -1) >= c.signal.stop)
                    ):
                        c.status = "RISK-TRIGGERED"
                        logger.debug(f"\t ::: <red>Stop triggered</red> for {instrument.symbol} at {c.signal.stop}")
                        return [TargetPosition(instrument.signal(0, group="Risk Manager", comment="Stop triggered"), 0)]

                if c.signal.take:
                    if (
                        pos > 0
                        and self._get_price(update, -1) >= c.signal.take
                        or (pos < 0 and self._get_price(update, +1) <= c.signal.take)
                    ):
                        c.status = "RISK-TRIGGERED"
                        logger.debug(f"\t ::: <green>Take triggered</green> for {instrument.symbol} at {c.signal.take}")
                        return [TargetPosition(instrument.signal(0, group="Risk Manager", comment="Take triggered"), 0)]

            case "DONE":
                logger.debug(f"\t ::: <yellow>Stop tracking</yellow> {instrument.symbol}")
                self._signals.pop(instrument)

        return []

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        c = self._signals.get(instrument)
        if c is None:
            return

        if abs(ctx.positions[instrument.symbol].quantity - c.target.target_position_size) <= instrument.min_size:
            c.status = "OPEN"

        if c.target.target_position_size == 0:
            c.status = "DONE"
