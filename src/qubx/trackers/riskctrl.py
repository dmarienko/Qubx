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


class StopTakePositionTracker(PositionsTracker):
    _signals: Dict[Instrument, SgnCtrl]

    def __init__(
        self,
        take_target: float | None = None,
        stop_risk: float | None = None,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
    ) -> None:
        self.take_target = take_target
        self.stop_risk = stop_risk
        self._signals = dict()
        super().__init__(sizer)
        self._take_target_fraction = take_target / 100 if take_target else None
        self._stop_risk_fraction = stop_risk / 100 if stop_risk else None

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        targets = []
        for s in signals:
            quote = ctx.quote(s.instrument.symbol)
            if quote is None:
                logger.warning(f"Quote not available for {s.instrument.symbol}. Skipping signal {s}")
                continue

            if s.signal > 0:
                entry = s.price if s.price else quote.ask
                if self._take_target_fraction:
                    s.take = entry * (1 + self._take_target_fraction)
                if self._stop_risk_fraction:
                    s.stop = entry * (1 - self._stop_risk_fraction)

            elif s.signal < 0:
                entry = s.price if s.price else quote.bid
                if self._take_target_fraction:
                    s.take = entry * (1 - self._take_target_fraction)
                if self._stop_risk_fraction:
                    s.stop = entry * (1 + self._stop_risk_fraction)

            target = self.get_position_sizer().calculate_target_positions(ctx, [s])[0]
            targets.append(target)
            self._signals[s.instrument] = SgnCtrl(s, target, "NEW")
            logger.debug(
                f"\t ::: <yellow>Start tracking {target}</yellow> of {s.instrument.symbol} take: {s.take} stop: {s.stop}"
            )

        return targets

    def is_active(self, instrument: Instrument) -> bool:
        return instrument in self._signals

    def reset(self, instrument: Instrument):
        self._signals.pop(instrument, None)

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

    def update(
        self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar
    ) -> List[TargetPosition] | TargetPosition:
        c = self._signals.get(instrument)
        if c is None:
            return []

        match c.status:
            case "NEW":
                # - nothing to do just waiting for position to be open
                pass

            case "RISK-TRIGGERED":
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
                        return TargetPosition.zero(
                            ctx,
                            instrument.signal(
                                0,
                                price=c.signal.stop,
                                group="Risk Manager",
                                comment="Stop triggered",
                                options=dict(fill_at_signal_price=True),
                            ),
                        )

                if c.signal.take:
                    if (
                        pos > 0
                        and self._get_price(update, -1) >= c.signal.take
                        or (pos < 0 and self._get_price(update, +1) <= c.signal.take)
                    ):
                        c.status = "RISK-TRIGGERED"
                        logger.debug(f"\t ::: <green>Take triggered</green> for {instrument.symbol} at {c.signal.take}")
                        return TargetPosition.zero(
                            ctx,
                            instrument.signal(
                                0,
                                price=c.signal.take,
                                group="Risk Manager",
                                comment="Take triggered",
                                options=dict(fill_at_signal_price=True),
                            ),
                        )

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


class AtrRiskTracker(StopTakePositionTracker):
    """
    ATR based risk management
    Take at entry +/- ATR[1] * take_target
    Stop at entry -/+ ATR[1] * stop_risk
    """

    def __init__(
        self,
        take_target: float | None,
        stop_risk: float | None,
        atr_timeframe: str,
        atr_period: int,
        atr_smoother="sma",
        sizer: IPositionSizer = FixedSizer(1.0),
    ) -> None:
        self.atr_timeframe = atr_timeframe
        self.atr_period = atr_period
        self.atr_smoother = atr_smoother
        super().__init__(take_target, stop_risk, sizer)

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        targets = []
        for s in signals:
            volatility = atr(
                ctx.ohlc(s.instrument, self.atr_timeframe),
                self.atr_period,
                smoother=self.atr_smoother,
                percentage=False,
            )
            if len(volatility) < 2:
                continue
            last_volatility = volatility[1]
            quote = ctx.quote(s.instrument.symbol)
            if last_volatility is None or not np.isfinite(last_volatility) or quote is None:
                continue

            if s.signal > 0:
                entry = s.price if s.price else quote.ask
                if self.stop_risk:
                    s.stop = entry - self.stop_risk * last_volatility
                if self.take_target:
                    s.take = entry + self.take_target * last_volatility

            elif s.signal < 0:
                entry = s.price if s.price else quote.bid
                if self.stop_risk:
                    s.stop = entry + self.stop_risk * last_volatility
                if self.take_target:
                    s.take = entry - self.take_target * last_volatility

            target = self.get_position_sizer().calculate_target_positions(ctx, [s])[0]
            targets.append(target)
            self._signals[s.instrument] = SgnCtrl(s, target, "NEW")
            logger.debug(
                f"\t ::: <yellow>Start tracking {target}</yellow> of {s.instrument.symbol} take: {s.take} stop: {s.stop}"
            )

        return targets


class MinAtrExitDistanceTracker(PositionsTracker):
    """
    Allow exit only if price has moved away from entry by the specified distance in ATR units.
    """

    _signals: dict[str, Signal]

    def __init__(
        self,
        take_target: float | None,
        stop_target: float | None,
        atr_timeframe: str,
        atr_period: int,
        atr_smoother="sma",
        sizer: IPositionSizer = FixedSizer(1.0),
    ) -> None:
        super().__init__(sizer)
        self.atr_timeframe = atr_timeframe
        self.atr_period = atr_period
        self.atr_smoother = atr_smoother
        self.take_target = take_target
        self.stop_target = stop_target
        self._signals = dict()

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        targets = []
        for s in signals:
            volatility = atr(
                ctx.ohlc(s.instrument, self.atr_timeframe),
                self.atr_period,
                smoother=self.atr_smoother,
                percentage=False,
            )
            if len(volatility) < 2:
                continue
            last_volatility = volatility[1]
            quote = ctx.quote(s.instrument.symbol)
            if last_volatility is None or not np.isfinite(last_volatility) or quote is None:
                continue

            self._signals[s.instrument.symbol] = s

            if s.signal != 0:
                # if signal is not 0, atr thresholds don't apply
                # set expected stop price in case sizer needs it
                if s.stop is None:
                    price = quote.ask if s.signal > 0 else quote.bid
                    s.stop = (
                        price - self.stop_target * last_volatility
                        if s.signal > 0
                        else price + self.stop_target * last_volatility
                    )

                target = self.get_position_sizer().calculate_target_positions(ctx, [s])[0]
                targets.append(target)
                continue

            if self.__check_exit(ctx, s.instrument):
                logger.debug(f"\t ::: <yellow>Min ATR distance reached</yellow> for {s.instrument.symbol}")
                targets.append(TargetPosition.zero(ctx, s))

        return targets

    def update(
        self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar
    ) -> List[TargetPosition] | TargetPosition:
        signal = self._signals.get(instrument.symbol)
        if signal is None or signal.signal != 0:
            return []
        if not self.__check_exit(ctx, instrument):
            return []
        logger.debug(f"\t ::: <yellow>Min ATR distance reached</yellow> for {instrument.symbol}")
        return TargetPosition.zero(
            ctx, instrument.signal(0, group="Risk Manager", comment=f"Original signal price: {signal.reference_price}")
        )

    def __check_exit(self, ctx: StrategyContext, instrument: Instrument) -> bool:
        volatility = atr(
            ctx.ohlc(instrument, self.atr_timeframe),
            self.atr_period,
            smoother=self.atr_smoother,
            percentage=False,
        )
        if len(volatility) < 2:
            return False

        last_volatility = volatility[1]
        quote = ctx.quote(instrument.symbol)
        if last_volatility is None or not np.isfinite(last_volatility) or quote is None:
            return False

        pos = ctx.positions.get(instrument.symbol)
        if pos is None:
            return False

        entry = pos.position_avg_price
        allow_exit = False
        if pos.quantity > 0:
            stop = entry - self.stop_target * last_volatility
            take = entry + self.take_target * last_volatility
            if quote.bid <= stop or quote.ask >= take:
                allow_exit = True
        else:
            stop = entry + self.stop_target * last_volatility
            take = entry - self.take_target * last_volatility
            if quote.ask >= stop or quote.bid <= take:
                allow_exit = True
        return allow_exit
