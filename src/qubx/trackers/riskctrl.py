from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal

import numpy as np

from qubx import logger
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition
from qubx.core.series import Bar, Quote, Trade
from qubx.core.strategy import IPositionSizer, PositionsTracker, StrategyContext
from qubx.trackers.sizers import FixedRiskSizer, FixedSizer

from qubx.ta.indicators import atr


class State(Enum):
    NEW = 0
    OPEN = 1
    RISK_TRIGGERED = 2
    DONE = 3


@dataclass
class SgnCtrl:
    signal: Signal
    target: TargetPosition
    status: State = State.NEW
    take_order_id: str | None = None
    stop_order_id: str | None = None
    take_executed_price: float | None = None
    stop_executed_price: float | None = None


class StopTakePositionTracker(PositionsTracker):
    """
    Basic stop-take position tracker. It observes position opening or closing and controls stop-take logic.
    It doesn't use any limit or stop order but processes market quotes in real time and verifies
    if any risk level is crossed, then generates a zero position.
    One drawback of this approach is the additional slippage that might occur due to stop-take logic latency.
    """

    _trackings: Dict[Instrument, SgnCtrl]

    def __init__(
        self,
        take_target: float | None = None,
        stop_risk: float | None = None,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
    ) -> None:
        self.take_target = take_target
        self.stop_risk = stop_risk
        self._trackings = dict()
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

            # - final step - calculate actual target position and check if tracker can approve it
            target = self.get_position_sizer().calculate_target_positions(ctx, [s])[0]
            if self._handle_new_target(ctx, s, target):
                targets.append(target)

        return targets

    def _handle_new_target(self, ctx: StrategyContext, signal: Signal, target: TargetPosition) -> bool:
        """
        As it doesn't use any referenced orders for position - new target is always approved
        """
        self._trackings[signal.instrument] = SgnCtrl(signal, target, State.NEW)
        logger.debug(
            f"<yellow>{self.__class__.__name__}</yellow> started tracking <cyan><b>{target}</b></cyan> of {signal.instrument.symbol} take: {signal.take} stop: {signal.stop}"
        )
        return True

    def is_active(self, instrument: Instrument) -> bool:
        return instrument in self._trackings

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
        c = self._trackings.get(instrument)
        if c is None:
            return []

        match c.status:
            case State.NEW:
                # - nothing to do just waiting for position to be open
                pass

            case State.RISK_TRIGGERED:
                # - nothing to do just waiting for position to be closed
                pass

            case State.OPEN:
                pos = c.target.target_position_size
                if c.signal.stop:
                    if (
                        pos > 0
                        and self._get_price(update, +1) <= c.signal.stop
                        or (pos < 0 and self._get_price(update, -1) >= c.signal.stop)
                    ):
                        c.status = State.RISK_TRIGGERED
                        logger.debug(
                            f"<yellow>{self.__class__.__name__}</yellow> triggered <red>STOP LOSS</red> for <green>{instrument.symbol}</green> at {c.signal.stop}"
                        )
                        return TargetPosition.zero(
                            ctx,
                            instrument.signal(
                                0,
                                price=c.signal.stop,
                                group="Risk Manager",
                                comment="Stop triggered",
                            ),
                        )

                if c.signal.take:
                    if (
                        pos > 0
                        and self._get_price(update, -1) >= c.signal.take
                        or (pos < 0 and self._get_price(update, +1) <= c.signal.take)
                    ):
                        c.status = State.RISK_TRIGGERED
                        logger.debug(
                            f"<yellow>{self.__class__.__name__}</yellow> triggered <green>TAKE PROFIT</green> for <green>{instrument.symbol}</green> at {c.signal.take}"
                        )
                        return TargetPosition.zero(
                            ctx,
                            instrument.signal(
                                0,
                                price=c.signal.take,
                                group="Risk Manager",
                                comment="Take triggered",
                            ),
                        )

            case State.DONE:
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> stops tracking <green>{instrument.symbol}</green>"
                )
                self._trackings.pop(instrument)

        return []

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        c = self._trackings.get(instrument)
        if c is None:
            return

        pos = ctx.positions[instrument.symbol].quantity
        if abs(pos - c.target.target_position_size) <= instrument.min_size:
            c.status = State.OPEN

        if c.status == State.RISK_TRIGGERED and abs(pos) <= instrument.min_size:
            c.status = State.DONE


class AdvancedStopTakePositionTracker(StopTakePositionTracker):
    """
    Provides the same functionality as StopTakePositionTracker but sends take/stop
    as limit/stop orders immediately after the tracked position is opened.
    If new signal is received it should adjust take and stop orders.
    """

    def update(self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar) -> List[TargetPosition]:
        # fmt: off
        c = self._trackings.get(instrument)
        if c is None:
            return []

        match c.status:
            case State.NEW:
                # - nothing to do just waiting for position to be open
                pass

            case State.RISK_TRIGGERED:
                c.status = State.DONE
                # - send service signal that risk triggeres (it won't be processed by StrategyContext)
                if c.stop_executed_price: 
                    return [
                            TargetPosition.service(
                                ctx, instrument.signal(0, price=c.stop_executed_price, group="Risk Manager", comment="Stop triggered"),
                            )
                    ]
                elif c.take_executed_price: 
                    return [
                            TargetPosition.service(
                                ctx, instrument.signal(0, price=c.take_executed_price, group="Risk Manager", comment="Take triggered"),
                            )
                    ]

            case State.OPEN:
                pass

            case State.DONE:
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> stops tracking <green>{instrument.symbol}</green>"
                )
                self._trackings.pop(instrument)

        # fmt: on
        return []

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        c = self._trackings.get(instrument)
        if c is None:
            return

        pos = ctx.positions[instrument.symbol].quantity
        if abs(pos - c.target.target_position_size) <= instrument.min_size:
            c.status = State.OPEN
            if c.target.take:
                try:
                    logger.debug(
                        f"<yellow>{self.__class__.__name__}</yellow> is sending take limit order for <green>{instrument.symbol}</green> at {c.target.take}"
                    )
                    order = ctx.trade(instrument, -pos, c.target.take)
                    c.take_order_id = order.id
                except Exception as e:
                    logger.error(
                        f"<yellow>{self.__class__.__name__}</yellow> couldn't send take limit order for <green>{instrument.symbol}</green>: {str(e)}"
                    )

            if c.target.stop:
                try:
                    logger.debug(
                        f"<yellow>{self.__class__.__name__}</yellow> is sending stop order for <green>{instrument.symbol}</green> at {c.target.stop}"
                    )
                    # - for simulation purposes we assume that stop order will be executed at stop price
                    order = ctx.trade(instrument, -pos, c.target.stop, stop_type="market", fill_at_signal_price=True)
                    c.stop_order_id = order.id
                except Exception as e:
                    logger.error(
                        f"<yellow>{self.__class__.__name__}</yellow> couldn't send stop order for <green>{instrument.symbol}</green>: {str(e)}"
                    )

        if c.status == State.OPEN and abs(pos) <= instrument.min_size:
            if deal.order_id == c.take_order_id:
                c.status = State.RISK_TRIGGERED
                c.take_executed_price = deal.price
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> triggered <green>TAKE PROFIT</green> (<red>{c.take_order_id}</red>) for <green>{instrument.symbol}</green> at {c.take_executed_price}"
                )
                # - cancel stop if need
                self.__cncl_stop(ctx, c)

            elif deal.order_id == c.stop_order_id:
                c.status = State.RISK_TRIGGERED
                c.stop_executed_price = deal.price
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> triggered <magenta>STOP LOSS</magenta> (<red>{c.take_order_id}</red>) for <green>{instrument.symbol}</green> at {c.stop_executed_price}"
                )
                # - cancel take if need
                self.__cncl_take(ctx, c)

            else:
                # - closed by opposite signal or externally
                c.status = State.DONE
                self.__cncl_stop(ctx, c)
                self.__cncl_take(ctx, c)

    def __cncl_stop(self, ctx: StrategyContext, ctrl: SgnCtrl):
        if ctrl.stop_order_id is not None:
            logger.debug(
                f"<yellow>{self.__class__.__name__}</yellow> canceling stop order <red>{ctrl.stop_order_id}</red> for {ctrl.signal.instrument.symbol}"
            )
            ctx.cancel_order(ctrl.stop_order_id)
            ctrl.stop_order_id = None

    def __cncl_take(self, ctx: StrategyContext, ctrl: SgnCtrl):
        if ctrl.take_order_id is not None:
            logger.debug(
                f"<yellow>{self.__class__.__name__}</yellow> canceling take order <red>{ctrl.stop_order_id}</red> for {ctrl.signal.instrument.symbol}"
            )
            ctx.cancel_order(ctrl.take_order_id)
            ctrl.take_order_id = None

    def _handle_new_target(self, ctx: StrategyContext, signal: Signal, target: TargetPosition) -> bool:
        """
        If new target differs from current and take / stop were sent
        we need to cancel them first
        """
        ctr1 = self._trackings.get(signal.instrument)
        if ctr1 is not None and ctr1.target.target_position_size != target.target_position_size:
            self.__cncl_stop(ctx, ctr1)
            self.__cncl_take(ctx, ctr1)

        return super()._handle_new_target(ctx, signal, target)


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
            self._trackings[s.instrument] = SgnCtrl(s, target, State.NEW)
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
