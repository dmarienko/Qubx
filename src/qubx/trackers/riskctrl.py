from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal

import numpy as np

from qubx import logger
from qubx.core.basics import Deal, Instrument, OrderStatus, Signal, TargetPosition
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


class RiskCalculator:
    def calculate_risks(self, ctx: StrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        return signal


class RiskController(PositionsTracker):
    _trackings: Dict[Instrument, SgnCtrl]
    _waiting: Dict[Instrument, SgnCtrl]
    _risk_calculator: RiskCalculator

    def __init__(self, risk_calculator: RiskCalculator, sizer: IPositionSizer) -> None:
        self._risk_calculator = risk_calculator
        self._trackings = {}
        self._waiting = {}
        super().__init__(sizer)

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

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        targets = []
        for s in signals:
            quote = ctx.quote(s.instrument.symbol)
            if quote is None:
                logger.warning(f"Quote not available for {s.instrument.symbol}. Skipping signal {s}")
                continue

            # - calculate risk
            signal_with_risk = self._risk_calculator.calculate_risks(ctx, quote, s)
            if signal_with_risk is None:
                continue

            # - final step - calculate actual target position and check if tracker can approve it
            target = self.get_position_sizer().calculate_target_positions(ctx, [signal_with_risk])[0]
            if self.handle_new_target(ctx, s, target):
                targets.append(target)

        return targets

    def handle_new_target(self, ctx: StrategyContext, signal: Signal, target: TargetPosition) -> bool:
        """
        As it doesn't use any referenced orders for position - new target is always approved
        """
        # - add first in waiting list
        self._waiting[signal.instrument] = SgnCtrl(signal, target, State.NEW)
        logger.debug(
            f"<yellow>{self.__class__.__name__}</yellow> <g>new signal received:</g> <cyan><b>{target}</b></cyan> for {signal.instrument.symbol} take: {signal.take} stop: {signal.stop}"
        )

        return True

    def is_active(self, instrument: Instrument) -> bool:
        return instrument in self._trackings


class ClientSideRiskController(RiskController):
    """
    Risk is controlled on client (Qubx) side without using limit order for take and stop order for loss.
    So when risk is triggered, it uses market orders to close position immediatelly.
    As result it may lead to significant slippage.
    """

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
                            ctx, instrument.signal(0, group="Risk Manager", comment="Stop triggered")
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
                                group="Risk Manager",
                                comment="Take triggered",
                            ),
                        )

            case State.DONE:
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> -- stops tracking -- <green>{instrument.symbol}</green>"
                )
                self._trackings.pop(instrument)

        return []

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        pos = ctx.positions[instrument.symbol].quantity

        # - check what is in the waiting list
        if (c_w := self._waiting.get(instrument)) is not None:
            if abs(pos - c_w.target.target_position_size) <= instrument.min_size:
                c_w.status = State.OPEN
                self._trackings[instrument] = c_w  # add to tracking
                self._waiting.pop(instrument)  # remove from waiting
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> -- starts tracking -- <cyan><b>{c_w.target}</b></cyan> of {c_w.signal.instrument.symbol} take: {c_w.signal.take} stop: {c_w.signal.stop}"
                )
                return

        # - check what is in the tracking list
        if (c_t := self._trackings.get(instrument)) is not None:
            if c_t.status == State.RISK_TRIGGERED and abs(pos) <= instrument.min_size:
                c_t.status = State.DONE


class BrokerSideRiskController(RiskController):
    """
    Risk is managed on the broker's side by using limit orders for take and stop order for loss.
    For backtesting we assume that stop orders are executed by it's price.
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

                # - remove from the tracking list
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> -- stops tracking -- <green>{instrument.symbol}</green>"
                )
                self._trackings.pop(instrument)

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

            case State.DONE:
                logger.debug(
                    f"<yellow>{self.__class__.__name__}</yellow> -- stops tracking -- <green>{instrument.symbol}</green>"
                )
                self._trackings.pop(instrument)

        # fmt: on
        return []

    def __cncl_stop(self, ctx: StrategyContext, ctrl: SgnCtrl):
        if ctrl.stop_order_id is not None:
            logger.debug(
                f"<yellow>{self.__class__.__name__}</yellow> <m>-- canceling stop order --</m> <red>{ctrl.stop_order_id}</red> for {ctrl.signal.instrument.symbol}"
            )
            ctx.cancel_order(ctrl.stop_order_id)
            ctrl.stop_order_id = None

    def __cncl_take(self, ctx: StrategyContext, ctrl: SgnCtrl):
        if ctrl.take_order_id is not None:
            logger.debug(
                f"<yellow>{self.__class__.__name__}</yellow> <m>-- canceling take order --</m> <red>{ctrl.take_order_id}</red> for {ctrl.signal.instrument.symbol}"
            )
            ctx.cancel_order(ctrl.take_order_id)
            ctrl.take_order_id = None

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        pos = ctx.positions[instrument.symbol].quantity

        if (c_w := self._waiting.get(instrument)) is not None:
            if abs(pos - c_w.target.target_position_size) <= instrument.min_size:
                c_w.status = State.OPEN

                # - check if we need to cancel previous stop / take orders
                ctr1 = self._trackings.get(instrument)
                if ctr1 is not None:
                    self.__cncl_stop(ctx, ctr1)
                    self.__cncl_take(ctx, ctr1)

                self._trackings[instrument] = c_w  # add to tracking
                self._waiting.pop(instrument)  # remove from waiting

                if c_w.target.take:
                    try:
                        logger.debug(
                            f"<yellow>{self.__class__.__name__}</yellow> is sending take limit order for <green>{instrument.symbol}</green> at {c_w.target.take}"
                        )
                        order = ctx.trade(instrument, -pos, c_w.target.take)
                        c_w.take_order_id = order.id

                        # - if order was executed immediately we don't need to send stop order
                        if order.status == "CLOSED":
                            c_w.status = State.RISK_TRIGGERED
                            logger.debug(
                                f"<yellow>{self.__class__.__name__}</yellow> <g>TAKE PROFIT</g> was exected immediately for <green>{instrument.symbol}</green> at {c_w.target.take}"
                            )
                            return

                    except Exception as e:
                        logger.error(
                            f"<yellow>{self.__class__.__name__}</yellow> couldn't send take limit order for <green>{instrument.symbol}</green>: {str(e)}"
                        )

                if c_w.target.stop:
                    try:
                        logger.debug(
                            f"<yellow>{self.__class__.__name__}</yellow> is sending stop order for <green>{instrument.symbol}</green> at {c_w.target.stop}"
                        )
                        # - for simulation purposes we assume that stop order will be executed at stop price
                        order = ctx.trade(
                            instrument, -pos, c_w.target.stop, stop_type="market", fill_at_signal_price=True
                        )
                        c_w.stop_order_id = order.id
                    except Exception as e:
                        logger.error(
                            f"<yellow>{self.__class__.__name__}</yellow> couldn't send stop order for <green>{instrument.symbol}</green>: {str(e)}"
                        )

        # - check tracked signal
        if (c_t := self._trackings.get(instrument)) is not None:
            if c_t.status == State.OPEN and abs(pos) <= instrument.min_size:
                if deal.order_id == c_t.take_order_id:
                    c_t.status = State.RISK_TRIGGERED
                    c_t.take_executed_price = deal.price
                    logger.debug(
                        f"<yellow>{self.__class__.__name__}</yellow> triggered <green>TAKE PROFIT</green> (<red>{c_t.take_order_id}</red>) for <green>{instrument.symbol}</green> at {c_t.take_executed_price}"
                    )
                    # - cancel stop if need
                    self.__cncl_stop(ctx, c_t)

                elif deal.order_id == c_t.stop_order_id:
                    c_t.status = State.RISK_TRIGGERED
                    c_t.stop_executed_price = deal.price
                    logger.debug(
                        f"<yellow>{self.__class__.__name__}</yellow> triggered <magenta>STOP LOSS</magenta> (<red>{c_t.take_order_id}</red>) for <green>{instrument.symbol}</green> at {c_t.stop_executed_price}"
                    )
                    # - cancel take if need
                    self.__cncl_take(ctx, c_t)

                else:
                    # - closed by opposite signal or externally
                    c_t.status = State.DONE
                    self.__cncl_stop(ctx, c_t)
                    self.__cncl_take(ctx, c_t)


class GenericRiskControllerDecorator(PositionsTracker, RiskCalculator):
    riskctrl: RiskController

    def __init__(
        self,
        sizer: IPositionSizer,
        riskctrl: RiskController,
    ) -> None:
        super().__init__(sizer)
        self.riskctrl = riskctrl

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        return self.riskctrl.process_signals(ctx, signals)

    def is_active(self, instrument: Instrument) -> bool:
        return self.riskctrl.is_active(instrument)

    def update(
        self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar
    ) -> List[TargetPosition] | TargetPosition:
        return self.riskctrl.update(ctx, instrument, update)

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        return self.riskctrl.on_execution_report(ctx, instrument, deal)

    def calculate_risks(self, ctx: StrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        raise NotImplementedError("calculate_risks should be implemented by subclasses")


class StopTakePositionTracker(GenericRiskControllerDecorator):
    """
    Basic fixed stop-take position tracker. It observes position opening or closing and controls stop-take logic.
    It may use either limit and stop orders for managing risk or market orders depending on 'risk_controlling_side' parameter.
    """

    def __init__(
        self,
        take_target: float | None = None,
        stop_risk: float | None = None,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
        risk_controlling_side: str = "broker",
    ) -> None:
        self.take_target = take_target
        self.stop_risk = stop_risk
        self._take_target_fraction = take_target / 100 if take_target else None
        self._stop_risk_fraction = stop_risk / 100 if stop_risk else None

        super().__init__(
            sizer,
            (
                BrokerSideRiskController(self, sizer)
                if risk_controlling_side == "broker"
                else ClientSideRiskController(self, sizer)
            ),
        )

    def calculate_risks(self, ctx: StrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        if signal.signal > 0:
            entry = signal.price if signal.price else quote.ask
            if self._take_target_fraction:
                signal.take = entry * (1 + self._take_target_fraction)
            if self._stop_risk_fraction:
                signal.stop = entry * (1 - self._stop_risk_fraction)

        elif signal.signal < 0:
            entry = signal.price if signal.price else quote.bid
            if self._take_target_fraction:
                signal.take = entry * (1 - self._take_target_fraction)
            if self._stop_risk_fraction:
                signal.stop = entry * (1 + self._stop_risk_fraction)

        return signal


class AtrRiskTracker(GenericRiskControllerDecorator):
    """
    ATR based risk management
    Take at entry +/- ATR[1] * take_target
    Stop at entry -/+ ATR[1] * stop_risk
    It may use either limit and stop orders for managing risk or market orders depending on 'risk_controlling_side' parameter.
    """

    def __init__(
        self,
        take_target: float | None,
        stop_risk: float | None,
        atr_timeframe: str,
        atr_period: int,
        atr_smoother="sma",
        sizer: IPositionSizer = FixedSizer(1.0),
        risk_controlling_side: str = "broker",
    ) -> None:
        self.take_target = take_target
        self.stop_risk = stop_risk
        self.atr_timeframe = atr_timeframe
        self.atr_period = atr_period
        self.atr_smoother = atr_smoother

        super().__init__(
            sizer,
            (
                BrokerSideRiskController(self, sizer)
                if risk_controlling_side == "broker"
                else ClientSideRiskController(self, sizer)
            ),
        )

    def calculate_risks(self, ctx: StrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        volatility = atr(
            ctx.ohlc(signal.instrument, self.atr_timeframe),
            self.atr_period,
            smoother=self.atr_smoother,
            percentage=False,
        )
        if len(volatility) < 2:
            return None

        last_volatility = volatility[1]
        if last_volatility is None or not np.isfinite(last_volatility) or quote is None:
            return None

        if signal.signal > 0:
            entry = signal.price if signal.price else quote.ask
            if self.stop_risk:
                signal.stop = entry - self.stop_risk * last_volatility
            if self.take_target:
                signal.take = entry + self.take_target * last_volatility

        elif signal.signal < 0:
            entry = signal.price if signal.price else quote.bid
            if self.stop_risk:
                signal.stop = entry + self.stop_risk * last_volatility
            if self.take_target:
                signal.take = entry - self.take_target * last_volatility

        return signal


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
