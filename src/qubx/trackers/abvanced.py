from dataclasses import dataclass

import numpy as np

from qubx import logger
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition
from qubx.core.interfaces import IPositionSizer, IStrategyContext, PositionsTracker
from qubx.core.series import OHLCV, Bar, Quote, Trade
from qubx.trackers.riskctrl import State, StopTakePositionTracker


@dataclass
class _AdvCtrl:
    signal: Signal
    entry: float
    stop: float
    entry_bar_time: int
    is_being_tracked: bool = True

    def is_long(self) -> bool:
        return self.signal.signal > 0


class ImprovedEntryTracker(PositionsTracker):
    """
    Advanced entry tracker with additional features like tracking price improvements, reassigning stops, and stop-take functionality.

    Provides the same functionality as StopTakePositionTracker but sends take/stop as limit/stop orders
    immediately after the tracked position is opened. If new signal is received it should adjust take and stop orders.

    TODO: Need test for ImprovedEntryTracker (AdvancedTrackers)
    """

    timeframe: str
    _tracker: StopTakePositionTracker
    _ohlcs: dict[Instrument, OHLCV]
    _entries: dict[Instrument, _AdvCtrl]

    def __init__(
        self,
        timeframe: str,
        sizer: IPositionSizer,
        track_price_improvements: bool = True,
        reassign_stops: bool = True,  # set stop to signal's bar low / high
        take_target: float | None = None,
        stop_risk: float | None = None,
    ) -> None:
        super().__init__(sizer)
        self.track_price_improvements = track_price_improvements
        self.reassign_stops = reassign_stops
        self.timeframe = timeframe
        self._ohlcs = {}
        self._tracker = StopTakePositionTracker(
            take_target=take_target, stop_risk=stop_risk, sizer=sizer, risk_controlling_side="broker"
        )
        self._entries = {}

    def ohlc(self, instrument: Instrument) -> OHLCV:
        """
        We need this for make it works in multiprocessing env
        """
        r = self._ohlcs.get(instrument)
        if r is None:
            self._ohlcs[instrument] = (r := OHLCV(instrument.symbol, self.timeframe, 3))
        return r

    def is_active(self, instrument: Instrument) -> bool:
        if instrument in self._entries:
            return self._entries[instrument].is_being_tracked
        return False

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        _sgs = []
        for s in signals:
            instrument = s.instrument
            if s.signal == 0:
                if instrument in self._entries:
                    logger.debug(
                        f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <W>REMOVING FROM</W> got 0 signal - stop entry processing"
                    )
                    self._entries.pop(instrument)
            else:
                # - buy at signal bar high or sell at signal bar low
                signal_bar: Bar = self.ohlc(instrument)[0]
                # - reassign new entry and stop to signal
                ent_price = signal_bar.high if s.signal > 0 else signal_bar.low
                stp_price = signal_bar.low if s.signal > 0 else signal_bar.high
                s.price = ent_price  # new entry
                if self.reassign_stops or s.stop is None:  # reassign new stop to signal if signal has no stops
                    s.stop = stp_price

                # - case 1: there's already tracked entry
                if self._is_entry_tracked_for(instrument):
                    # - ask tracker to cancel entry stop order
                    logger.debug(
                        f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <Y><k> Cancel entry because new signal </k></Y></g>"
                    )
                    _sgs.append(instrument.signal(0, comment=f"{s.comment} - Cancel entry because new signal"))

                # - case 2: position is opened and StopTake tracker tracks it
                if self._tracker_has_position(instrument):
                    logger.debug(
                        f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <Y><k> Close position because new signal </k></Y>"
                    )
                    _sgs.append(instrument.signal(0, comment=f"{s.comment} Close positon because new signal"))

                # - new entry for tracking
                self._entries[instrument] = _AdvCtrl(s, entry=ent_price, stop=s.stop, entry_bar_time=signal_bar.time)

            _sgs.append(s)

        return self._tracker.process_signals(ctx, _sgs)

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar
    ) -> list[TargetPosition] | TargetPosition:
        _new_bar = False
        if isinstance(update, Bar):
            _new_bar = self.ohlc(instrument).update_by_bar(
                update.time, update.open, update.high, update.low, upd := update.close, update.volume
            )
        elif isinstance(update, Quote):
            _new_bar = self.ohlc(instrument).update(update.time, upd := update.mid_price(), 0)
        elif isinstance(update, Trade):
            _new_bar = self.ohlc(instrument).update(update.time, upd := update.price, update.size)
        else:
            raise ValueError(f"Unknown update type: {type(update)}")

        _res = []
        if self._is_entry_tracked_for(instrument):
            s = self._entries[instrument]
            _is_long = s.is_long()

            if (_is_long and upd < s.stop) or (not _is_long and upd > s.stop):
                logger.debug(
                    f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <R><k>CANCELING ENTRY</k></R> : {s.signal} ||| {upd}"
                )
                self._entries.pop(instrument)
                return self._tracker.process_signals(
                    ctx,
                    [
                        instrument.signal(
                            0, comment=f"Cancel: price {upd} broke entry's {'low' if _is_long else 'high'} at {s.stop}"
                        )
                    ],
                )

            # - if we need to improve entry on new bar
            if _new_bar and self.track_price_improvements:
                bar = self.ohlc(instrument)[1]
                if _is_long and bar.high < s.entry:
                    logger.debug(
                        f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <M><k>IMPROVING LONG ENTRY</k></M> : {s.signal.price} -> {bar.high}"
                    )
                    s.signal.price = bar.high
                    s.signal.take = self.get_new_take_on_improve(ctx, s.signal, bar)
                    s.entry = bar.high
                    _res.extend(self._tracker.process_signals(ctx, [s.signal]))
                elif not _is_long and bar.low > s.entry:
                    logger.debug(
                        f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <M><k>IMPROVING SHORT ENTRY</k></M> : {s.signal.price} -> {bar.low}"
                    )
                    s.signal.price = bar.low
                    s.signal.take = self.get_new_take_on_improve(ctx, s.signal, bar)
                    s.entry = bar.low
                    _res.extend(self._tracker.process_signals(ctx, [s.signal]))

        if _tu := self._tracker.update(ctx, instrument, update):
            _res.extend(_tu)
        return _res

    def get_new_take_on_improve(self, ctx: IStrategyContext, s: Signal, b: Bar) -> float | None:
        """
        What's new take target after improve (remains the same by default)
        """
        return s.take

    def _is_entry_tracked_for(self, instrument: Instrument):
        return instrument in self._entries and (s := self._entries[instrument]).is_being_tracked

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal):
        self._tracker.on_execution_report(ctx, instrument, deal)

        # - tracker becomes active - so position is open
        if self._tracker_has_position(instrument):
            logger.debug(f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <M><w> Position is open </w></M>")
            if instrument in self._entries:
                self._entries[instrument].is_being_tracked = False

        if self._tracker_triggers_risk(instrument):
            logger.debug(f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: <R><k> Risk triggered </k></R>")
            if instrument in self._entries:
                self._entries.pop(instrument)

    def _tracker_has_position(self, instrument: Instrument) -> bool:
        # TODO: implementation is too StopTakePositionTracker dependent !
        # TODO: need generic mehod to check if tracker is active / has open positions
        if instrument in self._tracker.riskctrl._trackings:
            st = self._tracker.riskctrl._trackings[instrument].status
            return st == State.OPEN
        return False

    def _tracker_triggers_risk(self, instrument: Instrument) -> bool:
        if instrument in self._tracker.riskctrl._trackings:
            st = self._tracker.riskctrl._trackings[instrument].status
            return st == State.RISK_TRIGGERED
        return False


class ImprovedEntryTrackerDynamicTake(ImprovedEntryTracker):
    """
    Updates take profit target on improved entries

    TODO: Need test for ImprovedEntryTrackerDynamicTake (AdvancedTrackers)
    """

    risk_reward_ratio: float

    def __init__(
        self,
        timeframe: str,
        sizer: IPositionSizer,
        risk_reward_ratio: float,
        track_price_improvements: bool = True,
        reassign_stops: bool = True,  # set stop to signal's bar low / high
    ) -> None:
        super().__init__(timeframe, sizer, track_price_improvements, reassign_stops, None, None)
        self.risk_reward_ratio = risk_reward_ratio

    def get_new_take_on_improve(self, ctx: IStrategyContext, s: Signal, b: Bar) -> float | None:
        """
        Use risk reward ratio to calculate new take target
        """
        if (t := s.take) is not None and s.price is not None and s.stop is not None:
            t = s.price + np.sign(s.signal) * self.risk_reward_ratio * abs(s.price - s.stop)
        return t
