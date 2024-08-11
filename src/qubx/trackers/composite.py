from collections import defaultdict
from typing import Callable
from qubx.core.series import Bar, Quote, Trade
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition
from qubx.core.strategy import IPositionSizer, PositionsTracker, StrategyContext


Targets = list[TargetPosition] | TargetPosition | None


class CompositeTracker(PositionsTracker):
    """
    Combines multiple trackers. Returns the most conservative target position.
    """

    def __init__(self, *trackers: PositionsTracker) -> None:
        self.trackers = trackers

    def process_signals(self, ctx: StrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        _index_to_targets: dict[int, Targets] = {
            index: tracker.process_signals(ctx, signals) for index, tracker in enumerate(self.trackers)
        }
        return self._select_min_targets(_index_to_targets)

    def update(self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar) -> list[TargetPosition]:
        _index_to_targets: dict[int, Targets] = {
            index: tracker.update(ctx, instrument, update) for index, tracker in enumerate(self.trackers)
        }
        return self._select_min_targets(_index_to_targets)

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        for tracker in self.trackers:
            tracker.on_execution_report(ctx, instrument, deal)

    def _select_min_targets(self, tracker_to_targets: dict[int, Targets]) -> list[TargetPosition]:
        _symbol_to_targets: dict[str, list[TargetPosition]] = defaultdict(list)
        for targets in tracker_to_targets.values():
            if isinstance(targets, list):
                for target in targets:
                    _symbol_to_targets[target.instrument.symbol].append(target)
            elif isinstance(targets, TargetPosition):
                _symbol_to_targets[targets.instrument.symbol].append(targets)

        _symbol_to_targets = self._process_override_signals(_symbol_to_targets)

        _symbol_to_min_target = {
            symbol: min(targets, key=lambda target: abs(target.target_position_size))
            for symbol, targets in _symbol_to_targets.items()
        }

        return list(_symbol_to_min_target.values())

    def _process_override_signals(
        self, symbol_to_targets: dict[str, list[TargetPosition]]
    ) -> dict[str, list[TargetPosition]]:
        """
        Filter out signals that allow override if there is more than one signal for the same symbol.
        """
        filt_symbol_to_targets = {}
        for symbol, targets in symbol_to_targets.items():
            if len(targets) == 1 or all(t.signal.options.get("allow_override", False) for t in targets):
                filt_symbol_to_targets[symbol] = targets
                continue
            filt_symbol_to_targets[symbol] = [
                target for target in targets if not target.signal.options.get("allow_override", False)
            ]
        return filt_symbol_to_targets


class ConditionalTracker(PositionsTracker):
    """
    Wraps a single tracker. Can be used to add some logic before or after the wrapped tracker.
    """

    def __init__(self, condition: Callable[[Signal], bool], tracker: PositionsTracker) -> None:
        self.condition = condition
        self.tracker = tracker

    def process_signals(self, ctx: StrategyContext, signals: list[Signal]) -> list[TargetPosition] | TargetPosition:
        filtered_signals = []
        for signal in signals:
            cond = self.condition(signal)
            if cond:
                filtered_signals.append(signal)
            elif self.tracker.is_active(signal.instrument):
                # This is important for instance if we get an opposite signal
                # we need to at least close the open position
                filtered_signals.append(
                    Signal(
                        instrument=signal.instrument,
                        signal=0,
                        price=signal.price,
                        stop=signal.stop,
                        take=signal.take,
                        reference_price=signal.reference_price,
                        group=signal.group,
                        comment=signal.comment,
                        options=dict(allow_override=True),
                    )
                )
        return self.tracker.process_signals(ctx, filtered_signals)

    def update(
        self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar
    ) -> list[TargetPosition] | TargetPosition:
        return self.tracker.update(ctx, instrument, update)

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        self.tracker.on_execution_report(ctx, instrument, deal)


class LongTracker(ConditionalTracker):
    def __init__(self, tracker: PositionsTracker) -> None:
        super().__init__(self._condition, tracker)

    def _condition(self, signal: Signal) -> bool:
        return signal.signal > 0


class ShortTracker(ConditionalTracker):
    def __init__(self, tracker: PositionsTracker) -> None:
        super().__init__(self._condition, tracker)

    def _condition(self, signal: Signal) -> bool:
        return signal.signal < 0


class CompositeTrackerPerSide(CompositeTracker):
    def __init__(
        self,
        trackers: list[PositionsTracker] | None = None,
        long_trackers: list[PositionsTracker] | None = None,
        short_trackers: list[PositionsTracker] | None = None,
    ):
        if trackers is None and long_trackers is None and short_trackers is None:
            raise ValueError("At least one of trackers, long_trackers or short_trackers must be provided.")
        self.trackers = trackers or []
        self.long_trackers = LongTracker(CompositeTracker(*long_trackers)) if long_trackers else None
        self.short_trackers = ShortTracker(CompositeTracker(*short_trackers)) if short_trackers else None
        if self.long_trackers is not None:
            self.trackers.append(self.long_trackers)
        if self.short_trackers is not None:
            self.trackers.append(self.short_trackers)
        super().__init__(*self.trackers)
