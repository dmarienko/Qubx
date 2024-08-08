from collections import defaultdict
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

        _symbol_to_min_target = {
            symbol: min(targets, key=lambda target: abs(target.target_position_size))
            for symbol, targets in _symbol_to_targets.items()
        }

        return list(_symbol_to_min_target.values())
