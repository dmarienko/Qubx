from typing import Any, Optional, List

from qubx import lookup, logger
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.pandaz.utils import *
from qubx.core.utils import recognize_time

from qubx.core.series import Quote
from qubx.core.strategy import IStrategy, PositionsTracker, StrategyContext, TriggerEvent
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader, AsTimestampedRecords, AsQuotes, RestoreTicksFromOHLC
from qubx.core.basics import ZERO_COSTS, Deal, Instrument, Order, ITimeProvider, Position, Signal

from qubx.backtester.ome import OrdersManagementEngine

from qubx.ta.indicators import sma, ema
from qubx.backtester.simulator import simulate
from qubx.trackers.sizers import FixedSizer


def Q(time: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(time), bid, ask, 0, 0)


class DebugStratageyCtx(StrategyContext):
    def __init__(self, instrs) -> None:
        self.positions = {i.symbol: i for i in instrs}

        self.instruments = instrs
        self.positions = {i.symbol: Position(i) for i in instrs}
        # acc: AccountProcessor
        self._n_orders = 0
        self._n_orders_buy = 0
        self._n_orders_sell = 0
        self._orders_size = 0

    def trade(
        self, instr_or_symbol: Instrument | str, amount: float, price: float | None = None, time_in_force="gtc"
    ) -> Order:
        self._n_orders += 1
        self._orders_size += amount
        if amount > 0:
            self._n_orders_buy += 1
        if amount < 0:
            self._n_orders_sell += 1
        return Order(
            "test",
            "MARKET",
            instr_or_symbol.symbol if isinstance(instr_or_symbol, Instrument) else instr_or_symbol,
            np.datetime64(0, "ns"),
            amount,
            price if price is not None else 0,
            "BUY" if amount > 0 else "SELL",
            "CLOSED",
            "gtc",
            "test1",
        )


class TestTrackersAndGatherers:

    def test_simple_tracker_sizer(self):
        ctx = DebugStratageyCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")])
        tracker = PositionsTracker(FixedSizer(1000.0))

        gathering = SimplePositionGatherer()
        i = instrs[0]
        tracker.process_signals(ctx, gathering, [i.signal(1), i.signal(0.5), i.signal(-0.5)])

        assert ctx._n_orders == 3
        assert ctx._orders_size == 1000.0
        assert ctx._n_orders_buy == 2
        assert ctx._n_orders_sell == 1
