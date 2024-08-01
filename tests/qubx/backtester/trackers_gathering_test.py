from typing import Any, Optional, List

from qubx import lookup, logger
from qubx.core.account import AccountProcessor
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.pandaz.utils import *
from qubx.core.utils import recognize_time

from qubx.core.series import Quote
from qubx.core.strategy import IPositionGathering, IStrategy, PositionsTracker, StrategyContext, TriggerEvent
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader, AsTimestampedRecords, AsQuotes, RestoreTicksFromOHLC
from qubx.core.basics import ZERO_COSTS, Deal, Instrument, Order, ITimeProvider, Position, Signal

from qubx.backtester.ome import OrdersManagementEngine

from qubx.ta.indicators import sma, ema
from qubx.backtester.simulator import simulate
from qubx.trackers.rebalancers import PortfolioRebalancerTracker
from qubx.trackers.sizers import FixedRiskSizer, FixedSizer


def Q(time: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(time), bid, ask, 0, 0)


class TestingPositionGatherer(IPositionGathering):
    def alter_position_size(
        self, ctx: StrategyContext, instrument: Instrument, new_size: float, at_price: float | None = None
    ) -> float:
        position = ctx.positions[instrument.symbol]
        current_position = position.quantity
        to_trade = new_size - current_position
        if abs(to_trade) < instrument.min_size:
            logger.warning(
                f"Can't change position size for {instrument}. Current position: {current_position}, requested size: {new_size}"
            )
        else:
            # position.quantity = new_size
            position.update_position(ctx.time(), new_size, ctx.quote(instrument.symbol).mid_price())
            r = ctx.trade(instrument, to_trade, at_price)
            logger.info(
                f"{instrument.symbol} >>> (TESTS) Adjusting position from {current_position} to {new_size} : {r}"
            )
        return current_position

    def update_by_deal_data(self, instrument: Instrument, deal: Deal): ...


class DebugStratageyCtx(StrategyContext):
    def __init__(self, instrs, capital) -> None:
        self.positions = {i.symbol: i for i in instrs}

        self.instruments = instrs
        self.positions = {i.symbol: Position(i) for i in instrs}
        self.capital = capital

        self.acc = AccountProcessor("test", "USDT", reserves={})  # , initial_capital=10000.0)
        self.acc.update_balance("USDT", capital, 0)
        self.acc.attach_positions(*self.positions.values())
        self._n_orders = 0
        self._n_orders_buy = 0
        self._n_orders_sell = 0
        self._orders_size = 0

    def quote(self, symbol: str) -> Quote | None:
        return Q("2020-01-01", 1000.0, 1000.5)

    def get_capital(self) -> float:
        return self.capital

    def time(self) -> np.datetime64:
        return np.datetime64("2020-01-01T00:00:00", "ns")

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

    def get_reserved(self, instrument: Instrument) -> float:
        return 0.0


class TestTrackersAndGatherers:

    def test_simple_tracker_sizer(self):
        ctx = DebugStratageyCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        tracker = PositionsTracker(FixedSizer(1000.0))

        gathering = SimplePositionGatherer()
        i = instrs[0]

        res = gathering.alter_positions(ctx, tracker.process_signals(ctx, [i.signal(1), i.signal(0.5), i.signal(-0.5)]))

        assert ctx._n_orders == 3
        assert ctx._orders_size == 1000.0
        assert ctx._n_orders_buy == 2
        assert ctx._n_orders_sell == 1

    def test_fixed_risk_sizer(self):
        ctx = DebugStratageyCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        i = instrs[0]
        sizer = FixedRiskSizer(10.0)
        s = sizer.calculate_target_positions(ctx, [i.signal(1, stop=900.0)])
        _entry, _stop, _cap_in_risk = 1000.5, 900, 10000 * 10 / 100
        assert s[0].target_position_size == (_cap_in_risk / ((_entry - _stop) / _entry)) / _entry

    def test_rebalancer(self):
        ctx = DebugStratageyCtx(
            I := [
                lookup.find_symbol("BINANCE.UM", "BTCUSDT"),
                lookup.find_symbol("BINANCE.UM", "ETHUSDT"),
                lookup.find_symbol("BINANCE.UM", "SOLUSDT"),
            ],
            30000,
        )
        assert I[0] is not None and I[1] is not None and I[2] is not None

        tracker = PortfolioRebalancerTracker(30000, 0)
        targets = tracker.process_signals(ctx, [I[0].signal(+0.5), I[1].signal(+0.3), I[2].signal(+0.2)])

        gathering = TestingPositionGatherer()
        gathering.alter_positions(ctx, targets)

        print(" - - - - - - - - - - - - - - - - - - - - - - - - -")

        tracker.process_signals(
            ctx,
            [
                I[0].signal(+0.1),
                I[1].signal(+0.8),
                I[2].signal(+0.1),
            ],
        )

        print(" - - - - - - - - - - - - - - - - - - - - - - - - -")

        targets = tracker.process_signals(
            ctx,
            [
                I[0].signal(0),
                I[1].signal(0),
                I[2].signal(0),
            ],
        )
        gathering.alter_positions(ctx, targets)

        assert ctx.positions[I[0].symbol].quantity == 0
        assert ctx.positions[I[1].symbol].quantity == 0
        assert ctx.positions[I[2].symbol].quantity == 0
