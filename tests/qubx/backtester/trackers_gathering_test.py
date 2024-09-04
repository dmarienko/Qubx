from typing import Any, Optional, List

from pandas import Timestamp

from qubx import QubxLogConfig, lookup, logger
from qubx.core.account import AccountProcessor
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.pandaz.utils import *
from qubx.core.utils import recognize_time, time_to_str

from qubx.core.series import OHLCV, Quote
from qubx.core.strategy import IPositionGathering, IStrategy, PositionsTracker, StrategyContext, TriggerEvent
from qubx.data.readers import (
    AsOhlcvSeries,
    CsvStorageDataReader,
    AsTimestampedRecords,
    AsQuotes,
    RestoreTicksFromOHLC,
    AsPandasFrame,
)
from qubx.core.basics import ZERO_COSTS, Deal, Instrument, Order, ITimeProvider, Position, Signal, TargetPosition

from qubx.backtester.ome import OrdersManagementEngine

from qubx.ta.indicators import sma, ema
from qubx.backtester.simulator import simulate
from qubx.trackers.composite import CompositeTracker, CompositeTrackerPerSide, LongTracker
from qubx.trackers.rebalancers import PortfolioRebalancerTracker
from qubx.trackers.riskctrl import AtrRiskTracker, StopTakePositionTracker
from qubx.trackers.sizers import FixedRiskSizer, FixedSizer


def Q(time: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(time), bid, ask, 0, 0)


class TestingPositionGatherer(IPositionGathering):
    def alter_position_size(self, ctx: StrategyContext, target: TargetPosition) -> float:
        instrument, new_size, at_price = target.instrument, target.target_position_size, target.price
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

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal): ...


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
        self,
        instr_or_symbol: Instrument | str,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **optional,
    ) -> Order:
        # fmt: off
        self._n_orders += 1
        self._orders_size += amount
        if amount > 0: self._n_orders_buy += 1
        if amount < 0: self._n_orders_sell += 1
        return Order(
            "test", "MARKET", instr_or_symbol.symbol if isinstance(instr_or_symbol, Instrument) else instr_or_symbol,
            np.datetime64(0, "ns"), amount, price if price is not None else 0, "BUY" if amount > 0 else "SELL", "CLOSED", "gtc", "test1")
        # fmt: on

    def get_reserved(self, instrument: Instrument) -> float:
        return 0.0


class ZeroTracker(PositionsTracker):
    def __init__(self) -> None:
        pass

    def process_signals(self, ctx: StrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        return [TargetPosition.create(ctx, s, target_size=0) for s in signals]


class TestTrackersAndGatherers:

    def test_simple_tracker_sizer(self):
        ctx = DebugStratageyCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        tracker = PositionsTracker(FixedSizer(1000.0, amount_in_quote=False))

        gathering = SimplePositionGatherer()
        i = instrs[0]
        assert i is not None

        res = gathering.alter_positions(ctx, tracker.process_signals(ctx, [i.signal(1), i.signal(0.5), i.signal(-0.5)]))

        assert ctx._n_orders == 3
        assert ctx._orders_size == 1000.0
        assert ctx._n_orders_buy == 2
        assert ctx._n_orders_sell == 1

    def test_fixed_risk_sizer(self):
        ctx = DebugStratageyCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        i = instrs[0]
        assert i is not None

        sizer = FixedRiskSizer(10.0)
        s = sizer.calculate_target_positions(ctx, [i.signal(1, stop=900.0)])
        _entry, _stop, _cap_in_risk = 1000.5, 900, 10000 * 10 / 100
        assert s[0].target_position_size == i.round_size_down((_cap_in_risk / ((_entry - _stop) / _entry)) / _entry)

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

    def test_atr_tracker(self):

        I = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert I is not None

        r = CsvStorageDataReader("tests/data/csv")

        class StrategyForTracking(IStrategy):
            timeframe: str = "1Min"
            fast_period = 5
            slow_period = 12

            def on_event(self, ctx: StrategyContext, event: TriggerEvent) -> List[Signal] | None:
                signals = []
                for i in ctx.instruments:
                    ohlc = ctx.ohlc(i, self.timeframe)
                    fast = sma(ohlc.close, self.fast_period)
                    slow = sma(ohlc.close, self.slow_period)
                    pos = ctx.positions[i.symbol].quantity

                    if pos <= 0 and (fast[0] > slow[0]) and (fast[1] < slow[1]):
                        signals.append(i.signal(+1, stop=min(ohlc[0].low, ohlc[1].low)))

                    if pos >= 0 and (fast[0] < slow[0]) and (fast[1] > slow[1]):
                        signals.append(i.signal(-1, stop=max(ohlc[0].high, ohlc[1].high)))

                return signals

            def tracker(self, ctx: StrategyContext) -> PositionsTracker:
                return PositionsTracker(FixedRiskSizer(1, 10_000, reinvest_profit=True))

        rep = simulate(
            {
                "Strategy ST client": [
                    StrategyForTracking(timeframe="15Min", fast_period=10, slow_period=25),
                    t0 := StopTakePositionTracker(
                        None, None, sizer=FixedRiskSizer(1, 10_000), risk_controlling_side="client"
                    ),
                ],
                "Strategy ST broker": [
                    StrategyForTracking(timeframe="15Min", fast_period=10, slow_period=25),
                    t1 := StopTakePositionTracker(
                        None, None, sizer=FixedRiskSizer(1, 10_000), risk_controlling_side="broker"
                    ),
                ],
                "Strategy ATR client": [
                    StrategyForTracking(timeframe="15Min", fast_period=10, slow_period=25),
                    t2 := AtrRiskTracker(
                        5,
                        5,
                        "15Min",
                        25,
                        atr_smoother="kama",
                        sizer=FixedRiskSizer(1, 10_000),
                        risk_controlling_side="client",
                    ),
                ],
                "Strategy ATR broker": [
                    StrategyForTracking(timeframe="15Min", fast_period=10, slow_period=25),
                    t3 := AtrRiskTracker(
                        5,
                        5,
                        "15Min",
                        25,
                        atr_smoother="kama",
                        sizer=FixedRiskSizer(1, 10_000),
                        risk_controlling_side="broker",
                    ),
                ],
            },
            r,
            10000,
            ["BINANCE.UM:BTCUSDT"],
            dict(type="ohlc", timeframe="15Min", nback=0),
            "-1Sec",
            "vip0_usdt",
            "2024-01-01",
            "2024-01-03 13:00",
        )

        assert rep[2].executions_log.iloc[-1].price < rep[3].executions_log.iloc[-1].price
        assert t0.is_active(I) and t1.is_active(I)
        assert not t2.is_active(I) and not t3.is_active(I)

    def test_composite_tracker(self):
        ctx = DebugStratageyCtx(
            I := [
                lookup.find_symbol("BINANCE.UM", "BTCUSDT"),
                lookup.find_symbol("BINANCE.UM", "ETHUSDT"),
                lookup.find_symbol("BINANCE.UM", "SOLUSDT"),
            ],
            30000,
        )
        assert I[0] is not None and I[1] is not None and I[2] is not None

        # 1. Check that we get 0 targets for all symbols
        tracker = CompositeTracker(ZeroTracker(), StopTakePositionTracker())
        targets = tracker.process_signals(ctx, [I[0].signal(+0.5), I[1].signal(+0.3), I[2].signal(+0.2)])
        assert all(t.target_position_size == 0 for t in targets)

        # 2. Check that we get nonzero target positions
        tracker = CompositeTracker(StopTakePositionTracker(sizer=FixedSizer(1.0, amount_in_quote=False)))
        targets = tracker.process_signals(ctx, [I[0].signal(+0.5), I[1].signal(+0.3), I[2].signal(+2.0)])
        assert targets[0].target_position_size == 0.5
        assert targets[1].target_position_size == 0.3
        assert (  # SOL has 1 as min_size_step so anything below 1 would be rounded to 0
            targets[2].target_position_size == 2.0
        )

        # 3. Check that allow_override works
        tracker = CompositeTracker(StopTakePositionTracker())
        targets = tracker.process_signals(ctx, [I[0].signal(0, options=dict(allow_override=True)), I[0].signal(+0.5)])
        assert targets[0].target_position_size == 0.5

    def test_long_short_trackers(self):
        ctx = DebugStratageyCtx(
            I := [
                lookup.find_symbol("BINANCE.UM", "BTCUSDT"),
            ],
            30000,
        )
        assert I[0] is not None

        # 1. Check that tracker skips the signal if it is not long
        tracker = LongTracker(StopTakePositionTracker())
        targets = tracker.process_signals(ctx, [I[0].signal(-0.5)])
        assert not targets

        # 2. Check that tracker sends 0 target if it was active before
        tracker = LongTracker(StopTakePositionTracker())
        _ = tracker.process_signals(ctx, [I[0].signal(+0.5)])
        targets = tracker.process_signals(ctx, [I[0].signal(-0.5)])
        assert isinstance(targets, list) and targets[0].target_position_size == 0

    def test_composite_per_side_tracker(self):
        ctx = DebugStratageyCtx(
            I := [
                lookup.find_symbol("BINANCE.UM", "BTCUSDT"),
                lookup.find_symbol("BINANCE.UM", "ETHUSDT"),
            ],
            30000,
        )
        assert I[0] is not None and I[1] is not None

        # 1. Check that long and short signals are processed by corresponding trackers
        tracker = CompositeTrackerPerSide(
            long_trackers=[StopTakePositionTracker(10, 5)], short_trackers=[StopTakePositionTracker(5, 5)]
        )
        targets = tracker.process_signals(ctx, [I[0].signal(-0.5), I[1].signal(+0.5)])
        short_target = StopTakePositionTracker(5, 5).process_signals(ctx, [I[0].signal(-0.5)])
        long_target = StopTakePositionTracker(10, 5).process_signals(ctx, [I[1].signal(+0.5)])
        assert targets[0].signal.stop == short_target[0].signal.stop
        assert targets[1].signal.stop == long_target[0].signal.stop

        # 2. Check that sending an opposite side signal is processed correctly
        targets = tracker.process_signals(ctx, [I[0].signal(+0.5)])
        assert targets[0].target_position_size == 0.5

    def test_tracker_with_stop_loss_in_advance(self):
        # from qubx.core.series import st
        class GuineaPig(IStrategy):
            tests = {}

            def on_fit(
                self, ctx: StrategyContext, fit_time: str | Timestamp, previous_fit_time: str | Timestamp | None = None
            ):
                self.tests = {recognize_time(k): v for k, v in self.tests.items()}

            def on_event(self, ctx: StrategyContext, event: TriggerEvent) -> List[Signal] | None:
                r = []
                for k in list(self.tests.keys()):
                    if event.time >= k:
                        r.append(self.tests.pop(k))
                        # print(time_to_str(event.time))
                return r

        I = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert I is not None
        ohlc = CsvStorageDataReader("tests/data/csv").read(
            "BTCUSDT_ohlcv_M1", start="2024-01-01", stop="2024-01-15", transform=AsPandasFrame()
        )
        assert isinstance(ohlc, pd.DataFrame)

        result = simulate(
            {
                "TEST_StopTakePositionTracker": [
                    GuineaPig(tests={"2024-01-01 20:00:00": I.signal(-1, stop=43800)}),
                    t1 := StopTakePositionTracker(None, None, sizer=FixedRiskSizer(1), risk_controlling_side="client"),
                ],
                "TEST2_AdvancedStopTakePositionTracker": [
                    GuineaPig(
                        tests={
                            "2024-01-01 20:00:00": I.signal(-1, stop=43800, take=43400),
                            "2024-01-01 23:10:00": I.signal(+1, stop=43400, take=44200),
                            "2024-01-02 00:00:00": I.signal(+1, stop=43500, take=45500),
                            "2024-01-02 01:10:00": I.signal(-1, stop=45500, take=44800),
                        }
                    ),
                    t2 := StopTakePositionTracker(None, None, sizer=FixedRiskSizer(1), risk_controlling_side="broker"),
                ],
            },
            {f"BINANCE.UM:BTCUSDT": ohlc},
            10000,
            instruments=[f"BINANCE.UM:BTCUSDT"],
            subscription=dict(type="ohlc", timeframe="1Min"),
            trigger="-1Sec",
            silent=True,
            debug="DEBUG",
            commissions="vip0_usdt",
            start="2024-01-01",
            stop="2024-01-03",
        )
        assert len(result[0].executions_log) == 2
        assert not t1.is_active(I)

        assert len(result[1].executions_log) == 7
        assert len(result[1].signals_log) == 7
        assert result[1].signals_log.iloc[-1]["service"] == True
        assert not t2.is_active(I)
