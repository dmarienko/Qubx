import pandas as pd
from pytest import approx

from qubx import logger, lookup
from qubx.backtester.simulator import simulate
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    DataType,
    Deal,
    Instrument,
    Order,
    Position,
    Signal,
    TargetPosition,
)
from qubx.core.interfaces import (
    IPositionGathering,
    IStrategy,
    IStrategyContext,
    PositionsTracker,
    TriggerEvent,
)
from qubx.core.metrics import portfolio_metrics
from qubx.core.series import OHLCV, Quote
from qubx.core.utils import recognize_time, time_to_str
from qubx.data.readers import (
    AsPandasFrame,
    CsvStorageDataReader,
)
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.pandaz.utils import *
from qubx.ta.indicators import ema, sma
from qubx.trackers.composite import CompositeTracker, CompositeTrackerPerSide, LongTracker
from qubx.trackers.riskctrl import AtrRiskTracker, StopTakePositionTracker
from qubx.trackers.sizers import FixedLeverageSizer, FixedRiskSizer, FixedSizer

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)


def Q(time: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(time), bid, ask, 0, 0)


class TestingPositionGatherer(IPositionGathering):
    def alter_position_size(self, ctx: IStrategyContext, target: TargetPosition) -> float:
        instrument, new_size, at_price = target.instrument, target.target_position_size, target.price
        position = ctx.positions[instrument]
        current_position = position.quantity
        to_trade = new_size - current_position
        if abs(to_trade) < instrument.min_size:
            logger.warning(
                f"Can't change position size for {instrument}. Current position: {current_position}, requested size: {new_size}"
            )
        else:
            # position.quantity = new_size
            q = ctx.quote(instrument)
            assert q is not None
            position.update_position(ctx.time(), new_size, q.mid_price())
            r = ctx.trade(instrument, to_trade, at_price)
            logger.info(
                f"{instrument.symbol} >>> (TESTS) Adjusting position from {current_position} to {new_size} : {r}"
            )
        return current_position

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal): ...


class DebugStratageyCtx(IStrategyContext):
    def __init__(self, instrs, capital) -> None:
        self._instruments = instrs
        self.capital = capital

        positions = {i: Position(i) for i in instrs}
        self.account = BasicAccountProcessor("test", "USDT")  # , initial_capital=10000.0)
        self.account.update_balance("USDT", capital, 0)
        self.account.attach_positions(*positions.values())
        self._n_orders = 0
        self._n_orders_buy = 0
        self._n_orders_sell = 0
        self._orders_size = 0

    @property
    def instruments(self) -> list[Instrument]:
        return self._instruments

    def quote(self, symbol: str) -> Quote | None:
        return Q("2020-01-01", 1000.0, 1000.5)

    def get_capital(self) -> float:
        return self.capital

    def get_total_capital(self) -> float:
        return self.capital

    def time(self) -> np.datetime64:
        return np.datetime64("2020-01-01T00:00:00", "ns")

    def trade(
        self,
        instrument: Instrument,
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
            "test", "MARKET", instrument,
            np.datetime64(0, "ns"), amount, price if price is not None else 0, "BUY" if amount > 0 else "SELL", "CLOSED", "gtc", "test1")
        # fmt: on


class ZeroTracker(PositionsTracker):
    def __init__(self) -> None:
        pass

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        return [TargetPosition.create(ctx, s, target_size=0) for s in signals]


class GuineaPig(IStrategy):
    """
    Simple signals player
    """

    tests = {}

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1Min"])

    def on_fit(self, ctx: IStrategyContext):
        self.tests = {recognize_time(k): v for k, v in self.tests.items()}

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | None:
        r = []
        for k in list(self.tests.keys()):
            if event.time >= k:
                r.append(self.tests.pop(k))
        return r


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

    def test_atr_tracker(self):
        I = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert I is not None

        r = CsvStorageDataReader("tests/data/csv")

        class StrategyForTracking(IStrategy):
            timeframe: str = "1Min"
            fast_period = 5
            slow_period = 12

            def on_init(self, ctx: IStrategyContext) -> None:
                ctx.set_base_subscription(DataType.OHLC[self.timeframe])

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | None:
                signals = []
                for i in ctx.instruments:
                    ohlc = ctx.ohlc(i, self.timeframe)
                    fast = sma(ohlc.close, self.fast_period)
                    slow = sma(ohlc.close, self.slow_period)
                    pos = ctx.positions[i].quantity

                    if pos <= 0 and (fast[0] > slow[0]) and (fast[1] < slow[1]):
                        signals.append(i.signal(+1, stop=min(ohlc[0].low, ohlc[1].low)))

                    if pos >= 0 and (fast[0] < slow[0]) and (fast[1] > slow[1]):
                        signals.append(i.signal(-1, stop=max(ohlc[0].high, ohlc[1].high)))

                return signals

            def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
                return PositionsTracker(FixedRiskSizer(1, 10_000, reinvest_profit=True))

        rep = simulate(
            strategies={
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
            data={"ohlc": r, "quote": r},
            capital=10000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2024-01-01",
            stop="2024-01-03 13:00",
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
        tracker = LongTracker(StopTakePositionTracker(risk_controlling_side="client"))
        targets = tracker.process_signals(ctx, [I[0].signal(-0.5)])
        assert not targets

        # 2. Check that tracker sends 0 target if it was active before
        tracker = LongTracker(StopTakePositionTracker(risk_controlling_side="client"))
        _ = tracker.process_signals(ctx, [I[0].signal(+0.5)])

        # - now tracker works only by execution reports, so we 'emulate' it here
        ctx.positions[I[0]].quantity = +0.5
        tracker.on_execution_report(ctx, I[0], Deal(0, "0", np.datetime64(10000, "ns"), +0.5, 1.0, True))

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
            {"ohlc": {"BTCUSDT": ohlc}},
            10000,
            instruments=["BINANCE.UM:BTCUSDT"],
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

    def test_stop_loss_broker_side(self):
        T = pd.Timestamp
        reader = CsvStorageDataReader("tests/data/csv")
        assert isinstance(ohlc := reader.read("BTCUSDT_ohlcv_M1", transform=AsPandasFrame()), pd.DataFrame)
        assert (i1 := lookup.find_symbol("BINANCE.UM", "BTCUSDT")) is not None

        # fmt: off
        S = pd.DataFrame({ 
            i1: {
                    T("2024-01-10 15:08:59.716000"): 1,
                    T("2024-01-10 15:10:52.679000"): 1,
                    T("2024-01-10 15:32:44.798000"): 1,
                    T("2024-01-10 15:59:55.303000"): 1,
                    T("2024-01-10 16:09:00.970000"): 1,
                    T("2024-01-10 16:12:34.233000"): 1,
                    T("2024-01-10 19:04:00.905000"): 1,
                    T("2024-01-10 19:44:37.785000"): 1,
                    T("2024-01-10 20:06:00.322000"): 1,
                }
            }
        )

        rep = simulate(
            {
                "liq_buy_bounces_c": [S, StopTakePositionTracker(2.5, 0.5, FixedLeverageSizer(0.1), "client")],
                "liq_buy_bounces_b": [S, StopTakePositionTracker(2.5, 0.5, FixedLeverageSizer(0.1), "broker")],
            },
            {"ohlc": {"BTCUSDT": ohlc}}, 10000, ["BINANCE.UM:BTCUSDT"], "vip9_usdt",
            S.index[0] - pd.Timedelta("5Min"), S.index[-1] + pd.Timedelta("5Min"),
            signal_timeframe="1Min", debug="DEBUG"
        )
        assert len(rep[0].executions_log) == len(rep[1].executions_log)

        mtrx0 = portfolio_metrics(
            rep[0].portfolio_log, rep[0].executions_log, rep[0].capital, account_transactions=False, commission_factor=1
        )
        mtrx1 = portfolio_metrics(
            rep[1].portfolio_log, rep[1].executions_log, rep[1].capital, account_transactions=False, commission_factor=1
        )
        assert N(mtrx0["gain"]) == 22.6519
        assert N(mtrx1["gain"]) == 23.3365

        # fmt: on
