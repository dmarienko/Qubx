from typing import Any, Optional, List

from qubx import lookup, logger
from qubx.pandaz.utils import *

from qubx.core.series import Quote
from qubx.core.utils import recognize_time
from qubx.core.strategy import IStrategy, StrategyContext, TriggerEvent
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader, AsTimestampedRecords, AsQuotes, RestoreTicksFromOHLC
from qubx.core.basics import ZERO_COSTS, Deal, Instrument, Order, ITimeProvider

from qubx.backtester.ome import OrdersManagementEngine

from qubx.ta.indicators import sma, ema
from qubx.backtester.simulator import simulate


class _TimeService(ITimeProvider):
    def g(self, quote: Quote) -> Quote:
        self._time = quote.time
        return quote

    def time(self) -> np.datetime64:
        return self._time


def Q(time: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(time), bid, ask, 0, 0)


class TestBacktesterStuff:

    def test_basic_ome(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)

        q0 = Q("2020-01-01 10:00", 32000, 32001)
        ome.update_bbo(t.g(q0))

        r0 = ome.place_order("BUY", "MARKET", 0.04, 0, "Test1")
        assert r0.order.status == "CLOSED"
        assert r0.exec is not None
        assert r0.exec.amount == 0.04

        r1 = ome.place_order("SELL", "LIMIT", 0.1, q0.bid, "Test2")
        assert r1.order.status == "CLOSED"
        assert r1.exec is not None
        assert r1.exec.amount == -0.1

        r2 = ome.place_order("BUY", "LIMIT", 0.04, q0.bid - 100, "Test2")
        assert r2.order.status == "OPEN"
        assert r2.exec is None

        r3 = ome.place_order("BUY", "LIMIT", 0.1, q0.bid - 100, "Test3")
        assert r3.order.status == "OPEN"
        assert r3.exec is None

        r4 = ome.place_order("SELL", "LIMIT", 0.04, q0.ask + 100, "Test4")
        assert r4.order.status == "OPEN"
        assert r4.exec is None

        r5 = ome.place_order("SELL", "LIMIT", 0.14, q0.ask + 50, "Test5")
        assert r5.order.status == "OPEN"
        assert r5.order.client_id == "Test5"
        assert r5.exec is None

        r6 = ome.place_order("SELL", "LIMIT", 0.3, q0.ask, "Test6")
        assert r6.order.status == "OPEN"
        assert r6.exec is None

        r7 = ome.place_order("BUY", "LIMIT", 0.12, q0.bid, "Test7")
        assert r7.order.status == "OPEN"
        assert r7.exec is None

        assert len(ome.get_open_orders()) == 6

        ome.cancel_order(r6.order.id)
        assert len(ome.get_open_orders()) == 5

        try:
            ome.cancel_order(r6.order.id)
            assert False
        except:
            assert True

        ome.cancel_order(r7.order.id)
        ome.cancel_order(r5.order.id)
        ome.cancel_order(r4.order.id)
        ome.cancel_order(r3.order.id)
        rc2 = ome.cancel_order(r2.order.id)
        assert rc2.order.status == "CANCELED"
        assert len(ome.get_open_orders()) == 0

    def test_ome_execution(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)

        q0 = Q("2020-01-01 10:00", 32000, 32001)
        ome.update_bbo(t.g(q0))

        r1 = ome.place_order("SELL", "LIMIT", 0.3, 32001, "Test1")
        r2 = ome.place_order("BUY", "LIMIT", 0.3, 32000, "Test2")

        # - nothing changed - no reports
        rs = ome.update_bbo(t.g(Q("2020-01-01 10:01", 32000, 32001)))
        assert not rs

        rs = ome.update_bbo(t.g(Q("2020-01-01 10:01", 32002, 32003)))
        assert rs[0].exec is not None
        assert rs[0].exec.aggressive == False
        assert rs[0].exec.price == 32001

        rs = ome.update_bbo(t.g(Q("2020-01-01 10:01", 31899, 31900)))
        assert rs[0].exec is not None
        assert rs[0].exec.aggressive == False
        assert rs[0].exec.price == 32000

        assert len(ome.get_open_orders()) == 0

    def test_ome_loop(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        r = CsvStorageDataReader("tests/data/csv")
        stream = r.read("BTCUSDT_ohlcv_M1", transform=RestoreTicksFromOHLC(trades=False, spread=instr.min_tick))

        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)
        ome.update_bbo(t.g(stream[0]))
        l1 = ome.place_order("BUY", "LIMIT", 0.5, 39500.0, "Test1")
        l2 = ome.place_order("SELL", "LIMIT", 0.5, 52000.0, "Test2")

        execs = []
        for i in range(len(stream)):
            rs = ome.update_bbo(t.g(stream[i]))
            if rs:
                execs.append(rs[0].exec)

        assert l1.order.status == "CLOSED"
        assert l2.order.status == "CLOSED"
        assert execs[0].price == 39500.0
        assert execs[1].price == 52000.0

    def test_simulator(self):

        class CrossOver(IStrategy):
            timeframe: str = "1Min"
            fast_period = 5
            slow_period = 12

            def on_event(self, ctx: StrategyContext, event: TriggerEvent):
                for i in ctx.instruments:
                    ohlc = ctx.ohlc(i, self.timeframe)
                    fast = ema(ohlc.close, self.fast_period)
                    slow = ema(ohlc.close, self.slow_period)
                    pos = ctx.positions[i.symbol].quantity
                    if pos <= 0:
                        if (fast[0] > slow[0]) and (fast[1] < slow[1]):
                            ctx.trade(i, abs(pos) + i.min_size * 10)
                    if pos >= 0:
                        if (fast[0] < slow[0]) and (fast[1] > slow[1]):
                            ctx.trade(i, -pos - i.min_size * 10)
                return None

            def ohlcs(self, timeframe: str) -> Dict[str, pd.DataFrame]:
                return {s.symbol: self.ctx.ohlc(s, timeframe).pd() for s in self.ctx.instruments}

        r = CsvStorageDataReader("tests/data/csv")
        ohlc = r.read("BINANCE.UM:BTCUSDT", "2024-01-01", "2024-01-02", AsOhlcvSeries("5Min"))
        fast = ema(ohlc.close, 5)
        slow = ema(ohlc.close, 15)
        sigs = (((fast > slow) + (fast.shift(1) < slow.shift(1))) == 2) - (
            ((fast < slow) + (fast.shift(1) > slow.shift(1))) == 2
        )
        sigs = sigs.pd()
        sigs = sigs[sigs != 0]
        s2 = shift_series(sigs, "4Min59Sec").rename("BTCUSDT") / 100
        rep1 = simulate(
            {
                # - generated signals as series
                "test0": CrossOver(timeframe="5Min", fast_period=5, slow_period=15),
                "test1": s2,
            },
            r,
            10000,
            ["BINANCE.UM:BTCUSDT"],
            dict(type="ohlc", timeframe="5Min", nback=0),
            "5Min -1Sec",
            "vip0_usdt",
            "2024-01-01",
            "2024-01-02",
            n_jobs=1,
        )

        assert all(
            rep1[0].executions_log[["filled_qty", "price", "side"]]
            == rep1[1].executions_log[["filled_qty", "price", "side"]]
        )
