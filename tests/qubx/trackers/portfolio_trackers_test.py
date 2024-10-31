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


from qubx.core.metrics import portfolio_metrics
from qubx.ta.indicators import sma, ema
from qubx.backtester.simulator import simulate
from qubx.trackers.composite import CompositeTracker, CompositeTrackerPerSide, LongTracker
from qubx.trackers.rebalancers import PortfolioRebalancerTracker
from qubx.trackers.riskctrl import AtrRiskTracker, StopTakePositionTracker
from qubx.trackers.sizers import FixedLeverageSizer, FixedRiskSizer, FixedSizer, LongShortRatioPortfolioSizer

from pytest import approx

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)


def Q(time: str, p: float) -> Quote:
    return Quote(recognize_time(time), p, p, 0, 0)


def S(ctx, sdict: dict):
    return [ctx.get_instrument(s).signal(v) for s, v in sdict.items()]


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
    _d_qts: Dict[str, Quote]
    _c_time: int = 0
    _o_id: int = 10000

    def __init__(self, symbols: List[str], capital) -> None:
        self.instruments = [x for s in symbols if (x := lookup.find_symbol("BINANCE.UM", s)) is not None]
        self._d_qts = {i.symbol: None for i in self.instruments}
        self.positions = {i.symbol: Position(i) for i in self.instruments}
        self.capital = capital
        self.acc = AccountProcessor("test", "USDT", reserves={})
        self.acc.update_balance("USDT", capital, 0)
        self.acc.attach_positions(*self.positions.values())

    def quote(self, symbol: str) -> Quote | None:
        return self._d_qts.get(symbol)

    def get_capital(self) -> float:
        return self.capital

    def time(self) -> np.datetime64:
        return np.datetime64(self._c_time, "ns")

    def get_reserved(self, instrument: Instrument) -> float:
        return 0.0

    def get_instrument(self, symbol: str) -> Instrument | None:
        return self.positions.get(symbol).instrument

    def push(self, quotes: Dict[str, Quote]) -> None:
        for s, q in quotes.items():
            self._d_qts[s] = q
            self.positions[s].update_market_price_by_tick(q)
            self._c_time = max(q.time, self._c_time)

    def reset(self):
        [p.reset() for p in self.positions.values()]

    def trade(
        self,
        instr_or_symbol: Instrument | str,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **optional,
    ) -> Order:
        side = "BUY" if amount > 0 else "SELL"
        key = instr_or_symbol.symbol if isinstance(instr_or_symbol, Instrument) else instr_or_symbol
        print(" >>> ", side, key, amount)

        order = Order(
            f"test_{self._o_id}",
            "MARKET",
            key,
            np.datetime64(0, "ns"),
            amount,
            price if price is not None else 0,
            side,
            "CLOSED",
            "gtc",
            "test1",
        )

        self._o_id += 1
        d = Deal(str(self.time()), order.id, self.time(), amount, self.quote(key).mid_price(), True)
        self.acc.process_deals(key, [d])
        return order

    def pos_board(self):
        _p_l, _p_s = 0, 0
        for p in self.positions.values():
            print(f"\t{p}")
            _p_l += abs(p.market_value_funds) if p.market_value_funds > 0 else 0
            _p_s += abs(p.market_value_funds) if p.market_value_funds < 0 else 0
        print(f"\tTOTAL net value: {_p_l + _p_s} | L: {_p_l} S: {_p_s}")
        return _p_l, _p_s


class TestPortfolioRelatedStuff:

    def test_portfolio_rebalancer_process_signals(self):
        ctx = DebugStratageyCtx(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            30000,
        )

        I = [ctx.get_instrument(s) for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
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

    def test_LongShortRatioPortfolioSizer(self):
        ctx = DebugStratageyCtx(["MATICUSDT", "CRVUSDT", "NEARUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"], 100000.0)
        prt = PortfolioRebalancerTracker(100_000, 10, LongShortRatioPortfolioSizer(longs_to_shorts_ratio=1))
        g = SimplePositionGatherer()

        # - market data
        ctx.push(
            {
                "MATICUSDT": Q("2022-01-01", 2.5774),
                "CRVUSDT": Q("2022-01-01", 5.569),
                "NEARUSDT": Q("2022-01-01", 14.7149),
                "ETHUSDT": Q("2022-01-01", 3721.67),
                "XRPUSDT": Q("2022-01-01", 0.8392),
                "LTCUSDT": Q("2022-01-01", 149.08),
            }
        )

        # - 'trade'
        c_positions = g.alter_positions(
            ctx,
            prt.process_signals(
                ctx,
                S(
                    ctx,
                    # - how it comes from the strategy
                    {
                        "MATICUSDT": 0.6143356287746169,
                        "CRVUSDT": 0.07459699350250036,
                        "NEARUSDT": 0.31106737772288284,
                        "ETHUSDT": -0.33,
                        "XRPUSDT": -0.33,
                        "LTCUSDT": -0.33,
                    },
                ),
            ),
        )

        _v_l, _v_s = ctx.pos_board()

        # - expected ratio should be close near 1
        assert N(_v_l / _v_s, 10) == 1.0

        # - new data
        ctx.push(
            {
                "MATICUSDT": Q("2022-01-10", 2.0926),
                "CRVUSDT": Q("2022-01-10", 4.497),
                "NEARUSDT": Q("2022-01-10", 13.363),
                "ETHUSDT": Q("2022-01-10", 3135.26),
                "XRPUSDT": Q("2022-01-10", 0.749),
                "LTCUSDT": Q("2022-01-10", 129.8),
            }
        )
        g.alter_positions(
            ctx,
            prt.process_signals(
                ctx,
                S(
                    ctx,
                    {
                        "MATICUSDT": 0.1,
                        "CRVUSDT": 0.2,
                        "NEARUSDT": 0.6,
                        "ETHUSDT": -0.2,
                        "XRPUSDT": -0.1,
                        "LTCUSDT": -0.8,
                    },
                ),
            ),
        )

        _v_l, _v_s = ctx.pos_board()

        # - expected ratio should be close near 1 after rebalance
        assert N(_v_l / _v_s, 10) == 1.0

        # - close short leg
        g.alter_positions(
            ctx,
            prt.process_signals(
                ctx,
                S(
                    ctx,
                    {
                        "MATICUSDT": 0.1,
                        "CRVUSDT": 0.2,
                        "NEARUSDT": 0.6,
                        "ETHUSDT": 0,
                        "XRPUSDT": 0,
                        "LTCUSDT": 0,
                    },
                ),
            ),
        )

        # - no short exposure
        _v_l, _v_s = ctx.pos_board()
        assert _v_s == 0

        # - long exposure ~ cap / 2
        assert N(_v_l, 10) == ctx.acc.get_total_capital() / 2
