from typing import Callable, Iterable, List, Set, Tuple, Union, Dict, Any

from qubx import logger, lookup
from qubx.core.basics import Instrument, Signal, TriggerEvent, MarketEvent, SubscriptionType
from qubx.core.interfaces import IStrategy, IStrategyContext

from qubx import logger, lookup
from qubx.data import loader
from qubx.backtester.simulator import simulate


class Issue1(IStrategy):
    exchange: str = "BINANCE.UM"
    _idx = 0
    _err = False
    _to_test: List[List[Instrument]] = []

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(SubscriptionType.OHLC, timeframe="1h")
        ctx.set_fit_schedule("59 22 * */1 L7")  # Run at 22:59 every month on Sunday
        ctx.set_event_schedule("55 23 * * *")  # Run at 23:55 every day
        self._to_test = [
            [self.find_instrument(s) for s in ["BTCUSDT", "ETHUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "BCHUSDT", "LTCUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "AAVEUSDT", "ETHUSDT"]],
        ]

    def on_fit(self, ctx: IStrategyContext):
        ctx.set_universe(self._to_test[self._idx])
        logger.info(f" -> SET NEW UNIVERSE {','.join(i.symbol for i in self._to_test[self._idx])}")
        self._idx += 1
        if self._idx > 2:
            self._idx = 0

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal]:
        for s in ctx.instruments:
            q = ctx.quote(s)
            # - quotes should be in ctx already !!!
            if q is None:
                # print(f"\n{s.symbol} -> NO QUOTE\n")
                logger.error(f"\n{s.symbol} -> NO QUOTE\n")
                self._err = True

        return []

    def find_instrument(self, symbol: str) -> Instrument:
        i = lookup.find_symbol(self.exchange, symbol)
        assert i is not None, f"Could not find {self.exchange}:{symbol}"
        return i


class Issue2(IStrategy):
    _fits_called = 0
    _events_called = 0

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(SubscriptionType.OHLC, timeframe="1h")
        ctx.set_fit_schedule("59 22 * * *")  # Run at 22:59 every month on Sunday
        ctx.set_event_schedule("55 23 * * *")  # Run at 23:55 every day
        self._fits_called = 0
        self._events_called = 0

    def on_fit(self, ctx: IStrategyContext):
        logger.info(f" > [{ctx.time()}] On Fit is called")
        self._fits_called += 1

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal]:
        logger.info(f" > [{ctx.time()}] On event is called")
        self._events_called += 1
        return []


class Issue3(IStrategy):
    _fits_called = 0
    _triggers_called = 0
    _market_called = 0

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(SubscriptionType.OHLC, timeframe="1h")
        self._fits_called = 0
        self._triggers_called = 0

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal]:
        self._triggers_called += 1
        return []

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> List[Signal]:
        self._market_called += 1
        return []


class TestSimulator:
    def test_fit_event_quotes(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        test0 = simulate(
            {
                "fail1": (stg := Issue1()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-08-01",
            debug="DEBUG",
            n_jobs=1,
        )

        assert not stg._err, "Got Errors during the simulation"

    def test_scheduled_events(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        test0 = simulate(
            {
                "fail2": (stg := Issue2()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-06-10",
            debug="DEBUG",
            n_jobs=1,
        )

        assert stg._fits_called >= 9, "Got Errors during the simulation"
        assert stg._events_called >= 9, "Got Errors during the simulation"

    def test_market_updates(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        test0 = simulate(
            {
                "fail3": (stg := Issue3()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-06-10",
            debug="DEBUG",
            n_jobs=1,
        )

        assert stg._triggers_called * 4 == stg._market_called, "Got Errors during the simulation"
