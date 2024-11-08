from typing import Callable, Iterable, List, Set, Tuple, Union, Dict, Any

from qubx import logger, lookup
from qubx.core.basics import Instrument, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, SubscriptionType

from qubx import logger, lookup
from qubx.data import loader
from qubx.backtester.simulator import simulate


class Issue1(IStrategy):
    _to_test: List[List[Instrument]] = []
    _idx = 0
    _err = False

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(SubscriptionType.OHLC, timeframe="1h")
        ctx.set_fit_schedule("59 22 * */1 L7")  # Run at 22:59 every month on Sunday
        ctx.set_event_schedule("55 23 * * *")  # Run at 23:55 every day

        self._idx = 0
        self._to_test = [
            [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "ETHUSDT"]],  # type: ignore
            [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "BCHUSDT", "LTCUSDT"]],  # type: ignore
            [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "AAVEUSDT", "ETHUSDT"]],  # type: ignore
        ]

    def on_fit(self, ctx: IStrategyContext, fit_time, previous_fit_time=None):
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
