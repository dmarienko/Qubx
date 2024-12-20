from typing import Any

from qubx import logger, lookup
from qubx.backtester.broker import SimulatedAccountProcessor
from qubx.backtester.simulator import simulate
from qubx.backtester.utils import SimulatedScheduler, SimulatedTimeProvider, recognize_simulation_data_config
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import DataType, Instrument, ITimeProvider, Signal, TriggerEvent, dt_64
from qubx.core.context import StrategyContext
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.data.helpers import loader


class Tester1(IStrategy):
    _idx = 0
    _err = False
    _to_test: list[list[Instrument]] = []

    def on_init(self, ctx: IStrategyContext) -> None:
        logger.info(f"Exchange:{ctx.exchanges}")

    def on_fit(self, ctx: IStrategyContext):
        pass
        # logger.info(f" -> SET NEW UNIVERSE {','.join(i.symbol for i in self._to_test[self._idx])}")
        # self._idx += 1
        # if self._idx > 2:
        #     self._idx = 0

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        for s in ctx.instruments:
            q = ctx.quote(s)
            # - quotes should be in ctx already !!!
            if q is None:
                logger.error(f"\n{s.symbol} -> NO QUOTE\n")
                self._err = True

        return []

    # def find_instrument(self, symbol: str) -> Instrument:
    #     i = lookup.find_symbol(ctx.exchange[0], symbol)
    #     assert i is not None, f"Could not find {self.exchange}:{symbol}"
    # return i


class TestStrategyContext:
    def test_context_exchanges(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        simulate(
            {
                "fail1": (stg := Tester1()),
            },
            {"ohlc(4h)": ld},
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-08-01",
            debug="DEBUG",
            n_jobs=1,
        )

        # assert not stg._err, "Got Errors during the simulation"
