from collections import defaultdict
from typing import Callable, Iterable, List, Set, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd

from qubx import logger, lookup
from qubx.core.basics import Instrument, Signal, TriggerEvent, MarketEvent, Subtype
from qubx.core.interfaces import IStrategy, IStrategyContext

from qubx import logger, lookup
from qubx.data import loader
from qubx.backtester.simulator import simulate
from qubx.core.series import OHLCV, Quote


class Issue1(IStrategy):
    exchange: str = "BINANCE.UM"
    _idx = 0
    _err = False
    _to_test: List[List[Instrument]] = []

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(Subtype.OHLC["1h"])
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
        ctx.set_base_subscription(Subtype.OHLC["1h"])
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
    _last_trigger_event: TriggerEvent | None = None
    _last_market_event: MarketEvent | None = None
    _market_events: list[MarketEvent]

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(Subtype.OHLC["1h"])
        self._fits_called = 0
        self._triggers_called = 0
        self._market_events = []

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
        self._triggers_called += 1
        self._last_trigger_event = event

    def on_market_data(self, ctx: IStrategyContext, event: MarketEvent):
        self._market_called += 1
        self._last_market_event = event
        self._market_events.append(event)


class Issue4(IStrategy):
    _issues = 0

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(Subtype.OHLC["1h"])

    def on_market_data(self, ctx: IStrategyContext, event: MarketEvent):
        try:
            if event.type != Subtype.QUOTE:
                return
            quote = event.data
            assert isinstance(quote, Quote)
            _ohlc = ctx.ohlc(event.instrument)
            assert _ohlc[0].close == quote.mid_price(), f"OHLC: {_ohlc[0].close} != Quote: {quote.mid_price()}"
        except:
            self._issues += 1


class Issue5(IStrategy):
    _err_time: int = 0
    _err_bars: int = 0
    _out: OHLCV | None = None

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(Subtype.OHLC["1d"])
        ctx.set_event_schedule("0 0 * * *")  # Run at 00:00 every day

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal]:
        # logger.info(f"On Event: {ctx.time()}")

        data = ctx.ohlc(ctx.instruments[0], "1d", 10)
        # logger.info(f"On Event: {len(data)} -> {data[0].open} ~ {data[0].close}")
        print(f"On Event: {ctx.time()}\n{str(data)}")

        # - at 00:00 bar[0] must be previous day's bar !
        if data[0].time >= ctx.time().item() - pd.Timedelta("1d").asm8:
            self._err_time += 1

        # - check bar's consitency
        if data[0].open == data[0].close and data[0].open == data[0].low and data[0].open == data[0].low:
            self._err_bars += 1

        self._out = data
        return []


class Test6_HistOHLC(IStrategy):
    _U: List[List[Instrument]] = []
    _out: Dict[Any, OHLCV] = {}
    _out_fit: Dict[Any, Any] = {}

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(Subtype.OHLC["1d"])
        # ctx.set_fit_schedule("59 22 * */1 L7")
        # ctx.set_event_schedule("55 23 * * *")
        # ctx.set_fit_schedule("0 0 * */1 L1")
        ctx.set_fit_schedule("0 0 * * *")
        ctx.set_event_schedule("0 0 * * *")
        # ctx.set_warmup(SubscriptionType.OHLC, "1d")
        self._U = [
            [self.find_instrument(s) for s in ["BTCUSDT", "ETHUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "BCHUSDT", "LTCUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "AAVEUSDT", "ETHUSDT"]],
        ]
        self._out = {}
        self._out_fit = {}

    def find_instrument(self, symbol: str) -> Instrument:
        assert (i := lookup.find_symbol("BINANCE.UM", symbol)) is not None, f"Could not find BINANCE.UM:{symbol}"
        return i

    def on_fit(self, ctx: IStrategyContext):
        print(f" - - - - - - - ||| FIT : {ctx.time()}")
        self._out_fit |= self.get_ohlc(ctx, self._U[0])
        ctx.set_universe(self._U[0])

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal]:
        print(f" - - - - - - - {ctx.time()}")
        self._out |= self.get_ohlc(ctx, ctx.instruments)
        return []

    def get_ohlc(self, ctx: IStrategyContext, instruments: List[Instrument]) -> dict:
        closes = defaultdict(dict)
        for i in instruments:
            data = ctx.ohlc(i, "1d", 4)
            print(f":: :: :: {i.symbol} :: :: ::\n{str(data)}")
            closes[pd.Timestamp(data[0].time, unit="ns")] |= {i.symbol: data[0].close}
        return closes


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

        # +1 because first event is used for on_fit and skipped for on_market_data
        assert stg._triggers_called * 4 == stg._market_called + 1, "Got Errors during the simulation"

    def test_ohlc_quote_update(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        # fmt: off
        test0 = simulate(
            { "fail4": (stg := Issue4()), },
            ld, aux_data=ld,
            capital=100_000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt", debug="DEBUG", n_jobs=1,
            start="2023-06-01", stop="2023-06-10",
        )
        # fmt: on

        assert stg._issues == 0, "Got Errors during the simulation"

    def test_ohlc_data(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        test4 = simulate(
            {
                "Issue5": (stg := Issue5()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-07-01",
            stop="2023-07-10",
            debug="DEBUG",
            silent=True,
            n_jobs=1,
        )
        assert stg._err_time == 0, "Got wrong OHLC bars time"
        assert stg._err_bars == 0, "OHLC bars were not consistent"

        r = ld[["BTCUSDT"], "2023-06-30":"2023-07-10"]("1d")["BTCUSDT"]  # type: ignore
        assert all(
            stg._out.pd()[["open", "high", "low", "close"]] == r[["open", "high", "low", "close"]]
        ), "Out OHLC differ"

    def test_ohlc_hist_data(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)
        ld_test = loader("BINANCE.UM", "1d", source="csv::tests/data/csv_1h/", n_jobs=1)

        # fmt: off
        simulate({ "Issue5": (stg := Test6_HistOHLC()), },
            ld, 
            capital=100_000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt",
            start="2023-07-01", stop="2023-07-04",
            debug="DEBUG", silent=True, n_jobs=1,
        )
        # fmt: on

        r = ld_test[["BTCUSDT", "ETHUSDT"], "2023-06-30":"2023-07-03 23:59:59"]("1d")  # type: ignore
        # print(r.display(-1))
        assert all(pd.DataFrame.from_dict(stg._out_fit, orient="index") == r.close)

        r = ld_test[["BTCUSDT", "ETHUSDT"], "2023-07-01":"2023-07-03 23:59:59"]("1d")  # type: ignore
        assert all(pd.DataFrame.from_dict(stg._out, orient="index") == r.close)
