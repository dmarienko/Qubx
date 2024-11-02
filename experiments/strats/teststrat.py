from typing import List, Dict
import pandas as pd
import numpy as np
from tabulate import tabulate

from qubx import logger
from qubx.core.interfaces import IStrategy, PositionsTracker, TriggerEvent, IStrategyContext
from qubx.core.basics import Instrument, Position, Signal
from qubx.pandaz import srows, scols, ohlc_resample, retain_columns_and_join
from qubx.trackers import Capital, PortfolioRebalancerTracker
from qubx.utils.misc import quotify, dequotify


def priceframe(ctx: IStrategyContext, field: str, timeframe: str) -> pd.DataFrame:
    data = []
    for i in ctx.instruments:
        d = ctx.ohlc(i.symbol, timeframe)
        if hasattr(d, field):
            b = getattr(d, field).pd()
            data.append(b)
        else:
            logger.error(f"No {field} column in OHLC for {i.symbol}")
    return scols(*data)


class FlipFlopStrat(IStrategy):
    capital_invested: float = 100.0
    trading_allowed: bool = False
    _tracker: PortfolioRebalancerTracker

    def on_start(self, ctx: IStrategyContext):
        logger.info(f"> Started with capital {self.capital_invested}")
        self._tracker = self.tracker(ctx)

    def on_fit(
        self, ctx: "StrategyContext", fit_time: str | pd.Timestamp, previous_fit_time: str | pd.Timestamp | None = None
    ):
        closes = priceframe(ctx, "close", "5Min")
        logger.info(f"> Fit is called | fit_time: {fit_time} / prev: {previous_fit_time}")
        logger.info(f"{str(closes)}")

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal] | None:
        logger.info(f"{event.time} -> {event}")
        ohlcs = self.ohlcs("15Min")

        symbols_to_close, symbols_to_open = [], []
        for s, p in ctx.positions.items():
            if p.quantity != 0:
                symbols_to_close.append(s)
            else:
                symbols_to_open.append(s)

        if not symbols_to_close:  # first run just open half from all universe
            symbols_to_open = symbols_to_open[: len(symbols_to_open) // 2]

        cap = self._tracker.estimate_capital_to_trade(ctx, symbols_to_close)
        capital_per_symbol = np.clip(round(cap.capital / len(symbols_to_open)), 5, np.inf)

        logger.info(
            f"\n>>> to close: {symbols_to_close}\n>>> to open: {symbols_to_open} | {capital_per_symbol} per symbol"
        )

        c_time = ctx.time()
        to_open = pd.DataFrame({s: {c_time: capital_per_symbol / ohlcs[s].close.iloc[-1]} for s in symbols_to_open})
        to_close = pd.DataFrame({s: {c_time: 0} for s in symbols_to_close})
        signals = scols(to_close, to_open)

        self.reporting(signals, cap)

        # - process signals
        if self.trading_allowed:
            self._tracker.process_signals(ctx, signals)
        else:
            logger.warning("Trading is disabled - no postions will be changed")

        return None

    def ohlcs(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        return {s.symbol: self.ctx.ohlc(s, timeframe).pd() for s in self.ctx.instruments}

    def on_stop(self, ctx: IStrategyContext):
        logger.info(f"> test is stopped")

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
        return PortfolioRebalancerTracker(self.capital_invested, 0)

    def reporting(self, signals: pd.DataFrame, wealth: Capital):
        _str_pos = tabulate(signals.tail(1), dequotify(list(signals.columns.values)), tablefmt="rounded_grid")  # type: ignore
        _mesg = ""
        if wealth.symbols_to_close:
            _mesg = f"({wealth.released_amount:.2f} will be released from closing <red>{wealth.symbols_to_close}</red>)"
        logger.info(
            f"Positions to process for {wealth.capital:.2f} {self.ctx.trading_service.get_base_currency()} {_mesg}:\n<blue>{_str_pos}</blue>"
        )
