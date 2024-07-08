from typing import List, Dict
import pandas as pd
import numpy as np
from tabulate import tabulate

from qubx import logger
from qubx.core.strategy import IStrategy, TriggerEvent, StrategyContext
from qubx.core.basics import Instrument, Position, Signal
from qubx.pandaz import srows, scols, ohlc_resample, retain_columns_and_join
from qubx.trackers import Capital, PortfolioRebalancerTracker
from qubx.utils.misc import quotify, dequotify
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV


def priceframe(ctx: StrategyContext, field: str, timeframe: str) -> pd.DataFrame:
    data = []
    for i in ctx.instruments:
        d = ctx.ohlc(i.symbol, timeframe)
        if hasattr(d, field):
            b = getattr(d, field).pd()
            data.append(b)
        else:
            logger.error(f"No {field} column in OHLC for {i.symbol}")
    return scols(*data)


class TradeTestStrat(IStrategy):
    capital_invested: float = 100.0
    trading_allowed: bool = False
    _tracker: PortfolioRebalancerTracker

    def on_start(self, ctx: StrategyContext):
        logger.info(f"> Started with capital {self.capital_invested}")
        self._tracker = self.tracker(ctx)

    def on_fit(
        self, ctx: "StrategyContext", fit_time: str | pd.Timestamp, previous_fit_time: str | pd.Timestamp | None = None
    ):
        closes = priceframe(ctx, "close", "5Min")
        logger.info(f"> Fit is called | fit_time: {fit_time} / prev: {previous_fit_time}")
        logger.info(f"{str(closes)}")

    def on_event(self, ctx: StrategyContext, event: TriggerEvent) -> List[Signal] | None:
        if event.type == "trade":
            trade: Trade = event.data
            assert event.instrument is not None
            logger.info(f"{event.time} {event.instrument.symbol} -> {trade}")

        return None

    def ohlcs(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        return {s.symbol: self.ctx.ohlc(s, timeframe).pd() for s in self.ctx.instruments}

    def on_stop(self, ctx: StrategyContext):
        logger.info(f"> test is stopped")

    def tracker(self, ctx: StrategyContext) -> PortfolioRebalancerTracker:
        return PortfolioRebalancerTracker(ctx, self.capital_invested, 0)

    def reporting(self, signals: pd.DataFrame, wealth: Capital):
        _str_pos = tabulate(signals.tail(1), dequotify(list(signals.columns.values)), tablefmt="rounded_grid")  # type: ignore
        _mesg = ""
        if wealth.symbols_to_close:
            _mesg = f"({wealth.released_amount:.2f} will be released from closing <red>{wealth.symbols_to_close}</red>)"
        logger.info(
            f"Positions to process for {wealth.capital:.2f} {self.ctx.exchange_service.get_base_currency()} {_mesg}:\n<blue>{_str_pos}</blue>"
        )
