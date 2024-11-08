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
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV


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


class TradeTestStrat(IStrategy):
    capital_invested: float = 100.0
    trading_allowed: bool = False
    _tracker: PortfolioRebalancerTracker

    def on_start(self, ctx: IStrategyContext):
        logger.info(f"> Started with capital {self.capital_invested}")
        self._tracker = self.tracker(ctx)

    def on_fit(self, ctx: IStrategyContext):
        closes = priceframe(ctx, "close", "5Min")
        logger.info(f"> Fit is called | fit_time: {ctx.time()}")
        logger.info(f"{str(closes)}")

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal] | None:
        match event.type:
            case "trade":
                trade: Trade = event.data
                assert event.instrument is not None
                logger.info(f"{event.time} {event.instrument.symbol} -> Triggered on trade {trade}")
            case "time":
                logger.info(f"{event.time} -> Triggered on time event")
            case _:
                logger.info(f"{event.time} -> Triggered on unknown event {event}")

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
