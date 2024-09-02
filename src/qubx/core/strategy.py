"""
 # All interfaces related to strategy etc
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import FunctionType
from collections import defaultdict
from dataclasses import dataclass
from threading import Thread
from multiprocessing.pool import ThreadPool
import traceback

import pandas as pd

from qubx import lookup, logger
from qubx.core.account import AccountProcessor
from qubx.core.helpers import BasicScheduler, set_parameters_to_object
from qubx.core.basics import (
    TargetPosition,
    TriggerEvent,
    Deal,
    Instrument,
    Order,
    Position,
    Signal,
    dt_64,
    td_64,
    ITimeProvider,
    IComminucationManager,
)
from qubx.core.series import Trade, Quote, Bar, OHLCV
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_seconds_to_str


_SW = Stopwatch()


class ITradingServiceProvider(ITimeProvider, IComminucationManager):
    acc: AccountProcessor

    def set_account(self, account: AccountProcessor):
        self.acc = account

    def get_account(self) -> AccountProcessor:
        return self.acc

    def get_name(self) -> str:
        raise NotImplementedError("get_name is not implemented")

    def get_account_id(self) -> str:
        raise NotImplementedError("get_account_id is not implemented")

    def get_capital(self) -> float:
        return self.acc.get_free_capital()

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **optional,
    ) -> Order:
        raise NotImplementedError("send_order is not implemented")

    def cancel_order(self, order_id: str) -> Order | None:
        raise NotImplementedError("cancel_order is not implemented")

    def get_orders(self, symbol: str | None = None) -> List[Order]:
        raise NotImplementedError("get_orders is not implemented")

    def get_position(self, instrument: Instrument | str) -> Position:
        raise NotImplementedError("get_position is not implemented")

    def get_base_currency(self) -> str:
        raise NotImplementedError("get_basic_currency is not implemented")

    def process_execution_report(self, symbol: str, report: Dict[str, Any]) -> Tuple[Order, List[Deal]]:
        raise NotImplementedError("process_execution_report is not implemented")

    @staticmethod
    def _extract_price(update: float | Quote | Trade | Bar) -> float:
        if isinstance(update, float):
            return update
        elif isinstance(update, Quote):
            return 0.5 * (update.bid + update.ask)  # type: ignore
        elif isinstance(update, Trade):
            return update.price  # type: ignore
        elif isinstance(update, Bar):
            return update.close  # type: ignore
        else:
            raise ValueError(f"Unknown update type: {type(update)}")

    def update_position_price(self, symbol: str, timestamp: dt_64, update: float | Quote | Trade | Bar):
        self.acc.update_position_price(timestamp, symbol, ITradingServiceProvider._extract_price(update))


class IBrokerServiceProvider(IComminucationManager, ITimeProvider):
    trading_service: ITradingServiceProvider

    def __init__(self, exchange_id: str, trading_service: ITradingServiceProvider) -> None:
        self._exchange_id = exchange_id
        self.trading_service = trading_service

    def subscribe(self, subscription_type: str, instruments: List[Instrument], **kwargs) -> bool:
        raise NotImplementedError("subscribe")

    def unsubscribe(self, subscription_type: str, instruments: List[Instrument]) -> bool:
        raise NotImplementedError("unsubscribe")

    def has_subscription(self, subscription_type: str, instrument: Instrument) -> bool:
        raise NotImplementedError("has_subscription")

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> List[Bar]:
        raise NotImplementedError("get_historical_ohlcs")

    def get_quote(self, symbol: str) -> Quote | None:
        raise NotImplementedError("get_quote")

    def get_trading_service(self) -> ITradingServiceProvider:
        return self.trading_service

    def close(self):
        pass

    def get_scheduler(self) -> BasicScheduler:
        raise NotImplementedError("schedule_event")

    @property
    def is_simulated_trading(self) -> bool:
        return False


class SubscriptionType:
    """
    Subscription type constants
    """

    QUOTE = "quote"
    TRADE = "trade"
    AGG_TRADE = "agg_trade"
    OHLC = "ohlc"


class StrategyContext(ITimeProvider):
    """
    Strategy context interface
    """

    instruments: List[Instrument]  # list of instruments this strategy trades
    positions: Dict[str, Position]  # positions of the strategy (instrument -> position)
    acc: AccountProcessor

    def process_data(self, symbol: str, d_type: str, data: Any) -> bool: ...

    def ohlc(self, instrument: str | Instrument, timeframe: str) -> OHLCV: ...

    def start(self, blocking: bool = False): ...

    def stop(self): ...

    def time(self) -> dt_64: ...

    def trade(
        self,
        instr_or_symbol: Instrument | str,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **options,
    ) -> Order: ...

    def cancel(self, instr_or_symbol: Instrument | str): ...

    def cancel_order(self, order_id: str): ...

    def quote(self, symbol: str) -> Quote | None: ...

    def get_capital(self) -> float: ...

    def get_reserved(self, instrument: Instrument) -> float: ...

    def get_historical_ohlcs(self, instrument: Instrument | str, timeframe: str, length: int) -> OHLCV | None: ...

    def set_universe(self, instruments: list[Instrument]): ...

    def subscribe(self, subscription_type: str, instr_or_symbol: Instrument | str, **kwargs) -> bool: ...

    def unsubscribe(self, subscription_type: str, instr_or_symbol: Instrument | str) -> bool: ...

    def has_subscription(self, subscription_type: str, instr_or_symbol: Instrument | str) -> bool: ...

    @staticmethod
    def get_latencies_report():
        scope_to_latency_sec = {scope: _SW.latency_sec(scope) for scope in _SW.latencies.keys()}
        scope_to_count = {l: _SW.counts[l] for l in scope_to_latency_sec.keys()}
        scope_to_total_time = {scope: scope_to_count[scope] * lat for scope, lat in scope_to_latency_sec.items()}
        # create pandas datafrmae from dictionaries
        lats = pd.DataFrame(
            {
                "scope": list(scope_to_latency_sec.keys()),
                "latency": list(scope_to_latency_sec.values()),
                "count": list(scope_to_count.values()),
                "total_time": list(scope_to_total_time.values()),
            }
        )
        lats["latency"] = lats["latency"].apply(lambda x: f"{x:.4f}")
        lats["total_time (min)"] = lats["total_time"].apply(lambda x: f"{x / 60:.4f}")
        lats.drop(columns=["total_time"], inplace=True)
        return lats


class IPositionGathering:
    """
    Common interface for position gathering
    """

    def alter_position_size(self, ctx: StrategyContext, target: TargetPosition) -> float: ...

    def alter_positions(
        self, ctx: StrategyContext, targets: List[TargetPosition] | TargetPosition
    ) -> Dict[Instrument, float]:
        if not isinstance(targets, list):
            targets = [targets]

        res = {}
        if targets:
            for t in targets:
                if t.is_service:  # we skip processing service positions
                    continue
                try:
                    res[t.instrument] = self.alter_position_size(ctx, t)
                except Exception as ex:
                    logger.error(f"[{ctx.time()}]: Failed processing target position {t} : {ex}")
                    logger.opt(colors=False).error(traceback.format_exc())
        return res

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal): ...


class IPositionSizer:
    """
    Common interface for get actual positions from signals
    """

    def calculate_target_positions(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition]:
        """
        Target position size calculator

        :param ctx: strategy context object
        :param signals: list of signals to process
        """
        raise NotImplementedError("calculate_target_positions is not implemented")


class PositionsTracker:
    """
    Process signals from strategy and track position. It can contains logic for risk management for example.
    """

    _sizer: IPositionSizer

    def __init__(self, sizer: IPositionSizer) -> None:
        self._sizer = sizer

    def get_position_sizer(self) -> IPositionSizer:
        return self._sizer

    def is_active(self, instrument: Instrument) -> bool:
        return True

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]) -> List[TargetPosition] | TargetPosition:
        """
        Default implementation just returns calculated target positions
        """
        return self.get_position_sizer().calculate_target_positions(ctx, signals)

    def update(
        self, ctx: StrategyContext, instrument: Instrument, update: Quote | Trade | Bar
    ) -> List[TargetPosition] | TargetPosition:
        """
        Tracker is being updated by new market data.
        It may require to change position size or create new position because of interior tracker's logic (risk management for example).
        """
        ...

    def on_execution_report(self, ctx: StrategyContext, instrument: Instrument, deal: Deal):
        """
        Tracker is notified when execution report is received
        """
        ...


class IStrategy:
    ctx: StrategyContext

    def __init__(self, **kwargs) -> None:
        set_parameters_to_object(self, **kwargs)

    def on_start(self, ctx: StrategyContext):
        """
        This method is called strategy is started
        """
        pass

    def on_fit(
        self, ctx: StrategyContext, fit_time: str | pd.Timestamp, previous_fit_time: str | pd.Timestamp | None = None
    ):
        """
        This method is called when it's time to fit model
        :param fit_time: last time of fit data to use
        :param previous_fit_time: last time of fit data used in previous fit call
        """
        return None

    def on_universe_change(
        self, ctx: StrategyContext, add_instruments: list[Instrument], rm_instruments: list[Instrument]
    ) -> None:
        """
        This method is called when the trading universe is updated.
        """
        return None

    def on_event(self, ctx: StrategyContext, event: TriggerEvent) -> List[Signal] | Signal | None:
        return None

    def on_stop(self, ctx: StrategyContext):
        pass

    def tracker(self, ctx: StrategyContext) -> PositionsTracker | None:
        pass
