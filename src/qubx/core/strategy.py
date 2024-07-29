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


class StrategyContext:
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

    def get_latencies_report(self): ...

    def time(self) -> dt_64: ...

    def trade(
        self, instr_or_symbol: Instrument | str, amount: float, price: float | None = None, time_in_force="gtc"
    ) -> Order: ...

    def cancel(self, instr_or_symbol: Instrument | str): ...

    def quote(self, symbol: str) -> Quote | None: ...

    def get_capital(self) -> float: ...

    def get_reserved(self, instrument: Instrument) -> float: ...

    def get_historical_ohlcs(self, instrument: Instrument | str, timeframe: str, length: int) -> OHLCV | None: ...


class IPositionAdjuster:
    """
    Common interface for adjusting position
    """

    def adjust_position_size(
        self, ctx: StrategyContext, instrument: Instrument, new_size: float, at_price: float | None = None
    ) -> float: ...


class PositionsTracker:
    """
    Tracks position and processing signals. It can contains logic for risk management for example.
    """

    def process_signals(self, ctx: StrategyContext, signals: List[Signal]): ...

    def update(self, ctx: StrategyContext, quote: Quote):  # TODO: ???
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

    def on_event(self, ctx: StrategyContext, event: TriggerEvent) -> Optional[List[Signal]]:
        return None

    def on_stop(self, ctx: StrategyContext):
        pass

    def tracker(self, ctx: StrategyContext) -> PositionsTracker | None:
        pass
