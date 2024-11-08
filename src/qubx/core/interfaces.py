"""
This module defines interfaces and classes related to trading strategies.

This module includes:
    - Trading service providers
    - Broker service providers
    - Market data providers
    - Strategy contexts
    - Position tracking and management
"""

import traceback
import pandas as pd

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
    SW,
)
from qubx.core.series import OrderBook, Trade, Quote, Bar, OHLCV
from enum import StrEnum


class ITradingServiceProvider(ITimeProvider, IComminucationManager):
    """Trading service provider interface for managing trading operations.

    Handles account operations, order placement, and position tracking.
    """

    acc: AccountProcessor

    def set_account(self, account: AccountProcessor):
        """Sets the account processor for the trading service provider.

        Args:
            account: AccountProcessor instance to be set.
        """
        self.acc = account

    def get_account(self) -> AccountProcessor:
        """Retrieve the current account processor.

        Returns:
            AccountProcessor: The current AccountProcessor object.
        """
        return self.acc

    def get_name(self) -> str:
        """Get the name of the trading service provider.

        Returns:
            str: The name of the trading service provider.
        """
        raise NotImplementedError("get_name is not implemented")

    def get_account_id(self) -> str:
        """Get the account ID associated with the trading service provider.

        Returns:
            str: The account ID.
        """
        raise NotImplementedError("get_account_id is not implemented")

    def get_capital(self) -> float:
        """Get the available capital in the account.

        Returns:
            float: The free capital.
        """
        return self.acc.get_capital()

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
        """Sends an order to the trading service.

        Args:
            instrument: The instrument to trade.
            order_side: Order side ("buy" or "sell").
            order_type: Type of order ("market" or "limit").
            amount: Amount of instrument to trade.
            price: Price for limit orders.
            client_id: Client-specified order ID.
            time_in_force: Time in force for order (default: "gtc").
            **optional: Additional order parameters.

        Returns:
            Order: The created order object.
        """
        raise NotImplementedError("send_order is not implemented")

    def cancel_order(self, order_id: str) -> Order | None:
        """Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            Order | None: The cancelled Order object if successful, None otherwise.
        """
        raise NotImplementedError("cancel_order is not implemented")

    def get_orders(self, instrument: Instrument | None = None) -> List[Order]:
        """Get a list of current orders, optionally filtered by symbol.

        Args:
            symbol: The symbol to filter orders by (optional).

        Returns:
            List[Order]: A list of Order objects.
        """
        raise NotImplementedError("get_orders is not implemented")

    def get_position(self, instrument: Instrument) -> Position:
        """Get the current position for a given instrument.

        Args:
            instrument: The instrument or symbol to get the position for.

        Returns:
            Position: A Position object representing the current position.
        """
        raise NotImplementedError("get_position is not implemented")

    def get_base_currency(self) -> str:
        """Get the base currency for the account.

        Returns:
            str: The base currency.
        """
        raise NotImplementedError("get_basic_currency is not implemented")

    def process_execution_report(self, instrument: Instrument, report: dict[str, Any]) -> Tuple[Order, List[Deal]]:
        """Process an execution report for a given symbol.

        Args:
            symbol: The symbol the execution report is for.
            report: A dictionary containing the execution report details.

        Returns:
            Tuple[Order, List[Deal]]: A tuple containing the updated Order and a list of Deal objects.
        """
        raise NotImplementedError("process_execution_report is not implemented")

    @staticmethod
    def _extract_price(update: float | Quote | Trade | Bar) -> float:
        """Extract the price from various types of market data updates.

        Args:
            update: The market data update, which can be a float, Quote, Trade, or Bar.

        Returns:
            float: The extracted price.

        Raises:
            ValueError: If the update type is unknown.
        """
        if isinstance(update, float):
            return update
        elif isinstance(update, Quote) or isinstance(update, OrderBook):
            return update.mid_price()
        elif isinstance(update, Trade):
            return update.price  # type: ignore
        elif isinstance(update, Bar):
            return update.close  # type: ignore
        else:
            raise ValueError(f"Unknown update type: {type(update)}")

    def update_position_price(self, instrument: Instrument, timestamp: dt_64, update: float | Quote | Trade | Bar):
        """Updates the price of a position.

        Args:
            symbol: Symbol of the position.
            timestamp: Timestamp of the update.
            update: Price update (float, Quote, Trade, or Bar).
        """
        self.acc.update_position_price(timestamp, instrument, ITradingServiceProvider._extract_price(update))


class IBrokerServiceProvider(IComminucationManager, ITimeProvider):
    trading_service: ITradingServiceProvider

    def __init__(self, exchange_id: str, trading_service: ITradingServiceProvider) -> None:
        self._exchange_id = exchange_id
        self.trading_service = trading_service

    def subscribe(
        self,
        instruments: list[Instrument],
        subscription_type: str,
        warmup_period: str | None = None,
        ohlc_warmup_period: str | None = None,
        **kwargs,
    ) -> None:
        """
        Subscribe to market data for a list of instruments.

        Args:
            subscription_type: Type of subscription
            instruments: List of instruments to subscribe to
            warmup_period: Warmup period for the subscription
            ohlc_warmup_period: Warmup period for OHLC data
            **kwargs: Additional subscription parameters
        """
        ...

    def unsubscribe(self, instruments: list[Instrument], subscription_type: str | None) -> None:
        """
        Unsubscribe from market data for a list of instruments.

        Args:
            instruments: List of instruments to unsubscribe from
            subscription_type: Type of subscription to unsubscribe from (optional)
        """
        ...

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """
        Check if an instrument has a subscription.

        Args:
            instrument: Instrument to check
            subscription_type: Type of subscription to check

        Returns:
            bool: True if instrument has the subscription
        """
        ...

    def commit(self) -> None:
        """
        Apply all pending subscription changes.
        """
        ...

    def get_historical_ohlcs(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]: ...

    def get_quote(self, instrument: Instrument) -> Quote | None: ...

    def get_trading_service(self) -> ITradingServiceProvider:
        return self.trading_service

    def close(self):
        pass

    def get_scheduler(self) -> BasicScheduler: ...

    @property
    def is_simulated_trading(self) -> bool: ...


class SubscriptionType(StrEnum):
    """Subscription type constants."""

    QUOTE = "quote"
    TRADE = "trade"
    OHLC = "ohlc"
    ORDERBOOK = "orderbook"


class IMarketDataProvider(ITimeProvider):
    """Interface for market data providing class"""

    def ohlc(self, instrument: Instrument, timeframe: str) -> OHLCV:
        """Get OHLCV data for an instrument.

        Args:
            instrument: The instrument to get data for
            timeframe: The timeframe of the OHLCV data

        Returns:
            OHLCV: The OHLCV data series
        """
        ...

    def quote(self, instrument: Instrument) -> Quote | None:
        """Get latest quote for an instrument.

        Args:
            instrument: The instrument to get quote for

        Returns:
            Quote | None: The latest quote or None if not available
        """
        ...

    def get_historical_ohlcs(self, instrument: Instrument, timeframe: str, length: int) -> OHLCV:
        """Get historical OHLCV data for an instrument.

        Args:
            instrument: The instrument to get data for
            timeframe: The timeframe of the data
            length: Number of bars to retrieve

        Returns:
            OHLCV: Historical OHLCV data series
        """
        ...

    def get_aux_data(self, data_id: str, **parametes) -> pd.DataFrame | None:
        """Get auxiliary data by ID.

        Args:
            data_id: Identifier for the auxiliary data
            **parametes: Additional parameters for the data request

        Returns:
            pd.DataFrame | None: The auxiliary data or None if not found
        """
        ...

    def get_instruments(self) -> list[Instrument]:
        """Get list of all available instruments.

        Returns:
            list[Instrument]: List of available instruments
        """
        ...

    def get_instrument(self, symbol: str, exchange: str) -> Instrument | None:
        """Get instrument by symbol and exchange.

        Args:
            symbol: The symbol to look up
            exchange: The exchange to look up

        Returns:
            Instrument | None: The instrument if found, None otherwise
        """
        ...


class ITradingManager:
    """Manages order operations."""

    def trade(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **options,
    ) -> Order:
        """Place a trade order.

        Args:
            instrument: The instrument to trade
            amount: Amount to trade (positive for buy, negative for sell)
            price: Optional limit price
            time_in_force: Time in force for the order
            **options: Additional order options

        Returns:
            Order: The created order
        """
        ...

    def cancel(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument.

        Args:
            instrument: The instrument to cancel orders for
        """
        ...

    def cancel_order(self, order_id: str) -> None:
        """Cancel a specific order.

        Args:
            order_id: ID of the order to cancel
        """
        ...


class IUniverseManager:
    """Manages universe updates."""

    def set_universe(self, instruments: list[Instrument]):
        """Set the trading universe.

        Args:
            instruments: List of instruments in the universe
        """
        ...

    @property
    def instruments(self) -> list[Instrument]:
        """
        Get the list of instruments in the universe.
        """
        ...


class ISubscriptionManager:
    """Manages subscriptions."""

    def subscribe(self, instrument: Instrument, subscription_type: str, **kwargs) -> bool:
        """Subscribe to market data for an instrument.

        Args:
            instrument: Instrument to subscribe to
            subscription_type: Type of subscription
            **kwargs: Additional subscription parameters
        """
        ...

    def unsubscribe(self, instrument: Instrument, subscription_type: str | None = None) -> bool:
        """Unsubscribe from market data for an instrument.

        Args:
            instrument: Instrument to unsubscribe from
            subscription_type: Type of subscription to unsubscribe from (optional)
        """
        ...

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """Check if subscription exists.

        Args:
            subscription_type: Type of subscription
            instrument: Instrument to check

        Returns:
            bool: True if subscription exists
        """
        ...

    def get_base_subscription(self) -> tuple[SubscriptionType, dict]:
        """
        Get the main subscription which should be used for the simulation.
        This data is used for updating the internal OHLCV data series.
        By default, simulation uses 1h OHLCV bars and live trading uses orderbook data.
        """
        ...

    def set_base_subscription(self, subscription_type: SubscriptionType, **kwargs) -> None:
        """
        Set the main subscription which should be used for the simulation.

        Args:
            subscription_type: Type of subscription
            **kwargs: Additional subscription parameters (e.g. timeframe for OHLCV)
        """
        ...

    def get_warmup(self, subscription_type: str) -> str:
        """
        Get the warmup period for a subscription type.

        Args:
            subscription_type: Type of subscription (e.g. SubscriptionType.OHLC, or something custom like "liquidation")

        Returns:
            str: Warmup period
        """
        ...

    def set_warmup(self, subscription_type: str, period: str) -> None:
        """
        Set the warmup period for a subscription type (default is 0).

        Args:
            subscription_type: Type of subscription (e.g. SubscriptionType.OHLC, or something custom like "liquidation")
            period: Warmup period (e.g. "1d")
        """
        ...


class IProcessingManager:
    """Manages event processing."""

    def process_data(self, symbol: str, d_type: str, data: Any) -> bool:
        """
        Process incoming data.

        Args:
            symbol: Symbol of the data
            d_type: Type of the data
            data: The data to process

        Returns:
            bool: True if processing should be halted
        """
        ...

    def set_fit_schedule(self, schedule: str) -> None:
        """
        Set the schedule for fitting the strategy model (default is to trigger fit only at start).
        """
        ...

    def set_event_schedule(self, schedule: str) -> None:
        """
        Set the schedule for triggering events (default is to only trigger on data events).
        """
        ...


class IAccountViewer:
    @property
    def positions(self) -> dict[Instrument, Position]:
        """
        Get the current positions.
        """
        ...

    def get_capital(self) -> float:
        """
        Get the available free capital in the account.
        """
        ...

    def get_total_capital(self) -> float:
        """
        Get the total capital in the account.
        """
        ...

    def get_reserved(self, instrument: Instrument) -> float:
        """
        Get the reserved amount for an instrument.
        """
        ...


class IStrategyContext(
    IMarketDataProvider, ITradingManager, IUniverseManager, ISubscriptionManager, IProcessingManager, IAccountViewer
):
    account: AccountProcessor
    strategy: "IStrategy"

    def start(self, blocking: bool = False):
        """
        Starts the strategy context.

        Args:
            blocking: Whether to block the main thread
        """
        ...

    def stop(self):
        """Stops the strategy context."""
        ...

    @property
    def is_simulation(self) -> bool:
        """
        Check if the strategy is running in simulation mode.
        """
        ...

    @property
    def positions(self) -> dict[Instrument, Position]:
        """
        Get the current positions.
        """
        return self.account.positions

    @staticmethod
    def latency_report() -> pd.DataFrame:
        """
        Get latency report for the strategy.
        """
        if (report := SW.latency_report()) is None:
            raise ValueError("No latency report available")
        return report


class IPositionGathering:
    """
    Common interface for position gathering
    """

    def alter_position_size(self, ctx: IStrategyContext, target: TargetPosition) -> float: ...

    def alter_positions(
        self, ctx: IStrategyContext, targets: List[TargetPosition] | TargetPosition
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

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal): ...


class IPositionSizer:
    """Interface for calculating target positions from signals."""

    def calculate_target_positions(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        """Calculates target position sizes.

        Args:
            ctx: Strategy context object.
            signals: List of signals to process.

        Returns:
            List of target positions.
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

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition] | TargetPosition:
        """
        Default implementation just returns calculated target positions
        """
        return self.get_position_sizer().calculate_target_positions(ctx, signals)

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar
    ) -> List[TargetPosition] | TargetPosition:
        """
        Tracker is being updated by new market data.
        It may require to change position size or create new position because of interior tracker's logic (risk management for example).
        """
        ...

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal):
        """
        Tracker is notified when execution report is received
        """
        ...


class IStrategy:
    """Base class for trading strategies."""

    ctx: IStrategyContext

    def __init__(self, **kwargs) -> None:
        set_parameters_to_object(self, **kwargs)

    def on_init(self, ctx: IStrategyContext):
        """
        This method is called when strategy is initialized.
        It is useful for setting the base subscription and warmup periods via the subscription manager.
        """
        ...

    def on_start(self, ctx: IStrategyContext):
        """
        This method is called strategy is started. You can already use the market data provider.
        """
        pass

    def on_fit(self, ctx: IStrategyContext, fit_time: dt_64, previous_fit_time: dt_64 | None = None):
        """Called when it's time to fit the model.

        Args:
            ctx: Strategy context.
            fit_time: Last time of fit data to use.
            previous_fit_time: Last time of fit data used in previous fit.

        Returns:
            None
        """
        return None

    def on_universe_change(
        self, ctx: IStrategyContext, add_instruments: list[Instrument], rm_instruments: list[Instrument]
    ) -> None:
        """
        This method is called when the trading universe is updated.
        """
        return None

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> List[Signal] | Signal | None:
        """Called on strategy events.

        Args:
            ctx: Strategy context.
            event: Trigger event to process.

        Returns:
            List of signals, single signal, or None.
        """
        return None

    def on_stop(self, ctx: IStrategyContext):
        pass

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker | None:
        pass
