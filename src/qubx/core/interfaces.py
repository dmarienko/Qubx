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
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from qubx import logger
from qubx.core.basics import (
    AssetBalance,
    CtrlChannel,
    Deal,
    Instrument,
    ITimeProvider,
    MarketEvent,
    MarketType,
    Order,
    OrderRequest,
    Position,
    Signal,
    TargetPosition,
    TriggerEvent,
    dt_64,
)
from qubx.core.helpers import set_parameters_to_object
from qubx.core.series import OHLCV, Bar, Quote, Trade


class IAccountViewer:
    account_id: str

    def get_base_currency(self) -> str:
        """Get the base currency for the account.

        Returns:
            str: The base currency.
        """
        ...

    ########################################################
    # Capital information
    ########################################################
    def get_capital(self) -> float:
        """Get the available free capital in the account.

        Returns:
            float: The amount of free capital available for trading
        """
        ...

    def get_total_capital(self) -> float:
        """Get the total capital in the account including positions value.

        Returns:
            float: Total account capital
        """
        ...

    ########################################################
    # Balance and position information
    ########################################################
    def get_balances(self) -> dict[str, AssetBalance]:
        """Get all currency balances.

        Returns:
            dict[str, AssetBalance]: Dictionary mapping currency codes to AssetBalance objects
        """
        ...

    def get_positions(self) -> dict[Instrument, Position]:
        """Get all current positions.

        Returns:
            dict[Instrument, Position]: Dictionary mapping instruments to their positions
        """
        ...

    def get_position(self, instrument: Instrument) -> Position:
        """Get the current position for a specific instrument.

        Args:
            instrument: The instrument to get the position for

        Returns:
            Position: The position object
        """
        ...

    @property
    def positions(self) -> dict[Instrument, Position]:
        """[Deprecated: Use get_positions()] Get all current positions.

        Returns:
            dict[Instrument, Position]: Dictionary mapping instruments to their positions
        """
        return self.get_positions()

    def get_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        """Get active orders, optionally filtered by instrument.

        Args:
            instrument: Optional instrument to filter orders by

        Returns:
            dict[str, Order]: Dictionary mapping order IDs to Order objects
        """
        ...

    def position_report(self) -> dict:
        """Get detailed report of all positions.

        Returns:
            dict: Dictionary containing position details including quantities, prices, PnL etc.
        """
        ...

    ########################################################
    # Leverage information
    ########################################################
    def get_leverage(self, instrument: Instrument) -> float:
        """Get the leverage used for a specific instrument.

        Args:
            instrument: The instrument to check

        Returns:
            float: Current leverage ratio for the instrument
        """
        ...

    def get_leverages(self) -> dict[Instrument, float]:
        """Get leverages for all instruments.

        Returns:
            dict[Instrument, float]: Dictionary mapping instruments to their leverage ratios
        """
        ...

    def get_net_leverage(self) -> float:
        """Get the net leverage across all positions.

        Returns:
            float: Net leverage ratio
        """
        ...

    def get_gross_leverage(self) -> float:
        """Get the gross leverage across all positions.

        Returns:
            float: Gross leverage ratio
        """
        ...

    ########################################################
    # Margin information
    # Used for margin, swap, futures, options trading
    ########################################################
    def get_total_required_margin(self) -> float:
        """Get total margin required for all positions.

        Returns:
            float: Total required margin
        """
        ...

    def get_available_margin(self) -> float:
        """Get available margin for new positions.

        Returns:
            float: Available margin
        """
        ...

    def get_margin_ratio(self) -> float:
        """Get current margin ratio.

        Formula: (total capital + positions value) / total required margin

        Example:
            If total capital is 1000, positions value is 2000, and total required margin is 3000,
            the margin ratio would be (1000 + 2000) / 3000 = 1.0

        Returns:
            float: Current margin ratio
        """
        ...


class IBroker:
    """Broker provider interface for managing trading operations.

    Handles account operations, order placement, and position tracking.
    """

    channel: CtrlChannel

    @property
    def is_simulated_trading(self) -> bool:
        """
        Check if the broker is in simulation mode.
        """
        ...

    # TODO: think about replacing with async methods
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

    def cancel_orders(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument.

        Args:
            instrument: The instrument to cancel orders for.
        """
        raise NotImplementedError("cancel_orders is not implemented")

    def update_order(self, order_id: str, price: float | None = None, amount: float | None = None) -> Order:
        """Update an existing order.

        Args:
            order_id: The ID of the order to update.
            price: New price for the order.
            amount: New amount for the order.

        Returns:
            Order: The updated Order object if successful

        Raises:
            NotImplementedError: If the method is not implemented
            OrderNotFound: If the order is not found
            BadRequest: If the request is invalid
        """
        raise NotImplementedError("update_order is not implemented")


class IDataProvider:
    time_provider: ITimeProvider
    channel: CtrlChannel

    def subscribe(
        self,
        subscription_type: str,
        instruments: Set[Instrument],
        reset: bool = False,
    ) -> None:
        """
        Subscribe to market data for a list of instruments.

        Args:
            subscription_type: Type of subscription
            instruments: Set of instruments to subscribe to
            reset: Reset existing instruments for the subscription type. Default is False.
        """
        ...

    def unsubscribe(self, subscription_type: str | None, instruments: Set[Instrument]) -> None:
        """
        Unsubscribe from market data for a list of instruments.

        Args:
            subscription_type: Type of subscription to unsubscribe from (optional)
            instruments: Set of instruments to unsubscribe from
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

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        """
        Get all subscriptions for an instrument.

        Args:
            instrument (optional): Instrument to get subscriptions for. If None, all subscriptions are returned.

        Returns:
            List[str]: List of subscriptions
        """
        ...

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> List[Instrument]:
        """
        Get a list of instruments that are subscribed to a specific subscription type.

        Args:
            subscription_type: Type of subscription to filter by (optional)

        Returns:
            List[Instrument]: List of subscribed instruments
        """
        ...

    def warmup(self, configs: Dict[Tuple[str, Instrument], str]) -> None:
        """
        Run warmup for subscriptions.

        Args:
            configs: Dictionary of (subscription type, instrument) pairs and warmup periods.

        Example:
            warmup({
                (DataType.OHLC["1h"], instr1): "30d",
                (DataType.OHLC["1Min"], instr1): "6h",
                (DataType.OHLC["1Sec"], instr2): "5Min",
                (DataType.TRADE, instr2): "1h",
            })
        """
        ...

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]:
        """
        Get historical OHLC data for an instrument.
        """
        ...

    def get_quote(self, instrument: Instrument) -> Quote:
        """
        Get the latest quote for an instrument.
        """
        ...

    @property
    def is_simulation(self) -> bool:
        """
        Check if data provider is in simulation mode.
        """
        ...

    def close(self):
        """
        Close the data provider.
        """
        ...


class IMarketManager(ITimeProvider):
    """Interface for market data providing class"""

    def ohlc(self, instrument: Instrument, timeframe: str | None = None, length: int | None = None) -> OHLCV:
        """Get OHLCV data for an instrument. If length is larger then available cached data, it will be requested from the broker.

        Args:
            instrument: The instrument to get data for
            timeframe (optional): The timeframe of the data. If None, the default timeframe is used.
            length (optional): Number of bars to retrieve. If None, full cached data is returned.

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

    def get_data(self, instrument: Instrument, sub_type: str) -> list[Any]:
        """Get data for an instrument. This method is used for getting data for custom subscription types.
        Could be used for orderbook, trades, liquidations, funding rates, etc.

        Args:
            instrument: The instrument to get data for
            sub_type: The subscription type of data to get

        Returns:
            List[Any]: The data
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

    def submit_orders(self, order_requests: list[OrderRequest]) -> list[Order]:
        """Submit multiple orders to the exchange."""
        ...

    def set_target_position(
        self, instrument: Instrument, target: float, price: float | None = None, **options
    ) -> Order:
        """Set target position for an instrument.

        Args:
            instrument: The instrument to set target position for
            target: Target position size
            price: Optional limit price
            time_in_force: Time in force for the order
            **options: Additional order options

        Returns:
            Order: The created order
        """
        ...

    def close_position(self, instrument: Instrument) -> None:
        """Close position for an instrument.

        Args:
            instrument: The instrument to close position for
        """
        ...

    def close_positions(self, market_type: MarketType | None = None) -> None:
        """Close all positions."""
        ...

    def cancel_order(self, order_id: str) -> None:
        """Cancel a specific order.

        Args:
            order_id: ID of the order to cancel
        """
        ...

    def cancel_orders(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument.

        Args:
            instrument: The instrument to cancel orders for
        """
        ...


class IUniverseManager:
    """Manages universe updates."""

    def set_universe(self, instruments: list[Instrument], skip_callback: bool = False):
        """Set the trading universe.

        Args:
            instruments: List of instruments in the universe
        """
        ...

    def add_instruments(self, instruments: list[Instrument]):
        """Add instruments to the trading universe.

        Args:
            instruments: List of instruments to add
        """
        ...

    def remove_instruments(self, instruments: list[Instrument]):
        """Remove instruments from the trading universe.

        Args:
            instruments: List of instruments to remove
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

    def subscribe(self, subscription_type: str, instruments: List[Instrument] | Instrument | None = None) -> None:
        """Subscribe to market data for an instrument.

        Args:
            subscription_type: Type of subscription. If None, the base subscription type is used.
            instruments: A list of instrument of instrument to subscribe to
        """
        ...

    def unsubscribe(self, subscription_type: str, instruments: List[Instrument] | Instrument | None = None) -> None:
        """Unsubscribe from market data for an instrument.

        Args:
            subscription_type: Type of subscription to unsubscribe from (e.g. DataType.OHLC)
            instruments (optional): A list of instruments or instrument to unsubscribe from.
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

    def get_base_subscription(self) -> str:
        """
        Get the main subscription which should be used for the simulation.
        This data is used for updating the internal OHLCV data series.
        By default, simulation uses 1h OHLCV bars and live trading uses orderbook data.
        """
        ...

    def set_base_subscription(self, subscription_type: str) -> None:
        """
        Set the main subscription which should be used for the simulation.

        Args:
            subscription_type: Type of subscription (e.g. DataType.OHLC, DataType.OHLC["1h"])
        """
        ...

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        """
        Get all subscriptions for an instrument.

        Args:
            instrument: Instrument to get subscriptions for (optional)

        Returns:
            List[str]: List of subscriptions
        """
        ...

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> List[Instrument]:
        """
        Get a list of instruments that are subscribed to a specific subscription type.

        Args:
            subscription_type: Type of subscription to filter by (optional)

        Returns:
            List[Instrument]: List of subscribed instruments
        """
        ...

    def get_warmup(self, subscription_type: str) -> str:
        """
        Get the warmup period for a subscription type.

        Args:
            subscription_type: Type of subscription (e.g. DataType.OHLC["1h"], etc.)

        Returns:
            str: Warmup period
        """
        ...

    def set_warmup(self, configs: dict[Any, str]) -> None:
        """
        Set the warmup period for different subscriptions.

        If there are multiple ohlc configs specified, they will be warmed up in parallel.

        Args:
            configs: Dictionary of subscription types and warmup periods.
                     Keys can be subscription types of dictionaries with subscription parameters.

        Example:
            set_warmup({
                DataType.OHLC["1h"]: "30d",
                DataType.OHLC["1Min"]: "6h",
                DataType.OHLC["1Sec"]: "5Min",
                DataType.TRADE: "1h",
            })
        """
        ...

    def commit(self) -> None:
        """
        Apply all pending changes.
        """
        ...

    @property
    def auto_subscribe(self) -> bool:
        """
        Get whether new instruments are automatically subscribed to existing subscriptions.

        Returns:
            bool: True if auto-subscription is enabled
        """
        ...

    @auto_subscribe.setter
    def auto_subscribe(self, value: bool) -> None:
        """
        Enable or disable automatic subscription of new instruments.

        Args:
            value: True to enable auto-subscription, False to disable
        """
        ...


class IAccountProcessor(IAccountViewer):
    time_provider: ITimeProvider

    def start(self):
        """
        Start the account processor.
        """
        ...

    def stop(self):
        """
        Stop the account processor.
        """
        ...

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        """Set the subscription manager for the account processor.

        Args:
            manager: ISubscriptionManager instance to set
        """
        ...

    def update_balance(self, currency: str, total: float, locked: float):
        """Update balance for a specific currency.

        Args:
            currency: Currency code
            total: Total amount of currency
            locked: Amount of locked currency
        """
        ...

    # TODO: refactor interface to accept float, Quote, Trade
    def update_position_price(self, time: dt_64, instrument: Instrument, price: float) -> None:
        """Update position price for an instrument.

        Args:
            time: Timestamp of the update
            instrument: Instrument being updated
            price: New price
        """
        ...

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        """Process executed deals for an instrument.

        Args:
            instrument: Instrument the deals belong to
            deals: List of deals to process
        """
        ...

    def process_order(self, order: Order) -> None:
        """Process order updates.

        Args:
            order: Order to process
        """
        ...

    def attach_positions(self, *position: Position) -> "IAccountProcessor":
        """Attach positions to the account.

        Args:
            *position: Position objects to attach

        Returns:
            I"IAccountProcessor": Self for chaining
        """
        ...

    def add_active_orders(self, orders: Dict[str, Order]) -> None:
        """Add active orders to the account.

        Warning only use in the beginning for state restoration because it does not update locked balances.

        Args:
            orders: Dictionary mapping order IDs to Order objects
        """
        ...


class IProcessingManager:
    """Manages event processing."""

    def process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool) -> bool:
        """
        Process incoming data.

        Args:
            instrument: Instrument the data is for
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

    def get_event_schedule(self, event_id: str) -> str | None:
        """
        Get defined schedule for event id.
        """
        ...

    def is_fitted(self) -> bool:
        """
        Check if the strategy is fitted.
        """
        ...


class IStrategyContext(
    IMarketManager,
    ITradingManager,
    IUniverseManager,
    ISubscriptionManager,
    IProcessingManager,
    IAccountViewer,
):
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

    def is_running(self) -> bool:
        """
        Check if the strategy is running.
        """
        ...

    @property
    def is_simulation(self) -> bool:
        """
        Check if the strategy is running in simulation mode.
        """
        ...


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

    def on_fit(self, ctx: IStrategyContext):
        """
        Called when it's time to fit the model.
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

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> List[Signal] | Signal | None:
        """
        Called when new market data is received.

        Args:
            ctx: Strategy context.
            data: The market data received.

        Returns:
            List of signals, single signal, or None.
        """
        return None

    def on_order_update(self, ctx: IStrategyContext, order: Order) -> list[Signal] | Signal | None:
        """
        Called when an order update is received.

        Args:
            ctx: Strategy context.
            order: The order update.
        """
        return None

    def on_stop(self, ctx: IStrategyContext):
        pass

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker | None:
        pass
