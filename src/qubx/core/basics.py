from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from queue import Empty, Queue
from threading import Event, Lock
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qubx.core.exceptions import QueueTimeout
from qubx.core.series import Quote, Trade, time_as_nsec
from qubx.core.utils import prec_ceil, prec_floor, time_delta_to_str
from qubx.utils.misc import Stopwatch
from qubx.utils.ntp import start_ntp_thread, time_now

dt_64 = np.datetime64
td_64 = np.timedelta64
ns_to_dt_64 = lambda ns: np.datetime64(ns, "ns")

OPTION_FILL_AT_SIGNAL_PRICE = "fill_at_signal_price"

SW = Stopwatch()


@dataclass
class Signal:
    """
    Class for presenting signals generated by strategy

    Attributes:
        reference_price: float - exact price when signal was generated

        Options:
        - allow_override: bool - if True, and there is another signal for the same instrument, then override current.
    """

    instrument: "Instrument"
    signal: float
    price: float | None = None
    stop: float | None = None
    take: float | None = None
    reference_price: float | None = None
    group: str = ""
    comment: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        _p = f" @ { self.price }" if self.price is not None else ""
        _s = f" stop: { self.stop }" if self.stop is not None else ""
        _t = f" take: { self.take }" if self.take is not None else ""
        _r = f" {self.reference_price:.2f}" if self.reference_price is not None else ""
        _c = f" [{self.comment}]" if self.take is not None else ""
        return (
            f"{self.group}{_r} {self.signal:+f} {self.instrument.symbol}{_p}{_s}{_t} on {self.instrument.exchange}{_c}"
        )


@dataclass
class TargetPosition:
    """
    Class for presenting target position calculated from signal
    """

    time: dt_64  # time when position was set
    signal: Signal  # original signal
    target_position_size: float  # actual position size after processing in sizer
    _is_service: bool = False

    @staticmethod
    def create(ctx: "ITimeProvider", signal: Signal, target_size: float) -> "TargetPosition":
        return TargetPosition(ctx.time(), signal, signal.instrument.round_size_down(target_size))

    @staticmethod
    def zero(ctx: "ITimeProvider", signal: Signal) -> "TargetPosition":
        return TargetPosition(ctx.time(), signal, 0.0)

    @staticmethod
    def service(ctx: "ITimeProvider", signal: Signal, size: float | None = None) -> "TargetPosition":
        """
        Generate just service position target (for logging purposes)
        """
        return TargetPosition(ctx.time(), signal, size if size else signal.signal, _is_service=True)

    @property
    def instrument(self) -> "Instrument":
        return self.signal.instrument

    @property
    def price(self) -> float | None:
        return self.signal.price

    @property
    def stop(self) -> float | None:
        return self.signal.stop

    @property
    def take(self) -> float | None:
        return self.signal.take

    @property
    def is_service(self) -> bool:
        """
        Some target may be used just for informative purposes (post-factum risk management etc)
        """
        return self._is_service

    def __str__(self) -> str:
        return f"{'::: INFORMATIVE ::: ' if self.is_service else ''}Target for {self.signal} -> {self.target_position_size} at {self.time}"


class AssetType(StrEnum):
    CRYPTO = "CRYPTO"
    STOCK = "STOCK"
    FX = "FX"
    INDEX = "INDEX"


class MarketType(StrEnum):
    SPOT = "SPOT"
    MARGIN = "MARGIN"
    SWAP = "SWAP"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


@dataclass
class Instrument:
    symbol: str
    asset_type: AssetType
    market_type: MarketType
    exchange: str
    base: str
    quote: str
    settle: str
    exchange_symbol: str  # symbol used by the exchange
    tick_size: float  # minimal price step
    lot_size: float  # minimal position size
    min_size: float  # minimal allowed position size
    min_notional: float = 0.0  # minimal notional value
    initial_margin: float = 0.0  # initial margin
    maint_margin: float = 0.0  # maintenance margin
    liquidation_fee: float = 0.0  # liquidation fee
    contract_size: float = 1.0  # contract size
    onboard_date: datetime | None = None
    delivery_date: datetime | None = None

    @property
    def price_precision(self):
        if not hasattr(self, "_price_precision"):
            self._price_precision = int(abs(np.log10(self.tick_size)))
        return self._price_precision

    @property
    def size_precision(self):
        if not hasattr(self, "_size_precision"):
            self._size_precision = int(abs(np.log10(self.lot_size)))
        return self._size_precision

    def is_futures(self) -> bool:
        return self.market_type in [MarketType.FUTURE, MarketType.SWAP]

    def is_spot(self) -> bool:
        # TODO: handle margin better
        return self.market_type in [MarketType.SPOT, MarketType.MARGIN]

    def round_size_down(self, size: float) -> float:
        """
        Round down size to specified precision

        i.size_precision == 3
        i.round_size_up(0.1234) -> 0.123
        """
        return prec_floor(size, self.size_precision)

    def round_size_up(self, size: float) -> float:
        """
        Round up size to specified precision

        i.size_precision == 3
        i.round_size_up(0.1234) -> 0.124
        """
        return prec_ceil(size, self.size_precision)

    def round_price_down(self, price: float) -> float:
        """
        Round down price to specified precision

        i.price_precision == 3
        i.round_price_down(1.234999, 3) -> 1.234
        """
        return prec_floor(price, self.price_precision)

    def round_price_up(self, price: float) -> float:
        """
        Round up price to specified precision

        i.price_precision == 3
        i.round_price_up(1.234999) -> 1.235
        """
        return prec_ceil(price, self.price_precision)

    def signal(
        self,
        signal: float,
        price: float | None = None,
        stop: float | None = None,
        take: float | None = None,
        group: str = "",
        comment: str = "",
        options: dict[str, Any] | None = None,  # - probably we need to remove it ?
        **kwargs,
    ) -> Signal:
        return Signal(
            instrument=self,
            signal=signal,
            price=price,
            stop=stop,
            take=take,
            group=group,
            comment=comment,
            options=(options or {}) | kwargs,
        )

    def __hash__(self) -> int:
        return hash((self.symbol, self.exchange, self.market_type))

    def __eq__(self, other: Any) -> bool:
        if other is None or not isinstance(other, Instrument):
            return False
        return str(self) == str(other)

    def __str__(self) -> str:
        return ":".join([self.exchange, self.market_type, self.symbol])

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class BatchEvent:
    time: dt_64 | pd.Timestamp
    data: list[Any]


class TransactionCostsCalculator:
    """
    A class for calculating transaction costs for a trading strategy.
    Attributes
    ----------
    name : str
        The name of the transaction costs calculator.
    maker : float
        The maker fee, as a percentage of the transaction value.
    taker : float
        The taker fee, as a percentage of the transaction value.

    """

    name: str
    maker: float
    taker: float

    def __init__(self, name: str, maker: float, taker: float):
        self.name = name
        self.maker = maker / 100.0
        self.taker = taker / 100.0

    def get_execution_fees(
        self, instrument: Instrument, exec_price: float, amount: float, crossed_market=False, conversion_rate=1.0
    ):
        if crossed_market:
            return abs(amount * exec_price) * self.taker / conversion_rate
        else:
            return abs(amount * exec_price) * self.maker / conversion_rate

    def get_overnight_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def get_funding_rates_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def __repr__(self):
        return f"<{self.name}: {self.maker * 100:.4f} / {self.taker * 100:.4f}>"


ZERO_COSTS = TransactionCostsCalculator("Zero", 0.0, 0.0)


@dataclass
class TriggerEvent:
    """
    Event data for strategy trigger
    """

    time: dt_64
    type: str
    instrument: Optional[Instrument]
    data: Optional[Any]


@dataclass
class MarketEvent:
    """
    Market data update.
    """

    time: dt_64
    type: str
    instrument: Instrument | None
    data: Any
    is_trigger: bool = False

    def to_trigger(self) -> TriggerEvent:
        return TriggerEvent(self.time, self.type, self.instrument, self.data)

    def __repr__(self):
        _items = [
            f"time={self.time}",
            f"type={self.type}",
        ]
        if self.instrument is not None:
            _items.append(f"instrument={self.instrument}")
        _items.append(f"data={self.data}")
        return f"MarketEvent({', '.join(_items)})"


@dataclass
class Deal:
    id: str  # trade id
    order_id: str  # order's id
    time: dt_64  # time of trade
    amount: float  # signed traded amount: positive for buy and negative for selling
    price: float
    aggressive: bool
    fee_amount: float | None = None
    fee_currency: str | None = None


OrderType = Literal["MARKET", "LIMIT", "STOP_MARKET", "STOP_LIMIT"]
OrderSide = Literal["BUY", "SELL"]
OrderStatus = Literal["OPEN", "CLOSED", "CANCELED", "NEW"]


@dataclass
class OrderRequest:
    instrument: Instrument
    quantity: float
    price: float | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    id: str
    type: OrderType
    instrument: Instrument
    time: dt_64
    quantity: float
    price: float
    side: OrderSide
    status: OrderStatus
    time_in_force: str
    client_id: str | None = None
    cost: float = 0.0
    options: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.id}] {self.type} {self.side} {self.quantity} of {self.instrument} {('@ ' + str(self.price)) if self.price > 0 else ''} ({self.time_in_force}) [{self.status}]"


@dataclass
class AssetBalance:
    free: float = 0.0
    locked: float = 0.0
    total: float = 0.0

    def __str__(self) -> str:
        return f"free={self.free:.2f} locked={self.locked:.2f} total={self.total:.2f}"

    def lock(self, lock_amount: float) -> None:
        self.locked += lock_amount
        self.free = self.total - self.locked

    def __add__(self, amount: float) -> "AssetBalance":
        self.total += amount
        self.free += amount
        return self

    def __sub__(self, amount: float) -> "AssetBalance":
        self.total -= amount
        self.free -= amount
        return self


MARKET_TYPE = Literal["SPOT", "MARGIN", "SWAP", "FUTURES", "OPTION"]


class Position:
    instrument: Instrument  # instrument for this position
    quantity: float = 0.0  # quantity positive for long and negative for short
    pnl: float = 0.0  # total cumulative position PnL in portfolio basic funds currency
    r_pnl: float = 0.0  # total cumulative position PnL in portfolio basic funds currency
    market_value: float = 0.0  # position's market value in quote currency
    market_value_funds: float = 0.0  # position market value in portfolio funded currency
    position_avg_price: float = 0.0  # average position price
    position_avg_price_funds: float = 0.0  # average position price
    commissions: float = 0.0  # cumulative commissions paid for this position

    last_update_time: int = np.nan  # when price updated or position changed    # type: ignore
    last_update_price: float = np.nan  # last update price (actually instrument's price) in quoted currency
    last_update_conversion_rate: float = np.nan  # last update conversion rate

    # margin requirements
    maint_margin: float = 0.0

    # - helpers for position processing
    _qty_multiplier: float = 1.0
    __pos_incr_qty: float = 0

    def __init__(
        self,
        instrument: Instrument,
        quantity: float = 0.0,
        pos_average_price: float = 0.0,
        r_pnl: float = 0.0,
    ) -> None:
        self.instrument = instrument

        self.reset()
        if quantity != 0.0 and pos_average_price > 0.0:
            self.quantity = quantity
            self.position_avg_price = pos_average_price
            self.r_pnl = r_pnl

    def reset(self) -> None:
        """
        Reset position to zero
        """
        self.quantity = 0.0
        self.pnl = 0.0
        self.r_pnl = 0.0
        self.market_value = 0.0
        self.market_value_funds = 0.0
        self.position_avg_price = 0.0
        self.position_avg_price_funds = 0.0
        self.commissions = 0.0
        self.last_update_time = np.nan  # type: ignore
        self.last_update_price = np.nan
        self.last_update_conversion_rate = np.nan
        self.maint_margin = 0.0
        self.__pos_incr_qty = 0
        self._qty_multiplier = self.instrument.contract_size

    def reset_by_position(self, pos: "Position") -> None:
        self.quantity = pos.quantity
        self.pnl = pos.pnl
        self.r_pnl = pos.r_pnl
        self.market_value = pos.market_value
        self.market_value_funds = pos.market_value_funds
        self.position_avg_price = pos.position_avg_price
        self.position_avg_price_funds = pos.position_avg_price_funds
        self.commissions = pos.commissions
        self.last_update_time = pos.last_update_time
        self.last_update_price = pos.last_update_price
        self.last_update_conversion_rate = pos.last_update_conversion_rate
        self.maint_margin = pos.maint_margin
        self.__pos_incr_qty = pos.__pos_incr_qty

    @property
    def notional_value(self) -> float:
        return self.quantity * self.last_update_price

    def _price(self, update: Quote | Trade) -> float:
        if isinstance(update, Quote):
            return update.bid if np.sign(self.quantity) > 0 else update.ask
        elif isinstance(update, Trade):
            return update.price
        raise ValueError(f"Unknown update type: {type(update)}")

    def change_position_by(
        self, timestamp: dt_64, amount: float, exec_price: float, fee_amount: float = 0, conversion_rate: float = 1
    ) -> tuple[float, float]:
        return self.update_position(
            timestamp,
            self.instrument.round_size_down(self.quantity + amount),
            exec_price,
            fee_amount,
            conversion_rate=conversion_rate,
        )

    def update_position(
        self, timestamp: dt_64, position: float, exec_price: float, fee_amount: float = 0, conversion_rate: float = 1
    ) -> tuple[float, float]:
        # - realized PnL of this fill
        deal_pnl = 0
        quantity = self.quantity
        comms = 0

        if quantity != position:
            pos_change = position - quantity
            direction = np.sign(pos_change)
            prev_direction = np.sign(quantity)

            # how many shares are closed/open
            qty_closing = min(abs(self.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
            qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing

            # - extract realized part of PnL
            if qty_closing != 0:
                _abs_qty_close = abs(qty_closing)
                deal_pnl = qty_closing * (self.position_avg_price - exec_price)

                quantity += qty_closing
                self.__pos_incr_qty -= _abs_qty_close

                # - reset average price to 0 if smaller than minimal price change to avoid cumulative error
                if abs(quantity) < self.instrument.lot_size:
                    quantity = 0.0
                    self.position_avg_price = 0.0
                    self.__pos_incr_qty = 0

            # - if it has something to add to position let's update price and cost
            if qty_opening != 0:
                _abs_qty_open = abs(qty_opening)
                pos_avg_price_raw = (_abs_qty_open * exec_price + self.__pos_incr_qty * self.position_avg_price) / (
                    self.__pos_incr_qty + _abs_qty_open
                )
                # - round position average price to be in line with how it's calculated by broker
                self.position_avg_price = (
                    self.instrument.round_price_down(pos_avg_price_raw)
                    if direction < 0
                    else self.instrument.round_price_up(pos_avg_price_raw)
                )
                self.__pos_incr_qty += _abs_qty_open

            # - update position and position's price
            self.position_avg_price_funds = self.position_avg_price / conversion_rate
            self.quantity = position

            # - convert PnL to fund currency
            self.r_pnl += deal_pnl / conversion_rate

            # - update pnl
            self.update_market_price(time_as_nsec(timestamp), exec_price, conversion_rate)

            # - calculate transaction costs
            comms = fee_amount / conversion_rate
            self.commissions += comms

        return deal_pnl, comms

    def update_market_price_by_tick(self, tick: Quote | Trade, conversion_rate: float = 1) -> float:
        return self.update_market_price(tick.time, self._price(tick), conversion_rate)

    def update_position_by_deal(self, deal: Deal, conversion_rate: float = 1) -> tuple[float, float]:
        time = deal.time.as_unit("ns").asm8 if isinstance(deal.time, pd.Timestamp) else deal.time
        return self.change_position_by(
            timestamp=time,
            amount=deal.amount,
            exec_price=deal.price,
            fee_amount=deal.fee_amount or 0,
            conversion_rate=conversion_rate,
        )
        # - deal contains cumulative amount
        # return self.update_position(time, deal.amount, deal.price, deal.aggressive, conversion_rate)

    def update_market_price(self, timestamp: dt_64, price: float, conversion_rate: float) -> float:
        self.last_update_time = timestamp  # type: ignore
        self.last_update_price = price
        self.last_update_conversion_rate = conversion_rate

        if not np.isnan(price):
            u_pnl = self.unrealized_pnl()
            self.pnl = u_pnl + self.r_pnl
            if self.instrument.is_futures():
                # for derivatives market value of the position is the current unrealized PnL
                self.market_value = u_pnl
            else:
                # for spot: market value is the current value of the position
                # TODO: implement market value calculation for margin
                self.market_value = self.quantity * self.last_update_price * self._qty_multiplier

            # calculate mkt value in funded currency
            self.market_value_funds = self.market_value / conversion_rate

            # - update margin requirements
            self._update_maint_margin()

        return self.pnl

    def total_pnl(self) -> float:
        # TODO: account for commissions
        return self.r_pnl + self.unrealized_pnl()

    def unrealized_pnl(self) -> float:
        if not np.isnan(self.last_update_price):
            return self.quantity * (self.last_update_price - self.position_avg_price) / self.last_update_conversion_rate  # type: ignore
        return 0.0

    def is_open(self) -> bool:
        return abs(self.quantity) > self.instrument.min_size

    def get_amount_released_funds_after_closing(self, to_remain: float = 0.0) -> float:
        """
        Estimate how much funds would be released if part of position closed
        """
        d = np.sign(self.quantity)
        funds_release = self.market_value_funds
        if to_remain != 0 and self.quantity != 0 and np.sign(to_remain) == d:
            qty_to_release = max(self.quantity - to_remain, 0) if d > 0 else min(self.quantity - to_remain, 0)
            funds_release = qty_to_release * self.last_update_price / self.last_update_conversion_rate
        return abs(funds_release)

    @staticmethod
    def _t2s(t) -> str:
        return (
            np.datetime64(t, "ns").astype("datetime64[ms]").item().strftime("%Y-%m-%d %H:%M:%S")
            if not np.isnan(t)
            else "???"
        )

    def __str__(self):
        return " ".join(
            [
                f"{self._t2s(self.last_update_time)}",
                f"[{self.instrument}]",
                f"qty={self.quantity:.{self.instrument.size_precision}f}",
                f"entryPrice={self.position_avg_price:.{self.instrument.price_precision}f}",
                f"price={self.last_update_price:.{self.instrument.price_precision}f}",
                f"pnl={self.unrealized_pnl():.2f}",
                f"value={self.market_value_funds:.2f}",
            ]
        )

    def __repr__(self):
        return self.__str__()

    def _update_maint_margin(self) -> None:
        if self.instrument.maint_margin:
            self.maint_margin = (
                self.instrument.maint_margin * self._qty_multiplier * abs(self.quantity) * self.last_update_price
            )


class CtrlChannel:
    """
    Controlled data communication channel
    """

    control: Event
    _queue: Queue  # we need something like disruptor here (Queue is temporary)
    name: str
    lock: Lock

    def __init__(self, name: str, sentinel=(None, None, None)):
        self.name = name
        self.control = Event()
        self.lock = Lock()
        self._sent = sentinel
        self._queue = Queue()
        self.start()

    def register(self, callback):
        pass

    def stop(self):
        if self.control.is_set():
            self.control.clear()
            self._queue.put(self._sent)  # send sentinel

    def start(self):
        self.control.set()

    def send(self, data):
        if self.control.is_set():
            self._queue.put(data)

    def receive(self, timeout: int | None = None) -> Any:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            raise QueueTimeout(f"Timeout waiting for data on {self.name} channel")


class ITimeProvider:
    """
    Generic interface for providing current time
    """

    def time(self) -> dt_64:
        """
        Returns current time
        """
        ...


class Subtype(StrEnum):
    """
    Subscription type constants. Used for specifying the type of data to subscribe to.
    Special value `Subtype.ALL` can be used to subscribe to all available data types
    that are currently in use by the broker for other instruments.
    """

    ALL = "__all__"
    NONE = "__none__"
    QUOTE = "quote"
    TRADE = "trade"
    OHLC = "ohlc"
    ORDERBOOK = "orderbook"
    LIQUIDATION = "liquidation"
    FUNDING_RATE = "funding_rate"
    OHLC_TICKS = "ohlc_ticks"  # when we want to emulate ticks from OHLC data

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Subtype):
            return self.value == other.value
        return self.value == Subtype.from_str(other)[0].value

    def __hash__(self) -> int:
        return hash(self.value)

    def __getitem__(self, *args, **kwargs) -> str:
        match self:
            case Subtype.OHLC | Subtype.OHLC_TICKS:
                tf = args[0] if args else kwargs.get("timeframe")
                if not tf:
                    raise ValueError("Timeframe is not provided for OHLC subscription")
                return f"{self.value}({tf})"
            case Subtype.ORDERBOOK:
                if len(args) == 2:
                    tick_size_pct, depth = args
                elif len(args) > 0:
                    raise ValueError(f"Invalid arguments for ORDERBOOK subscription: {args}")
                else:
                    tick_size_pct = kwargs.get("tick_size_pct", 0.01)
                    depth = kwargs.get("depth", 200)
                return f"{self.value}({tick_size_pct}, {depth})"
            case _:
                return self.value

    @staticmethod
    def from_str(value: Union[str, "Subtype"]) -> tuple["Subtype", dict[str, Any]]:
        """
        Parse subscription type from string.
        Returns: (subtype, params)

        Example:
        >>> Subtype.from_str("ohlc(1Min)")
        (Subtype.OHLC, {"timeframe": "1Min"})

        >>> Subtype.from_str("orderbook(0.01, 100)")
        (Subtype.ORDERBOOK, {"tick_size_pct": 0.01, "depth": 100})

        >>> Subtype.from_str("quote")
        (Subtype.QUOTE, {})
        """
        if isinstance(value, Subtype):
            return value, {}
        try:
            _value = value.lower()
            _has_params = Subtype._str_has_params(value)
            if not _has_params and value.upper() not in Subtype.__members__:
                return Subtype.NONE, {}
            elif not _has_params:
                return Subtype(_value), {}
            else:
                type_name, params_str = value.split("(", 1)
                params = [p.strip() for p in params_str.rstrip(")").split(",")]
                match type_name.lower():
                    case Subtype.OHLC.value:
                        return Subtype.OHLC, {"timeframe": time_delta_to_str(pd.Timedelta(params[0]).asm8.item())}

                    case Subtype.OHLC_TICKS.value:
                        return Subtype.OHLC_TICKS, {"timeframe": time_delta_to_str(pd.Timedelta(params[0]).asm8.item())}

                    case Subtype.ORDERBOOK.value:
                        return Subtype.ORDERBOOK, {"tick_size_pct": float(params[0]), "depth": int(params[1])}

                    case _:
                        return Subtype.NONE, {}
        except IndexError:
            raise ValueError(f"Invalid subscription type: {value}")

    @staticmethod
    def _str_has_params(value: str) -> bool:
        return "(" in value


class TradingSessionResult:
    id: int
    name: str
    start: str | pd.Timestamp
    stop: str | pd.Timestamp
    exchange: str
    instruments: list[Instrument]
    capital: float
    leverage: float
    base_currency: str
    commissions: str
    portfolio_log: pd.DataFrame
    executions_log: pd.DataFrame
    signals_log: pd.DataFrame
    is_simulation: bool

    def __init__(
        self,
        id: int,
        name: str,
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        exchange: str,
        instruments: list[Instrument],
        capital: float,
        leverage: float,
        base_currency: str,
        commissions: str,
        portfolio_log: pd.DataFrame,
        executions_log: pd.DataFrame,
        signals_log: pd.DataFrame,
        is_simulation=True,
    ):
        self.id = id
        self.name = name
        self.start = start
        self.stop = stop
        self.exchange = exchange
        self.instruments = instruments
        self.capital = capital
        self.leverage = leverage
        self.base_currency = base_currency
        self.commissions = commissions
        self.portfolio_log = portfolio_log
        self.executions_log = executions_log
        self.signals_log = signals_log
        self.is_simulation = is_simulation


@dataclass
class Liquidation:
    time: dt_64
    quantity: float
    price: float
    side: int


@dataclass
class FundingRate:
    time: dt_64
    rate: float
    interval: str
    next_funding_time: dt_64
    mark_price: float | None = None
    index_price: float | None = None


class LiveTimeProvider(ITimeProvider):
    def __init__(self):
        self._start_ntp_thread()

    def time(self) -> dt_64:
        return time_now()

    def _start_ntp_thread(self):
        start_ntp_thread()
