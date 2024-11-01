from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from threading import Event, Lock
from queue import Queue

from qubx.utils.misc import Stopwatch
from qubx.core.series import Quote, Trade, time_as_nsec
from qubx.core.utils import prec_ceil, prec_floor


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


@dataclass
class FuturesInfo:
    contract_type: Optional[str] = None  # contract type
    delivery_date: Optional[datetime] = None  # delivery date
    onboard_date: Optional[datetime] = None  # futures contract size
    contract_size: float = 1.0  # futures contract size
    maint_margin: float = 0.0  # maintanance margin
    required_margin: float = 0.0  # required margin
    liquidation_fee: float = 0.0  # liquidation cost

    def __str__(self) -> str:
        return f"{self.contract_type} ({self.contract_size}) {self.onboard_date.isoformat()} -> {self.delivery_date.isoformat()}"


@dataclass
class Instrument:
    symbol: str  # instrument's name
    market_type: str  # market type (CRYPTO, STOCK, FX, etc)
    exchange: str  # exchange id
    base: str  # base symbol
    quote: str  # quote symbol
    margin_symbol: str  # margin asset
    min_tick: float = 0.0  # tick size - minimal price change
    min_size_step: float = 0.0  # minimal position change step size
    min_size: float = 0.0  # minimal allowed position size

    # - futures section
    futures_info: Optional[FuturesInfo] = None

    _aux_instrument: Optional["Instrument"] = None  # instrument used for conversion to main asset basis
    #  | let's say we trade BTC/ETH with main account in USDT
    #  | so we need to use ETH/USDT for convert profits/losses to USDT
    _tick_precision: int = field(repr=False, default=-1)  # type: check
    _size_precision: int = field(repr=False, default=-1)

    @property
    def is_futures(self) -> bool:
        return self.futures_info is not None

    @property
    def price_precision(self):
        if self._tick_precision < 0:
            self._tick_precision = int(abs(np.log10(self.min_tick)))
        return self._tick_precision

    @property
    def size_precision(self):
        if self._size_precision < 0:
            self._size_precision = int(abs(np.log10(self.min_size_step)))
        return self._size_precision

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
            self,
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
        if other is None:
            return False
        if type(other) != type(self):
            return False
        return self.symbol == other.symbol and self.exchange == other.exchange and self.market_type == other.market_type

    def __str__(self) -> str:
        return f"{self.exchange}:{self.symbol} [{self.market_type} {str(self.futures_info) if self.futures_info else 'SPOT ' + self.base + '/' + self.quote }]"


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
class Deal:
    id: str | int  # trade id
    order_id: str | int  # order's id
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
class Order:
    id: str
    type: OrderType
    symbol: str
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
        return f"[{self.id}] {self.type} {self.side} {self.quantity} of {self.symbol} {('@ ' + str(self.price)) if self.price > 0 else ''} ({self.time_in_force}) [{self.status}]"


class Position:
    instrument: Instrument  # instrument for this poisition
    quantity: float = 0.0  # quantity positive for long and negative for short
    pnl: float = 0.0  # total cumulative position PnL in portfolio basic funds currency
    r_pnl: float = 0.0  # total cumulative position PnL in portfolio basic funds currency
    market_value: float = 0.0  # position's market value in quote currency
    market_value_funds: float = 0.0  # position market value in portfolio funded currency
    position_avg_price: float = 0.0  # average position price
    position_avg_price_funds: float = 0.0  # average position price
    commissions: float = 0.0  # cumulative commissions paid for this position

    last_update_time: int = np.nan  # when price updated or position changed
    last_update_price: float = np.nan  # last update price (actually instrument's price) in quoted currency
    last_update_conversion_rate: float = np.nan  # last update conversion rate

    # - helpers for position processing
    _formatter: str
    _prc_formatter: str
    _qty_multiplier: float = 1.0
    __pos_incr_qty: float = 0

    def __init__(self, instrument: Instrument, quantity=0.0, pos_average_price=0.0, r_pnl=0.0) -> None:
        self.instrument = instrument

        # - size/price formaters
        #                 time         [symbol]                                                        qty
        self._formatter = f"%s [{instrument.exchange}:{instrument.symbol}] %{instrument.size_precision+8}.{instrument.size_precision}f"
        #                           pos_avg_px                 pnl  | mkt_price mkt_value
        self._formatter += f"%10.{instrument.price_precision}f %+10.4f | %s  %10.2f"
        self._prc_formatter = f"%.{instrument.price_precision}f"
        if instrument.is_futures:
            self._qty_multiplier = instrument.futures_info.contract_size  # type: ignore

        self.reset()
        if quantity != 0.0 and pos_average_price > 0.0:
            self.quantity = quantity
            self.position_avg_price = pos_average_price
            self.r_pnl = r_pnl

    def reset(self):
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
        self.__pos_incr_qty = 0

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
                if abs(quantity) < self.instrument.min_size_step:
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
                self.position_avg_price = self.instrument.round_price_down(pos_avg_price_raw)
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
            self.pnl = self.quantity * (price - self.position_avg_price) / conversion_rate + self.r_pnl
            self.market_value = self.quantity * self.last_update_price * self._qty_multiplier

            # calculate mkt value in funded currency
            self.market_value_funds = self.market_value / conversion_rate
        return self.pnl

    def total_pnl(self) -> float:
        # TODO: account for commissions
        pnl = self.r_pnl
        if not np.isnan(self.last_update_price):  # type: ignore
            pnl += self.quantity * (self.last_update_price - self.position_avg_price) / self.last_update_conversion_rate  # type: ignore
        return pnl

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
        _mkt_price = (self._prc_formatter % self.last_update_price) if self.last_update_price else "---"
        return self._formatter % (
            Position._t2s(self.last_update_time),
            self.quantity,
            self.position_avg_price_funds,
            self.pnl,
            _mkt_price,
            self.market_value_funds,
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

    def receive(self) -> Any:
        return self._queue.get()


class SimulatedCtrlChannel(CtrlChannel):
    """
    Simulated communication channel. Here we don't use queue but it invokes callback directly
    """

    _callback: Callable[[Tuple], bool]

    def register(self, callback):
        self._callback = callback

    def send(self, data):
        # - when data is sent, invoke callback
        return self._callback.process_data(*data)

    def receive(self) -> Any:
        raise ValueError("This method should not be called in a simulated environment.")

    def stop(self):
        self.control.clear()

    def start(self):
        self.control.set()


class IComminucationManager:
    databus: CtrlChannel

    def get_communication_channel(self) -> CtrlChannel:
        return self.databus

    def set_communication_channel(self, channel: CtrlChannel):
        self.databus = channel


class ITimeProvider:
    """
    Generic interface for providing current time
    """

    def time(self) -> dt_64:
        """
        Returns current time
        """
        ...


class TradingSessionResult:
    id: int
    name: str
    start: str | pd.Timestamp
    stop: str | pd.Timestamp
    exchange: str
    instruments: List[Instrument]
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
        instruments: List[Instrument],
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
