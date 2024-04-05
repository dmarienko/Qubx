from datetime import datetime
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field

import asyncio
# from multiprocessing import Queue, Process, Event, Lock
from threading import Thread, Event, Lock
from queue import Queue

from qubx.core.series import Quote, Trade, time_as_nsec
from qubx.core.utils import time_to_str, time_delta_to_str, recognize_timeframe


dt_64 = np.datetime64
td_64 = np.timedelta64


@dataclass
class FuturesInfo:
    contract_type: Optional[str] = None             # contract type  
    delivery_date: Optional[datetime] = None        # delivery date
    onboard_date: Optional[datetime] = None         # futures contract size
    contract_size: float = 1.0                      # futures contract size
    maint_margin: float = 0.0                       # maintanance margin
    required_margin: float = 0.0                    # required margin
    liquidation_fee: float = 0.0                    # liquidation cost

    def __str__(self) -> str:
        return f"{self.contract_type} ({self.contract_size}) {self.onboard_date.isoformat()} -> {self.delivery_date.isoformat()}"


@dataclass
class Instrument:
    symbol: str                                     # instrument's name
    market_type: str                                # market type (CRYPTO, STOCK, FX, etc)
    exchange: str                                   # exchange id
    base: str                                       # base symbol
    quote: str                                      # quote symbol
    margin_symbol: str                              # margin asset
    min_tick: float = 0.0                           # tick size - minimal price change
    min_size_step: float = 0.0                      # minimal position change step size
    min_size: float = 0.0                           # minimal allowed position size

    # - futures section
    futures_info: Optional[FuturesInfo] = None          

    _aux_instrument: Optional['Instrument'] = None  # instrument used for conversion to main asset basis
                                                    #  | let's say we trade BTC/ETH with main account in USDT
                                                    #  | so we need to use ETH/USDT for convert profits/losses to USDT
    _tick_precision: int = field(repr=False ,default=-1) #type: check
    _size_precision: int = field(repr=False ,default=-1)

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

    def __str__(self) -> str:
        return f"{self.exchange}:{self.symbol} [{self.market_type} {str(self.futures_info) if self.futures_info else 'SPOT ' + self.base + '/' + self.quote }]" 


@dataclass
class Signal: 
    """
    Class for presenting signals generated by strategy
    """
    instrument: Instrument
    signal: float
    price: Optional[float] = None
    stop: Optional[float] = None
    take: Optional[float] = None
    group: Optional[str] = None 
    comment: Optional[str] = None


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

    def get_execution_fees(self, instrument: Instrument, exec_price: float, amount: float, crossed_market=False, conversion_rate=1.0):
        if crossed_market:
            return abs(amount * exec_price) * self.taker / conversion_rate
        else:
            return abs(amount * exec_price) * self.maker / conversion_rate

    def get_overnight_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def get_funding_rates_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def __repr__(self):
        return f'<{self.name}: {self.maker * 100:.4f} / {self.taker * 100:.4f}>'


ZERO_COSTS = TransactionCostsCalculator('Zero', 0.0, 0.0)


@dataclass
class Deal:
    time: dt_64
    amount: float         # signed traded amount: positive for buy and negative for selling
    price: float
    aggressive: bool


@dataclass
class Order:
    id: str
    type: str
    symbol: str
    time: dt_64
    quantity: float
    price: float
    side: str
    status: str
    time_in_force: str
    client_id: str | None = None
    cost: float = 0.0
    # - use execution report
    execution: Deal | None = None
    
    def __str__(self) -> str:
        return f"[{self.id}] {self.type} {self.side} {self.quantity} of {self.symbol} {('@ ' + str(self.price)) if self.price > 0 else ''} ({self.time_in_force}) [{self.status}]"


def round_down(x, n):
    dvz = 10**(-n)
    return (int(x / dvz)) * dvz


class Position:
    instrument: Instrument                      # instrument for this poisition
    quantity: float = 0.0                       # quantity positive for long and negative for short
    tcc: TransactionCostsCalculator             # transaction costs calculator
    pnl: float = 0.0                            # total cumulative position PnL in portfolio basic funds currency
    r_pnl: float = 0.0                          # total cumulative position PnL in portfolio basic funds currency
    market_value: float = 0.0                   # position's market value in quote currency
    market_value_funds: float = 0.0             # position market value in portfolio funded currency
    position_avg_price: float = 0.0             # average position price
    position_avg_price_funds: float = 0.0       # average position price
    commissions: float = 0.0                    # cumulative commissions paid for this position

    last_update_time: int = np.nan              # when price updated or position changed
    last_update_price: float = np.nan           # last update price (actually instrument's price) in quoted currency

    # - helpers for position processing 
    _formatter: str
    _prc_formatter: str
    _qty_multiplier: float = 1.0
    __pos_incr_qty: float = 0

    def __init__(self, instrument: Instrument, tcc: TransactionCostsCalculator, 
                 quantity=0.0, pos_average_price=0.0, r_pnl=0.0
                 ) -> None:
        self.instrument = instrument
        self.tcc = tcc
        
        # - size/price formaters
        #                 time         [symbol]                                                        qty                                                  
        self._formatter = f'%s [{instrument.exchange}:{instrument.symbol}] %{instrument.size_precision+8}.{instrument.size_precision}f'
        #                           pos_avg_px                 pnl  | mkt_price mkt_value
        self._formatter += f'%10.{instrument.price_precision}f %+10.4f | %s  %10.2f'
        self._prc_formatter = f"%.{instrument.price_precision}f"
        if instrument.is_futures:
            self._qty_multiplier = instrument.futures_info.contract_size # type: ignore

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
        self.last_update_time = np.nan # type: ignore
        self.last_update_price = np.nan
        self.__pos_incr_qty = 0

    def _price(self, update: Quote | Trade) -> float:
        if isinstance(update, Quote):
            return update.bid if np.sign(self.quantity) > 0 else update.ask
        elif isinstance(update, Trade):
            return update.price
        raise ValueError(f"Unknown update type: {type(update)}")

    def change_position_by(self, timestamp: dt_64, amount: float, exec_price: float, aggressive=True, conversion_rate:float=1) -> float:
        return self.update_position(timestamp, self.quantity + amount, exec_price, aggressive=aggressive, conversion_rate=conversion_rate)

    def update_position(self, timestamp: dt_64, position: float, exec_price: float, aggressive=True, conversion_rate:float=1) -> float:
        # - realized PnL of this fill
        deal_pnl = 0
        quantity = self.quantity

        if quantity != position:
            pos_change = position - quantity
            direction = np.sign(pos_change)
            prev_direction = np.sign(quantity)

            # how many shares are closed/open
            qty_closing = min(abs(self.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
            qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing

            # - extract realized part of PnL
            if qty_closing != 0:
                deal_pnl = qty_closing * (self.position_avg_price - exec_price)
                quantity += qty_closing
                # - reset average price to 0 if smaller than minimal price change to avoid cumulative error
                if abs(quantity) < self.instrument.min_size_step:
                    quantity = 0.0
                    self.position_avg_price = 0.0
                    self.__pos_incr_qty = 0

            # - if it has something to add to position let's update price and cost
            if qty_opening != 0:
                _abs_qty_open = abs(qty_opening)
                pos_avg_price_raw = (_abs_qty_open * exec_price + self.__pos_incr_qty * self.position_avg_price) / (self.__pos_incr_qty + _abs_qty_open)
                # - round position average price to be in line with how it's calculated by broker
                self.position_avg_price = round_down(pos_avg_price_raw, self.instrument.price_precision)
                self.__pos_incr_qty += _abs_qty_open

            # - update position and position's price
            self.position_avg_price_funds = self.position_avg_price / conversion_rate
            self.quantity = position

            # - convert PnL to fund currency
            self.r_pnl += deal_pnl / conversion_rate

            # - update pnl
            self.update_market_price(time_as_nsec(timestamp), exec_price, conversion_rate)

            # - calculate transaction costs
            comms = self.tcc.get_execution_fees(self.instrument, exec_price, pos_change, aggressive, conversion_rate)
            self.commissions += comms

        return deal_pnl

    def update_market_price_by_tick(self, tick: Quote | Trade, conversion_rate:float=1) -> float:
        return self.update_market_price(tick.time, self._price(tick), conversion_rate)

    def update_position_by_deal(self, deal: Deal, conversion_rate:float=1) -> float:
        time = deal.time.as_unit('ns').asm8 if isinstance(deal.time, pd.Timestamp) else deal.time
        return self.change_position_by(time, deal.amount, deal.price, deal.aggressive, conversion_rate)

    def update_market_price(self, timestamp: dt_64, price: float, conversion_rate:float) -> float:
        self.last_update_time = timestamp # type: ignore
        self.last_update_price = price

        if not np.isnan(price):
            self.pnl = self.quantity * (price - self.position_avg_price) / conversion_rate + self.r_pnl
            self.market_value = self.quantity * self.last_update_price * self._qty_multiplier

            # calculate mkt value in funded currency
            self.market_value_funds = self.market_value / conversion_rate
        return self.pnl

    def total_pnl(self, conversion_rate:float=1.0) -> float:
        pnl = self.r_pnl
        if not np.isnan(self.last_update_price): # type: ignore
            pnl += self.quantity * (self.last_update_price - self.position_avg_price) / conversion_rate # type: ignore
        return pnl

    @staticmethod
    def _t2s(t) -> str:
        return np.datetime64(t, 'ns').astype('datetime64[ms]').item().strftime('%Y-%m-%d %H:%M:%S') if t else '---'

    def __str__(self):
        _mkt_price = (self._prc_formatter % self.last_update_price) if self.last_update_price else "---"
        return self._formatter % (Position._t2s(self.last_update_time), self.quantity, self.position_avg_price_funds,self.pnl, _mkt_price,  self.market_value_funds)
    

class CtrlChannel:
    """
    Controlled data communication channel
    """
    control: Event
    queue: Queue     # we need something like disruptor here (Queue is temporary)
    name: str
    lock: Lock

    def __init__(self, name: str):
        self.name = name
        self.control = Event()
        self.queue = Queue()
        self.lock = Lock()
        self.start()

    def stop(self):
        if self.control.is_set():
            self.control.clear()

    def start(self):
        self.control.set()


class AsyncioThreadRunner(Thread):
    channel: Optional[CtrlChannel]

    def __init__(self, channel: Optional[CtrlChannel]):
        self.result = None
        self.channel = channel
        self.loops = []
        super().__init__()

    def add(self, func, *args, **kwargs) -> 'AsyncioThreadRunner':
        self.loops.append(func(self.channel, *args, **kwargs))
        # self.f = func
        # self.ar = args
        # self.kw = kwargs
        return self

    async def run_loop(self):
        self.result = await asyncio.gather(*self.loops)

    def run(self):
        if self.channel:
            self.channel.control.set()
        asyncio.run(self.run_loop())
        # self.f(self.channel, *self.ar, **self.kw)

    def stop(self):
        if self.channel:
            self.channel.control.clear()
            self.channel.queue.put((None, None)) # send sentinel