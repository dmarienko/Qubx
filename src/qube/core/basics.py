from datetime import datetime
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import math
from dataclasses import dataclass, field
from qube.core.series import Quote, Trade, time_as_nsec
from qube.core.utils import time_to_str, time_delta_to_str, recognize_timeframe


dt_64 = np.datetime64


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
    def __init__(self, taker: float, maker: float):
        self.taker = taker
        self.maker = maker

    def get_execution_fees(self, instrument: Instrument, exec_price: float, amount: float, crossed_market=False, conversion_rate=1.0):
        if instrument.is_futures:
            amount = amount * instrument.futures_info.contract_size
        else:
            amount = amount * exec_price

        if crossed_market:
            return conversion_rate * abs(amount) * self.taker
        else:
            return conversion_rate * abs(amount) * self.maker

    def get_overnight_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def get_funding_rates_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def __repr__(self):
        return f'<TCC: {self.maker * 100:.4f} / {self.taker * 100:.4f}>'


ZERO_COSTS = TransactionCostsCalculator(0.0, 0.0)


class Position:
    instrument: Instrument                      # instrument for this poisition
    quantity: float = 0.0                       # quantity positive for long and negative for short
    tcc: TransactionCostsCalculator             # transaction costs calculator
    pnl: float = 0.0                            # total cumulative position PnL in portfolio funds currency
    r_pnl: float = 0.0                          # total cumulative position PnL in portfolio funds currency
    market_value: float = 0.0                   # position's market value in quote currency
    market_value_funds: float = 0.0             # position market value in portfolio funded currency
    cost_quoted: float = 0.0                    # position's cost in quote currency
    cost_funds: float = 0.0                     # position cost in basic currency
    commissions: float = 0.0                    # cumulative commissions paid for this position

    last_update_time: Optional[int] = None      # when price updated or position changed
    last_update_price: Optional[float] = None   # last update price (actually instrument's price) in quoted currency

    # - helpers for position formatting
    _formatter: str
    _prc_formatter: str

    def __init__(self, instrument: Instrument, tcc: TransactionCostsCalculator, 
                 quantity=0.0, average_price=0.0, aux_price=1.0, 
                 ) -> None:
        self.instrument = instrument
        self.tcc = tcc
        
        # - size/price formaters
        self._formatter = f'[{instrument.exchange}:{instrument.symbol}] %s %8.{instrument.price_precision}f %s %8.2f $%.2f / %.{instrument.price_precision}f'
        self._prc_formatter = f"%.{instrument.price_precision}f"

        if quantity != 0.0 and average_price > 0.0:
            self.quantity = quantity
            raise ValueError("[TODO] Position: restore state by quantity and avg price !!!!")

    def _price(self, update: Union[Quote, Trade]) -> float:
        if isinstance(update, Quote):
            return update.bid if np.sign(self.quantity) > 0 else update.ask
        elif isinstance(update, Trade):
            return update.price
        raise ValueError(f"Unknown update type: {type(update)}")

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

            # if self.instrument.is_futures:
                # qty_closing /= exec_price
                # qty_opening /= exec_price
                # quantity /= exec_price

            new_cost = self.cost_quoted + qty_opening * exec_price

            # if we have closed some shares
            if self.quantity != 0:
                pe = self.cost_quoted / quantity
                new_cost += qty_closing * pe 
                deal_pnl = qty_closing * (pe - exec_price)

                if self.instrument.is_futures:
                    deal_pnl *= pe

            self.cost_quoted = new_cost
            self.quantity = position

            # convert current position's cost to funds currency
            self.cost_funds = self.cost_quoted / conversion_rate

            # convert PnL to fund currency
            self.r_pnl += deal_pnl / conversion_rate

            # update pnl
            self._update_market_price(time_as_nsec(timestamp), exec_price, conversion_rate)

            # calculate transaction costs
            comms = self.tcc.get_execution_fees(self.instrument, exec_price, pos_change, aggressive, conversion_rate)
            self.commissions += comms

        return deal_pnl

    def update_market_price(self, price: Union[Quote, Trade], conversion_rate:float=1) -> float:
        return self._update_market_price(price.time, self._price(price), conversion_rate)

    def _update_market_price(self, timestamp: dt_64, price: float, conversion_rate:float) -> float:
        self.last_update_time = timestamp
        self.last_update_price = price

        self.market_value = 0
        if not np.isnan(self.last_update_price):
            self.market_value = self.quantity * self.last_update_price
            _qty_in_base = 1.0
            if self.instrument.is_futures:
                _qty_in_base = self.instrument.futures_info.contract_size * self.quantity / self.cost_quoted if self.cost_quoted != 0.0 else 0.0

            self.pnl = _qty_in_base * (self.market_value - self.cost_quoted) / conversion_rate + self.r_pnl

        # calculate mkt value in funded currency
        self.market_value_funds = self.market_value / conversion_rate
        return self.pnl


    @staticmethod
    def _t2s(t) -> str:
        return np.datetime64(t, 'ns').astype('datetime64[ms]').item().strftime('%Y-%m-%d %H:%M:%S.%f') if t else '---'

    def __str__(self):
        _p_str = (self._prc_formatter % self.last_update_price) if self.last_update_price else "---"
        return self._formatter % (Position._t2s(self.last_update_time), self.quantity, _p_str, self.pnl, self.market_value_funds, self.cost_funds)
    
