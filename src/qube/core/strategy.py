"""
 # All interfaces related to strategy etc
"""
import numpy as np
from dataclasses import dataclass

dt_64 = np.datetime64


@dataclass
class Instrument:
    # instrument name
    symbol: str
    # exchange
    exchange: str

    # tick size
    tick_size: float

    # true for futures
    is_futures: bool

    # futures contract size
    futures_contract_size: float = 1

    # instrument used for conversion to main basis
    # let's say we trade BTC/ETH with main account in USDT
    # so we need to use ETH/USDT for convert profits/losses to USDT
    currency_conversion_instrument: 'Instrument' = None


@dataclass
class Event:
    time: dt_64