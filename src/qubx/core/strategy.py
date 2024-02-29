"""
 # All interfaces related to strategy etc
"""
from typing import Callable, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from qubx.core.basics import ZERO_COSTS, Instrument, Position, Signal, TransactionCostsCalculator, dt_64

E_TIMER = 1
E_QUOTE = 2
E_TRADE = 3
E_OPENBOOK = 4
E_HIST_DATA_READY = 100
E_HIST_DATA_ERROR = -100
@dataclass
class Event:
    time: dt_64
    type: int               # ??
    instrument: Instrument

DataListener = Callable[[Instrument, int], None]
ExchListener = Callable[[Instrument, int], None]

class DataProvider:
    def add_data_listener(self, listener: DataListener):
        pass

    def request_historical_data(self, 
                                instruments: List[Instrument], 
                                timeframe: str,
                                start: Union[str, int, dt_64], 
                                stop: Union[str, int, dt_64]):
        pass


class ExchangeServiceProvider:
    def add_exchange_listener(self, listener: ExchListener):
        pass

    def get_position(self, instrument: Instrument) -> Position:
        pass

    def get_tcc(self, instrument: Instrument) -> TransactionCostsCalculator:
        return ZERO_COSTS

    def time(self) -> dt_64:
        """
        Returns current time
        """
        pass
    

class IStrategy:
    ctx: 'TradingContext'

    def on_init(self):
        pass

    def process_event(self, time: dt_64, event: Event) -> Optional[List[Signal]]:
        return None

        
class TradingContext:
    strategy: IStrategy 
    exchange: ExchangeServiceProvider
    data: DataProvider
    instruments: List[Instrument]
    positions: Dict[str, Position]

    def __init__(
            self, 
            strategy: IStrategy, 
            exchange: ExchangeServiceProvider, 
            data: DataProvider,
            instruments: List[Instrument]
        ) -> None:
        self.strategy = strategy
        self.exchange = exchange
        self.data = data

        self.instruments = []
        self.positions = {}
        for instr in instruments:
            # process instruments - need to find convertors etc
            # . . . . 
            self.instruments.append(instr)
            p = exchange.get_position(instr)
            self.positions[instr.symbol] = Position(instr) 

