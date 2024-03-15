"""
 # All interfaces related to strategy etc
"""
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from qubx import lookup
from qubx.core.lookups import InstrumentsLookup
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
ExecutionListener = Callable[[Instrument, int], None]


class IDataProvider:
    def add_data_listener(self, listener: DataListener):
        pass

    def subscribe(self, instruments: List[Instrument]):
        pass

    def request_historical_data(self, 
                                instruments: List[Instrument], 
                                timeframe: str,
                                start: Union[str, int, dt_64], 
                                stop: Union[str, int, dt_64]):
        pass


class IExchangeServiceProvider:
    def add_execution_listener(self, listener: ExecutionListener):
        raise ValueError("add_execution_listener is not implemented")

    def sync_position(self, position: Position) -> Position:
        raise ValueError("sync_position is not implemented")

    def time(self) -> dt_64:
        """
        Returns current time
        """
        raise ValueError("time is not implemented")

    def get_name(self) -> str:
        raise ValueError("get_name is not implemented")


class IStrategy:
    ctx: 'StrategyContext'

    def populate_parameters(self, **kwargs):
        for k,v in kwargs.items():
            if k.startswith('_'):
                raise ValueError("Internal variable can't be set from external parameter !")
            if hasattr(self, k):
                self.__dict__[k] = v

    def on_init(self):
        pass

    def process_event(self, time: dt_64, event: Event) -> Optional[List[Signal]]:
        return None

 
class StrategyContext:
    strategy: IStrategy 
    exchange: IExchangeServiceProvider
    data: IDataProvider
    instruments: List[Instrument]
    positions: Dict[str, Position]

    def __init__(
            self, 
            strategy: IStrategy, 
            config: Optional[Dict[str, Any]],
            data: IDataProvider,
            exchange: IExchangeServiceProvider, 
            instruments: List[Instrument],
            # - need account class for holding all this data ?
            fees_spec: str, base_currency: str,
        ) -> None:
        self.strategy = strategy
        self.exchange = exchange
        self.data = data
        self.config = config
        self.base_currency = base_currency
        self.fees_spec = fees_spec

        self.instruments = []
        self.positions = {}

        for instr in instruments:
            # process instruments - need to find convertors etc
            self._create_synced_position(instr)

    def _create_synced_position(self, instrument: Instrument):
        symb = instrument.symbol

        if instrument not in self.instruments:
            self.positions[symb] = self.exchange.sync_position(
                Position(instrument, self.get_tcc(instrument))
            )
            self.instruments.append(instrument)

            # check if we need any aux instrument for this one
            # used to calculate PnL in base currency for crosses like ETH/BTC and USDT funded account
            aux = lookup.find_aux_instrument_for(instrument, self.base_currency)
            if aux is not None:
                instrument._aux_instrument = aux
                self.instruments.append(aux)
                aux_pos = Position(aux, self.get_tcc(aux))
                self.positions[aux.symbol] = self.exchange.sync_position(aux_pos)
        
        return self.positions.get(symb)

    def get_tcc(self, instrument: Instrument) -> TransactionCostsCalculator:
        tcc = lookup.fees.find(self.exchange.get_name(), self.fees_spec)
        if tcc is None:
            raise ValueError(f"Can't find fees calculator using given schema: '{self.fees_spec}' for {instrument}!")
        return tcc 

