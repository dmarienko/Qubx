"""
 # All interfaces related to strategy etc
"""
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass

import asyncio
from threading import Thread, Event, Lock
from queue import Queue
# from multiprocessing import Queue #as Queue

from qubx import lookup, logger
from qubx.core.lookups import InstrumentsLookup
from qubx.core.basics import Instrument, Position, Signal, TransactionCostsCalculator, dt_64
from qubx.core.series import TimeSeries, Trade, Quote, Bar

from enum import Enum

class EventType(Enum):
    E_TIMER = 1
    E_QUOTE = 2
    E_TRADE = 3
    E_OPENBOOK = 4
    E_OHLC_BAR = 5
    E_HIST_DATA_READY = 100
    E_HIST_DATA_ERROR = -100

@dataclass
class TriggerEvent:
    time: dt_64
    type: EventType
    instrument: Instrument


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

    def stop(self):
        if self.control.is_set():
            self.control.clear()

    def start(self):
        self.control.set()


class AsyncioThreadRunner(Thread):
    def __init__(self, channel: CtrlChannel):
        self.result = None
        self.channel = channel
        self.loops = []
        super().__init__()

    def add(self, func, *args, **kwargs) -> 'AsyncioThreadRunner':
        self.loops.append(func(self.channel, *args, **kwargs))
        return self

    async def run_loop(self):
        self.result = await asyncio.gather(*self.loops)

    def run(self):
        self.channel.control.set()
        asyncio.run(self.run_loop())

    def stop(self):
        self.channel.control.clear()
        self.channel.queue.put((None, None)) # send sentinel


class IDataProvider:
    ch_data: CtrlChannel

    def __init__(self):
        self.ch_data = CtrlChannel('marketdata') 

    def subscribe(self, subscription_type: str, symbols: List[str], **kwargs) -> AsyncioThreadRunner:
        return None


class IExchangeServiceProvider:
    def sync_position(self, position: Position) -> Position:
        return position
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

    def on_init(self):
        pass

    def process_event(self, time: dt_64, event: TriggerEvent) -> Optional[List[Signal]]:
        return None

 
class StrategyContext:
    strategy: IStrategy 
    exchange: IExchangeServiceProvider
    mktdata: IDataProvider
    instruments: List[Instrument]
    positions: Dict[str, Position]

    def __init__(
            self, 
            strategy: IStrategy, 
            config: Optional[Dict[str, Any]],
            mktdata: IDataProvider,
            exchange: IExchangeServiceProvider, 
            instruments: List[Instrument],
            # - need account class for holding all this data ?
            fees_spec: str, base_currency: str,

            # - context's parameters
            market_data_subscription:str='ohlc',
            subscription_params: Dict[str,Any] = None,

        ) -> None:

        # - set parameters to strategy (not sure we will do it here)
        self.strategy = strategy
        if isinstance(strategy, type):
            self.strategy = strategy()
        self.populate_parameters_to_strategy(self.strategy, **config)

        self.exchange = exchange
        self.mktdata = mktdata
        self.config = config
        self.base_currency = base_currency
        self.fees_spec = fees_spec

        self.instruments = []
        self.positions = {}


        # - get actual positions from exchange
        symbols = []
        for instr in instruments:
            # process instruments - need to find convertors etc
            self._create_synced_position(instr)
            symbols.append(instr.symbol)

        # - subscribe to market data
        subscription_params = {} if subscription_params is None else subscription_params
        logger.info(f"Subscribing on {market_data_subscription} data using {subscription_params} for \n\t{symbols} ")
        mdrunner = self.mktdata.subscribe(market_data_subscription, symbols, **subscription_params)

    def _market_data_processor(self):
        self.sers = {}
        logger.info("Start listening to market data")
        channel = self.mktdata.ch_data
        while channel.control.is_set():
            s, data = channel.queue.get()

            # TODO: processing quote
            if isinstance(data, Quote):
                pass

            # TODO: processing trade
            if isinstance(data, Trade):
                pass

            # TODO: processing ohlc bars
            if isinstance(data, Bar):
                # print(f"{s} {pd.Timestamp(data.time, unit='ns')}: {data}", flush=True)
                if s not in self.sers:
                    self.sers[s] = {}
                self.sers[s][data.time] = data

        logger.info("Stop market data listening")

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
        tcc = lookup.fees.find(self.exchange.get_name().lower(), self.fees_spec)
        if tcc is None:
            raise ValueError(f"Can't find fees calculator using given schema: '{self.fees_spec}' for {instrument}!")
        return tcc 

    def populate_parameters_to_strategy(self, strategy: IStrategy, **kwargs):
        for k,v in kwargs.items():
            if k.startswith('_'):
                raise ValueError("Internal variable can't be set from external parameter !")
            if hasattr(strategy, k):
                strategy.__dict__[k] = v
                logger.info(f"Set {k} -> {v}")
