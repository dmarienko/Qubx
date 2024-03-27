"""
 # All interfaces related to strategy etc
"""
from typing import Any, Callable, Dict, List, Optional, Union, Self
import numpy as np
from dataclasses import dataclass
from enum import Enum

import asyncio
from threading import Thread, Event, Lock
from queue import Queue

import pandas as pd
# from multiprocessing import Queue #as Queue

from qubx import lookup, logger
from qubx.core.lookups import InstrumentsLookup
from qubx.core.basics import Instrument, Position, Signal, TransactionCostsCalculator, dt_64
from qubx.core.series import TimeSeries, Trade, Quote, Bar


class TriggerType(Enum):
    BAR = 1
    TIME = 3
    QUOTE = 4
    TRADE = 5
    ORDERBOOK = 6

    _bar_timeframe: Optional[str] = None
    _inside_bar_delay: Optional[pd.Timedelta] = None
    _scheduled_time: Optional[str] = None

    def delay(self, delay: str) -> Self:
        if self != TriggerType.BAR:
            raise RuntimeError("'delay' can be set only for BAR trigger")
        self._inside_bar_delay = pd.Timedelta(delay)
        return self

    def timeframe(self, tframe: str) -> Self:
        if self != TriggerType.BAR and self != TriggerType.TIME:
            raise RuntimeError("'timeframe' can be set either for BAR or TIME triggers")
        self._bar_timeframe = tframe
        return self

    def time(self, time: str) -> Self:
        if self != TriggerEvent.TIME:
            raise RuntimeError("'time' can be set only for TIME trigger")
        self._scheduled_time = time


@dataclass
class TriggerEvent:
    time: dt_64
    type: TriggerType
    instrument: Optional[Instrument]
    data: Optional[Any] 


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
    channel: Optional[CtrlChannel]

    def __init__(self, channel: Optional[CtrlChannel]):
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
        if self.channel:
            self.channel.control.set()
        asyncio.run(self.run_loop())

    def stop(self):
        if self.channel:
            self.channel.control.clear()
            self.channel.queue.put((None, None)) # send sentinel


class IDataProvider:

    def subscribe(self, subscription_type: str, symbols: List[str], **kwargs) -> AsyncioThreadRunner:
        return None

    def get_communication_channel(self) -> CtrlChannel:
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

    def on_start(self, ctx: 'StrategyContext'):
        pass

    def on_event(self, ctx: 'StrategyContext', event: TriggerEvent) -> Optional[List[Signal]]:
        return None

    def on_stop(self, ctx: 'StrategyContext'):
        pass

 
class StrategyContext:
    strategy: IStrategy 
    exchange_service: IExchangeServiceProvider
    data_provider: IDataProvider
    instruments: List[Instrument]
    positions: Dict[str, Position]

    _t_mdata_processor: Optional[AsyncioThreadRunner] = None
    _t_mdata_subscriber: Optional[AsyncioThreadRunner] = None

    def __init__(self, 
            # - strategy with parameters
            strategy: IStrategy, config: Optional[Dict[str, Any]],
            # - - - - - - - - - - - - - - - - - - - - -

            # - data provider and exchange service
            data_provider: IDataProvider,
            exchange_service: IExchangeServiceProvider, 
            instruments: List[Instrument],
            # - - - - - - - - - - - - - - - - - - - - -

            # - need account class for holding all this data ?
            fees_spec: str, base_currency: str,
            # - - - - - - - - - - - - - - - - - - - - -

            # - context's parameters
            trigger_on: TriggerType, 
            md_subscription_type:str='ohlc',
            md_subscription_params: Dict[str,Any] = None,
            # - - - - - - - - - - - - - - - - - - - - -

        ) -> None:

        # - set parameters to strategy (not sure we will do it here)
        self.strategy = strategy
        if isinstance(strategy, type):
            self.strategy = strategy()
        self.populate_parameters_to_strategy(self.strategy, **config)

        # - other initialization
        self.exchange_service = exchange_service
        self.data_provider = data_provider
        self.config = config
        self.base_currency = base_currency
        self.fees_spec = fees_spec
        self.md_subscription_type = md_subscription_type
        self.md_subscription_params = md_subscription_params
        self.instruments = instruments
        self.positions = {}

        # - check how it's configured to be triggered
        self.trigger = trigger_on
        self._trig_interval_in_bar_nsec = 0
        match trigger_on:

            case TriggerType.BAR:
                if trigger_on._bar_timeframe is None:
                    raise ValueError(f"Timeframe is required for {trigger_on.name} trigger: use TriggerType.timeframe(...)")

                if trigger_on._inside_bar_delay is None:
                    raise ValueError(f"Delay is required for {trigger_on.name} trigger: use TriggerType.delay(...)")

                if abs(trigger_on._inside_bar_delay) > pd.Timedelta(trigger_on._bar_timeframe):
                    raise ValueError(f"Delay must be less or equal to bar's timeframe for {trigger_on.name} trigger: you set delay {trigger_on._inside_bar_delay} for {trigger_on._bar_timeframe}")

                # for positive delay - trigger strategy when this interval passed after new bar's open
                if trigger_on._inside_bar_delay >= pd.Timedelta(0): 
                    self._trig_interval_in_bar_nsec = trigger_on._inside_bar_delay.asm8.item()
                # for negative delay - trigger strategy when time is closer to bar's closing time more than this interval
                else:
                    self._trig_interval_in_bar_nsec = (pd.Timedelta(trigger_on._bar_timeframe) + trigger_on._inside_bar_delay).asm8.item()

            case TriggerType.TIME:
                if trigger_on._scheduled_time is None:
                    raise ValueError(f"Scheduled time is required for {trigger_on.name} type of trigger: use TriggerType.time(...)")
                raise ValueError(f"{trigger_on} NOT IMPLEMENTED")

            case _:
                raise ValueError(f"{trigger_on} NOT IMPLEMENTED")

        # - states 
        self._is_initilized = False

    async def _market_data_processor(self, channel: CtrlChannel):
        self.sers = {}
        logger.info("Start listening to market data")

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

                # - test
                t = self.exchange_service.time()
                self.strategy.on_event(self, TriggerEvent(t, TriggerType.E_OHLC_BAR, s, self.sers[s]))

        logger.info("Stop market data listening")

    def _create_synced_position(self, instrument: Instrument):
        symb = instrument.symbol

        if instrument not in self.instruments:
            self.positions[symb] = self.exchange_service.sync_position(
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
                self.positions[aux.symbol] = self.exchange_service.sync_position(aux_pos)
        
        return self.positions.get(symb)

    def get_tcc(self, instrument: Instrument) -> TransactionCostsCalculator:
        tcc = lookup.fees.find(self.exchange_service.get_name().lower(), self.fees_spec)
        if tcc is None:
            raise ValueError(f"Can't find fees calculator using given schema: '{self.fees_spec}' for {instrument}!")
        return tcc 

    def start(self):
        if self._t_mdata_processor:
            raise ValueError("Context is already started !")

        # - get actual positions from exchange
        for instr in self.instruments:
            # process instruments - need to find convertors etc
            self._create_synced_position(instr)
        
        symbols = [i.symbol for i in self.instruments]

        # - create incoming market data processing
        self._t_mdata_processor = AsyncioThreadRunner(self.data_provider.get_communication_channel())
        self._t_mdata_processor.add(self._market_data_processor)

        # - subscribe to market data
        md_subscription_params = {} if self.md_subscription_params is None else self.md_subscription_params
        logger.info(f"Subscribing on {self.md_subscription_type} data using {md_subscription_params} for \n\t{symbols} ")
        self._t_mdata_subscriber = self.data_provider.subscribe(self.md_subscription_type, symbols, **md_subscription_params)

        # - initialize on very first data (???)
        if not self._is_initilized:
            self.strategy.on_start(self)
            self._is_initilized = True

        self._t_mdata_processor.start()
        logger.info("Market data processor started")

        self._t_mdata_subscriber.start()
        logger.info("Market data ubscribers started")

    def stop(self):
        self._t_mdata_subscriber.stop()
        self._t_mdata_processor.stop()
        self._t_mdata_processor = None
        self._t_mdata_subscriber = None

    def populate_parameters_to_strategy(self, strategy: IStrategy, **kwargs):
        for k,v in kwargs.items():
            if k.startswith('_'):
                raise ValueError("Internal variable can't be set from external parameter !")
            if hasattr(strategy, k):
                strategy.__dict__[k] = v
                logger.info(f"Set {k} -> {v}")

    def time(self) -> dt_64:
        return self.exchange_service.time()