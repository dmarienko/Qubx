import pandas as pd

from qubx import logger, lookup
from qubx.core.basics import Instrument, Position, Subtype, TargetPosition
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.interfaces import (
    IAccountProcessor,
    IBrokerServiceProvider,
    IPositionGathering,
    IStrategy,
    IStrategyContext,
    ISubscriptionManager,
    ITimeProvider,
    ITradingManager,
    ITradingServiceProvider,
    IUniverseManager,
)
from qubx.core.loggers import StrategyLogging


class UniverseManager(IUniverseManager):
    _context: IStrategyContext
    _strategy: IStrategy
    _broker: IBrokerServiceProvider
    _trading_service: ITradingServiceProvider
    _cache: CachedMarketDataHolder
    _logging: StrategyLogging
    _subscription_manager: ISubscriptionManager
    _trading_manager: ITradingManager
    _time_provider: ITimeProvider
    _positions: dict[Instrument, Position]
    _position_gathering: IPositionGathering

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        broker: IBrokerServiceProvider,
        trading_service: ITradingServiceProvider,
        cache: CachedMarketDataHolder,
        logging: StrategyLogging,
        subscription_manager: ISubscriptionManager,
        trading_manager: ITradingManager,
        time_provider: ITimeProvider,
        account_processor: IAccountProcessor,
        position_gathering: IPositionGathering,
    ):
        self._context = context
        self._strategy = strategy
        self._broker = broker
        self._trading_service = trading_service
        self._cache = cache
        self._logging = logging
        self._subscription_manager = subscription_manager
        self._trading_manager = trading_manager
        self._time_provider = time_provider
        self._positions = account_processor.positions
        self._position_gathering = position_gathering
        self._instruments = []

    def set_universe(self, instruments: list[Instrument], skip_callback: bool = False) -> None:
        new_set = set(instruments)
        prev_set = set(self._instruments)
        rm_instr = list(prev_set - new_set)
        add_instr = list(new_set - prev_set)

        self.__add_instruments(add_instr)
        self.__remove_instruments(rm_instr)

        if not skip_callback and (add_instr or rm_instr):
            self._strategy.on_universe_change(self._context, add_instr, rm_instr)

        self._subscription_manager.commit()  # apply pending changes

        # set new instruments
        self._instruments.clear()
        self._instruments.extend(instruments)

    def add_instruments(self, instruments: list[Instrument]):
        self.__add_instruments(instruments)
        self._strategy.on_universe_change(self._context, instruments, [])
        self._subscription_manager.commit()
        self._instruments.extend(instruments)

    def remove_instruments(self, instruments: list[Instrument]):
        self.__remove_instruments(instruments)
        self._strategy.on_universe_change(self._context, [], instruments)
        self._subscription_manager.commit()
        self._instruments = list(set(self._instruments) - set(instruments))

    @property
    def instruments(self) -> list[Instrument]:
        return self._instruments

    def __remove_instruments(self, instruments: list[Instrument]) -> None:
        """
        Remove symbols from universe. Steps:
        - close all open positions
        - unsubscribe from market data
        - remove from data cache

        We are still keeping the symbols in the positions dictionary.
        """
        # - close all open positions
        exit_targets = [
            TargetPosition.zero(self._context, instr.signal(0, group="Universe", comment="Universe change"))
            for instr in instruments
            if instr.symbol in self._positions and abs(self._positions[instr.symbol].quantity) > instr.min_size
        ]
        self._position_gathering.alter_positions(self._context, exit_targets)

        # - if still open positions close them manually
        for instr in instruments:
            pos = self._positions.get(instr)
            if pos and abs(pos.quantity) > instr.min_size:
                self._trading_manager.trade(instr, -pos.quantity)

        # - unsubscribe from market data
        for instr in instruments:
            self._subscription_manager.unsubscribe(Subtype.ALL, instr)

        # - remove from data cache
        for instr in instruments:
            self._cache.remove(instr)

    def __add_instruments(self, instruments: list[Instrument]) -> None:
        # - create positions for instruments
        self._create_and_update_positions(instruments)

        # - get actual positions from exchange
        for instr in instruments:
            self._cache.init_ohlcv(instr)

        # - subscribe to market data
        self._subscription_manager.subscribe(
            (
                Subtype.ALL
                if self._subscription_manager.auto_subscribe
                else self._subscription_manager.get_base_subscription()
            ),
            instruments,
        )

        # - reinitialize strategy loggers
        self._logging.initialize(
            self._time_provider.time(), self._positions, self._trading_service.get_account().get_balances()
        )

    def _create_and_update_positions(self, instruments: list[Instrument]):
        for instrument in instruments:
            _ = self._trading_service.get_position(instrument)

            # - check if we need any aux instrument for calculating pnl ?
            # TODO: test edge cases for aux symbols
            aux = lookup.find_aux_instrument_for(instrument, self._trading_service.get_base_currency())
            if aux is not None:
                instrument._aux_instrument = aux
                instruments.append(aux)
                _ = self._trading_service.get_position(aux)
