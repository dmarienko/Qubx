import pandas as pd
from qubx import lookup, logger
from qubx.core.account import AccountProcessor
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.loggers import StrategyLogging
from qubx.core.basics import TargetPosition, Instrument, Position
from qubx.core.loggers import StrategyLogging
from qubx.core.interfaces import (
    IBrokerServiceProvider,
    IPositionGathering,
    IStrategy,
    ITradingServiceProvider,
    IStrategyContext,
    IUniverseManager,
    ISubscriptionManager,
    ITradingManager,
    ITimeProvider,
)


class UniverseManager(IUniverseManager):
    __context: IStrategyContext
    __strategy: IStrategy
    __broker: IBrokerServiceProvider
    __trading_service: ITradingServiceProvider
    __cache: CachedMarketDataHolder
    __logging: StrategyLogging
    __subscription_manager: ISubscriptionManager
    __trading_manager: ITradingManager
    __time_provider: ITimeProvider
    __positions: dict[Instrument, Position]
    __position_gathering: IPositionGathering

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
        account_processor: AccountProcessor,
        position_gathering: IPositionGathering,
    ):
        self.__context = context
        self.__strategy = strategy
        self.__broker = broker
        self.__trading_service = trading_service
        self.__cache = cache
        self.__logging = logging
        self.__subscription_manager = subscription_manager
        self.__trading_manager = trading_manager
        self.__time_provider = time_provider
        self.__positions = account_processor.positions
        self.__position_gathering = position_gathering
        self._instruments = []

    def set_universe(self, instruments: list[Instrument], skip_callback: bool = False) -> None:
        new_set = set(instruments)
        prev_set = set(self._instruments)
        rm_instr = list(prev_set - new_set)
        add_instr = list(new_set - prev_set)

        self.__add_instruments(add_instr)
        self.__remove_instruments(rm_instr)

        if not skip_callback and (add_instr or rm_instr):
            self.__strategy.on_universe_change(self.__context, add_instr, rm_instr)

        self.__broker.commit()  # apply pending changes

        # set new instruments
        self._instruments.clear()
        self._instruments.extend(instruments)

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
            TargetPosition.zero(self.__context, instr.signal(0, group="Universe", comment="Universe change"))
            for instr in instruments
            if instr.symbol in self.__positions and abs(self.__positions[instr.symbol].quantity) > instr.min_size
        ]
        self.__position_gathering.alter_positions(self.__context, exit_targets)

        # - if still open positions close them manually
        for instr in instruments:
            pos = self.__positions.get(instr)
            if pos and abs(pos.quantity) > instr.min_size:
                self.__trading_manager.trade(instr, -pos.quantity)

        # - unsubscribe from market data
        for instr in instruments:
            self.__subscription_manager.unsubscribe(instr)

        # - remove from data cache
        for instr in instruments:
            self.__cache.remove(instr)

    def __add_instruments(self, instruments: list[Instrument]) -> None:
        # - create positions for instruments
        self._create_and_update_positions(instruments)

        # - get actual positions from exchange
        for instr in instruments:
            self.__cache.init_ohlcv(instr)

        # - subscribe to market data
        self.__subscription_manager.subscribe(instruments)

        # - reinitialize strategy loggers
        self.__logging.initialize(
            self.__time_provider.time(), self.__positions, self.__trading_service.get_account().get_balances()
        )

    def _create_and_update_positions(self, instruments: list[Instrument]):
        for instrument in instruments:
            _ = self.__trading_service.get_position(instrument)

            # - check if we need any aux instrument for calculating pnl ?
            # TODO: test edge cases for aux symbols
            aux = lookup.find_aux_instrument_for(instrument, self.__trading_service.get_base_currency())
            if aux is not None:
                instrument._aux_instrument = aux
                instruments.append(aux)
                _ = self.__trading_service.get_position(aux)
