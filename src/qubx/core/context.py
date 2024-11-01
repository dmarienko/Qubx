import traceback

from typing import Any, Callable
from threading import Thread

from qubx import logger
from qubx.core.account import AccountProcessor
from qubx.core.helpers import BasicScheduler, CachedMarketDataHolder, set_parameters_to_object
from qubx.core.loggers import LogsWriter, StrategyLogging
from qubx.core.basics import Instrument, dt_64, SW, CtrlChannel
from qubx.core.loggers import StrategyLogging
from qubx.core.interfaces import (
    IBrokerServiceProvider,
    IPositionGathering,
    IStrategy,
    ITradingServiceProvider,
    PositionsTracker,
    IStrategyContext,
)
from qubx.data.readers import DataReader
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.trackers.sizers import FixedSizer
from .mixins import ProcessingManager, SubscriptionManager, TradingManager, UniverseManager, MarketDataProvider


class StrategyContext(
    IStrategyContext, MarketDataProvider, ProcessingManager, SubscriptionManager, TradingManager, UniverseManager
):
    DEFAULT_POSITION_TRACKER: Callable[[], PositionsTracker] = lambda: PositionsTracker(
        FixedSizer(1.0, amount_in_quote=False)
    )

    __trading_service: ITradingServiceProvider  # service for exchange API: orders managemewnt
    __logging: StrategyLogging  # recording all activities for the strat: execs, positions, portfolio
    __broker: IBrokerServiceProvider  # market data provider
    __cache: CachedMarketDataHolder
    __scheduler: BasicScheduler
    __initial_instruments: list[Instrument]

    __thread_data_loop: Thread | None = None  # market data loop
    __is_initialized: bool = False

    def __init__(
        self,
        strategy: IStrategy,
        broker: IBrokerServiceProvider,
        account: AccountProcessor,
        instruments: list[Instrument],
        logging: StrategyLogging,
        config: dict[str, Any] | None = None,
        position_gathering: IPositionGathering | None = None,
        aux_data_provider: DataReader | None = None,
    ) -> None:
        self.account = account
        self.strategy = self.__instantiate_strategy(strategy, config)

        self.__broker = broker
        self.__logging = logging
        self.__scheduler = broker.get_scheduler()
        self.__trading_service = broker.get_trading_service()
        self.__trading_service.set_account(self.account)
        self.__initial_instruments = instruments

        self.__cache = CachedMarketDataHolder()

        # fmt: off
        __position_tracker = self.strategy.tracker(self) or self.DEFAULT_POSITION_TRACKER()
        __position_gathering = position_gathering or SimplePositionGatherer()
        MarketDataProvider.__init__(
            self,
            cache=self.__cache,
            broker=self.__broker,
            universe_manager=self,
            aux_data_provider=aux_data_provider,
        )
        UniverseManager.__init__(
            self,
            context=self,
            strategy=self.strategy,
            broker=self.__broker,
            trading_service=self.__trading_service,
            cache=self.__cache,
            logging=self.__logging,
            subscription_manager=self,
            trading_manager=self,
            time_provider=self,
            account_processor=self.account,
            position_gathering=__position_gathering,
        )
        SubscriptionManager.__init__(
            self,
            broker=self.__broker
        )
        TradingManager.__init__(
            self,
            time_provider=self,
            trading_service=self.__trading_service,
            strategy_name=self.strategy.__class__.__name__,
        )
        ProcessingManager.__init__(
            self,
            context=self,
            strategy=self.strategy,
            logging=self.__logging,
            market_data=self,
            subscription_manager=self,
            time_provider=self,
            position_tracker=__position_tracker,
            position_gathering=__position_gathering,
            cache=self.__cache,
            scheduler=self.__broker.get_scheduler(),
            is_simulation=self.__broker.is_simulated_trading,
        )
        # fmt: on
        self.__post_init__()

    def __post_init__(self):
        self.strategy.on_init(self)
        # - update cache default timeframe
        _, __sub_params = self.get_base_subscription()
        __default_timeframe = __sub_params.get("timeframe", "1Sec")
        self.__cache.update_default_timeframe(__default_timeframe)

    def time(self) -> dt_64:
        return self.__trading_service.time()

    def start(self, blocking: bool = False):
        if self.__is_initialized:
            raise ValueError("Strategy is already started !")

        # - run cron scheduler
        self.__scheduler.run()

        # - create incoming market data processing
        databus = self.__broker.get_communication_channel()
        databus.register(self)

        # - initialize strategy (should we do that after any first market data received ?)
        if not self.__is_initialized:
            try:
                self.strategy.on_start(self)
                self.__is_initialized = True
            except Exception as strat_error:
                logger.error(
                    f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_start: {strat_error}"
                )
                logger.error(traceback.format_exc())
                return

        # - update universe with initial instruments after the strategy is initialized
        self.set_universe(self.__initial_instruments)

        # - for live we run loop
        if not self.__broker.is_simulated_trading:
            self.__thread_data_loop = Thread(target=self.__process_incoming_data_loop, args=(databus,), daemon=True)
            self.__thread_data_loop.start()
            logger.info("(StrategyContext) strategy is started in thread")
            if blocking:
                self.__thread_data_loop.join()

    def stop(self):
        if self.__thread_data_loop:
            self.__broker.get_communication_channel().stop()
            self.__thread_data_loop.join()
            try:
                self.strategy.on_stop(self)
            except Exception as strat_error:
                logger.error(
                    f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_stop: {strat_error}"
                )
                logger.opt(colors=False).error(traceback.format_exc())
            self.__thread_data_loop = None

        # - close logging
        self.__logging.close()

    def __process_incoming_data_loop(self, channel: CtrlChannel):
        logger.info("(StrategyContext) Start processing market data")
        while channel.control.is_set():
            with SW("StrategyContext._process_incoming_data"):
                # - waiting for incoming market data
                instrument, d_type, data = channel.receive()
                if self.process_data(instrument, d_type, data):
                    channel.stop()
                    break
        logger.info("(StrategyContext) Market data processing stopped")

    def __instantiate_strategy(self, strategy: IStrategy, config: dict[str, Any] | None) -> IStrategy:
        __strategy = strategy() if isinstance(strategy, type) else strategy
        __strategy.ctx = self
        set_parameters_to_object(__strategy, **config if config else {})
        return __strategy
