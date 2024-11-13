import traceback

from typing import Any, Callable, List, Dict
from threading import Thread

from qubx import logger
from qubx.core.account import AccountProcessor
from qubx.core.helpers import BasicScheduler, CachedMarketDataHolder, set_parameters_to_object
from qubx.core.loggers import StrategyLogging
from qubx.core.basics import Instrument, dt_64, SW, CtrlChannel
from qubx.core.loggers import StrategyLogging
from qubx.data.readers import DataReader
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.trackers.sizers import FixedSizer
from qubx.core.interfaces import (
    IBrokerServiceProvider,
    IMarketDataProvider,
    IPositionGathering,
    IStrategy,
    ITradingServiceProvider,
    IStrategyContext,
    IUniverseManager,
    ISubscriptionManager,
    ITradingManager,
    IProcessingManager,
    PositionsTracker,
)
from .mixins import ProcessingManager, SubscriptionManager, TradingManager, UniverseManager, MarketDataProvider


class StrategyContext(IStrategyContext):
    DEFAULT_POSITION_TRACKER: Callable[[], PositionsTracker] = lambda: PositionsTracker(
        FixedSizer(1.0, amount_in_quote=False)
    )

    __market_data_provider: IMarketDataProvider
    __universe_manager: IUniverseManager
    __subscription_manager: ISubscriptionManager
    __trading_manager: ITradingManager
    __processing_manager: IProcessingManager

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
        position_gathering: IPositionGathering | None = None,  # TODO: make position gathering part of the strategy
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

        __position_tracker = self.strategy.tracker(self)
        if __position_tracker is None:
            __position_tracker = StrategyContext.DEFAULT_POSITION_TRACKER()

        __position_gathering = position_gathering if position_gathering is not None else SimplePositionGatherer()

        self.__market_data_provider = MarketDataProvider(
            cache=self.__cache,
            broker=self.__broker,
            universe_manager=self,
            aux_data_provider=aux_data_provider,
        )
        self.__universe_manager = UniverseManager(
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
        self.__subscription_manager = SubscriptionManager(broker=self.__broker)
        self.__trading_manager = TradingManager(
            time_provider=self,
            trading_service=self.__trading_service,
            strategy_name=self.strategy.__class__.__name__,
        )
        self.__processing_manager = ProcessingManager(
            context=self,
            strategy=self.strategy,
            broker=self.__broker,
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
        self.__post_init__()

    def __post_init__(self) -> None:
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

        # - update universe with initial instruments after the strategy is initialized
        self.set_universe(self.__initial_instruments, skip_callback=True)

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

        # - for live we run loop
        if not self.__broker.is_simulated_trading:
            self.__thread_data_loop = Thread(target=self.__process_incoming_data_loop, args=(databus,), daemon=True)
            self.__thread_data_loop.start()
            logger.info("(StrategyContext) strategy is started in thread")
            if blocking:
                self.__thread_data_loop.join()

    def stop(self):
        if self.__thread_data_loop:
            self.__broker.close()
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

    @property
    def is_simulation(self) -> bool:
        return self.__broker.is_simulated_trading

    # IAccountViewer delegation
    @property
    def positions(self):
        return self.account.positions

    def get_capital(self) -> float:
        return self.account.get_capital()

    def get_total_capital(self) -> float:
        return self.account.get_total_capital()

    def get_reserved(self, instrument: Instrument) -> float:
        return self.account.get_reserved(instrument)

    # IMarketDataProvider delegation
    def ohlc(self, instrument: Instrument, timeframe: str | None = None):
        return self.__market_data_provider.ohlc(instrument, timeframe)

    def quote(self, instrument: Instrument):
        return self.__market_data_provider.quote(instrument)

    def get_historical_ohlcs(self, instrument: Instrument, timeframe: str, length: int):
        return self.__market_data_provider.get_historical_ohlcs(instrument, timeframe, length)

    def get_aux_data(self, data_id: str, **parameters):
        return self.__market_data_provider.get_aux_data(data_id, **parameters)

    def get_instruments(self):
        return self.__market_data_provider.get_instruments()

    def get_instrument(self, symbol: str, exchange: str):
        return self.__market_data_provider.get_instrument(symbol, exchange)

    # ITradingManager delegation
    def trade(self, instrument: Instrument, amount: float, price: float | None = None, time_in_force="gtc", **options):
        return self.__trading_manager.trade(instrument, amount, price, time_in_force, **options)

    def cancel(self, instrument: Instrument):
        return self.__trading_manager.cancel(instrument)

    def cancel_order(self, order_id: str):
        return self.__trading_manager.cancel_order(order_id)

    # IUniverseManager delegation
    def set_universe(self, instruments: list[Instrument], skip_callback: bool = False):
        return self.__universe_manager.set_universe(instruments, skip_callback)

    @property
    def instruments(self):
        return self.__universe_manager.instruments

    # ISubscriptionManager delegation
    def subscribe(self, instruments: List[Instrument] | Instrument, subscription_type: str | None = None, **kwargs):
        return self.__subscription_manager.subscribe(instruments, subscription_type, **kwargs)

    def unsubscribe(self, instruments: List[Instrument] | Instrument, subscription_type: str | None = None):
        return self.__subscription_manager.unsubscribe(instruments, subscription_type)

    def has_subscription(self, instrument: Instrument, subscription_type: str):
        return self.__subscription_manager.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument) -> Dict[str, Dict[str, Any]]:
        return self.__subscription_manager.get_subscriptions(instrument)

    def get_base_subscription(self):
        return self.__subscription_manager.get_base_subscription()

    def set_base_subscription(self, subscription_type, **kwargs):
        return self.__subscription_manager.set_base_subscription(subscription_type, **kwargs)

    def get_warmup(self, subscription_type: str):
        return self.__subscription_manager.get_warmup(subscription_type)

    def set_warmup(self, subscription_type: str, period: str):
        return self.__subscription_manager.set_warmup(subscription_type, period)

    # IProcessingManager delegation
    def process_data(self, instrument: Instrument, d_type: str, data: Any):
        return self.__processing_manager.process_data(instrument, d_type, data)

    def set_fit_schedule(self, schedule: str):
        return self.__processing_manager.set_fit_schedule(schedule)

    def set_event_schedule(self, schedule: str):
        return self.__processing_manager.set_event_schedule(schedule)

    # private methods
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
