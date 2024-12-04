import traceback
from threading import Thread
from typing import Any, Callable, Dict, List, Union

from qubx import logger
from qubx.core.basics import SW, CtrlChannel, Instrument, Subtype, dt_64
from qubx.core.helpers import (
    BasicScheduler,
    CachedMarketDataHolder,
    set_parameters_to_object,
)
from qubx.core.interfaces import (
    IAccountProcessor,
    IBrokerServiceProvider,
    IMarketDataProvider,
    IPositionGathering,
    IProcessingManager,
    IStrategy,
    IStrategyContext,
    ISubscriptionManager,
    ITradingManager,
    ITradingServiceProvider,
    IUniverseManager,
    PositionsTracker,
)
from qubx.core.loggers import StrategyLogging
from qubx.data.readers import DataReader
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.trackers.sizers import FixedSizer

from .mixins import (
    MarketDataProvider,
    ProcessingManager,
    SubscriptionManager,
    TradingManager,
    UniverseManager,
)


class StrategyContext(IStrategyContext):
    DEFAULT_POSITION_TRACKER: Callable[[], PositionsTracker] = lambda: PositionsTracker(
        FixedSizer(1.0, amount_in_quote=False)
    )

    _market_data_provider: IMarketDataProvider
    _universe_manager: IUniverseManager
    _subscription_manager: ISubscriptionManager
    _trading_manager: ITradingManager
    _processing_manager: IProcessingManager

    _trading_service: ITradingServiceProvider  # service for exchange API: orders managemewnt
    _logging: StrategyLogging  # recording all activities for the strat: execs, positions, portfolio
    _broker: IBrokerServiceProvider  # market data provider
    _cache: CachedMarketDataHolder
    _scheduler: BasicScheduler
    _initial_instruments: list[Instrument]

    _thread_data_loop: Thread | None = None  # market data loop
    _is_initialized: bool = False

    def __init__(
        self,
        strategy: IStrategy,
        broker: IBrokerServiceProvider,
        account: IAccountProcessor,
        instruments: list[Instrument],
        logging: StrategyLogging,
        config: dict[str, Any] | None = None,
        position_gathering: IPositionGathering | None = None,  # TODO: make position gathering part of the strategy
        aux_data_provider: DataReader | None = None,
    ) -> None:
        self.account = account
        self.strategy = self.__instantiate_strategy(strategy, config)

        self._broker = broker
        self._logging = logging
        self._scheduler = broker.get_scheduler()
        self._trading_service = broker.get_trading_service()
        self._initial_instruments = instruments

        self._cache = CachedMarketDataHolder()

        __position_tracker = self.strategy.tracker(self)
        if __position_tracker is None:
            __position_tracker = StrategyContext.DEFAULT_POSITION_TRACKER()

        __position_gathering = position_gathering if position_gathering is not None else SimplePositionGatherer()

        self._subscription_manager = SubscriptionManager(broker=self._broker)
        self.account.set_subscription_manager(self._subscription_manager)

        self._market_data_provider = MarketDataProvider(
            cache=self._cache,
            broker=self._broker,
            universe_manager=self,
            aux_data_provider=aux_data_provider,
        )
        self._universe_manager = UniverseManager(
            context=self,
            strategy=self.strategy,
            broker=self._broker,
            trading_service=self._trading_service,
            cache=self._cache,
            logging=self._logging,
            subscription_manager=self,
            trading_manager=self,
            time_provider=self,
            account_processor=self.account,
            position_gathering=__position_gathering,
        )
        self._trading_manager = TradingManager(
            time_provider=self,
            trading_service=self._trading_service,
            strategy_name=self.strategy.__class__.__name__,
        )
        self._processing_manager = ProcessingManager(
            context=self,
            strategy=self.strategy,
            logging=self._logging,
            market_data=self,
            subscription_manager=self,
            time_provider=self,
            position_tracker=__position_tracker,
            position_gathering=__position_gathering,
            cache=self._cache,
            scheduler=self._broker.get_scheduler(),
            is_simulation=self._broker.is_simulated_trading,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        self.strategy.on_init(self)
        # - update cache default timeframe
        sub_type = self.get_base_subscription()
        _, params = Subtype.from_str(sub_type)
        __default_timeframe = params.get("timeframe", "1Sec")
        self._cache.update_default_timeframe(__default_timeframe)

    def time(self) -> dt_64:
        return self._trading_service.time()

    def start(self, blocking: bool = False):
        if self._is_initialized:
            raise ValueError("Strategy is already started !")

        # - run cron scheduler
        self._scheduler.run()

        # - create incoming market data processing
        databus = self._broker.get_communication_channel()
        databus.register(self)

        # - start account processing
        self.account.start()

        # - update universe with initial instruments after the strategy is initialized
        self.set_universe(self._initial_instruments, skip_callback=True)

        # - initialize strategy (should we do that after any first market data received ?)
        if not self._is_initialized:
            try:
                self.strategy.on_start(self)
                self._is_initialized = True
            except Exception as strat_error:
                logger.error(
                    f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_start: {strat_error}"
                )
                logger.error(traceback.format_exc())
                return

        # - for live we run loop
        if not self._broker.is_simulated_trading:
            self._thread_data_loop = Thread(target=self.__process_incoming_data_loop, args=(databus,), daemon=True)
            self._thread_data_loop.start()
            logger.info("(StrategyContext) strategy is started in thread")
            if blocking:
                self._thread_data_loop.join()

    def stop(self):
        if self._thread_data_loop:
            self._broker.close()
            self._broker.get_communication_channel().stop()
            self._thread_data_loop.join()
            try:
                self.strategy.on_stop(self)
            except Exception as strat_error:
                logger.error(
                    f"(StrategyContext) Strategy {self.strategy.__class__.__name__} raised an exception in on_stop: {strat_error}"
                )
                logger.opt(colors=False).error(traceback.format_exc())
            self._thread_data_loop = None

        # - stop account processing
        self.account.stop()

        # - close logging
        self._logging.close()

    def is_running(self):
        return self._thread_data_loop is not None and self._thread_data_loop.is_alive()

    @property
    def is_simulation(self) -> bool:
        return self._broker.is_simulated_trading

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
    def ohlc(self, instrument: Instrument, timeframe: str | None = None, length: int | None = None):
        return self._market_data_provider.ohlc(instrument, timeframe, length)

    def quote(self, instrument: Instrument):
        return self._market_data_provider.quote(instrument)

    def get_data(self, instrument: Instrument, sub_type: str) -> List[Any]:
        return self._market_data_provider.get_data(instrument, sub_type)

    def get_aux_data(self, data_id: str, **parameters):
        return self._market_data_provider.get_aux_data(data_id, **parameters)

    def get_instruments(self):
        return self._market_data_provider.get_instruments()

    def get_instrument(self, symbol: str, exchange: str):
        return self._market_data_provider.get_instrument(symbol, exchange)

    # ITradingManager delegation
    def trade(self, instrument: Instrument, amount: float, price: float | None = None, time_in_force="gtc", **options):
        return self._trading_manager.trade(instrument, amount, price, time_in_force, **options)

    def cancel(self, instrument: Instrument):
        return self._trading_manager.cancel(instrument)

    def cancel_order(self, order_id: str):
        return self._trading_manager.cancel_order(order_id)

    # IUniverseManager delegation
    def set_universe(self, instruments: list[Instrument], skip_callback: bool = False):
        return self._universe_manager.set_universe(instruments, skip_callback)

    @property
    def instruments(self):
        return self._universe_manager.instruments

    # ISubscriptionManager delegation
    def subscribe(self, subscription_type: str, instruments: List[Instrument] | Instrument | None = None):
        return self._subscription_manager.subscribe(subscription_type, instruments)

    def unsubscribe(self, subscription_type: str, instruments: List[Instrument] | Instrument | None = None):
        return self._subscription_manager.unsubscribe(subscription_type, instruments)

    def has_subscription(self, instrument: Instrument, subscription_type: str):
        return self._subscription_manager.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        return self._subscription_manager.get_subscriptions(instrument)

    def get_base_subscription(self) -> str:
        return self._subscription_manager.get_base_subscription()

    def set_base_subscription(self, subscription_type: str):
        return self._subscription_manager.set_base_subscription(subscription_type)

    def get_warmup(self, subscription_type: str) -> str:
        return self._subscription_manager.get_warmup(subscription_type)

    def set_warmup(self, configs: dict[Any, str]):
        return self._subscription_manager.set_warmup(configs)

    def commit(self):
        return self._subscription_manager.commit()

    @property
    def auto_subscribe(self) -> bool:
        return self._subscription_manager.auto_subscribe

    @auto_subscribe.setter
    def auto_subscribe(self, value: bool):
        self._subscription_manager.auto_subscribe = value

    # IProcessingManager delegation
    def process_data(self, instrument: Instrument, d_type: str, data: Any):
        return self._processing_manager.process_data(instrument, d_type, data)

    def set_fit_schedule(self, schedule: str):
        return self._processing_manager.set_fit_schedule(schedule)

    def set_event_schedule(self, schedule: str):
        return self._processing_manager.set_event_schedule(schedule)

    def is_fitted(self) -> bool:
        return self._processing_manager.is_fitted()

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
