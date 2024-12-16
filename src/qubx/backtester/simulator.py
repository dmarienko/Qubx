from typing import Literal

import numpy as np
import pandas as pd
from joblib import delayed

from qubx import QubxLogConfig, logger, lookup
from qubx.core.basics import DataType, TradingSessionResult
from qubx.core.context import StrategyContext
from qubx.core.exceptions import SimulationError
from qubx.core.helpers import extract_parameters_from_object, full_qualified_class_name
from qubx.core.interfaces import IStrategy
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.data.helpers import InMemoryCachedReader, TimeGuardedWrapper
from qubx.data.readers import DataReader
from qubx.utils.misc import ProgressParallel

from .account import SimulatedAccountProcessor
from .broker import SimulatedBroker
from .data import SimulatedDataProvider
from .utils import (
    DataDecls_t,
    ExchangeName_t,
    SetupTypes,
    SignalsProxy,
    SimulatedCtrlChannel,
    SimulatedLogFormatter,
    SimulatedScheduler,
    SimulatedTimeProvider,
    SimulationSetup,
    StrategiesDecls_t,
    SymbolOrInstrument_t,
    find_instruments_and_exchanges,
    recognize_simulation_configuration,
    recognize_simulation_data_config,
)


def simulate(
    strategies: StrategiesDecls_t,
    data: DataDecls_t,
    capital: float,
    instruments: list[SymbolOrInstrument_t] | dict[ExchangeName_t, list[SymbolOrInstrument_t]],
    commissions: str,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp | None = None,
    exchange: ExchangeName_t | None = None,
    base_currency: str = "USDT",
    leverage: float = 1.0,  # TODO: we need to add support for leverage
    n_jobs: int = 1,
    silent: bool = False,
    enable_event_batching: bool = True,
    aux_data: DataReader | None = None,
    accurate_stop_orders_execution: bool = False,
    signal_timeframe: str = "1Min",
    open_close_time_indent_secs=1,
    debug: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = "WARNING",
) -> list[TradingSessionResult]:
    """
    Backtest utility for trading strategies or signals using historical data.

    Args:
        strategies (StrategiesDecls_t): Trading strategy or signals configuration.
        data (DataDecls_t): Historical data for simulation, either as a dictionary of DataFrames or a DataReader object.
        capital (float): Initial capital for the simulation.
        instruments (list[SymbolOrInstrument_t] | dict[ExchangeName_t, list[SymbolOrInstrument_t]]): List of trading instruments or a dictionary mapping exchanges to instrument lists.
        commissions (str): Commission structure for trades.
        start (str | pd.Timestamp): Start time of the simulation.
        stop (str | pd.Timestamp | None): End time of the simulation. If None, simulates until the last accessible data.
        exchange (ExchangeName_t | None): Exchange name if not specified in the instruments list.
        base_currency (str): Base currency for the simulation, default is "USDT".
        leverage (float): Leverage factor for trading, default is 1.0.
        n_jobs (int): Number of parallel jobs for simulation, default is 1.
        silent (bool): If True, suppresses output during simulation.
        enable_event_batching (bool): If True, enables event batching for optimization.
        aux_data (DataReader | None): Auxiliary data provider (default is None).
        accurate_stop_orders_execution (bool): If True, enables more accurate stop order execution simulation.
        signal_timeframe (str): Timeframe for signals, default is "1Min".
        open_close_time_indent_secs (int): Time indent in seconds for open/close times, default is 1.
        debug (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None): Logging level for debugging.

    Returns:
        list[TradingSessionResult]: A list of TradingSessionResult objects containing the results of each simulation setup.
    """

    # - setup logging
    QubxLogConfig.set_log_level(debug.upper() if debug else "WARNING")

    # - process instruments:
    _instruments, _exchanges = find_instruments_and_exchanges(instruments, exchange)

    # - check we have exchange
    if not _exchanges:
        logger.error(
            _msg
            := "No exchange information provided - you can specify it by exchange parameter or use <yellow>EXCHANGE:SYMBOL</yellow> format for symbols"
        )
        raise SimulationError(_msg)

    # - check if instruments are from the same exchange (mmulti-exchanges is not supported yet)
    if len(_exchanges) > 1:
        logger.error(
            _msg := f"Multiple exchanges found: {', '.join(_exchanges)} - this mode is not supported yet in Qubx !"
        )
        raise SimulationError(_msg)

    exchange = _exchanges[0]

    # - recognize provided data
    _schedule, _base_subscription, _typed_readers = recognize_simulation_data_config(data, _instruments, exchange)

    # - recognize setup: it can be either a strategy or set of signals
    setups = recognize_simulation_configuration(
        "", strategies, _instruments, exchange, capital, leverage, base_currency, commissions
    )
    if not setups:
        logger.error(
            _msg
            := "Can't recognize setup - it should be a strategy, a set of signals or list of signals/strategies + tracker !"
        )
        raise SimulationError(_msg)

    # - check stop time : here we try to backtest till now (may be we need to get max available time from data reader ?)
    if stop is None:
        stop = pd.Timestamp.now(tz="UTC").astimezone(None)

    # - run simulations
    return _run_setups(
        setups,
        start,
        stop,
        _typed_readers,
        _schedule,
        _base_subscription,
        n_jobs=n_jobs,
        silent=silent,
        enable_event_batching=enable_event_batching,
        accurate_stop_orders_execution=accurate_stop_orders_execution,
        aux_data=aux_data,
        signal_timeframe=signal_timeframe,
        open_close_time_indent_secs=open_close_time_indent_secs,
    )


def _run_setups(
    setups: list[SimulationSetup],
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    typed_data_config: dict[str, DataReader],
    default_schedule: str,
    default_base_subscription: str,
    n_jobs: int = -1,
    silent: bool = False,
    enable_event_batching: bool = True,
    accurate_stop_orders_execution: bool = False,
    aux_data: DataReader | None = None,
    open_close_time_indent_secs=1,
    **kwargs,
) -> list[TradingSessionResult]:
    # loggers don't work well with joblib and multiprocessing in general because they contain
    # open file handlers that cannot be pickled. I found a solution which requires the usage of enqueue=True
    # in the logger configuration and specifying backtest "multiprocessing" instead of the default "loky"
    # for joblib. But it works now.
    # See: https://stackoverflow.com/questions/59433146/multiprocessing-logging-how-to-use-loguru-with-joblib-parallel
    _main_loop_silent = len(setups) == 1
    n_jobs = 1 if _main_loop_silent else n_jobs

    reports = ProgressParallel(n_jobs=n_jobs, total=len(setups), silent=_main_loop_silent, backend="multiprocessing")(
        delayed(_run_setup)(
            id,
            setup,
            start,
            stop,
            typed_data_config,
            default_schedule,
            default_base_subscription,
            silent=silent,
            enable_event_batching=enable_event_batching,
            accurate_stop_orders_execution=accurate_stop_orders_execution,
            aux_data_provider=aux_data,
            open_close_time_indent_secs=open_close_time_indent_secs,
            **kwargs,
        )
        for id, setup in enumerate(setups)
    )
    return reports  # type: ignore


def _run_setup(
    setup_id: int,
    setup: SimulationSetup,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    typed_data_config: dict[str, DataReader],
    default_schedule: str,
    default_base_subscription: str,
    silent: bool = False,
    enable_event_batching: bool = True,
    accurate_stop_orders_execution: bool = False,
    aux_data_provider: InMemoryCachedReader | None = None,
    signal_timeframe: str = "1Min",
    open_close_time_indent_secs=1,
    account_id: str = "Simulated0",
) -> TradingSessionResult:
    _stop = pd.Timestamp(stop)
    logger.debug(
        f"<red>{pd.Timestamp(start)}</red> Initiating simulated trading for {setup.exchange} for {setup.capital} x {setup.leverage} in {setup.base_currency}..."
    )

    channel = SimulatedCtrlChannel("databus", sentinel=(None, None, None))
    tcc = lookup.fees.find(setup.exchange.lower(), setup.commissions)
    assert tcc is not None, f"Can't find transaction costs calculator for {setup.exchange} with {setup.commissions} !"

    time_provider = SimulatedTimeProvider(np.datetime64(start, "ns"))

    # - we want to see simulate time in log messages
    QubxLogConfig.setup_logger(QubxLogConfig.get_log_level(), SimulatedLogFormatter(time_provider).formatter)

    account = SimulatedAccountProcessor(
        account_id=account_id,
        channel=channel,
        base_currency=setup.base_currency,
        initial_capital=setup.capital,
        time_provider=time_provider,
        tcc=tcc,
        accurate_stop_orders_execution=accurate_stop_orders_execution,
    )
    scheduler = SimulatedScheduler(channel, lambda: time_provider.time().item())
    broker = SimulatedBroker(channel, account)
    data_provider = SimulatedDataProvider(
        exchange_id=setup.exchange,
        channel=channel,
        scheduler=scheduler,
        time_provider=time_provider,
        account=account,
        readers=typed_data_config,
        open_close_time_indent_secs=open_close_time_indent_secs,
    )

    # - it will store simulation results into memory
    logs_writer = InMemoryLogsWriter("test", setup.name, "0")
    strat: IStrategy | None = None

    match setup.setup_type:
        case SetupTypes.STRATEGY:
            strat = setup.generator  # type: ignore

        case SetupTypes.STRATEGY_AND_TRACKER:
            strat = setup.generator  # type: ignore
            strat.tracker = lambda ctx: setup.tracker  # type: ignore

        case SetupTypes.SIGNAL:
            strat = SignalsProxy(timeframe=signal_timeframe)
            data_provider.set_generated_signals(setup.generator)  # type: ignore
            # - we don't need any unexpected triggerings
            _stop = min(setup.generator.index[-1], _stop)  # type: ignore

            # - no historical data for generated signals, so disable it
            enable_event_batching = False

        case SetupTypes.SIGNAL_AND_TRACKER:
            strat = SignalsProxy(timeframe=signal_timeframe)
            strat.tracker = lambda ctx: setup.tracker
            data_provider.set_generated_signals(setup.generator)  # type: ignore
            # - we don't need any unexpected triggerings
            _stop = min(setup.generator.index[-1], _stop)  # type: ignore

            # - no historical data for generated signals, so disable it
            enable_event_batching = False

        case _:
            raise SimulationError(f"Unsupported setup type: {setup.setup_type} !")

    # - check aux data provider
    _aux_data = None
    if aux_data_provider is not None:
        if not isinstance(aux_data_provider, InMemoryCachedReader):
            logger.error("Aux data provider should be an instance of InMemoryCachedReader! Skipping it.")
        _aux_data = TimeGuardedWrapper(aux_data_provider, time_guard=time_provider)

    assert isinstance(strat, IStrategy), f"Strategy should be an instance of IStrategy, but got {strat} !"

    ctx = StrategyContext(
        strategy=strat,
        broker=broker,
        data_provider=data_provider,
        account=account,
        scheduler=scheduler,
        time_provider=time_provider,
        instruments=setup.instruments,
        logging=StrategyLogging(logs_writer),
        aux_data_provider=_aux_data,
    )

    # - setup base subscription from spec
    if ctx.get_base_subscription() == DataType.NONE:
        logger.debug(f" | Setting up default base subscription: {default_base_subscription}")
        ctx.set_base_subscription(default_base_subscription)

    # - set default on_event schedule if detected and strategy didn't set it's own schedule
    if not ctx.get_event_schedule("time") and default_schedule:
        logger.debug(f" | Setting default schedule: {default_schedule}")
        ctx.set_event_schedule(default_schedule)

    # - start context at this point
    ctx.start()

    def _is_known_type(t: str):
        try:
            DataType(t)
            return True
        except:  # noqa: E722
            return False

    # - if any custom data providers are in the data spec
    for t, r in typed_data_config.items():
        if not _is_known_type(t) or t in [DataType.TRADE, DataType.OHLC_TRADES, DataType.OHLC_QUOTES, DataType.QUOTE]:
            logger.debug(f" | Subscribing to: {t}")
            ctx.subscribe(t, ctx.instruments)

    try:
        data_provider.run(start, _stop, silent=silent, enable_event_batching=enable_event_batching)  # type: ignore
    except KeyboardInterrupt:
        logger.error("Simulated trading interrupted by user !")

    # - get strategy parameters for this run
    _s_class, _s_params = "", None
    if setup.setup_type in [SetupTypes.STRATEGY, SetupTypes.STRATEGY_AND_TRACKER]:
        _s_params = extract_parameters_from_object(setup.generator)
        _s_class = full_qualified_class_name(setup.generator)

    return TradingSessionResult(
        setup_id,
        setup.name,
        start,
        stop,
        setup.exchange,
        setup.instruments,
        setup.capital,
        setup.leverage,
        setup.base_currency,
        setup.commissions,
        logs_writer.get_portfolio(as_plain_dataframe=True),
        logs_writer.get_executions(),
        logs_writer.get_signals(),
        strategy_class=_s_class,
        parameters=_s_params,
        is_simulation=True,
    )
