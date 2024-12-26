import inspect
import socket
import time
from pathlib import Path

from qubx import formatter, logger, lookup
from qubx.backtester.account import SimulatedAccountProcessor
from qubx.backtester.simulator import SimulatedBroker
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.core.basics import CtrlChannel, Instrument, ITimeProvider, LiveTimeProvider, TransactionCostsCalculator
from qubx.core.context import StrategyContext
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider, IStrategyContext
from qubx.core.loggers import StrategyLogging
from qubx.data import DataReader
from qubx.data.helpers import __KNOWN_READERS
from qubx.utils.misc import class_import
from qubx.utils.runner.configs import ExchangeConfig, load_strategy_config_from_yaml

from .accounts import AccountConfigurationManager
from .configs import AuxConfig, LoggingConfig, StrategyConfig


def run_strategy_yaml(
    config_file: Path,
    account_file: Path | None = None,
    paper: bool = False,
    blocking: bool = False,
) -> IStrategyContext:
    """
    Run the strategy with the given configuration file.

    Args:
        config_file (Path): The path to the configuration file.
        account_file (Path, optional): The path to the account configuration file. Defaults to None.
        paper (bool, optional): Whether to run in paper trading mode. Defaults to False.
        jupyter (bool, optional): Whether to run in a Jupyter console. Defaults to False.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    if account_file is not None and not account_file.exists():
        raise FileNotFoundError(f"Account configuration file not found: {account_file}")

    acc_manager = AccountConfigurationManager(account_file, config_file.parent, search_qubx_dir=True)
    stg_config = load_strategy_config_from_yaml(config_file)
    return run_strategy(stg_config, acc_manager, paper=paper, blocking=blocking)


def run_strategy_yaml_in_jupyter(config_file: Path, account_file: Path | None = None, paper: bool = False) -> None:
    """
    Helper for run this in jupyter console
    """
    try:
        from jupyter_console.app import ZMQTerminalIPythonApp
    except ImportError:
        logger.error(
            "Can't find <r>ZMQTerminalIPythonApp</r> module - try to install <g>jupyter-console</g> package first"
        )
        return
    try:
        import nest_asyncio
    except ImportError:
        logger.error("Can't find <r>nest_asyncio</r> module - try to install it first")
        return

    class TerminalRunner(ZMQTerminalIPythonApp):
        def __init__(self, **kwargs) -> None:
            self.init_code = kwargs.pop("init_code")
            super().__init__(**kwargs)

        def init_banner(self):
            pass

        def initialize(self, argv=None):
            super().initialize(argv=[])
            self.shell.run_cell(self.init_code)

    _base = Path(__file__).parent.absolute()
    with open(_base / "_jupyter_runner.pyt", "r") as f:
        content = f.read()

    content_with_values = content.format_map({"config_file": config_file, "account_file": account_file, "paper": paper})
    logger.info("Running in Jupyter console")
    TerminalRunner.launch_instance(init_code=content_with_values)


def run_strategy(
    config: StrategyConfig,
    account_manager: AccountConfigurationManager,
    paper: bool = False,
    blocking: bool = False,
) -> IStrategyContext:
    """
    Run the strategy with the given configuration.

    Args:
        config (StrategyConfig): The configuration of the strategy.
        account_manager (AccountManager): The account manager to use.
        paper (bool, optional): Whether to run in paper trading mode. Defaults to False.
        jupyter (bool, optional): Whether to run in a Jupyter console. Defaults to False.
    """
    ctx = create_strategy_context(config, account_manager, paper)
    if blocking:
        try:
            ctx.start(blocking=True)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            ctx.stop()
    else:
        ctx.start()

    return ctx


def create_strategy_context(
    config: StrategyConfig,
    account_manager: AccountConfigurationManager,
    paper: bool = False,
) -> IStrategyContext:
    """
    Create a strategy context from the given configuration.
    """
    stg_name = _get_strategy_name(config)
    _run_mode = "paper" if paper else "live"

    _strategy_class = class_import(config.strategy)

    _logging = _setup_strategy_logging(stg_name, config.logging)
    _aux_reader = _get_aux_reader(config.aux)

    _time = LiveTimeProvider()
    _chan = CtrlChannel("databus", sentinel=(None, None, None, None))
    _sched = BasicScheduler(_chan, lambda: _time.time().item())

    exchanges = list(config.exchanges.keys())
    if len(exchanges) > 1:
        raise ValueError("Multiple exchanges are not supported yet !")

    _exchange_to_tcc = {}
    _exchange_to_broker = {}
    _exchange_to_data_provider = {}
    _exchange_to_account = {}
    _instruments = []
    for exchange_name, exchange_config in config.exchanges.items():
        _exchange_to_tcc[exchange_name] = (tcc := _create_tcc(exchange_name, account_manager))
        _exchange_to_data_provider[exchange_name] = _create_data_provider(
            exchange_name,
            exchange_config,
            time_provider=_time,
            channel=_chan,
            account_manager=account_manager,
        )
        _exchange_to_account[exchange_name] = (
            account := _create_account_processor(
                exchange_name,
                exchange_config,
                channel=_chan,
                time_provider=_time,
                account_manager=account_manager,
                tcc=tcc,
                paper=paper,
            )
        )
        _exchange_to_broker[exchange_name] = _create_broker(
            exchange_name,
            exchange_config,
            _chan,
            time_provider=_time,
            account=account,
            account_manager=account_manager,
            paper=paper,
        )
        _instruments.extend(_create_instruments_for_exchange(exchange_name, exchange_config))

    # TODO: rework strategy context to support multiple exchanges
    _broker = _exchange_to_broker[exchanges[0]]
    _data_provider = _exchange_to_data_provider[exchanges[0]]
    _account = _exchange_to_account[exchanges[0]]

    logger.info(f"- Strategy: <blue>{stg_name}</blue>\n- Mode: {_run_mode}\n- Parameters: {config.parameters}")
    ctx = StrategyContext(
        strategy=_strategy_class,
        broker=_broker,
        data_provider=_data_provider,
        account=_account,
        scheduler=_sched,
        time_provider=_time,
        instruments=_instruments,
        logging=_logging,
        config=config.parameters,
        aux_data_provider=_aux_reader,
    )

    return ctx


def _get_strategy_name(cfg: StrategyConfig) -> str:
    return cfg.strategy.split(".")[-1]


def _setup_strategy_logging(stg_name: str, log_config: LoggingConfig) -> StrategyLogging:
    log_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    run_folder = f"logs/run_{log_id}"
    logger.add(f"{run_folder}/strategy/{stg_name}_{{time}}.log", format=formatter, rotation="100 MB", colorize=False)

    run_id = f"{socket.gethostname()}-{str(int(time.time() * 10**9))}"

    _log_writer_name = log_config.logger
    if "." not in _log_writer_name:
        _log_writer_name = f"qubx.core.loggers.{_log_writer_name}"

    logger.debug(f"Setup <g>{_log_writer_name}</g> logger...")
    _log_writer_class = class_import(_log_writer_name)
    _log_writer_params = {
        "account_id": "account",
        "strategy_id": stg_name,
        "run_id": run_id,
        "log_folder": run_folder,
    }
    _log_writer_sig_params = inspect.signature(_log_writer_class).parameters
    _log_writer_params = {k: v for k, v in _log_writer_params.items() if k in _log_writer_sig_params}
    _log_writer = _log_writer_class(**_log_writer_params)
    stg_logging = StrategyLogging(_log_writer, heartbeat_freq=log_config.heartbeat_interval)
    return stg_logging


def _get_aux_reader(aux_config: AuxConfig | None) -> DataReader | None:
    if aux_config is None:
        return None
    _reader_name = aux_config.reader
    _is_uri = "::" in _reader_name
    if _is_uri:
        # like: mqdb::nebula or csv::/data/rawdata/
        db_conn, db_name = _reader_name.split("::")
        return __KNOWN_READERS[db_conn](db_name, **aux_config.args)
    else:
        # like: sty.data.readers.MyCustomDataReader
        return class_import(_reader_name)(**aux_config.args)


def _create_tcc(exchange_name: str, account_manager: AccountConfigurationManager) -> TransactionCostsCalculator:
    settings = account_manager.get_exchange_settings(exchange_name)
    tcc = lookup.fees.find(exchange_name, settings.commissions)
    assert tcc is not None, f"Can't find fees calculator for {exchange_name} exchange"
    return tcc


def _create_data_provider(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    time_provider: ITimeProvider,
    channel: CtrlChannel,
    account_manager: AccountConfigurationManager,
) -> IDataProvider:
    settings = account_manager.get_exchange_settings(exchange_name)
    match exchange_config.connector.lower():
        case "ccxt":
            exchange = get_ccxt_exchange(exchange_name, use_testnet=settings.testnet)
            return CcxtDataProvider(exchange, time_provider, channel)
        case _:
            raise ValueError(f"Connector {exchange_config.connector} is not supported yet !")


def _create_account_processor(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account_manager: AccountConfigurationManager,
    tcc: TransactionCostsCalculator,
    paper: bool,
) -> IAccountProcessor:
    if paper:
        settings = account_manager.get_exchange_settings(exchange_name)
        return SimulatedAccountProcessor(
            account_id=exchange_name,
            channel=channel,
            base_currency=settings.base_currency,
            time_provider=time_provider,
            tcc=tcc,
            initial_capital=settings.initial_capital,
        )

    creds = account_manager.get_exchange_credentials(exchange_name)
    match exchange_config.connector.lower():
        case "ccxt":
            exchange = get_ccxt_exchange(
                exchange_name, use_testnet=creds.testnet, api_key=creds.api_key, secret=creds.secret
            )
            return CcxtAccountProcessor(
                exchange_name,
                exchange,
                channel,
                time_provider,
                base_currency=creds.base_currency,
                tcc=tcc,
            )
        case _:
            raise ValueError(f"Connector {exchange_config.connector} is not supported yet !")


def _create_broker(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    account_manager: AccountConfigurationManager,
    paper: bool,
) -> IBroker:
    if paper:
        assert isinstance(account, SimulatedAccountProcessor)
        return SimulatedBroker(channel=channel, account=account, exchange_id=exchange_name)

    creds = account_manager.get_exchange_credentials(exchange_name)

    match exchange_config.connector.lower():
        case "ccxt":
            exchange = get_ccxt_exchange(
                exchange_name, use_testnet=creds.testnet, api_key=creds.api_key, secret=creds.secret
            )
            return CcxtBroker(exchange, channel, time_provider, account)
        case _:
            raise ValueError(f"Connector {exchange_config.connector} is not supported yet !")


def _create_instruments_for_exchange(exchange_name: str, exchange_config: ExchangeConfig) -> list[Instrument]:
    exchange_name = exchange_name.upper()
    if exchange_name == "BINANCE.PM":
        # TODO: clean this up
        exchange_name = "BINANCE.UM"
    symbols = exchange_config.universe
    instruments = [lookup.find_symbol(exchange_name, symbol.upper()) for symbol in symbols]
    instruments = [i for i in instruments if i is not None]
    return instruments
