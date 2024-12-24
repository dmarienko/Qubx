import asyncio
import configparser
import inspect
import os
import socket
import time
import uuid
from os.path import exists, expanduser
from pathlib import Path
from typing import Tuple
import ccxt.pro as cxp

import click
import yaml
from dotenv import dotenv_values, find_dotenv, load_dotenv
from pydantic import BaseModel

from qubx import formatter, logger, lookup
from qubx.backtester.account import SimulatedAccountProcessor
from qubx.backtester.simulator import SimulatedBroker
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.core.basics import CtrlChannel, Instrument, LiveTimeProvider, TransactionCostsCalculator
from qubx.core.context import StrategyContext
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IStrategy
from qubx.core.loggers import CsvFileLogsWriter, InMemoryLogsWriter, LogsWriter, StrategyLogging
from qubx.data import DataReader
from qubx.data.helpers import __KNOWN_READERS
from qubx.utils.marketdata.ccxt import ccxt_build_qubx_exchange_name
from qubx.utils.misc import Struct, add_project_to_system_path, logo, version

LOGPATH = "logs"


class ExchangeConfig(BaseModel):
    name: str
    connector: str
    universe: list[str]
    instruments: list[Instrument] | None = None


class StrategyConfig(BaseModel):
    strategy: str
    name: str
    exchanges: list[ExchangeConfig]
    account: str
    parameters: dict | None = None
    aux: dict | None = None
    logger: str | None = None
    log_positions_interval: str | None = None
    log_portfolio_interval: str | None = None


def class_import(name):
    components = name.split(".")
    clz = components[-1]
    mod = __import__(".".join(components[:-1]), fromlist=[clz])
    mod = getattr(mod, clz)
    return mod


def _instruments_for_exchange(exch: str, symbols: list) -> list:
    instrs = []
    for s in symbols:
        instr = lookup.find_symbol(exch.upper(), s.upper())
        if instr is not None:
            instrs.append(instr)
        else:
            logger.warning(f"Can't find instrument for symbol {s} - try to refresh lookup first !")
    return instrs


def run_ccxt_trading(
    strategy: IStrategy,
    exchange_name: str,
    symbols: list[str],
    credentials: dict | None = None,
    strategy_config: dict | None = None,
    blocking: bool = True,
    account_id: str = "main",
    strategy_id: str | None = None,
    base_currency: str = "USDT",
    commissions: str | None = None,
    use_testnet: bool = False,
    paper: bool = False,
    paper_capital: float = 100_000,
    aux_config: dict | None = None,
    log: str = "InMemoryLogsWriter",
    log_folder: str = "logs",
    loop: asyncio.AbstractEventLoop | None = None,
) -> StrategyContext:
    strategy_id = (
        strategy_id or uuid.uuid4().hex[:8]
    )  # todo: not sure, but if take from tags cannot distinguish between different runs
    logger.info(f"Running {'paper' if paper else 'live'} strategy on {exchange_name} exchange ({strategy_id=})...")
    credentials = credentials if not paper else {}
    assert paper or credentials, "Credentials are required for live trading"

    instruments: list[Instrument] = (  # type: ignore
        symbols if isinstance(symbols[0], Instrument) else _get_instruments(symbols, exchange_name)
    )
    instruments = [i for i in instruments if i is not None]

    aux_reader = get_aux_reader(aux_config)
    logger.debug(f"Setup <g>{aux_reader.__class__.__name__}</g> reader...") if aux_reader is not None else None

    channel = CtrlChannel("databus", sentinel=(None, None, None, None))
    time_provider = LiveTimeProvider()
    scheduler = BasicScheduler(channel, lambda: time_provider.time().item())

    # - find proper fees calculator
    qubx_exchange_name = ccxt_build_qubx_exchange_name(exchange_name)
    fees_calculator = lookup.fees.find(qubx_exchange_name.lower(), commissions)
    assert fees_calculator is not None, f"Can't find fees calculator for {qubx_exchange_name} exchange"

    exchange = get_ccxt_exchange(exchange_name, use_testnet=use_testnet, loop=loop, **(credentials or {}))
    if exchange.apiKey:
        logger.info(f"Connected {exchange_name} exchange with {exchange.apiKey[:2]}...{exchange.apiKey[-2:]} API key")
    account, broker, data_provider = initialize_ccxt_components(
        account_id,
        base_currency,
        channel,
        exchange,
        fees_calculator,
        paper,
        paper_capital,
        time_provider,
        use_testnet
    )

    # - get logger
    run_id = (
        socket.gethostname()
        + "-"
        + str(int(time.time() * 10**9) if paper else broker.time_provider.time().item() // 100_000_000)
    )
    stg_logging = get_logger(log, account_id, run_id, strategy, log_folder)

    ctx = StrategyContext(
        strategy=strategy,
        broker=broker,
        data_provider=data_provider,
        account=account,
        scheduler=scheduler,
        time_provider=time_provider,
        instruments=instruments,
        logging=stg_logging,
        config=strategy_config,
        aux_data_provider=aux_reader,
    )

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


def initialize_ccxt_components(
        account_id: str,
        base_currency: str,
        channel: CtrlChannel,
        exchange: cxp.Exchange,
        fees_calculator: TransactionCostsCalculator,
        paper: bool,
        paper_capital: float,
        time_provider: LiveTimeProvider,
        use_testnet: bool = False
):
    if paper:
        account = SimulatedAccountProcessor(
            account_id=account_id,
            channel=channel,
            base_currency=base_currency,
            time_provider=time_provider,
            tcc=fees_calculator,
            initial_capital=paper_capital,
        )
        broker = SimulatedBroker(channel=channel, account=account, exchange_id=exchange.name.upper())
        logger.debug("Setup paper account...")
    else:
        account = CcxtAccountProcessor(account_id, exchange, channel, time_provider, base_currency, tcc=fees_calculator)
        broker = CcxtBroker(exchange, channel, time_provider, account)
        logger.debug(f"Setup live {'testnet ' if use_testnet else ''}account...")
    data_provider = CcxtDataProvider(exchange, time_provider, channel)
    return account, broker, data_provider


def get_aux_reader(aux_config: dict) -> DataReader | None:
    aux_reader = None
    if "::" in aux_config.get("reader", ""):
        # like: mqdb::nebula or csv::/data/rawdata/
        db_conn, path = aux_config["reader"].split("::")
        kwargs = aux_config.get("args", {})
        reader = __KNOWN_READERS.get(db_conn, **kwargs)
        if reader is None:
            logger.error(f"Unknown reader {db_conn} - try to use {__KNOWN_READERS.keys()} only")
        else:
            aux_reader = reader(path)
    elif aux_config.get("reader") is not None:
        # like: sty.data.readers.MyCustomDataReader
        kwargs = aux_config.get("args", {})
        aux_reader = class_import(aux_config["reader"])(**kwargs)
    return aux_reader


def _get_instruments(symbols: list[str], exchange: str) -> list[Instrument]:
    exchange_symbols = []
    for symbol in symbols:
        if ":" in symbol:
            exchange, symbol = symbol.split(":")
            exchange_symbols.append((exchange, symbol))
        else:
            exchange_symbols.append((exchange, symbol))

    instruments = [lookup.find_symbol(_exchange.upper(), _symbol.upper()) for _exchange, _symbol in exchange_symbols]
    instruments = [i for i in instruments if i is not None]
    return instruments


def load_strategy_config(filename: str, account: str) -> StrategyConfig:
    try:
        with open(filename, "r") as f:
            content = yaml.safe_load(f)
    except Exception as exc:
        logger.error(f"Can't read strategy config from {filename} file: {str(exc)}")
        raise Exception(f"Can't read strategy config from {filename} file: {str(exc)}") from exc

    config_raw = content["config"]
    strat = config_raw["strategy"]
    name = strat.split(".")[-1]
    config_raw["name"] = name
    excs = []
    for exc, exc_data in config_raw["exchanges"].items():
        exc_data["name"] = exc
        exc_data["instruments"] = _instruments_for_exchange(exc, exc_data["universe"])
        exc_data["instruments"] = [i for i in exc_data["instruments"] if i is not None]
        excs.append(ExchangeConfig.model_validate(exc_data))
    config_raw["exchanges"] = excs
    config_raw["account"] = account

    str_config = StrategyConfig.model_validate(config_raw)
    return str_config


def load_account_env_config(account_id: str, env_file: str) -> dict | None:
    env_f = find_dotenv(env_file) or find_dotenv(Path(env_file).name)
    env_data = dotenv_values(env_f)
    env_data.update(os.environ)
    account_data = {}
    for name, value in env_data.items():
        if name.upper().startswith(f"{account_id.upper()}__"):
            account_data[name.split("__")[-1].lower()] = value
    if "exchange" not in [k.lower() for k in account_data.keys()]:
        logger.error(f"Can't find exchange for {account_id} account")
        return None
    account_data["account_id"] = account_id
    return account_data


def get_account_config(account_id: str, accounts_cfg_file: str) -> dict | None:
    parser = configparser.ConfigParser()
    try:
        parser.optionxform = str  # type: ignore
        parser.read(accounts_cfg_file)
    except Exception as exc:
        logger.error(f"Can't find { accounts_cfg_file } file for reading {account_id} account info: {str(exc)}")
        return None

    if account_id not in parser:
        logger.error(f"No records for {account_id} found in {accounts_cfg_file} file")
        return None

    cfg = dict(parser[account_id])

    # - check if there any reserved funds
    reserves = {}
    if "reserves" in cfg:
        rs = cfg["reserves"]
        for r in rs.split(","):
            s, v = r.strip().split(":")
            reserves[s] = float(v)

    # - add account id and reserves
    return cfg | {"account_id": account_id, "reserves": reserves}


def get_strategy_configs(config_file: str, search_paths: list, account: str) -> Tuple[IStrategy, StrategyConfig]:
    cfg = load_strategy_config(config_file, account)
    search_paths = list(search_paths)
    search_paths.append(Path(config_file).parent)
    try:
        for p in search_paths:
            if exists(pe := expanduser(p)):
                add_project_to_system_path(pe)
        strategy = class_import(cfg.strategy)
    except Exception as err:
        logger.error(f"Can't load strategy {cfg.strategy} from {config_file} file: {str(err)}")
        raise err

    return strategy, cfg


def create_strategy_context(
        strategy: IStrategy,
        acc_config: dict,
        cfg: StrategyConfig,
        strategy_id: str | None = None,
        paper: bool = False,
        testnet: bool = False,
        base_currency: str = "USDT",
        paper_capital: float = 100_000,
        commissions: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
) -> StrategyContext | None:
    exch_cfg = next((e for e in cfg.exchanges if e.name == acc_config["exchange"]), None)
    if exch_cfg is None:
        logger.error(f"Can't find exchange configuration for {cfg.exchange.name}")
        return
    strategy_config, symbols, exchange_name, aux_config = cfg.parameters, exch_cfg.instruments, exch_cfg.name, cfg.aux

    log_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    log_folder = f"{LOGPATH.removesuffix('/')}/run_{log_id}"
    logger.add(f"{log_folder}/strategy/{cfg.name}_" + "{time}.log", format=formatter, rotation="100 MB", colorize=False)

    strategy_id = strategy_id or uuid.uuid4().hex[:8]  # todo: not sure, but if take from tags cannot distinguish between different runs
    logger.info(f"Running {'paper' if paper else 'live'} strategy on {exchange_name} exchange ({strategy_id=})...")
    creds = {k: v for k, v in acc_config.items() if k.upper() in ["APIKEY", "SECRET"]}
    assert paper or creds, "Credentials are required for live trading"

    aux_reader = get_aux_reader(aux_config)
    logger.debug(f"Setup <g>{aux_reader.__class__.__name__}</g> reader...") if aux_reader is not None else None

    channel = CtrlChannel("databus", sentinel=(None, None, None, None))
    time_provider = LiveTimeProvider()
    scheduler = BasicScheduler(channel, lambda: time_provider.time().item())

    # - find proper fees calculator
    qubx_exchange_name = ccxt_build_qubx_exchange_name(exchange_name)
    fees_calculator = lookup.fees.find(qubx_exchange_name.lower(), commissions)
    assert fees_calculator is not None, f"Can't find fees calculator for {qubx_exchange_name} exchange"

    # - check connector
    match exch_cfg.connector.lower():
        case "ccxt":
            # - TODO: we need some factory here
            exchange = get_ccxt_exchange(exchange_name, use_testnet=testnet, loop=loop, **(creds))
            if exchange.apiKey:
                logger.info(
                    f"Connected {exchange_name} exchange with {exchange.apiKey[:2]}...{exchange.apiKey[-2:]} API key")
            account, broker, data_provider = initialize_ccxt_components(
                acc_config["account_id"],
                base_currency,
                channel,
                exchange,
                fees_calculator,
                paper,
                paper_capital,
                time_provider,
                testnet,
            )
        case _:
            raise ValueError(f"Connector {exch_cfg.connector} is not supported yet !")

    # - generate new run id
    run_id = (
        socket.gethostname()
        + "-"
        + str(int(time.time() * 10**9) if paper else broker.time_provider.time().item() // 100_000_000)
    )

    # - get logger
    _w_class = cfg.logger
    stg_logging = get_logger(_w_class, acc_config["account_id"], run_id, strategy, log_folder)

    logger.info(
        f""" - - - <blue>Qubx</blue> (ver. <red>{version()}</red>) - - -\n - Strategy: {strategy}\n - Config: {cfg.parameters} """
    )
    ctx = StrategyContext(
        strategy=strategy,
        broker=broker,
        data_provider=data_provider,
        account=account,
        scheduler=scheduler,
        time_provider=time_provider,
        instruments=exch_cfg.instruments,
        logging=stg_logging,
        config=strategy_config,
        aux_data_provider=aux_reader,
    )

    return ctx


def get_logger(logger_name: str, account_id: str, run_id: str, strategy: IStrategy, log_folder: str = "logs") -> StrategyLogging:
    if logger_name is not None:
        _w_class = logger_name if "." in logger_name else "qubx.core.loggers." + logger_name
        try:
            w_class = class_import(_w_class)
            logs_kwargs = {"account_id": account_id, "strategy_id": strategy.__class__.__name__, "run_id": run_id}
            if "log_folder" in inspect.signature(w_class).parameters:
                logs_kwargs["log_folder"] = log_folder
            writer = w_class(**logs_kwargs)
            logger.debug(f"Setup <g>{writer.__class__.__name__}</g> logger...")
        except Exception as err:
            logger.warning(f"Can't instantiate specified writer {logger_name}: {str(err)}")
            writer = LogsWriter(account_id, strategy.__class__.__name__, run_id)
    else:
        writer = InMemoryLogsWriter(account_id, strategy.__class__.__name__, run_id)
    stg_logging = StrategyLogging(writer, heartbeat_freq="1m")
    return stg_logging


def _run_in_jupyter(filename: str, accounts: str, paths: list):
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

    content_with_values = content.format_map(
        {
            "filename": filename,
            "accounts": accounts,
            "paths": str(paths),
        }
    )
    logger.info("Running in Jupyter console")
    TerminalRunner.launch_instance(init_code=content_with_values)


@click.group()
def main():
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True), required=True)
@click.option(
    "--account",
    "-a",
    type=click.STRING,
    help="Account id for trading",
    default=None,
    show_default=True,
    required=True,
)
@click.option(
    "--acc_file",
    "-f",
    default=".env",
    type=click.STRING,
    help="env file with live accounts configuration data",
    show_default=True,
)
@click.option(
    "--paths",
    "-p",
    multiple=True,
    default=["../", "~/projects/"],
    type=click.STRING,
    help="Live accounts configuration file",
)
@click.option("--jupyter", "-j", is_flag=True, default=False, help="Run strategy in jupyter console", show_default=True)
@click.option("--testnet", "-t", is_flag=True, default=False, help="Use testnet for trading", show_default=True)
@click.option("--paper", "-p", is_flag=True, default=False, help="Use paper trading mode", show_default=True)
def run(filename: str, account: str, acc_file: str, paths: list, jupyter: bool, testnet: bool, paper: bool):
    """
    Run strategy in live trading mode with specified configuration.
    """
    if not account and not paper:
        logger.error("Account id is required for live trading")
        return
    paths = list(paths)

    if jupyter:
        _run_in_jupyter(filename, acc_file, paths)
        return

    # - show Qubx logo with current version
    logo()

    strategy, cfg = get_strategy_configs(filename, paths, account)

    # - read account creds
    acc_config = load_account_env_config(account, acc_file)
    if acc_config is None:
        logger.error("Can't read account configuration")
        return None

    log_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    log_folder = f"{LOGPATH.removesuffix('/')}/run_{log_id}"
    logger.add(f"{log_folder}/strategy/{cfg.name}_" + "{time}.log", format=formatter, rotation="100 MB", colorize=False)

    ctx = create_strategy_context(
        strategy=strategy,
        acc_config=acc_config,
        cfg=cfg,
        strategy_id=None,
        paper=paper,
        testnet=testnet,
        base_currency="USDT",
        paper_capital=100_000,
        commissions=acc_config.get("commissions"),
        loop=None,
    )

    try:
        ctx.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        ctx.stop()


if __name__ == "__main__":
    main()
