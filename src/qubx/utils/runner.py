import asyncio
import configparser
import os
import socket
import sys
import time
import uuid
from os.path import exists, expanduser
from pathlib import Path
from typing import Literal

import click
import pandas as pd
import yaml
from dotenv import dotenv_values, find_dotenv, load_dotenv

from qubx import formatter, logger, lookup
from qubx.backtester.account import SimulatedAccountProcessor
from qubx.backtester.simulator import SimulatedBroker
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.core.basics import CtrlChannel, Instrument, LiveTimeProvider
from qubx.core.context import StrategyContext
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IStrategy
from qubx.core.loggers import InMemoryLogsWriter, LogsWriter, StrategyLogging, CsvFileLogsWriter
from qubx.data.helpers import __KNOWN_READERS
from qubx.utils.marketdata.ccxt import ccxt_build_qubx_exchange_name
from qubx.utils.misc import Struct, add_project_to_system_path, logo, version

LOGFILE = "logs/"


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
    log: str = Literal["InMemoryLogsWriter", "CsvFileLogsWriter"],
    loop: asyncio.AbstractEventLoop | None = None,
) -> StrategyContext:
    strategy_id = strategy_id or uuid.uuid4().hex[:8]  # todo: not sure, but if take from tags cannot distinguish between different runs
    logger.info(f"Running {'paper' if paper else 'live'} strategy on {exchange_name} exchange ({strategy_id=})...")
    credentials = credentials if not paper else {}
    assert paper or credentials, "Credentials are required for live trading"

    # TODO: setup proper loggers to write out to files
    instruments: list[Instrument] = (  # type: ignore
        symbols if isinstance(symbols[0], Instrument) else _get_instruments(symbols, exchange_name)
    )
    instruments = [i for i in instruments if i is not None]

    logs_writer = globals().get(log, InMemoryLogsWriter)(account_id=account_id, strategy_id=strategy_id, run_id="0")
    logger.debug(f"Setup <g>{logs_writer.__class__.__name__}</g> logger...")
    stg_logging = StrategyLogging(logs_writer, heartbeat_freq="1m")

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
    logger.debug(f"Setup <g>{aux_reader.__class__.__name__}</g> reader...") if aux_reader is not None else None

    channel = CtrlChannel("databus", sentinel=(None, None, None, None))
    time_provider = LiveTimeProvider()
    scheduler = BasicScheduler(channel, lambda: time_provider.time().item())
    exchange = get_ccxt_exchange(exchange_name, use_testnet=use_testnet, loop=loop, **(credentials or {}))
    if exchange.apiKey:
        logger.info(f"Connected {exchange_name} exchange with {exchange.apiKey[:2]}...{exchange.apiKey[-2:]} API key")

    # - find proper fees calculator
    qubx_exchange_name = ccxt_build_qubx_exchange_name(exchange_name)
    fees_calculator = lookup.fees.find(qubx_exchange_name.lower(), commissions)
    assert fees_calculator is not None, f"Can't find fees calculator for {qubx_exchange_name} exchange"

    if paper:
        account = SimulatedAccountProcessor(
            account_id=account_id,
            channel=channel,
            base_currency=base_currency,
            time_provider=time_provider,
            tcc=fees_calculator,
            initial_capital=paper_capital,
        )
        broker = SimulatedBroker(channel=channel, account=account)
        logger.debug("Setup paper account...")
    else:
        account = CcxtAccountProcessor(account_id, exchange, channel, time_provider, base_currency, tcc=fees_calculator)
        broker = CcxtBroker(exchange, channel, time_provider, account)
        logger.debug(f"Setup live {'testnet ' if use_testnet else ''}account...")

    data_provider = CcxtDataProvider(exchange, time_provider, channel)

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


def load_strategy_config(filename: str, account: str) -> Struct:
    with open(filename, "r") as f:
        content = yaml.safe_load(f)

    config = content["config"]
    strat = config["strategy"]
    name = strat.split(".")[-1]
    r = Struct(
        strategy=strat,
        name=name,
        parameters=config.get("parameters", dict()),
        connector=config["connector"],
        exchange=config["parameters"]["exchange"],
        account=account,
        # md_subscr=config["subscription"], # todo: ask where to get?
        strategy_trigger=config["parameters"]["trigger_at"],
        strategy_fit_trigger=config["parameters"].get("fit_at", ""),
        aux=config.get("aux", None),
        portfolio_logger=config.get("logger", None),
        log_positions_interval=config.get("log_positions_interval", None),
        log_portfolio_interval=config.get("log_portfolio_interval", None),
    )

    universe = config["universe"]
    if isinstance(universe, dict):
        r.instruments = []
        for e, symbs in universe.items():
            r.instruments.extend(_instruments_for_exchange(e, symbs))
    else:
        r.instruments = _instruments_for_exchange(r.exchange, universe)

    return r


def get_account_env_config(account_id: str, env_file: str) -> dict | None:
    env_f = find_dotenv(env_file) or find_dotenv(Path(env_file).name)
    env_data = dotenv_values(env_f)
    env_data.update(os.environ)
    account_data = {}
    for name, value in env_data.items():
        if name.upper().startswith(f"{account_id.upper()}__"):
            account_data[name.split("__")[-1]] = value
    if not account_data:
        logger.error(f"No records for {account_id} found in env")
        # return None
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


def get_strategy(config_file: str, search_paths: list, account: str) -> (IStrategy, Struct):
    cfg = load_strategy_config(config_file, account)
    search_paths.append(Path(config_file).parent)
    try:
        for p in search_paths:
            if exists(pe := expanduser(p)):
                add_project_to_system_path(pe)
        strategy = class_import(cfg.strategy)
    except Exception as err:
        logger.error(str(err))
        return None

    return strategy, cfg


def create_strategy_context(config_file: str, accounts_cfg_file: str, search_paths: list) -> StrategyContext | None:
    cfg = load_strategy_config(config_file)
    search_paths.append(Path(config_file).parent)
    try:
        for p in search_paths:
            if exists(pe := expanduser(p)):
                add_project_to_system_path(pe)
        strategy = class_import(cfg.strategy)
    except Exception as err:
        logger.error(str(err))
        return None

    logger.add(LOGFILE + cfg.name + "_{time}.log", format=formatter, rotation="100 MB", colorize=False)

    # - read account creds
    acc_config = {}
    if cfg.account is not None:
        acc_config = get_account_config(cfg.account, accounts_cfg_file)
        if acc_config is None:
            return None

    # - check connector
    conn = cfg.connector.lower()
    match conn:
        case "ccxt":
            # - TODO: we need some factory here
            broker = CcxtBroker(cfg.exchange.lower(), **acc_config)
            exchange_connector = CcxtDataProvider(cfg.exchange.lower(), broker, **acc_config)
        case _:
            raise ValueError(f"Connector {conn} is not supported yet !")

    # - generate new run id
    run_id = socket.gethostname() + "-" + str(broker.time_provider.time().item() // 100_000_000)

    # - get logger
    writer = None
    _w_class = cfg.portfolio_logger
    if _w_class is not None:
        if "." not in _w_class:
            _w_class = "qubx.core.loggers." + _w_class
        try:
            w_class = class_import(_w_class)
            writer = w_class(acc_config["account_id"], strategy.__name__, run_id)
        except Exception as err:
            logger.warning(f"Can't instantiate specified writer {_w_class}: {str(err)}")
            writer = LogsWriter(acc_config["account_id"], strategy.__name__, run_id)

    logger.info(
        f""" - - - <blue>Qubx</blue> (ver. <red>{version()}</red>) - - -\n - Strategy: {strategy}\n - Config: {cfg.parameters} """
    )
    ctx = StrategyContextImpl(
        strategy,
        cfg.parameters,
        exchange_connector,
        instruments=cfg.instruments,
        md_subscription=cfg.md_subscr,
        trigger_spec=cfg.strategy_trigger,
        fit_spec=cfg.strategy_fit_trigger,
        logs_writer=writer,
        positions_log_freq=cfg.log_positions_interval,
        portfolio_log_freq=cfg.log_portfolio_interval,
    )

    return ctx


def _run_in_jupyter(filename: str, accounts: str, paths: list):
    """
    Helper for run this in jupyter console
    """
    try:
        from jupyter_console.app import ZMQTerminalIPythonApp
    except ImportError:
        logger.error("Can't find <red>ZMQTerminalIPythonApp</red> module - try to install jupyter package first")
        return
    try:
        import nest_asyncio
    except ImportError:
        logger.error("Can't find <red>nest_asyncio</red> module - try to install it first")
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

    logger.info("Running in Jupyter console")
    TerminalRunner.launch_instance(
        init_code=f"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import qubx
%qubxd
import pandas as pd
import nest_asyncio; nest_asyncio.apply()
from qubx.utils.misc import dequotify, quotify
from qubx.utils.runner import create_strategy_context
from qubx.pandaz.utils import *
import qubx.pandaz.ta as pta

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
ctx = create_strategy_context('{filename}', '{accounts}', {paths})
ctx.start()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def orders(symbol=None):
    return ctx.exchange_service.get_orders(symbol)

def trade(symbol, qty, price=None, tif='gtc'):
    return ctx.trade(symbol, qty, price, tif)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def pnl_report(all=True):
    from tabulate import tabulate
    d = dict()
    for s, p in ctx.positions.items():
        mv = round(p.market_value_funds, 3)
        if mv != 0.0 or all:
            d[dequotify(s)] = dict(
                Position=round(p.quantity, p.instrument.size_precision),  
                PnL=p.total_pnl(), 
                AvgPrice=round(p.position_avg_price_funds, p.instrument.price_precision), 
                LastPrice=round(p.last_update_price, p.instrument.price_precision),
                MktValue=mv
            )
    d = pd.DataFrame.from_dict(d).T
    # d = d[d['PnL'] != 0.0]
    if d.empty:
        print('-(no open positions yet)-')
        return
    d = d.sort_values('PnL' ,ascending=False)
    # d = pd.concat((d, pd.Series(dict(TOTAL=d['PnL'].sum()), name='PnL'))).fillna('')
    d = pd.concat((d, scols(pd.Series(dict(TOTAL=d['PnL'].sum()), name='PnL'), pd.Series(dict(TOTAL=d['MktValue'].sum()), name='MktValue')))).fillna('')
    print(tabulate(d, ['Position', 'PnL', 'AvgPrice', 'LastPrice', 'MktValue'], tablefmt='rounded_grid'))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__exit = exit
def exit():
    ctx.stop(); __exit()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
    )


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--account", "-a", type=click.STRING, help="Account id for trading", default=None, show_default=True)
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
    if not account and not paper:
        logger.error("Account id is required for live trading")
        return
    paths = list(paths)

    if jupyter:
        _run_in_jupyter(filename, acc_file, paths)
        return

    # - show Qubx logo with current version
    logo()

    strategy, cfg = get_strategy(filename, paths, account)
    if not all([strategy, cfg]):
        logger.error("Can't load strategy")
        return

    logger.add(LOGFILE + cfg.name + "_{time}.log", format=formatter, rotation="100 MB", colorize=False)

    # - read account creds
    acc_config = {}
    if cfg.account is not None:
        acc_config = get_account_env_config(account, acc_file)
        if acc_config is None:
            logger.error("Can't read account configuration")
            return None

    # - check connector
    conn = cfg.connector.lower()
    match conn:
        case "ccxt":
            run_ccxt_trading(
                strategy=strategy,
                exchange_name=cfg.exchange,
                symbols=cfg.instruments,
                credentials=acc_config,
                strategy_config=cfg.parameters,
                account_id=account,
                use_testnet=testnet,
                paper=paper,
                aux_config=cfg.aux,
                log=cfg.portfolio_logger,
            )
        case _:
            raise ValueError(f"Connector {conn} is not supported yet !")


if __name__ == "__main__":
    run()
