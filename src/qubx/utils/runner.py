import asyncio
from pathlib import Path
from dotenv import load_dotenv, find_dotenv, dotenv_values

import configparser
import socket
import sys
import time
from os.path import exists, expanduser

import click
import pandas as pd
import yaml

from qubx import formatter, logger, lookup
from qubx.backtester.simulator import SimulatedTrading
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.connector import CcxtBrokerServiceProvider
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.connectors.ccxt.trading import CcxtTradingConnector
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import Instrument
from qubx.core.context import StrategyContext
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.core.loggers import InMemoryLogsWriter, LogsWriter, StrategyLogging
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


def run_ccxt_paper_trading(
    strategy: IStrategy,
    exchange: str,
    symbols: list[str | Instrument],
    strategy_config: dict | None = None,
    blocking: bool = True,
    account_id: str = "main",
    base_currency: str = "USDT",
    capital: float = 100_000,
    commissions: str | None = None,
    use_testnet: bool = False,
) -> IStrategyContext:
    # TODO: setup proper loggers to write out to files
    instruments = symbols if isinstance(symbols[0], Instrument) else _get_instruments(symbols, exchange)
    instruments = [i for i in instruments if i is not None]

    logs_writer = InMemoryLogsWriter("test", "test", "0")

    _exchange = get_ccxt_exchange(exchange, use_testnet=use_testnet)

    trading_service = SimulatedTrading(
        account_processor=CcxtAccountProcessor(account_id, _exchange, base_currency),
        exchange_name=exchange,
        commissions=commissions,
        simulation_initial_time=pd.Timestamp.now().asm8,
    )

    broker = CcxtBrokerServiceProvider(_exchange, trading_service)

    account = BasicAccountProcessor(
        account_id=trading_service.get_account_id(),
        base_currency=base_currency,
        initial_capital=capital,
    )

    ctx = StrategyContext(
        strategy=strategy,
        broker=broker,
        account=account,
        instruments=instruments,
        logging=StrategyLogging(logs_writer, heartbeat_freq="1m"),
        config=strategy_config,
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


def run_ccxt_trading(
    strategy: IStrategy,
    exchange: str,
    symbols: list[str | Instrument],
    credentials: dict,
    strategy_config: dict | None = None,
    blocking: bool = True,
    account_id: str = "main",
    base_currency: str = "USDT",
    commissions: str | None = None,
    use_testnet: bool = False,
    loop: asyncio.AbstractEventLoop | None = None,
) -> StrategyContext:
    # TODO: setup proper loggers to write out to files
    instruments = symbols if isinstance(symbols[0], Instrument) else _get_instruments(symbols, exchange)

    logs_writer = InMemoryLogsWriter("test", "test", "0")
    stg_logging = StrategyLogging(logs_writer, heartbeat_freq="1m")

    _exchange = get_ccxt_exchange(exchange, use_testnet=use_testnet, loop=loop, **credentials)
    account = CcxtAccountProcessor(account_id, _exchange, base_currency)
    trading_service = CcxtTradingConnector(_exchange, account, commissions)
    broker = CcxtBrokerServiceProvider(_exchange, trading_service)

    ctx = StrategyContext(
        strategy=strategy,
        broker=broker,
        account=account,
        instruments=instruments,
        logging=stg_logging,
        config=strategy_config,
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


def load_strategy_config(filename: str) -> Struct:
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
        account=config.get("account"),
        # md_subscr=config["subscription"], # xxx
        strategy_trigger=config["parameters"]["trigger_at"], # ???
        strategy_fit_trigger=config["parameters"].get("fit_at", ""), # ???
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
    if not env_f:
        logger.error(f"Can't find {env_file} file for reading {account_id} account info")
        return None
    env_data = dotenv_values(env_f)
    account_data = {}
    for name, value in env_data.items():
        if name.upper().startswith(account_id.upper()):
            account_data[name.split("__")[-1]] = value
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


def get_strategy(config_file: str, search_paths: list) -> (IStrategy, Struct):
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
            broker = CcxtTradingConnector(cfg.exchange.lower(), **acc_config)
            exchange_connector = CcxtBrokerServiceProvider(cfg.exchange.lower(), broker, **acc_config)
        case _:
            raise ValueError(f"Connector {conn} is not supported yet !")

    # - generate new run id
    run_id = socket.gethostname() + "-" + str(broker.time().item() // 100_000_000)

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
@click.option("--accounts", "-a", default=".env", type=click.STRING, help=".env file with live accounts configuration data")
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
def run(filename: str, accounts: str, paths: list, jupyter: bool, testnet: bool, paper: bool):
    paths = list(paths)
    if jupyter:
        _run_in_jupyter(filename, accounts, paths)
        return

    # - show Qubx logo with current version
    logo()

    strategy, cfg = get_strategy(filename, paths)
    if not all([strategy, cfg]):
        logger.error("Can't load strategy")
        return

    logger.add(LOGFILE + cfg.name + "_{time}.log", format=formatter, rotation="100 MB", colorize=False)

    # - read account creds
    acc_config = {}
    if cfg.account is not None:
        acc_config = get_account_env_config(cfg.account, accounts)
        if acc_config is None:
            logger.error("Can't read account configuration")
            return None

    # - check connector
    conn = cfg.connector.lower()
    match conn:
        case "ccxt":
            if not paper:
                run_ccxt_trading(strategy, cfg.exchange, cfg.instruments, acc_config, cfg.parameters, use_testnet=testnet)
            else:
                run_ccxt_paper_trading(strategy, cfg.exchange, cfg.instruments, cfg.parameters, use_testnet=testnet)
        case _:
            raise ValueError(f"Connector {conn} is not supported yet !")


if __name__ == "__main__":
    run()
