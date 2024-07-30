import click, sys, yaml, sys, time
from os.path import exists, expanduser
import yaml, configparser, socket

from qubx import lookup, logger, formatter
from qubx.core.context import StrategyContextImpl
from qubx.impl.ccxt_connector import CCXTExchangesConnector
from qubx.impl.ccxt_trading import CCXTTradingConnector
from qubx.core.strategy import StrategyContext
from qubx.utils.misc import add_project_to_system_path, Struct, logo, version
from qubx.core.loggers import LogsWriter


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
        exchange=config["exchange"],
        account=config.get("account"),
        md_subscr=config["subscription"],
        strategy_trigger=config["trigger"],
        strategy_fit_trigger=config.get("fit", ""),
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


def create_strategy_context(config_file: str, accounts_cfg_file: str, search_paths: list) -> StrategyContext | None:
    cfg = load_strategy_config(config_file)
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
            broker = CCXTTradingConnector(cfg.exchange.lower(), **acc_config)
            exchange_connector = CCXTExchangesConnector(cfg.exchange.lower(), broker, **acc_config)
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
@click.option("--accounts", "-a", default="accounts.cfg", type=click.STRING, help="Live accounts configuration file")
@click.option(
    "--paths",
    "-p",
    multiple=True,
    default=["../", "~/projects/"],
    type=click.STRING,
    help="Live accounts configuration file",
)
@click.option("--jupyter", "-j", is_flag=True, default=False, help="Run strategy in jupyter console", show_default=True)
def run(filename: str, accounts: str, paths: list, jupyter: bool):
    if jupyter:
        _run_in_jupyter(filename, accounts, paths)
        return

    # - show Qubx logo with current version
    logo()

    # - create context
    ctx = create_strategy_context(filename, accounts, paths)
    if ctx is None:
        return

    # - run main loop
    try:
        ctx.start()

        # - just wake up every 60 sec and check if it's OK
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        ctx.stop()
        time.sleep(1)
        sys.exit(0)


if __name__ == "__main__":
    run()
