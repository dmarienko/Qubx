import click, sys, yaml, sys, time
from os.path import exists, expanduser
import yaml, configparser

from qubx import lookup, logger, formatter
from qubx.impl.ccxt_connector import CCXTConnector # TODO: need factory !
from qubx.core.strategy import StrategyContext
from qubx.utils.misc import add_project_to_system_path, Struct, version


LOGFILE = 'logs/'


def class_import(name):
    components = name.split('.')
    clz = components[-1]
    mod = __import__('.'.join(components[:-1]), fromlist=[clz])
    mod = getattr(mod, clz)
    return mod


def load_strategy_config(filename: str) -> Struct:
    with open(filename, 'r') as f:
        content = yaml.safe_load(f)

    config = content['config']
    execution = config['execution']
    strat = config['strategy']
    name = strat.split('.')[-1]
    r = Struct(
        strategy = strat,
        name = name,
        parameters = config.get('parameters', dict()),
        connector = execution['connector'],
        exchange = execution['exchange'],
        account = execution.get('account'),
        md_subscr = execution['subscription'],
        strategy_trigger = execution['trigger'],
    )

    universe = execution['universe']
    r.instruments = [lookup.find_symbol(r.exchange.upper(), s.upper()) for s in universe ]
    return r


def get_account_config(acc_name: str, accounts_cfg_file: str) -> dict | None:
    parser = configparser.ConfigParser()
    try:
        parser.optionxform=str  # type: ignore
        parser.read(accounts_cfg_file)
    except Exception as exc:
        logger.error(f"Can't find { accounts_cfg_file } file for reading {acc_name} account info: {str(exc)}")
        return None

    if acc_name not in parser:
        logger.error(f"No records for {acc_name} found in {accounts_cfg_file} file")
        return None

    cfg = dict(parser[acc_name])
    reserves = {}

    if 'reserves' in cfg: 
        rs = cfg['reserves']
        for r in rs.split(','):
            s,v = r.strip().split(':')
            reserves[s] = float(v)

    return cfg | {'reserves': reserves}


def create_strategy_context(config_file: str, accounts_cfg_file: str, search_paths: list) -> StrategyContext | None:
    cfg = load_strategy_config(config_file)
    try:
        for p in search_paths:
            if exists(pe:=expanduser(p)):
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
        case 'ccxt':
            connector = CCXTConnector(cfg.exchange.lower(), **acc_config)
        case _:
            raise ValueError(f"Connector {conn} is not supported yet !")
        
    logger.info(f""" - - - <blue>Qubx</blue> (ver. <red>{version()}</red>) - - -\n - Strategy: {strategy}\n - Config: {cfg.parameters} """)
    ctx = StrategyContext(
        strategy, cfg.parameters, connector, connector, 
        instruments=cfg.instruments, md_subscription=cfg.md_subscr, trigger=cfg.strategy_trigger
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
            self.init_code = kwargs.pop('init_code')
            super().__init__(**kwargs)
        def init_banner(self):
            pass
        def initialize(self, argv=None):
            super().initialize(argv=[])
            self.shell.run_cell(self.init_code)

    logger.info("Running in Jupyter console")
    TerminalRunner.launch_instance(init_code=f"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import qubx
%qubxd
import pandas as pd
import nest_asyncio; nest_asyncio.apply()
from qubx.utils.misc import dequotify, quotify
from qubx.utils.runner import create_strategy_context
from qubx.utils.pandas import *

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
""")


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('--accounts', '-a', default='accounts.cfg', type=click.STRING, help='Live accounts configuration file')
@click.option('--paths', '-p', multiple=True, default=['../', '~/projects/'], type=click.STRING, help='Live accounts configuration file')
@click.option('--jupyter', '-j', is_flag=True, default=False, help='Run strategy in jupyter console', show_default=True)
def run(filename: str, accounts: str, paths: list, jupyter: bool):
    if jupyter:
        _run_in_jupyter(filename, accounts, paths)
        return
    
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

