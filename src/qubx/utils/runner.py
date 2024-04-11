import sys, yaml, sys
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


def get_account_auth(acc_name: str, accounts_cfg_file: str):
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

    return dict(parser[acc_name])


def main_runner(config_file: str, accounts_cfg_file='accounts.cfg', *search_paths):
    cfg = load_strategy_config(config_file)
    try:
        for p in search_paths:
            add_project_to_system_path(p)
        strategy = class_import(cfg.strategy)
    except Exception as err:
        logger.error(str(err))
        return

    logger.add(LOGFILE + cfg.name + "_{time}.log", format=formatter, rotation="100 MB", colorize=False)

    # - read account creds
    acc_auth = {}
    if cfg.account is not None:
        acc_auth = get_account_auth(cfg.account, accounts_cfg_file)
        if acc_auth is None:
            return

    # - check connector
    conn = cfg.connector.lower()
    match conn:
        case 'ccxt':
            connector = CCXTConnector(cfg.exchange.lower(), **acc_auth)
        case _:
            raise ValueError(f"Connector {conn} is not supported yet !")
        
    logger.info(f""" - - - <blue>Qubx</blue> (ver. <red>{version()}</red>) - - -\n - Strategy: {strategy}\n - Config: {cfg.parameters} """)

    ctx = StrategyContext(
        strategy, cfg.parameters, connector, connector, 
        instruments=cfg.instruments, md_subscription=cfg.md_subscr, trigger=cfg.strategy_trigger)
    # ctx.start(join=True)
    ctx.start()
    return ctx

# @click.command()
# @click.argument('filename', type=click.Path(exists=True))

# def run(filename: str):
    # main_runner(filename)

# import asyncio

# async def observe(ctx):
#     while True:
#         for p in ctx.positions.values():
#             print('\t' + str(p))
#         print('- ' * 40)
#         try:
#             await asyncio.sleep(10)
#             print('\033[H')
#         except KeyboardInterrupt:
#             break
#         except Exception:
#             break


