import qubx
from qubx.core.basics import Instrument
%qubxd

import pandas as pd
import nest_asyncio;
nest_asyncio.apply()

from qubx.core.context import StrategyContext
from qubx.utils.misc import dequotify, quotify
from qubx.utils.runner import create_strategy_context
from qubx.pandaz.utils import *
import qubx.pandaz.ta as pta
import qubx.ta.indicators as ta

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
ctx: StrategyContext = create_strategy_context('{filename}', 'test', '{accounts}', {paths}) # type: ignore
assert ctx is not None, 'Strategy context is not created'
ctx.start()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def orders(instrument: Instrument | None=None):
    return ctx.get_orders(instrument)

def trade(instrument: Instrument, qty: float, price=None, tif='gtc'):
    return ctx.trade(instrument, qty, price, tif)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def pnl_report(all=True):
    from tabulate import tabulate

    d = dict()
    for s, p in ctx.get_positions().items():
        mv = round(p.market_value_funds, 3)
        if mv != 0.0 or all:
            d[dequotify(s.symbol)] = dict(
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
    ctx.stop()
    __exit()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -