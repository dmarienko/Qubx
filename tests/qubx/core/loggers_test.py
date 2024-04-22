from typing import Any, Dict, List, Union
from collections import defaultdict

import pandas as pd

from qubx import lookup

import time
import tabulate
from qubx import lookup
from qubx.core.basics import Position, ZERO_COSTS
from qubx.core.loggers import PositionsDumper, LogsWriter


class ConsolePositionsWriter(LogsWriter):
    """
    Simple positions - just writes current positions to the standard output as funcy table
    """
    def _dump_positions(self, account_id: str, strategy_id: str, data: List[Dict[str, Any]]):
        table = defaultdict(list)
        total_pnl, total_rpnl, total_mkv = 0, 0, 0
        
        for r in data:
            table['Symbol'].append(r['instrument_id'])
            table['Time'].append(r['timestamp'])
            table['Quantity'].append(r['quantity'])
            table['AvgPrice'].append(r['avg_position_price'])
            table['LastPrice'].append(r['current_price'])
            table['PnL'].append(r['pnl_quoted'])
            table['RealPnL'].append(r['realized_pnl_quoted'])
            table['MarketValue'].append(r['market_value_quoted'])
            total_pnl += r['pnl_quoted']
            total_rpnl += r['realized_pnl_quoted']
            total_mkv += r['market_value_quoted']

        table['Symbol'].append('TOTAL')
        table['PnL'].append(total_pnl)
        table['RealPnL'].append(total_rpnl)
        table['MarketValue'].append(total_mkv)

        # - write to database table here
        print(f" ::: Strategy {strategy_id} @ {account_id} :::")
        print(tabulate.tabulate(table, [
            'Symbol', 'Time', 'Quantity', 'AvgPrice', 'LastPrice', 'PnL', 'RealPnL', 'MarketValue'
        ], tablefmt='rounded_grid'))

    def write_data(self, log_type: str, account_id: str, strategy_id: str, data: List[Dict[str, Any]]):
        match log_type:

            case 'positions':
                self._dump_positions(account_id, strategy_id, data)

            case 'portfolio':
                pass

            case 'executions':
                pass


class TestPortfolioLoggers:

    def test_positions_dumper(self):
        # - initialize positions: this will be done in StrategyContext
        positions = [
            Position(lookup.find_symbol('BINANCE', s), ZERO_COSTS) for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] # type: ignore
        ] 
        positions[0].change_position_by(0, 0.05, 63000)
        positions[1].change_position_by(0, 0.5, 3200)
        positions[2].change_position_by(0, 10, 56)

        # - create dumper and attach positions
        writer = ConsolePositionsWriter()
        console_dumper = PositionsDumper(
            'Account1', 
            'Strategy1', 
            '1Sec',            # dumps positions once per 1 sec
            writer
        ).attach_positions(*positions)

        # - emulating updates from strategy (this will be done in StategyContext)
        for _ in range(30):
            t = pd.Timestamp('now').asm8
            for p in positions:
                # - selling 10% of position every tick 
                p.change_position_by(t, -p.quantity*0.1, p.last_update_price + 10)
            # - this method will be called inside the platform ! 
            console_dumper.store(t)
            time.sleep(0.25)


