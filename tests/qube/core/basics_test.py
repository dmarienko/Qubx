import pandas as pd
from dataclasses import dataclass
from typing import List, Union
from qube import lookup
from qube.core.basics import Instrument, Position, TransactionCostsCalculator, ZERO_COSTS
from qube.core.series import time_as_nsec, Trade, Quote


@dataclass
class Deal:
    time: int
    position: int
    exec_price: float
    def __init__(self, time, pos, price):
        self.time = time_as_nsec(time)
        self.position = pos
        self.exec_price = price

def run_deals_updates(p: Position, qs: List[Union[Deal, Trade, Quote]]) -> pd.Series:
    pnls = {}
    for q in qs:
        if isinstance(q, Deal): 
            pnls[pd.Timestamp(q.time, unit='ns')] = p.update_position(q.time, q.position, q.exec_price)
            print(p, '\t<-(exec)-')
        else: 
            pnls[pd.Timestamp(q.time, unit='ns')] = p.update_market_price(q)
            print(p)
    return pd.Series(pnls)


class TestBasics:

    def test_lookup(self):
        s0 = lookup['BINANCE:ETH.*']
        s1 = lookup['DUKAS:EURGBP']
        assert (
            lookup.find_aux_instrument_for(s0[0], 'USDT').symbol, 
            lookup.find_aux_instrument_for(s0[1], 'USDT'), 
            lookup.find_aux_instrument_for(s1[0], 'USD').symbol,
        ) == ('BTCUSDT', None, 'GBPUSD')

    def test_spot_positions(self):
        tcc = TransactionCostsCalculator(0.04/100, 0.04/100)
        i, s = lookup['BINANCE:BTCUSDT'][0], 1
        # i, s = lookup['BINANCE.UM:BTCUSDT'][0], 100
        # i, s = lookup['BINANCE.CM:BTCUSD_PERP'][0], 1
        D = '2024-01-01 '; qs = [
            Quote(D+'12:00:00', 45000, 45000.5, 100, 50),
            Deal( D+'12:00:30', s, 45010),
            Trade(D+'12:01:00', 45010, 10, 1),
            Trade(D+'12:02:00', 45015, 10, 1),
            Deal( D+'12:02:30', -s, 45015),
            Quote(D+'12:03:00', 45020, 45021, 0, 0),
            Deal( D+'12:03:30', -2*s, 45020),
            Quote(D+'12:04:00', 45120, 45121, 0, 0),
            Quote(D+'12:05:00', 45014, 45014, 0, 0),
            Deal( D+'12:06:30', 0, 45010),
            Quote(D+'12:10:00', 45020, 45020, 0, 0),
            Deal( D+'12:11:00', -1, 45020),
            Quote(D+'12:12:00', 45030, 45030, 0, 0),
            Deal( D+'12:13:00', 0, 45100),
        ]

        p = Position(i, tcc)
        pnls = run_deals_updates(p, qs)
        print(pnls)
        assert p.commissions == (1*45010+2*45015+1*45020+2*45010+45020+45100)*0.04/100
        assert p.pnl == -60

