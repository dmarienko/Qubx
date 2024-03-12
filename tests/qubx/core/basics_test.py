import pandas as pd
from dataclasses import dataclass
from typing import List, Union
from qubx import lookup

from tests.qubx.ta.utils_for_testing import N
from qubx.core.basics import Instrument, Position, TransactionCostsCalculator, ZERO_COSTS
from qubx.core.series import time_as_nsec, Trade, Quote


@dataclass
class Deal:
    time: int
    position: int
    exec_price: float
    aggr: bool
    def __init__(self, time, pos, price, agressive=True):
        self.time = time_as_nsec(time)
        self.position = pos
        self.exec_price = price
        self.aggr = agressive

def run_deals_updates(p: Position, qs: List[Union[Deal, Trade, Quote]]) -> dict:
    pnls = {}
    for q in qs:
        if isinstance(q, Deal): 
            pnls[pd.Timestamp(q.time, unit='ns')] = p.update_position(q.time, q.position, q.exec_price, aggressive=q.aggr)
            print(p, f'\t<-(exec -> {q.position})-')
        else: 
            pnls[pd.Timestamp(q.time, unit='ns')] = p.update_market_price(q); print(p)
    return pd.Series(pnls)


pos_round = lambda s, p, i: (p * round(s/p, i.size_precision), p, round(s/p, i.size_precision))


class TestBasics:

    def test_lookup(self):
        s0 = lookup.instruments['BINANCE:ETH.*']
        s1 = lookup.instruments['DUKAS:EURGBP']
        assert (
            lookup.find_aux_instrument_for(s0[0], 'USDT').symbol, 
            lookup.find_aux_instrument_for(s0[1], 'USDT'), 
            lookup.find_aux_instrument_for(s1[0], 'USD').symbol,
        ) == ('BTCUSDT', None, 'GBPUSD')

    def test_spot_positions(self):
        tcc = TransactionCostsCalculator('SPOT', 0.04, 0.04)
        i, s = lookup.instruments['BINANCE:BTCUSDT'][0], 1
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

    def test_futures_positions(self):
        D = '2024-01-01 '
        fi = lookup.instruments['BINANCE.UM:BTCUSDT'][0]
        pos = Position(fi, TransactionCostsCalculator('UM', 0.02, 0.05))
        q1 = pos_round(239.9, 47980, fi)[2]
        q2 = q1 + pos_round(143.6, 47860, fi)[2]
        q3 = q2 - pos_round(300, 48050, fi)[2]
        rpnls = run_deals_updates(pos, [
            Deal(D+'00:00', q1, 47980, False),
            Deal(D+'00:10', q2, 47860, False),
            Trade(D+'00:15', 47984.7, 1),
            Deal(D+'00:20', q3, 48050, False),
            Deal(D+'00:30', 0, 48158.7, True),
        ])
        assert N(rpnls.values) == [0.0, 0.0, 0.3976, 0.69, 0.4474]
        assert N(pos.pnl) == 1.1374
        assert N(pos.commissions) == 0.04815870 + 0.05766 + 0.028716 + 0.04798

        D = '2024-01-01 '
        i = lookup.instruments['BINANCE.UM:BTCUSDT'][0]
        px0 = Position(i, ZERO_COSTS)

        run_deals_updates(px0, [
            Deal( D+'12:00:00', 1000/45000.0, 45000.0),
            Deal( D+'12:01:00', 1000/45000.0 + 1000/46000.0, 46000.0),
            Deal( D+'12:03:00', 0, 47000.0),
            Trade( D+'12:04:00', 47000.0, 0),
            Trade( D+'12:06:00', 48000.0, 0),
        ])

        px1 = Position(i, ZERO_COSTS)
        px2 = Position(i, ZERO_COSTS)
        run_deals_updates(px1, [
            Deal( D+'12:00:00', 1000/45000, 45000),
            Deal( D+'12:03:00', 0, 47000),
        ])
        run_deals_updates(px2, [
            Deal( D+'12:01:00', 1000/46000, 46000),
            Deal( D+'12:03:00', 0, 47000),
        ])
        assert px0.total_pnl() ==  N(px1.total_pnl() + px2.total_pnl())

