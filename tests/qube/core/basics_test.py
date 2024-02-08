
from qube import lookup
from qube.core.basics import Instrument, Position, ZERO_COSTS


class TestBasics:

    def test_lookup(self):
        s0 = lookup['BINANCE:ETH.*']
        s1 = lookup['DUKAS:EURGBP']
        assert (
            lookup.find_aux_instrument_for(s0[0], 'USDT').symbol, 
            lookup.find_aux_instrument_for(s0[1], 'USDT'), 
            lookup.find_aux_instrument_for(s1[0], 'USD').symbol,
        ) == ('BTCUSDT', None, 'GBPUSD')

    def test_positions(self):
        s = lookup['BINANCE:BTCUSDT']
        pos = Position(s[0], ZERO_COSTS)
        print(pos)