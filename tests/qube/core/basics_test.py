
from qube import lookup
from qube.core.basics import Instrument, Position, ZERO_COSTS


class TestBasics:

    def test_positions(self):
        s = lookup['BINANCE:BTCUSDT']
        pos = Position(s[0], ZERO_COSTS)
        print(pos)