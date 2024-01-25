import numpy as np
import pandas as pd

# from qube.utils import reload_pyx_module
# reload_pyx_module('src/qube/core/')

from qube.core.series import (TimeSeries, sma, ema, tema, dema, kama, lag, compare, highest, lowest, OHLCV)
import tests.qube.ta.utils_for_testing as test


MIN1_UPDATES = [
    ('2024-01-01 00:00', 9), ('2024-01-01 00:00', 1),
    ('2024-01-01 00:01', 2), ('2024-01-01 00:01', 3),   ('2024-01-01 00:01', 2),
    ('2024-01-01 00:02', 3),
    ('2024-01-01 00:03', 4),
    ('2024-01-01 00:04', 5), ('2024-01-01 00:04', 5.1), ('2024-01-01 00:04:20', 5),
    ('2024-01-01 00:05', 6), ('2024-01-01 00:05', 7),   ('2024-01-01 00:05', 6),
    ('2024-01-01 00:07', 8), ('2024-01-01 00:07', -1),  ('2024-01-01 00:07', 8),
    ('2024-01-01 00:08', 8),
    ('2024-01-01 00:09', 8),
    ('2024-01-01 00:10', 12),('2024-01-01 00:10:01', 21),('2024-01-01 00:10:30', 1),('2024-01-01 00:10:31', 5),
    ('2024-01-01 00:11', 13),
    ('2024-01-01 00:12', 14),
    ('2024-01-01 00:13', 15),
    ('2024-01-01 00:14', 17),
    ('2024-01-01 00:15', 4),
]


class TestIndicators:

    def generate_random_series(self, n=100_000, freq='1Min'):
        T = pd.date_range('2024-01-01 00:00', freq=freq, periods=n)
        ds = 1 + (2*np.random.randn(len(T))).cumsum()
        data = list(zip(T, ds))
        return T, ds, data

    def test_indicators_on_series(self):
        _, _, data = self.generate_random_series()

        ts = TimeSeries('close', '1h')
        s1 = sma(ts, 50)
        e1 = ema(ts, 50)
        d1 = dema(ts, 50)
        t1 = tema(ts, 50)
        k1 = kama(ts, 50)
        ss1 = sma(s1, 50)
        ee1 = ema(e1, 50)
        test.push(ts, data)

        assert test.N(s1.to_series()[-20:]) == test.apply_to_frame(test.sma, ts.to_series(), 50)[-20:]
        assert test.N(e1.to_series()[-20:]) == test.apply_to_frame(test.ema, ts.to_series(), 50)[-20:]
        assert test.N(t1.to_series()[-20:]) == test.apply_to_frame(test.tema, ts.to_series(), 50)[-20:]
        assert test.N(k1.to_series()[-20:]) == test.apply_to_frame(test.kama, ts.to_series(), 50)[-20:]
        assert test.N(d1.to_series()[-20:]) == test.apply_to_frame(test.dema, ts.to_series(), 50)[-20:]
        # print(ss1.to_series())

    def test_indicators_lagged(self):
        _, _, data = self.generate_random_series()
        ts = TimeSeries('close', '1h')
        l1 = lag(ts, 1)
        l2 = lag(lag(ts, 1), 4)
        test.push(ts, data)
        assert all(lag(ts, 5).to_series().dropna() == l2.to_series().dropna())

    def test_indicators_comparison(self):
        _, _, data = self.generate_random_series()
        # - precalculated
        xs = test.push(TimeSeries('close', '10Min'), data)
        r = test.scols(xs.to_series(), lag(xs, 1).to_series(), names=['a', 'b'])
        assert len(compare(xs, lag(xs, 1)).to_series()) > 0
        assert all(np.sign(r.a - r.b).dropna() == compare(xs, lag(xs, 1)).to_series().dropna())

        # - on streamed data
        xs1 = TimeSeries('close', '10Min')
        c1 = compare(xs1, lag(xs1, 1))
        test.push(xs1, data)
        r = test.scols(xs1.to_series(), lag(xs1, 1).to_series(), names=['a', 'b'])
        assert len(c1.to_series()) > 0
        assert all(np.sign(r.a - r.b).dropna() == c1.to_series().dropna())

    def test_indicators_highest_lowest(self):
        _, _, data = self.generate_random_series()

        xs = TimeSeries('close', '12Min')
        hh = highest(xs, 13)
        ll = lowest(xs, 13)
        test.push(xs, data)

        rh = xs.pd().rolling(13).max()
        rl = xs.pd().rolling(13).min()
        assert all(abs(hh.pd().dropna() - rh.dropna()) <= 1e-4)
        assert all(abs(ll.pd().dropna() - rl.dropna()) <= 1e-4)

    def test_indicators_on_ohlc(self):
        ohlc = OHLCV('1Min')
        s1 = sma(ohlc.close, 5)
        test.push(ohlc, MIN1_UPDATES, 1)
        print(ohlc.to_series())
        print(s1.to_series())

        s2s = sma(ohlc.close, 5).to_series()
        print(s2s)

        # - TODO: fix this behaviour (nan) ! 
        assert test.N(s2s) == s1.to_series()
