import numpy as np
import pandas as pd

from qube.core.series import (TimeSeries, lag, compare, OHLCV)
from qube.ta.indicators import (sma, ema, tema, dema, kama, highest, lowest)
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
        ohlc = OHLCV('TEST', '1Min')
        s1 = sma(ohlc.close, 5)
        test.push(ohlc, MIN1_UPDATES, 1)
        print(ohlc.to_series())
        print(s1.to_series())

        s2s = sma(ohlc.close, 5).to_series()
        print(s2s)

        # - TODO: fix this behaviour (nan) ! 
        assert test.N(s2s) == s1.to_series()

    def test_bsf_calcs(self):
        _, _, data = self.generate_random_series(2000)

        def test_i(ts: TimeSeries):
            ds = ts - ts.shift(1)
            a1 = sma(ds * (ds > 0), 14) 
            a2 = ds 
            return (a1 - a2) / (a1 + a2)

        # - incremental calcs
        ts_i = TimeSeries('close', '1h')
        r1_i = test_i(ts_i)
        test.push(ts_i, data)

        # - calc on ready data
        ts_p = TimeSeries('close', '1h')
        test.push(ts_p, data)
        r1_p = test_i(ts_p)

        # - pandas 
        ds = ts_i.pd().diff()
        a1 = test.apply_to_frame(test.sma, ds * (ds > 0), 14)
        a2 = ds
        gauge =(a1 - a2) / (a1 + a2)

        s1 = sum(abs(r1_p.pd() - gauge).dropna())
        s2 = sum(abs(r1_i.pd() - gauge).dropna())
        print(s1, s2)
        assert s1 < 1e-10
        assert s2 < 1e-10

        # - another case
        def test_ii(ts: TimeSeries):
            a1 = sma(ts, 5) 
            a2 = sma(ts, 10) * 1000
            return a1 - a2

        ts_ii = TimeSeries('close', '10Min')
        r_ii = test_ii(ts_ii)
        test.push(ts_ii, data[:1000])

        a1 = test.apply_to_frame(test.sma, ts_ii.pd(), 5)
        a2 = 1000 * test.apply_to_frame(test.sma, ts_ii.pd(), 10)
        err = np.std(abs((a1 - a2) - r_ii.pd()).dropna())
        assert err < 1e-10
        

