import numpy as np
import pandas as pd

# from qube.utils import reload_pyx_module
# reload_pyx_module('src/qube/core/')

from qube.core.series import (TimeSeries, Sma, Ema, recognize_time, Tema, Dema, Kama, OHLCV)
from tests.qube.ta.utils_for_testing import sma, ema, tema, kama, apply_to_frame, N, push


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
        s1 = Sma(ts, 50)
        e1 = Ema(ts, 50)
        ss1 = Sma(s1, 50)
        ee1 = Ema(e1, 50)
        t1 = Tema(ts, 50)
        k1 = Kama(ts, 50)
        [ts.update(ti.asm8, vi) for ti, vi in data];

        assert N(s1.to_series()[-20:]) == apply_to_frame(sma, ts.to_series(), 50)[-20:]
        assert N(e1.to_series()[-20:]) == apply_to_frame(ema, ts.to_series(), 50)[-20:]
        assert N(t1.to_series()[-20:]) == apply_to_frame(tema, ts.to_series(), 50)[-20:]
        assert N(k1.to_series()[-20:]) == apply_to_frame(kama, ts.to_series(), 50)[-20:]

    def test_indicators_on_ohlc(self):
        ohlc = OHLCV('1Min')
        s1 = Sma(ohlc.close, 5)
        push(ohlc, MIN1_UPDATES, 1)
        print(ohlc.to_series())
        print(s1.to_series())
        print(Sma(ohlc.close, 5).to_series())
