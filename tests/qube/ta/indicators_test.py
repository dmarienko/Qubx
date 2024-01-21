import numpy as np
import pandas as pd

# from qube.utils import reload_pyx_module
# reload_pyx_module('src/qube/core/')

from qube.core.series import (TimeSeries, Sma, Ema, recognize_time, Tema, Dema, Kama, OHLCV)
from tests.qube.ta.utils_for_testing import sma, ema, tema, kama, apply_to_frame, N


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
        pass