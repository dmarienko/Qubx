import numpy as np

from qubx.core.series import TimeSeries, OHLCV
from qubx.core.utils import recognize_time
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader
from qubx.ta.indicators import psar, swings
from tests.qubx.ta.utils_for_testing import N, push


class TestCoreSeries:

    def test_basic_series(self):
        ts = TimeSeries("test", "10Min")
        ts.update(recognize_time("2024-01-01 00:00"), 1)
        ts.update(recognize_time("2024-01-01 00:01"), 5)
        ts.update(recognize_time("2024-01-01 00:06"), 2)
        ts.update(recognize_time("2024-01-01 00:12"), 3)
        ts.update(recognize_time("2024-01-01 00:21"), 4)
        ts.update(recognize_time("2024-01-01 00:22"), 5)
        ts.update(recognize_time("2024-01-01 00:31"), 6)
        ts.update(recognize_time("2024-01-01 00:33"), 7)
        ts.update(recognize_time("2024-01-01 00:45"), -12)
        ts.update(recognize_time("2024-01-01 00:55"), 12)
        ts.update(recognize_time("2024-01-01 01:00"), 12)
        assert all(ts.to_series().values == np.array([2, 3, 5, 7, -12, 12, 12]))

    def test_ohlc_series(self):
        ohlc = OHLCV("BTCUSDT", "1Min")
        push(
            ohlc,
            [
                ("2024-01-01 00:00", 9),
                ("2024-01-01 00:00", 1),
                ("2024-01-01 00:01", 2),
                ("2024-01-01 00:01", 3),
                ("2024-01-01 00:01", 2),
                ("2024-01-01 00:02", 3),
                ("2024-01-01 00:03", 4),
                ("2024-01-01 00:04", 5),
                ("2024-01-01 00:04", 5.1),
                ("2024-01-01 00:04:20", 5),
                ("2024-01-01 00:05", 6),
                ("2024-01-01 00:05", 7),
                ("2024-01-01 00:05", 6),
                ("2024-01-01 00:07", 8),
                ("2024-01-01 00:07", -1),
                ("2024-01-01 00:07", 8),
                ("2024-01-01 00:08", 8),
                ("2024-01-01 00:09", 8),
                ("2024-01-01 00:10", 12),
                ("2024-01-01 00:10:01", 21),
                ("2024-01-01 00:10:30", 1),
                ("2024-01-01 00:10:31", 5),
                ("2024-01-01 00:11", 13),
                ("2024-01-01 00:12", 14),
                ("2024-01-01 00:13", 15),
                ("2024-01-01 00:14", 17),
                ("2024-01-01 00:15", 4),
            ],
            1,
        )

        assert len(ohlc) == 15

        r = ohlc.to_series()

        ri = r.loc["2024-01-01 00:10:00"]
        assert ri.open == 12
        assert ri.high == 21
        assert ri.low == 1
        assert ri.close == 5
        assert ri.volume == 4

    def test_series_locator(self):
        ts = TimeSeries("test", "1Min")
        push(
            ts,
            [
                ("2024-01-01 00:00", 9),
                ("2024-01-01 00:01", 2),
                ("2024-01-01 00:02", 3),
                ("2024-01-01 00:03", 4),
                ("2024-01-01 00:04", 5),
                ("2024-01-01 00:05", 6),
                ("2024-01-01 00:07", 8),
                ("2024-01-01 00:08", 8),
                ("2024-01-01 00:09", 8),
                ("2024-01-01 00:10", 12),
                ("2024-01-01 00:11", 13),
                ("2024-01-01 00:12", 14),
                ("2024-01-01 00:13", 15),
                ("2024-01-01 00:14", 17),
                ("2024-01-01 00:15", 18),
            ],
        )
        assert ts[:] == ts.loc[:][:]
        assert len(ts.loc["2024-01-01 00:00":"2024-01-01 00:03"]) == 4
        assert len(ts.loc["2024-01-01 00:00":]) == 15
        assert len(ts.loc[:"2024-01-01 00:10"]) == 10
        assert len(ts.loc[0:4]) == 4
        assert len(ts.loc[0:100]) == 15
        assert ts.loc["2024-01-01 00:10"] == (np.datetime64("2024-01-01 00:10"), 12.0)
        assert len(ts.loc[:]) == 15

    def test_indicator_locator(self):
        ohlc = CsvStorageDataReader("tests/data/csv").read(
            "BTCUSDT_ohlcv_M1", start="2024-01-01", stop="2024-01-15", transform=AsOhlcvSeries("15Min")
        )
        assert isinstance(ohlc, OHLCV)
        sw = swings(ohlc, psar)

        cln1 = sw.loc[:]

        assert all(sw.tops.pd() == cln1.tops.pd())
        assert all(sw.bottoms.pd() == cln1.bottoms.pd())

        # - slices
        assert len(sw.loc["2024-01-01 00:30:00":"2024-01-01 11:30:00"]) == 45
        assert len(sw.loc[:"2024-01-02 00:00:00"]) == 97
