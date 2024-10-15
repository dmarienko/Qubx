import numpy as np
import pandas as pd

from qubx.core.series import OHLCV, Quote, Trade
from qubx.data.readers import (
    STOCK_DAILY_SESSION,
    AsPandasFrame,
    CsvStorageDataReader,
    InMemoryDataFrameReader,
    AsQuotes,
    AsOhlcvSeries,
    QuestDBConnector,
    RestoreTicksFromOHLC,
)
from pytest import approx

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)

T = lambda t: np.datetime64(t, "ns")


class TestDataReaders:

    def test_quotes_reader(self):
        r0 = CsvStorageDataReader("tests/data/csv")

        qts = r0.read("quotes")
        assert len(qts) == 500  # type: ignore

        qts = r0.read("quotes", "2017-08-24 13:09:29", transform=AsQuotes())
        assert len(qts) == 3  # type: ignore
        assert qts[-1].ask == 9.39  # type: ignore

    def test_ohlc_reader(self):
        r = CsvStorageDataReader("tests/data/csv/")
        assert "BTCUSDT_ohlcv_M1" in r.get_names()

        d0 = r.read("SPY")
        assert len(d0) == 4353

        d1 = r.read("BTCUSDT_ohlcv_M1", "2024-01-10 15:00", "2024-01-10 15:10")
        assert len(d1) == 11
        assert d1[0][0] == T("2024-01-10 15:00")
        assert d1[-1][0] == T("2024-01-10 15:10")

        # - let's read by 100 records
        _s = 0
        for x in r.read("BTCUSDT_ohlcv_M1", "2024-01-10", "2024-01-11", chunksize=100):
            _s += len(x)
            print(f"{x[0][0]} ~ {x[-1][0]} >> length is {len(x)} records")
        assert _s == 1441

        # - read as DataFrame
        d2: pd.DataFrame = r.read("SPY", transform=AsPandasFrame())  # type: ignore
        assert N(d2["close"].iloc[-1]) == 234.58999599999999

        # - read as OHLCV series
        d3 = r.read("BTCUSDT_ohlcv_M1", transform=AsOhlcvSeries("1d"))
        assert len(d3) == 46  # type: ignore

        # - DataFrame from quotes
        d4 = r.read("quotes", transform=AsPandasFrame())
        assert len(d4) == 500  # type: ignore
        assert d4["ask"].iloc[-1] == 9.39  # type: ignore

        # Let's make OHLC from quotes and make 2 Min bars out of them
        d5 = r.read("quotes", transform=AsOhlcvSeries("2Min"))
        assert len(d5) == 5

        # exact quote
        d6 = r.read("quotes", "2017-08-24 13:09:29", "2017-08-24 13:09:29")
        assert len(d6) == 1
        assert d6[0][0] == T("2017-08-24 13:09:29")

    def test_simulated_quotes_trades(self):
        r = CsvStorageDataReader("tests/data/csv/")

        tick_data = r.read("BTCUSDT_ohlcv_M1", transform=RestoreTicksFromOHLC(trades=True))
        ohlc_data = r.read("BTCUSDT_ohlcv_M1", transform=AsOhlcvSeries())

        restored_ohlc = OHLCV("restored", "1Min")
        for t in tick_data:
            if isinstance(t, Trade):
                restored_ohlc.update(t.time, t.price, t.size)
            else:
                restored_ohlc.update(t.time, t.mid_price())

        assert all((restored_ohlc.pd() - ohlc_data.pd()) < 1e-10)

        d1 = r.read("SPY", transform=RestoreTicksFromOHLC(daily_session_start_end=STOCK_DAILY_SESSION))[
            :4
        ]  # type: ignore

        assert d1[0].time == T("2000-01-03T09:30:00.100000000").item()
        assert d1[3].time == T("2000-01-03T15:59:59.900000000").item()

    def test_supported_data_id(self):
        r0 = CsvStorageDataReader("tests/data/csv/")
        assert not r0.get_aux_data_ids()

        r1 = InMemoryDataFrameReader({"TEST1": pd.DataFrame(), "TEST2": pd.DataFrame()})
        assert r1.get_aux_data_ids()
        assert set(r1.get_aux_data("symbols")) == {"TEST1", "TEST2"}
        try:
            r1.get_aux_data("some_arbitrary_data_id")
        except:
            assert True
