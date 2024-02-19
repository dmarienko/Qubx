import numpy as np

from qube.core.series import OHLCV, Quote, Trade
from qube.data.readers import CsvDataReader, QuotesDataProcessor, OhlcvDataProcessor, QuotesFromOHLCVDataProcessor

T = lambda t: np.datetime64(t, 'ns')


class TestDataReaders:

    def test_quotes_reader(self):
        r0 = CsvDataReader('tests/data/csv/quotes.csv', QuotesDataProcessor())

        qts = r0.read()
        assert len(qts) == 500

        qts = r0.read('2017-08-24 13:09:29')
        assert len(qts) == 3
        assert qts[-1].ask == 9.39

    def test_ohlc_reader(self):
        r1 = CsvDataReader('tests/data/csv/BTCUSDT_ohlcv_M1.csv.gz', OhlcvDataProcessor('test0'))
        r2 = CsvDataReader('tests/data/csv/BTCUSDT_ohlcv_M1_sec.csv.gz', OhlcvDataProcessor('test1'))
        # r3 = CsvDataReader('tests/data/csv/AAPL.csv', OhlcvDataProcessor('AAPL'), timestamp_parsers=["%m/%d/%Y", "%d-%m-%Y"])
        r4 = CsvDataReader('tests/data/csv/SPY.csv', OhlcvDataProcessor('SPY'), timestamp_parsers=["%Y-%m-%d"])

        d1 = r1.read('2024-01-08 11:05:00', '2024-01-08 11:10:00')
        T(d1.times[0]) == T('2024-01-08 11:10:00')
        T(d1.times[-1]) == T('2024-01-08 11:05:00')

        d2 = r2.read('2024-01-08 11:05:00', '2024-01-08 11:10:00')
        T(d2.times[0]) == T('2024-01-08 11:10:00')
        T(d2.times[-1]) == T('2024-01-08 11:05:00')

        d4 = r4.read('2100-01-08 11:05:00', '2124-01-08 11:10:00')
        assert d4 is None
        
        d4 = r4.read('2000-01-01 11:12:34', '2001-01-01 12:10:00')
        assert len(d4) == 252
        assert T(d4.times[-1]) == T('2000-01-03')
        assert T(d4.times[0]) == T('2000-12-29')

    def test_simulated_quotes_trades(self):
        r0 = CsvDataReader('tests/data/csv/BTCUSDT_ohlcv_M1.csv.gz', QuotesFromOHLCVDataProcessor(trades=True))
        r1 = CsvDataReader('tests/data/csv/BTCUSDT_ohlcv_M1.csv.gz', OhlcvDataProcessor('original'))

        tick_data = r0.read()
        ohlc_data = r1.read()

        restored_ohlc = OHLCV('restored', '1Min')
        for t in tick_data:
            if isinstance(t, Trade):
                restored_ohlc.update(t.time, t.price, t.size)
            else:
                restored_ohlc.update(t.time, t.mid_price())

        assert all((restored_ohlc.pd() - ohlc_data.pd()) < 1e-10)


