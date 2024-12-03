from dataclasses import dataclass
import pandas as pd

from qubx import lookup
from qubx.backtester.simulated_data import IterableSimulatorData, IteratedDataStreamsSlicer
from qubx.core.basics import Subtype
from qubx.data.helpers import loader


@dataclass
class DummyTimeEvent:
    time: int
    data: str

    @staticmethod
    def from_dict(data: dict[str | pd.Timedelta, str], start: str) -> list["DummyTimeEvent"]:
        _t0 = pd.Timestamp(start)
        return [DummyTimeEvent((_t0 + pd.Timedelta(t)).as_unit("ns").asm8.item(), d) for t, d in data.items()]

    @staticmethod
    def from_seq(start: str, n: int, ds: str, pfx: str) -> list["DummyTimeEvent"]:
        return DummyTimeEvent.from_dict({s * pd.Timedelta(ds): pfx for s in range(n + 1)}, start)

    def __repr__(self) -> str:
        return f"{pd.Timestamp(self.time, 'ns')} -> ({self.data})"


class TestSimulatedDataStuff:
    def test_iterator_slicer_1(self):
        # fmt: off
        slicer = IteratedDataStreamsSlicer()

        data1 = [
            DummyTimeEvent.from_seq("2020-01-01 00:05", 3, "1Min", "A1"),
            DummyTimeEvent.from_seq("2020-01-01 00:16", 3, "1Min", "A2"),
            DummyTimeEvent.from_seq("2020-01-01 00:19", 3, "1Min", "A3"),
        ]

        slicer += { "data1": iter(data1)}

        r = []
        for t in slicer:
            if not t: continue
            print(f"{pd.Timestamp(t[2].time, 'ns')} | id={t[0]} | {t[2].data}")
            r.append(t[2].data)

        assert r == [
            'A1', 'A1', 'A1', 'A1', 
            'A2', 'A2', 'A2', 'A2', 
            'A3', 'A3', 'A3', 'A3', 
        ]
        # fmt: on

    def test_iterator_slicer_2(self):
        # fmt: off
        slicer = IteratedDataStreamsSlicer()

        data1 = [
            DummyTimeEvent.from_seq("2020-01-01 00:05", 10, "1Min", "A1"),
            DummyTimeEvent.from_seq("2020-01-01 00:16", 10, "1Min", "A2"),
        ]

        data2 = [
            DummyTimeEvent.from_seq("2020-01-01 00:00", 10, "1Min", "B1"),
            DummyTimeEvent.from_seq("2020-01-01 00:11", 10, "1Min", "B2"),
        ]


        slicer += {
            'I0': iter(data1),
            'I1': iter(data2),
        }

        r = []
        for t in slicer:
            if not t: continue
            print(f"{pd.Timestamp(t[2].time, 'ns')} | id={t[0]} | {t[2].data}")
            r.append(t[2].data)

        assert r == [
            'B1', 'B1', 'B1', 'B1', 'B1', 
            'B1', 'A1', 
            'B1', 'A1', 'B1', 'A1', 'B1', 'A1', 'B1', 'A1', 'B1', 'A1', 'B2', 'A1', 'B2',
            'A1', 'B2', 'A1', 'B2', 'A1', 'B2', 'A1', 'B2', 'A2', 'B2', 
            'A2', 'B2', 'A2', 'B2', 'A2', 'B2', 'A2', 'B2', 
            'A2', 'A2', 'A2', 'A2', 'A2', 'A2'
        ]
        # fmt: on

    def test_iterator_slicer_3(self):
        # fmt: off
        slicer = IteratedDataStreamsSlicer()

        data1 = [
            DummyTimeEvent.from_seq("2020-01-01 00:00", 10, "1Min", "A1"),
            DummyTimeEvent.from_seq("2020-01-01 00:11", 10, "1Min", "A2"),
        ]

        data2 = [
            DummyTimeEvent.from_seq("2020-01-01 00:05", 10, "1Min", "B1"),
            DummyTimeEvent.from_seq("2020-01-01 00:16", 10, "1Min", "B2"),
        ]

        data3 = [
            DummyTimeEvent.from_seq("2020-01-01 00:08", 10, "1Min", "C1"),
            DummyTimeEvent.from_seq("2020-01-01 00:19", 10, "1Min", "C2"),
        ]

        slicer += {
            'i1': iter(data1),
            'i2': iter(data2),
            'i3': iter(data3),
        }

        r = []
        for t in slicer:
            if not t: continue
            print(f"{pd.Timestamp(t[2].time, 'ns')} | id={t[0]} | {t[2].data}")
            r.append(t[2].data)

        assert r == [
            'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 
                'B1', 'A1', 'B1', 'A1', 'B1', 'A1', 'B1', 'C1', 'A1', 'B1', 'C1', 'A1', 'B1', 'C1', 'A2', 'B1', 'C1', 'A2', 
                'B1', 'C1', 'A2', 'B1', 'C1', 'A2', 'B1', 'C1', 'A2', 'B1', 'C1', 'A2', 'B2', 'C1', 'A2', 'B2', 'C1', 'A2', 
            'B2', 'C1', 'A2', 'B2', 'C2', 'A2', 'B2', 'C2', 'A2', 'B2', 'C2', 
            'B2', 'C2', 'B2', 'C2', 'B2', 'C2', 'B2', 'C2', 'B2', 
            'C2', 'C2', 'C2', 'C2']
        # fmt: on

    def test_iterator_slicer_add_remove(self):
        # fmt: off
        slicer = IteratedDataStreamsSlicer()

        data1 = [
            DummyTimeEvent.from_seq("2020-01-01 00:00", 10, "1Min", "A1"),
            DummyTimeEvent.from_seq("2020-01-01 00:11", 10, "1Min", "A2"),
        ]

        data2 = [
            DummyTimeEvent.from_seq("2020-01-01 00:05", 10, "1Min", "B1"),
            DummyTimeEvent.from_seq("2020-01-01 00:16", 10, "1Min", "B2"),
        ]

        data3 = [
            DummyTimeEvent.from_seq("2020-01-01 00:08", 10, "1Min", "C1"),
            DummyTimeEvent.from_seq("2020-01-01 00:19", 10, "1Min", "C2"),
        ]

        data4 = [
            DummyTimeEvent.from_seq("2020-01-01 00:10", 10, "1Min", "D1"),
        ]

        slicer += {
            'x1': iter(data1),
            'x2': iter(data2),
            'x3': iter(data3),
        }

        r, k = [], 0
        ti = 0
        for t in slicer:
            if not t: continue
            print(f"{k:3d}: {pd.Timestamp(t[2].time, 'ns')} | id={t[0]} | {t[2].data}")
            assert t[2].time >= ti
            r.append(t[2].data)
            if k == 3: slicer.remove('x1')
            if k == 11: slicer += {'x10': iter(data4)}
            k += 1
            ti = t[2].time
        assert r == [
            'A1', 'A1', 'A1', 'A1', 
            'B1', 'B1', 'B1', 
            'B1', 'C1', 'B1', 'C1', 
            'B1', 'C1', 'D1', 
            'C1', 'D1', 'B1', 'C1', 'D1', 'B1', 'C1', 'D1', 'B1', 'C1', 'D1', 'B1', 'C1', 'D1', 'B1', 'C1', 'D1', 
            'B2', 'C1', 'D1', 'B2', 'C1', 'D1', 'B2', 'C2', 'D1', 'B2', 'C2', 'D1', 'B2', 
            'C2', 'B2', 'C2', 'B2', 'C2', 'B2', 'C2', 'B2', 'C2', 'B2', 'C2', 'B2', 
            'C2', 'C2', 'C2'
        ]
        # fmt: on

    def test_iterator_4_streams(self):
        # fmt: off
        slicer = IteratedDataStreamsSlicer()

        slicer += {
            "set.A": iter([DummyTimeEvent.from_seq("2020-01-01 00:00", 10, "1Min", "A1")]),
            "set.B": iter([DummyTimeEvent.from_seq("2020-01-01 00:05", 10, "1Min", "B1")]),
            "set.D": iter([DummyTimeEvent.from_seq("2020-01-01 00:03", 10, "1Min", "D1")]),
            "set.E": iter([DummyTimeEvent.from_seq("2020-01-01 00:03", 10, "1Min", "E1")]),
            "set.C": iter([DummyTimeEvent.from_seq("2020-01-01 00:01", 10, "1Min", "C1")]),
        }

        r = []
        for t in slicer:
            if not t:
                continue
            print(f"{pd.Timestamp(t[2].time, 'ns')} | id={t[0]} | {t[2].data}")
            r.append(t[2].data)

        assert r == [
            'A1', 'A1', 
            'C1', 'A1', 'C1', 'A1', 'C1', 'D1', 
            'E1', 'A1', 'C1', 'D1', 'E1', 'A1', 'C1', 'D1', 'E1', 'B1', 'A1', 'C1', 'D1', 'E1', 
            'B1', 'A1', 'C1', 'D1', 'E1', 'B1', 'A1', 'C1', 'D1', 'E1', 'B1', 'A1', 'C1', 'D1', 
            'E1', 'B1', 'A1', 'C1', 'D1', 'E1', 'B1', 'C1', 'D1', 'E1', 'B1', 'D1', 'E1', 'B1', 'D1', 'E1', 
            'B1', 'B1', 'B1'
        ]

        # fmt: on

    def test_iterable_simulation_data_management(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulatorData(ld, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        isd.add_instruments_for_subscription(Subtype.OHLC["1h"], [s1, s2])
        isd.add_instruments_for_subscription(Subtype.OHLC["1h"], s3)
        isd.add_instruments_for_subscription(Subtype.OHLC["4h"], s3)
        isd.add_instruments_for_subscription(Subtype.OHLC["1d"], s3)
        isd.add_instruments_for_subscription(Subtype.OHLC_TICKS["4h"], s1)

        # get all instruments for ANY subscription
        assert set(isd.get_instruments_for_subscription(Subtype.ALL)) == set([s1, s2, s3])

        # get subs for instrument
        assert isd.get_subscriptions_for_instrument(s3) == list(
            map(Subtype.from_str, [Subtype.OHLC["1h"], Subtype.OHLC["4h"], Subtype.OHLC["1d"]])
        )

        assert isd.get_instruments_for_subscription(Subtype.OHLC["4h"]) == [s3]
        assert isd.get_instruments_for_subscription(Subtype.OHLC["1h"]) == [s1, s2, s3]

        isd.remove_instruments_from_subscription(Subtype.OHLC["1h"], s3)
        assert isd.get_instruments_for_subscription(Subtype.OHLC["1h"]) == [s1, s2]

        isd.remove_instruments_from_subscription(Subtype.OHLC["1h"], [s1, s2, s3])
        assert isd.get_instruments_for_subscription(Subtype.OHLC["1h"]) == []

    def test_iterable_simulation_data_queue_with_warmup(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulatorData(ld, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        # set warmup period
        isd.set_warmup_period(Subtype.OHLC["1d"], "24h")
        isd.add_instruments_for_subscription(Subtype.OHLC["1d"], [s1, s2, s3])

        _n_hist = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            t = pd.Timestamp(d[2].time, "ns")
            data_type = d[1]
            if data_type.startswith("hist"):
                _n_hist += 1
            print(t, d[0], d[1])
        assert _n_hist == 3 * 4
