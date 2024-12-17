from dataclasses import dataclass

import numpy as np
import pandas as pd

from qubx import lookup
from qubx.backtester.simulated_data import EventBatcher, IterableSimulationData, IteratedDataStreamsSlicer
from qubx.core.basics import BatchEvent, DataType
from qubx.data.helpers import loader
from qubx.data.readers import InMemoryDataFrameReader


def get_event_dt(i: float, base: pd.Timestamp = pd.Timestamp("2021-01-01"), offset: str = "D") -> int:
    return (base + pd.Timedelta(i, offset)).as_unit("ns").asm8.item()  # type: ignore


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
        isd = IterableSimulationData({"ohlc": ld, "ohlc_quotes": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC["4h"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC["1d"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC_QUOTES["4h"], s1)

        # has subscription
        assert isd.has_subscription(s3, "ohlc(4h)")

        # has subscription
        assert not isd.has_subscription(s1, "ohlc(1d)")

        # get all instruments for ANY subscription
        assert set(isd.get_instruments_for_subscription(DataType.ALL)) == set([s1, s2, s3])

        # get subs for instrument
        assert set(isd.get_subscriptions_for_instrument(s3)) == set(
            [DataType.OHLC["1h"], DataType.OHLC["4h"], DataType.OHLC["1d"]]
        )

        assert isd.get_instruments_for_subscription(DataType.OHLC["4h"]) == [s3]
        assert isd.get_instruments_for_subscription(DataType.OHLC["1h"]) == [s1, s2, s3]

        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], s3)
        assert isd.get_instruments_for_subscription(DataType.OHLC["1h"]) == [s1, s2]

        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], [s1, s2, s3])
        assert isd.get_instruments_for_subscription(DataType.OHLC["1h"]) == []

        assert set(isd.get_subscriptions_for_instrument(None)) == set(
            [DataType.OHLC["4h"], DataType.OHLC_QUOTES["4h"], DataType.OHLC["1d"]]
        )

    def test_iterable_simulation_data_queue_with_warmup(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        # set warmup period
        isd.set_warmup_period(DataType.OHLC["1d"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["1d"], [s1, s2, s3])

        _n_hist = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            t = pd.Timestamp(d[2].time, "ns")
            data_type = d[1]
            is_hist = d[3]
            if is_hist:
                _n_hist += 1
            print(t, d[0], data_type, "HIST" if is_hist else "")
        assert _n_hist == 3 * 4

    def test_iterable_simulation_custom_subscription_type(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        s4 = lookup.find_symbol("BINANCE.UM", "AAVEUSDT")
        assert s1 is not None and s2 is not None and s3 is not None and s4 is not None

        # set warmup period
        isd.set_warmup_period(DataType.OHLC["1d"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["1d"], [s1, s2, s3])

        # - custom reader
        idx = pd.date_range(start="2023-06-01 00:00", end="2023-07-30", freq="1h", name="timestamp")
        c_data = pd.DataFrame({"value1": np.random.randn(len(idx)), "value2": np.random.randn(len(idx))}, index=idx)
        custom_reader_2 = InMemoryDataFrameReader({"BINANCE.UM:AAVEUSDT": c_data})

        _n_r, _got_hist, got_live = 0, False, False
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            t = pd.Timestamp(d[2].time, "ns")
            data_type = d[1]
            is_hist = d[3]
            _n_r += 1

            if _n_r == 20:
                print("- Subscribe on some shit here -")
                isd.set_typed_reader("some_my_custom_data", custom_reader_2)
                isd.set_warmup_period("some_my_custom_data", "24h")
                isd.add_instruments_for_subscription("some_my_custom_data", [s4])
            print(t, d[0], data_type, "HIST" if is_hist else "")

            if "some_my_custom_data" == data_type:
                got_live = True
                if is_hist:
                    got_hist = True

        assert got_live
        assert got_hist

    def test_batching_basic(self):
        events = [
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(1, offset="ms"), "data1"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(2, offset="ms"), "data2"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(5, offset="ms"), "data3"), False),
            ("ETHUSDT", "trade", DummyTimeEvent(get_event_dt(7, offset="s"), "data4"), False),
            ("ETHUSDT", "trade", DummyTimeEvent(get_event_dt(7.9, offset="s"), "data4"), False),
            ("BTCUSDT", "ohlc", DummyTimeEvent(get_event_dt(9, offset="s"), "data5"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(11, offset="s"), "data6"), False),
        ]

        # test 1
        batched_events = list(EventBatcher(events))
        expected_events = [
            (
                "BTCUSDT",
                "trade",
                BatchEvent(
                    get_event_dt(5, offset="ms"),  # type: ignore
                    [
                        DummyTimeEvent(get_event_dt(1, offset="ms"), "data1"),
                        DummyTimeEvent(get_event_dt(2, offset="ms"), "data2"),
                        DummyTimeEvent(get_event_dt(5, offset="ms"), "data3"),
                    ],
                ),
                False,
            ),
            (
                "ETHUSDT",
                "trade",
                BatchEvent(
                    get_event_dt(7.9, offset="s"),  # type: ignore
                    [
                        DummyTimeEvent(get_event_dt(7, offset="s"), "data4"),
                        DummyTimeEvent(get_event_dt(7.9, offset="s"), "data4"),
                    ],
                ),
                False,
            ),
            ("BTCUSDT", "ohlc", DummyTimeEvent(get_event_dt(9, offset="s"), "data5"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(11, offset="s"), "data6"), False),
        ]
        assert expected_events == batched_events

        # test 2 (check if batcher is disabled)
        nobatched_events = list(EventBatcher(events, passthrough=True))
        assert events == nobatched_events

    def test_batching_leftover_trades(self):
        events = [
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(1, offset="ms"), "data1"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(2, offset="ms"), "data2"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(5, offset="ms"), "data3"), False),
            ("ETHUSDT", "trade", DummyTimeEvent(get_event_dt(7, offset="s"), "data4"), False),
            ("ETHUSDT", "trade", DummyTimeEvent(get_event_dt(7.9, offset="s"), "data4"), False),
            ("BTCUSDT", "ohlc", DummyTimeEvent(get_event_dt(9, offset="s"), "data5"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(11, offset="s"), "data6"), False),
            ("ETHUSDT", "ohlc", DummyTimeEvent(get_event_dt(12, offset="s"), "data5"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(13, offset="s"), "data6"), False),
        ]
        expected_events = [
            (
                "BTCUSDT",
                "trade",
                BatchEvent(
                    get_event_dt(5, offset="ms"),  # type: ignore
                    [
                        DummyTimeEvent(get_event_dt(1, offset="ms"), "data1"),
                        DummyTimeEvent(get_event_dt(2, offset="ms"), "data2"),
                        DummyTimeEvent(get_event_dt(5, offset="ms"), "data3"),
                    ],
                ),
                False,
            ),
            (
                "ETHUSDT",
                "trade",
                BatchEvent(
                    get_event_dt(7.9, offset="s"),  # type: ignore
                    [
                        DummyTimeEvent(get_event_dt(7, offset="s"), "data4"),
                        DummyTimeEvent(get_event_dt(7.9, offset="s"), "data4"),
                    ],
                ),
                False,
            ),
            ("BTCUSDT", "ohlc", DummyTimeEvent(get_event_dt(9, offset="s"), "data5"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(11, offset="s"), "data6"), False),
            ("ETHUSDT", "ohlc", DummyTimeEvent(get_event_dt(12, offset="s"), "data5"), False),
            ("BTCUSDT", "trade", DummyTimeEvent(get_event_dt(13, offset="s"), "data6"), False),
        ]
        actual_output = list(EventBatcher(events))
        assert expected_events == actual_output
