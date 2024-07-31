import pandas as pd
from typing import Any, Iterator
from qubx.backtester.queue import SimulatedDataQueue, DataLoader, EventBatcher


class DummyEvent:
    time: pd.Timestamp
    data: str

    def __init__(self, time: pd.Timestamp, data: str):
        self.time = time
        self.data = data

    def __repr__(self) -> str:
        return f"{self.time} {self.data}"

    def __eq__(self, other: Any) -> bool:
        return self.time == other.time and self.data == other.data


class DummyDataLoader(DataLoader):
    def __init__(self, symbol: str, events: list[list[Any]]):
        self._events = events
        self._symbol = symbol
        for event in self._events:
            for e in event:
                e.symbol = symbol

    def load(self, start: str, end: str) -> Iterator:
        yield from self._events

    @property
    def symbol(self):
        return self._symbol

    @property
    def data_type(self):
        return "dummy"

    def __hash__(self) -> int:
        return hash((self._symbol, "dummy"))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DummyDataLoader):
            return False
        return self._symbol == other._symbol and "dummy" == other._data_type


def get_event_dt(i: float, base: pd.Timestamp = pd.Timestamp("2021-01-01"), offset: str = "D") -> pd.Timestamp:
    return base + pd.Timedelta(i, offset)  # type: ignore


class TestSimulatedQueueStuff:

    def test_dummy_basic_event_sync(self):
        q = SimulatedDataQueue()
        q += DummyDataLoader(
            "APPL",
            [
                [DummyEvent(get_event_dt(1), "data1")],
                [DummyEvent(get_event_dt(3), "data2"), DummyEvent(get_event_dt(5), "data3")],
            ],
        )
        q += DummyDataLoader(
            "MSFT",
            [
                [
                    DummyEvent(get_event_dt(2), "data4"),
                    DummyEvent(get_event_dt(4), "data5"),
                    DummyEvent(get_event_dt(5), "data6"),
                ]
            ],
        )
        expected_event_seq = [
            ("APPL", "dummy", DummyEvent(get_event_dt(1), "data1")),
            ("MSFT", "dummy", DummyEvent(get_event_dt(2), "data4")),
            ("APPL", "dummy", DummyEvent(get_event_dt(3), "data2")),
            ("MSFT", "dummy", DummyEvent(get_event_dt(4), "data5")),
            ("APPL", "dummy", DummyEvent(get_event_dt(5), "data3")),
            ("MSFT", "dummy", DummyEvent(get_event_dt(5), "data6")),
        ]
        actual_event_seq = list(q.create_iterable(get_event_dt(0), get_event_dt(10)))
        assert expected_event_seq == actual_event_seq

    def test_dummy_data_loader_add(self):
        q = SimulatedDataQueue()
        q += DummyDataLoader(
            "APPL",
            [
                [DummyEvent(get_event_dt(1), "data1")],
                [DummyEvent(get_event_dt(3), "data2"), DummyEvent(get_event_dt(5), "data3")],
            ],
        )
        actual_event_seq = []
        qiter = iter(q.create_iterable(get_event_dt(0), get_event_dt(10)))
        actual_event_seq.append(next(qiter))
        # now let's add another loader in the middle of the iteration
        q += DummyDataLoader(
            "MSFT",
            [
                [
                    DummyEvent(get_event_dt(2), "data4"),
                    DummyEvent(get_event_dt(4), "data5"),
                    DummyEvent(get_event_dt(5), "data6"),
                ]
            ],
        )
        while True:
            try:
                actual_event_seq.append(next(qiter))
            except StopIteration:
                break
        expected_event_seq = [
            ("APPL", "dummy", DummyEvent(get_event_dt(1), "data1")),
            ("MSFT", "dummy", DummyEvent(get_event_dt(2), "data4")),
            ("APPL", "dummy", DummyEvent(get_event_dt(3), "data2")),
            ("MSFT", "dummy", DummyEvent(get_event_dt(4), "data5")),
            ("APPL", "dummy", DummyEvent(get_event_dt(5), "data3")),
            ("MSFT", "dummy", DummyEvent(get_event_dt(5), "data6")),
        ]
        assert expected_event_seq == actual_event_seq

    def test_dummy_data_loader_remove(self):
        q = SimulatedDataQueue()
        l1 = DummyDataLoader(
            "APPL",
            [
                [DummyEvent(get_event_dt(1), "data1")],
                [DummyEvent(get_event_dt(3), "data2"), DummyEvent(get_event_dt(5), "data3")],
            ],
        )
        l2 = DummyDataLoader(
            "MSFT",
            [
                [
                    DummyEvent(get_event_dt(2), "data4"),
                    DummyEvent(get_event_dt(4), "data5"),
                    DummyEvent(get_event_dt(5), "data6"),
                ]
            ],
        )
        q += l1
        q += l2
        actual_event_seq = []
        qiter = iter(q.create_iterable(get_event_dt(0), get_event_dt(10)))
        for _ in range(3):
            actual_event_seq.append(next(qiter))
        q -= l2
        while True:
            try:
                actual_event_seq.append(next(qiter))
            except StopIteration:
                break

        expected_event_seq = [
            ("APPL", "dummy", DummyEvent(get_event_dt(1), "data1")),
            ("MSFT", "dummy", DummyEvent(get_event_dt(2), "data4")),
            ("APPL", "dummy", DummyEvent(get_event_dt(3), "data2")),
            ("APPL", "dummy", DummyEvent(get_event_dt(5), "data3")),
        ]
        assert expected_event_seq == actual_event_seq

    def test_batching_basic(self):
        events = [
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(1, offset="ms"), "data1")),
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(2, offset="ms"), "data2")),
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(5, offset="ms"), "data3")),
            ("ETHUSDT", "trade", DummyEvent(get_event_dt(7, offset="s"), "data4")),
            ("ETHUSDT", "trade", DummyEvent(get_event_dt(7.9, offset="s"), "data4")),
            ("BTCUSDT", "ohlc", DummyEvent(get_event_dt(9, offset="s"), "data5")),
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(11, offset="s"), "data6")),
        ]

        # test 1
        batched_events = list(EventBatcher(events))
        expected_events = [
            (
                "BTCUSDT",
                "trade",
                [
                    DummyEvent(get_event_dt(1, offset="ms"), "data1"),
                    DummyEvent(get_event_dt(2, offset="ms"), "data2"),
                    DummyEvent(get_event_dt(5, offset="ms"), "data3"),
                ],
            ),
            (
                "ETHUSDT",
                "trade",
                [
                    DummyEvent(get_event_dt(7, offset="s"), "data4"),
                    DummyEvent(get_event_dt(7.9, offset="s"), "data4"),
                ],
            ),
            ("BTCUSDT", "ohlc", DummyEvent(get_event_dt(9, offset="s"), "data5")),
            ("BTCUSDT", "trade", [DummyEvent(get_event_dt(11, offset="s"), "data6")]),
        ]
        assert expected_events == batched_events

        # test 2 (check if batcher is disabled)
        nobatched_events = list(EventBatcher(events, passthrough=True))
        assert events == nobatched_events

    def test_batching_leftover_trades(self):
        events = [
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(1, offset="ms"), "data1")),
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(2, offset="ms"), "data2")),
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(5, offset="ms"), "data3")),
            ("ETHUSDT", "trade", DummyEvent(get_event_dt(7, offset="s"), "data4")),
            ("ETHUSDT", "trade", DummyEvent(get_event_dt(7.9, offset="s"), "data4")),
            ("BTCUSDT", "ohlc", DummyEvent(get_event_dt(9, offset="s"), "data5")),
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(11, offset="s"), "data6")),
            ("ETHUSDT", "ohlc", DummyEvent(get_event_dt(12, offset="s"), "data5")),
            ("BTCUSDT", "trade", DummyEvent(get_event_dt(13, offset="s"), "data6")),
        ]
        expected_events = [
            (
                "BTCUSDT",
                "trade",
                [
                    DummyEvent(get_event_dt(1, offset="ms"), "data1"),
                    DummyEvent(get_event_dt(2, offset="ms"), "data2"),
                    DummyEvent(get_event_dt(5, offset="ms"), "data3"),
                ],
            ),
            (
                "ETHUSDT",
                "trade",
                [
                    DummyEvent(get_event_dt(7, offset="s"), "data4"),
                    DummyEvent(get_event_dt(7.9, offset="s"), "data4"),
                ],
            ),
            ("BTCUSDT", "ohlc", DummyEvent(get_event_dt(9, offset="s"), "data5")),
            ("BTCUSDT", "trade", [DummyEvent(get_event_dt(11, offset="s"), "data6")]),
            ("ETHUSDT", "ohlc", DummyEvent(get_event_dt(12, offset="s"), "data5")),
            ("BTCUSDT", "trade", [DummyEvent(get_event_dt(13, offset="s"), "data6")]),
        ]
        actual_output = list(EventBatcher(events))
        assert expected_events == actual_output
