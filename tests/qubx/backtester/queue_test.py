import pandas as pd
from typing import Any, Iterator
from qubx.backtester.queue import SimulatedDataQueue, DataLoader


class DummyEvent:
    time: str
    data: str

    def __init__(self, time: str, data: str):
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

    def __hash__(self) -> int:
        return hash((self._symbol, "dummy"))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DummyDataLoader):
            return False
        return self._symbol == other._symbol and "dummy" == other._data_type


def get_event_dt(i: int, base: pd.Timestamp = pd.Timestamp("2021-01-01")) -> str:
    return str(base + pd.Timedelta(i, "D"))


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
            ("APPL", DummyEvent(get_event_dt(1), "data1")),
            ("MSFT", DummyEvent(get_event_dt(2), "data4")),
            ("APPL", DummyEvent(get_event_dt(3), "data2")),
            ("MSFT", DummyEvent(get_event_dt(4), "data5")),
            ("APPL", DummyEvent(get_event_dt(5), "data3")),
            ("MSFT", DummyEvent(get_event_dt(5), "data6")),
        ]
        actual_event_seq = list(q.create_iterator(get_event_dt(0), get_event_dt(10)))
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
        qiter = q.create_iterator(get_event_dt(0), get_event_dt(10))
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
            ("APPL", DummyEvent(get_event_dt(1), "data1")),
            ("MSFT", DummyEvent(get_event_dt(2), "data4")),
            ("APPL", DummyEvent(get_event_dt(3), "data2")),
            ("MSFT", DummyEvent(get_event_dt(4), "data5")),
            ("APPL", DummyEvent(get_event_dt(5), "data3")),
            ("MSFT", DummyEvent(get_event_dt(5), "data6")),
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
        qiter = q.create_iterator(get_event_dt(0), get_event_dt(10))
        for _ in range(3):
            actual_event_seq.append(next(qiter))
        q -= l2
        while True:
            try:
                actual_event_seq.append(next(qiter))
            except StopIteration:
                break

        expected_event_seq = [
            ("APPL", DummyEvent(get_event_dt(1), "data1")),
            ("MSFT", DummyEvent(get_event_dt(2), "data4")),
            ("APPL", DummyEvent(get_event_dt(3), "data2")),
            ("APPL", DummyEvent(get_event_dt(5), "data3")),
        ]
        assert expected_event_seq == actual_event_seq
