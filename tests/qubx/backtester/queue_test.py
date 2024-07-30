from typing import Any, Iterator
from qubx.backtester.queue import SimulatedDataQueue, DataLoader


class Event:
    time: int
    data: str

    def __init__(self, time: int, data: str):
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


class TestSimulatedQueueStuff:

    def test_basic_event_sync(self):
        q = SimulatedDataQueue("1", "10")
        q += DummyDataLoader(
            "APPL",
            [
                [Event(1, "data1")],
                [Event(3, "data2"), Event(5, "data3")],
            ],
        )
        q += DummyDataLoader(
            "MSFT",
            [
                [
                    Event(2, "data4"),
                    Event(4, "data5"),
                    Event(5, "data6"),
                ]
            ],
        )
        expected_event_seq = [
            Event(1, "data1"),
            Event(2, "data4"),
            Event(3, "data2"),
            Event(4, "data5"),
            Event(5, "data3"),
            Event(5, "data6"),
        ]
        actual_event_seq = list(q)
        assert expected_event_seq == actual_event_seq
