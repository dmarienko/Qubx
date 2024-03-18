import typing
from threading import Lock, Event, RLock
from multiprocessing import Value, Lock as MpLock
from pyring.exceptions import SequenceNotFound, ReadCursorBlock
from pyring.ring_buffer import SimpleFactory, RingBufferInternal, RingFactory
from pyring.disruptor import DisruptorMethods


class DisruptorSubscriber:
    def __init__(self, ring_buffer: "MultiProducerDisruptor", read_cursor: int = 0):
        self._read_cursor = read_cursor
        self.__ring_buffer = ring_buffer
        self._read_cursor_barrier = Event()
        self._write_cursor_barrier = Event()

    def next(self, timeout: typing.Optional[float] = None):
        try:
            res = self.__ring_buffer._get(self._read_cursor)
        except SequenceNotFound:
            self._read_cursor_barrier.clear()
            success = self._read_cursor_barrier.wait(timeout=timeout)
            if not success:
                raise SequenceNotFound()

        res = self.__ring_buffer._get(self._read_cursor)

        # release the write barrier
        if not self._write_cursor_barrier.is_set():
            self._write_cursor_barrier.set()

        self._read_cursor += 1
        return res

    def unregister(self) -> None:
        self.__ring_buffer._unregister_subscriber()
        if not self._write_cursor_barrier.is_set():
            self._write_cursor_barrier.set()


class MultiProducerDisruptor(RingBufferInternal, DisruptorMethods):
    def __init__(self, size: int = 16, 
            factory: typing.Type[RingFactory] = SimpleFactory, 
            cursor_position_value: int = 0
    ):
        super().__init__(
            size=size, factory=factory, cursor_position_value=cursor_position_value
        )
        self._subscriber: DisruptorSubscriber = None

    def subscribe(self) -> DisruptorSubscriber:
        if self._subscriber is None:
            self._subscriber = DisruptorSubscriber(ring_buffer=self)
        return self._subscriber

    def _unregister_subscriber(self):
        self._subscriber = None

    def put(self, value, timeout: float = None):
        subscriber = self._subscriber

        if (self._get_cursor_position() - subscriber._read_cursor) == self.ring_size:
            subscriber._write_cursor_barrier.clear()
            success = subscriber._write_cursor_barrier.wait(timeout=timeout)
            if not success:
                raise ReadCursorBlock()

        # print(f" PUT ~ {value}", flush=True)

        result = super()._put(value)
        if (
            subscriber._read_cursor == result
            and not subscriber._read_cursor_barrier.is_set()
        ):
            subscriber._read_cursor_barrier.set()

        return result
