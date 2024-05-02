import signal, time
import asyncio, sys
from threading import Thread, Event, Lock
from queue import Queue
from typing import Awaitable, List, TypeVar
from ccxt import NetworkError
import stackprinter

from qubx import logger
from qubx.impl.ccxt_customizations import BinanceQV
import ccxt

import configparser

# def sigint_handler(signal, frame):
    # sys.stderr.write('\nInterrupted')
# signal.signal(signal.SIGINT, sigint_handler)

def get_account_auth(acc_name):
    parser = configparser.ConfigParser()
    parser.optionxform=str # type: ignore
    parser.read('.env')
    return dict(parser[acc_name])

T = TypeVar("T")

class CtrlChannel:
    def __init__(self) -> None:
        self.control = Event()
        self.queue = Queue()
    

# def _start_background_loop(loop: asyncio.AbstractEventLoop):
#     asyncio.set_event_loop(loop)
#     loop.run_forever()
# _DATA_LOOP = asyncio.new_event_loop()
# _DATA_LOOP_THREAD = Thread(target=_start_background_loop, args=(_DATA_LOOP,), daemon=True)
# _DATA_LOOP_THREAD.start()


def asyncio_run(coro: Awaitable[T], timeout=30) -> T:
    """
    Runs the coroutine in an event loop running on a background thread,
    and blocks the current thread until it returns a result.
    This plays well with gevent, since it can yield on the Future result call.

    :param coro: A coroutine, typically an async method
    :param timeout: How many seconds we should wait for a result before raising an error
    """
    return asyncio.run_coroutine_threadsafe(coro, _DATA_LOOP).result(timeout=timeout)


def asyncio_gather(*futures, return_exceptions=False) -> list:
    """
    A version of asyncio.gather that runs on the internal event loop
    """
    async def gather():
        return await asyncio.gather(*futures, return_exceptions=return_exceptions)

    return asyncio.run_coroutine_threadsafe(gather(), loop=_DATA_LOOP).result()


class Conn:
    def __init__(self, channel: CtrlChannel) -> None:
        self._DATA_LOOP = asyncio.new_event_loop()
        self.exchange = BinanceQV(
            get_account_auth('binance-mde') | {'asyncio_loop' : self._DATA_LOOP}
        )
        self.sync = ccxt.binance(get_account_auth('binance-mde'))
        self.sync.load_markets()
        self.channel = channel

    def subscirbe(self, symbols: List[str]):
        for s in symbols:
            asyncio.run_coroutine_threadsafe(self._listen_to_ohlcv(s), self._DATA_LOOP)

        for s in symbols:
            asyncio.run_coroutine_threadsafe(self._listen_to_execution_reports(s), self._DATA_LOOP)

        Thread(target=self._DATA_LOOP.run_forever, args=(), daemon=True).start()
        # Thread(target=self._SYNC_LOOP.run_forever, args=(), daemon=True).start()

    async def _listen_to_ohlcv(self, symbol: str):
        logger.info(f"(Con) start listening to ohlc : {symbol} | {id(self.exchange.asyncio_loop)}")
        ohlcv_i = await self.exchange.fetch_ohlcv(symbol, '1m', limit=10)        # type: ignore
        logger.info(f"Received initial snapshot : {len(ohlcv_i)}")
        while self.channel.control.is_set():
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, '1m')        # type: ignore
                logger.info(f"RECEIVED {symbol} {ohlcv[0]}")
                [self.channel.queue.put((symbol, (oh[0], oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))for oh in ohlcv]
            except Exception as e:
                logger.error(f"(Con) exception in _listen_to_ohlcv : {e}")
                logger.error(stackprinter.format(e))
                await self.exchange.close()        # type: ignore
                raise e
        logger.info(f"STOPPED 20")

    async def _listen_to_execution_reports(self, symbol: str):
        logger.info(f"(Con) start listening to execs : {symbol} | {id(self.exchange.asyncio_loop)}")
        while self.channel.control.is_set():
            try:
                exec = await self.exchange.watch_orders(symbol)        # type: ignore
                for report in exec:
                    self.channel.queue.put((symbol, report))
            except Exception as err:
                logger.error(f"(CCXTConnector) exception in _listen_to_execution_reports : {err}")
                logger.error(stackprinter.format(err))
        logger.info(f"STOPPED 21")

    def trade(self, symbol, size):
        logger.info(f" gonna trade {symbol} ~ {size}")
        # r = asyncio.run_coroutine_threadsafe(self.exch_data.fetch_bids_asks([symbol]), self._SYNC_LOOP)
        # r = self._SYNC_LOOP.create_task(self.exch_trade.fetch_bids_asks([symbol]))
        # r.add_done_callback(
            # lambda r: print('DDDDDDDDDDD', r.result())
        # )
        # r = self._SYNC_LOOP.run_until_complete(self.exch_data.fetch_bids_asks([symbol]))
        # print('--RES----> ', r.result())
        # self.exchange.fetch_bids_asks([symbols])
        # r = self.sync.create_order(symbol, 'market', 'sell', )
        # print('--RES----> ', r)

    # def _start_background_loop(self):
        # asyncio.set_event_loop(loop)
        # self._DATA_LOOP.run_forever()

N = 3
class Ctx:
    def __init__(self, symbols: List[str]) -> None:
        self.channel = CtrlChannel()
        self.channel.control.set()
        self.symbols = symbols
        self.conn = Conn(self.channel)

    def _context_loop(self):
        global N
        n = N
        while self.channel.control.is_set():
            symbol, data = self.channel.queue.get()
            n -= 1
            if n <= 0:
                self.conn.trade(symbol, 100)
                n = N
                continue
            logger.info(f" ---> {symbol} ::: {data}")
        logger.info(f"STOPPED 1")

    def start(self):
        self.conn.subscirbe(self.symbols)
        logger.info("Subscribed")
        TRADING_THREAD = Thread(target=self._context_loop, args=(), daemon=True)
        TRADING_THREAD.start()

def main():
    try:
        ctx = Ctx(['BTCUSDT', 'ETHUSDT'])
        ctx.start()
        while True: time.sleep(1000)

    except KeyboardInterrupt:
        ctx.channel.control.clear()
        ctx.channel.queue.put((None, None))
        time.sleep(1)
        sys.exit(0)

if __name__ == '__main__': main()