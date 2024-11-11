import asyncio
import pytest
import numpy as np
import time
from unittest.mock import MagicMock, AsyncMock, patch
from qubx import lookup
from qubx.core.basics import Instrument, CtrlChannel
from qubx.connectors.ccxt.ccxt_connector import CCXTExchangesConnector


OHLCV_RESPONSE = {"ETH/USDT": {"5m": [[1731239700000, 3222.69, 3227.58, 3218.18, 3220.01, 2866.3094, 10000.0, 5000.0]]}}


async def async_sleep(*args, seconds: int = 1, **kwargs):
    await asyncio.sleep(seconds)


class MockExchange:
    def __init__(self):
        self.name = "mock_exchange"
        self.watch_ohlcv_for_symbols = AsyncMock()
        self.watch_ohlcv_for_symbols.return_value = OHLCV_RESPONSE
        self.watch_trades_for_symbols = AsyncMock()
        self.watch_order_book_for_symbols = AsyncMock()
        self.watch_orders = AsyncMock()
        self.watch_orders.side_effect = async_sleep
        self.find_timeframe = MagicMock(return_value="1m")
        self.fetch_ohlcv = AsyncMock(return_value=[])


class TestCcxtExchangeConnector:
    connector: CCXTExchangesConnector

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_exchange = MockExchange()
        self.fixed_time = np.datetime64("2023-01-01T00:00:00.000000000")
        self.mock_trading_service = MagicMock()
        self.mock_trading_service.time = MagicMock(return_value=self.fixed_time)

        # Create event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Patch NTP functions and exchange
        with (
            patch("qubx.utils.ntp.start_ntp_thread"),
            patch("qubx.utils.ntp.time_now", return_value=self.fixed_time),
            patch("ccxt.pro.binanceqv", return_value=self.mock_exchange),
        ):
            self.connector = CCXTExchangesConnector(
                exchange_id="binanceqv", trading_service=self.mock_trading_service, loop=self.loop
            )

        # return from setup
        yield

        # teardown
        self.loop.stop()

    def test_subscribe(self):
        # Create test instrument
        i1, i2 = lookup.find_symbol("BINANCE.UM", "BTCUSDT"), lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        assert i1 is not None and i2 is not None

        # Subscribe to different data types
        # self.connector.subscribe([i1, i2], "trade", warmup_period="1m")
        # self.connector.subscribe([i1], "orderbook", warmup_period="1m")
        # self.connector.subscribe([i2], "orderbook", warmup_period="1m")
        self.connector.subscribe([i2], "ohlc", warmup_period="24h", timeframe="15Min")
        self.connector.subscribe([i1], "ohlc", warmup_period="24h", timeframe="15Min")

        # Commit subscriptions
        self.connector.commit()

        channel = self.connector.get_communication_channel()
        for _ in range(4):
            ohlc = channel.receive(3)
            print(ohlc)

        # Verify subscriptions were added
        # assert i1 in self.connector._subscriptions["trade"]
        # assert i1 in self.connector._subscriptions["orderbook"]
        assert i1 in self.connector._subscriptions["ohlc"]

        # Verify exchange methods were called
        # self.mock_exchange.watch_ohlcv_for_symbols.assert_awaited()
