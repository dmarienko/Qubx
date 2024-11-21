import pytest
from unittest.mock import Mock, call
from qubx import lookup
from qubx.core.mixins.subscription import SubscriptionManager
from qubx.core.basics import Instrument, Subtype


class TestSubscriptionStuff:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.exchange = "BINANCE.UM"
        self.mock_broker = Mock()
        self.mock_broker.is_simulated_trading = False
        self.mock_broker.get_subscribed_instruments.return_value = set()
        self.manager = SubscriptionManager(self.mock_broker)

    def _get_instrument(self, symbol: str) -> Instrument:
        instr = lookup.find_symbol(self.exchange, symbol)
        assert instr is not None
        return instr

    def test_basic_subscription(self):
        instrument = self._get_instrument("BTCUSDT")
        self.manager.subscribe(Subtype.ORDERBOOK, instrument)
        self.manager.commit()

        self.mock_broker.subscribe.assert_called_once_with(Subtype.ORDERBOOK, {instrument}, reset=True)

    def test_warmup_subscription(self):
        instrument = self._get_instrument("BTCUSDT")
        warmup_config = {Subtype.ORDERBOOK: "1d"}
        self.manager.set_warmup(warmup_config)
        self.manager.subscribe(Subtype.ORDERBOOK, instrument)
        self.manager.commit()

        expected_warmup = {(Subtype.ORDERBOOK, instrument): "1d"}
        self.mock_broker.warmup.assert_called_once_with(expected_warmup)

    def test_multiple_subscriptions(self):
        instruments = [self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")]
        self.manager.subscribe(Subtype.ORDERBOOK, instruments)
        self.manager.subscribe(Subtype.TRADE, instruments[0])
        self.manager.commit()

        expected_calls = [
            call(Subtype.ORDERBOOK, set(instruments), reset=True),
            call(Subtype.TRADE, set([instruments[0]]), reset=True),
        ]
        self.mock_broker.subscribe.assert_has_calls(expected_calls, any_order=True)

    def test_unsubscribe(self):
        instrument = self._get_instrument("BTCUSDT")
        self.mock_broker.get_subscribed_instruments.return_value = {instrument}

        self.manager.unsubscribe(Subtype.ORDERBOOK, instrument)
        self.manager.commit()

        self.mock_broker.subscribe.assert_called_once_with(Subtype.ORDERBOOK, set(), reset=True)

    def test_global_subscription(self):
        instruments = {self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")}
        self.mock_broker.get_subscribed_instruments.side_effect = lambda x=None: (instruments if x is None else set())

        self.manager.set_warmup({Subtype.TRADE: "1d"})
        self.manager.subscribe(Subtype.TRADE)
        self.manager.subscribe(Subtype.ORDERBOOK)
        self.manager.commit()

        self.mock_broker.warmup.assert_called_once_with({(Subtype.TRADE, i): "1d" for i in instruments})

        expected_calls = [
            call(Subtype.TRADE, instruments, reset=True),
            call(Subtype.ORDERBOOK, instruments, reset=True),
        ]
        self.mock_broker.subscribe.assert_has_calls(expected_calls, any_order=True)

    def test_subscribe_all(self):
        instruments = {self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")}
        self.mock_broker.get_subscribed_instruments.return_value = set()
        self.mock_broker.get_subscriptions.return_value = [Subtype.TRADE, Subtype.ORDERBOOK]

        self.manager.subscribe(Subtype.ALL, list(instruments))
        self.manager.commit()

        expected_calls = [
            call(Subtype.TRADE, instruments, reset=True),
            call(Subtype.ORDERBOOK, instruments, reset=True),
        ]
        self.mock_broker.subscribe.assert_has_calls(expected_calls, any_order=True)

    def test_ohlc_warmup(self):
        instruments = {self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")}
        self.mock_broker.get_subscribed_instruments.return_value = set()
        self.mock_broker.get_subscriptions.return_value = [Subtype.OHLC]

        # make sure that ohlc warmups are called even if base subscription is not ohlc
        self.manager.set_base_subscription(Subtype.TRADE)
        self.manager.set_warmup({Subtype.OHLC["1h"]: "30d", Subtype.OHLC["1m"]: "1d", Subtype.TRADE: "10m"})
        self.manager.subscribe(self.manager.get_base_subscription(), list(instruments))
        self.manager.commit()

        assert self.manager.get_base_subscription() == Subtype.TRADE

        expected_warmup = (
            {(Subtype.OHLC["1h"], i): "30d" for i in instruments}
            | {(Subtype.OHLC["1m"], i): "1d" for i in instruments}
            | {(Subtype.TRADE, i): "10m" for i in instruments}
        )
        self.mock_broker.warmup.assert_called_once_with(expected_warmup)
