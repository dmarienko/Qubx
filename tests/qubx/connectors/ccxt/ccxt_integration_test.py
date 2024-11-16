import pytest
from qubx.connectors.ccxt.ccxt_connector import CCXTExchangesConnector

import time
import pandas as pd
from typing import Any, Callable
from pathlib import Path
from collections import defaultdict

from qubx import lookup, logger, QubxLogConfig
from qubx.core.basics import TriggerEvent, Trade, MarketEvent, Instrument, SubscriptionType
from qubx.core.interfaces import IStrategyContext, IStrategy
from qubx.connectors.ccxt.ccxt_connector import CCXTExchangesConnector
from qubx.connectors.ccxt.ccxt_trading import CCXTTradingConnector
from qubx.utils.runner import get_account_config
from qubx.pandaz import scols
from qubx.backtester.simulator import SimulatedTrading
from qubx.utils.runner import run_ccxt_paper_trading
from qubx.utils.collections import TimeLimitedDeque
from qubx.utils.runner import run_ccxt_trading


class DebugStrategy(IStrategy):
    _data_counter: int = 0

    def on_init(self, ctx: IStrategyContext):
        ctx.set_base_subscription(SubscriptionType.OHLC, timeframe="1m")
        ctx.set_warmup(SubscriptionType.OHLC, "1h")

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        self._data_counter += 1
        if self._data_counter % 1000 == 0:
            logger.debug(f"Processed {self._data_counter} data points")

    def on_universe_change(
        self, ctx: IStrategyContext, add_instruments: list[Instrument], rm_instruments: list[Instrument]
    ):
        if add_instruments:
            _sub_to_params = ctx.get_subscriptions(ctx.instruments[0])
            for sub, params in _sub_to_params.items():
                ctx.subscribe(add_instruments, sub, **params)


class TestCcxtExchangeIntegrations:

    @pytest.mark.integration
    @pytest.mark.parametrize("exchange,symbols", [("BINANCE.UM", ["BTCUSDT"])])
    def test_basic_exchange_functions(
        self, exchange: str, symbols: list[str], exchange_credentials: dict[str, dict[str, str]]
    ):
        QubxLogConfig.set_log_level("DEBUG")

        ctx = run_ccxt_trading(
            strategy=(stg := DebugStrategy()),
            exchange=exchange,
            symbols=symbols,
            credentials=exchange_credentials[exchange],
            blocking=False,
            use_testnet=True,
            commissions="vip0_usdt",
        )

        self._wait(10, ctx.is_fitted)

        i1 = ctx.instruments[0]
        logger.info(f"Working with instrument {i1}")
        pos = ctx.positions[i1]
        if pos.quantity != 0:
            logger.info(f"Found existing position quantity {pos.quantity}")
            logger.info(f"Closing position")
            ctx.trade(i1, -pos.quantity)
            self._wait(10, pos.is_open)
            logger.info(f"Closed position")

        logger.info(f"Position is {pos.quantity}")
        qty1 = pos.quantity

        # enter market
        amount = i1.min_size * 2
        price = ctx.ohlc(i1)[0].close
        logger.info(f"Entering market with {amount} at market price {price}")
        order1 = ctx.trade(i1, amount=amount)
        assert order1 is not None and order1.price is not None

        try:
            self._wait(10, lambda pos=pos: pos.quantity != qty1)
        except TimeoutError:
            assert pos.quantity != qty1, "Position was not updated"

        # close position
        assert pos.quantity == qty1 + amount
        logger.info(f"Closing position of {pos.quantity} for {i1}")
        ctx.trade(i1, -pos.quantity)

        self._wait(10, lambda pos=pos: not pos.is_open())

        ctx.stop()

    def _wait(self, timeout: int, condition: Callable[[], bool], period: float = 0.1):
        start = time.time()
        while time.time() - start < timeout:
            if condition():
                return
            time.sleep(period)
        raise TimeoutError("Timeout reached")
