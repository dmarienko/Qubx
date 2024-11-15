import pytest
from qubx.connectors.ccxt.ccxt_connector import CCXTExchangesConnector

import time
import pandas as pd
from typing import Any
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


@pytest.mark.integration
@pytest.mark.parametrize("exchange,symbols", [("BINANCE", ["BTCUSDT"])])
def test_basic_exchange_functions(exchange: str, symbols: list[str], exchange_credentials: dict[str, dict[str, str]]):
    QubxLogConfig.set_log_level("DEBUG")
    ctx = run_ccxt_trading(
        strategy=(stg := DebugStrategy()),
        exchange=exchange,
        symbols=symbols,
        credentials=exchange_credentials[exchange],
        blocking=False,
        use_testnet=True,
    )
    ctx.stop()
