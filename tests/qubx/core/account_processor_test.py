from typing import Any

import pytest

from qubx import lookup
from qubx.backtester.simulator import (
    SimulatedBroker,
    SimulatedCtrlChannel,
    SimulatedDataProvider,
    find_instruments_and_exchanges,
    simulate,
)
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import Instrument, ITimeProvider, dt_64
from qubx.core.context import StrategyContext
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.core.mixins.trading import TradingManager
from qubx.data.readers import CsvStorageDataReader, DataReader
from qubx.pandaz.utils import *
from tests.qubx.core.utils_test import DummyTimeProvider


def run_debug_sim(
    strategy_id: str,
    strategy: IStrategy,
    data_reader: DataReader,
    exchange: str,
    symbols: list[str],
    commissions: str | None,
    start: str,
    stop: str,
    initial_capital: float,
    base_currency: str,
) -> tuple[IStrategyContext, InMemoryLogsWriter]:
    broker = SimulatedBroker(exchange, commissions, np.datetime64(start, "ns"))
    broker = SimulatedDataProvider(exchange, broker, data_reader)
    instruments, _ = find_instruments_and_exchanges(symbols, exchange)
    account = BasicAccountProcessor(
        account_id=broker.get_trading_service().get_account_id(),
        base_currency=base_currency,
        initial_capital=initial_capital,
    )
    logs_writer = InMemoryLogsWriter(strategy_id, strategy_id, "0")
    strategy_logging = StrategyLogging(logs_writer)
    ctx = StrategyContext(
        strategy=strategy,
        data_provider=broker,
        account=account,
        instruments=instruments,
        logging=strategy_logging,
    )
    ctx.start()
    broker.run(start, stop)
    return ctx, logs_writer


class TestAccountProcessorStuff:
    INITIAL_CAPITAL = 100_000

    def get_instrument(self, exchange: str, symbol: str) -> Instrument:
        instr = lookup.find_symbol(exchange, symbol)
        assert instr is not None
        return instr

    @pytest.fixture
    def trading_manager(self) -> TradingManager:
        name = "test"
        account = BasicAccountProcessor(
            account_id=name,
            base_currency="USDT",
            initial_capital=self.INITIAL_CAPITAL,
        )
        trading_service = SimulatedBroker(account, name)

        channel = SimulatedCtrlChannel("data")
        trading_service.set_communication_channel(channel)

        class PrintCallback:
            def process_data(self, instrument: Instrument, d_type: str, data: Any):
                print(data)

        channel.register(PrintCallback())

        return TradingManager(DummyTimeProvider(), trading_service, name)

    def test_spot_account_processor(self, trading_manager: TradingManager):
        trading_service = trading_manager._trading_service
        account = trading_service.account

        # - check initial state
        assert account.get_total_capital() == self.INITIAL_CAPITAL
        assert account.get_capital() == self.INITIAL_CAPITAL
        assert account.get_net_leverage() == 0
        assert account.get_gross_leverage() == 0
        assert account.get_balances()["USDT"].free == self.INITIAL_CAPITAL
        assert account.get_balances()["USDT"].locked == 0
        assert account.get_balances()["USDT"].total == self.INITIAL_CAPITAL

        ##############################################
        # 1. Buy BTC on spot for half of the capital
        ##############################################
        i1 = self.get_instrument("BINANCE", "BTCUSDT")

        # - update instrument price
        trading_service.update_position_price(
            i1,
            trading_service.time(),
            100_000.0,
        )

        # - execute trade for half of the initial capital
        o1 = trading_manager.trade(i1, 0.5)

        pos = account.positions[i1]
        assert pos.quantity == 0.5
        assert pos.market_value == pytest.approx(50_000)
        assert account.get_net_leverage() == pytest.approx(0.5)
        assert account.get_gross_leverage() == pytest.approx(0.5)
        assert account.get_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_total_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_balances()["USDT"].free == pytest.approx(self.INITIAL_CAPITAL / 2)
        assert account.get_balances()["BTC"].free == pytest.approx(0.5)

        ##############################################
        # 2. Test locking and unlocking of funds
        ##############################################
        o2 = trading_manager.trade(i1, 0.1, price=90_000)
        assert account.get_balances()["USDT"].locked == pytest.approx(9_000)
        trading_manager.cancel_order(o2.id)
        assert account.get_balances()["USDT"].locked == pytest.approx(0)

        ##############################################
        # 3. Sell BTC on spot
        ##############################################
        # - update instrument price
        o2 = trading_manager.trade(i1, -0.5)

        assert account.get_net_leverage() == 0
        assert account.get_gross_leverage() == 0
        assert account.get_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_total_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_balances()["USDT"].free == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_balances()["BTC"].free == pytest.approx(0)

    def test_swap_account_processor(self, trading_manager: TradingManager):
        trading_service = trading_manager._trading_service
        account = trading_service.account

        i1 = self.get_instrument("BINANCE.UM", "BTCUSDT")

        trading_service.update_position_price(
            i1,
            trading_service.time(),
            100_000.0,
        )

        # - execute trade for half of the initial capital
        o1 = trading_manager.trade(i1, 0.5)
        pos = account.positions[i1]

        # - check that market value of the position is close to 0 for swap
        assert pos.quantity == 0.5
        assert pos.market_value == pytest.approx(0, abs=1)

        # - check that USDT balance is actually left untouched
        balances = account.get_balances()
        assert len(balances) == 1
        assert balances["USDT"].free == pytest.approx(self.INITIAL_CAPITAL)

        # - check margin requirements
        assert account.get_total_required_margin() == pytest.approx(50_000 * i1.maint_margin)

        # increase price 2x
        trading_service.update_position_price(
            i1,
            trading_service.time(),
            200_000.0,
        )

        assert pos.market_value == pytest.approx(50_000, abs=1)
        assert pos.maint_margin == pytest.approx(100_000 * i1.maint_margin)

        # liquidate position
        o2 = trading_manager.trade(i1, -0.5)
        assert balances["USDT"].free == pytest.approx(self.INITIAL_CAPITAL + 50_000)
        assert pos.quantity == pytest.approx(0, abs=i1.min_size)
        assert pos.market_value == pytest.approx(0)

    def test_account_basics(self):
        initial_capital = 10_000

        ctx, _ = run_debug_sim(
            strategy_id="test0",
            strategy=IStrategy(),
            data_reader=CsvStorageDataReader("tests/data/csv"),
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            commissions=None,
            start="2024-01-01",
            stop="2024-01-02",
            initial_capital=initial_capital,
            base_currency="USDT",
        )

        # 1. Check account in the beginning
        assert 0 == ctx.account.get_net_leverage()
        assert 0 == ctx.account.get_gross_leverage()
        assert initial_capital == ctx.account.get_capital()
        assert initial_capital == ctx.account.get_total_capital()

        # 2. Execute a trade and check account
        leverage = 0.5
        instrument = ctx.instruments[0]
        quote = ctx.quote(instrument)
        assert quote is not None
        capital = ctx.account.get_total_capital()
        amount_in_base = capital * leverage
        amount = ctx.instruments[0].round_size_down(amount_in_base / quote.mid_price())
        leverage_adj = amount * quote.ask / capital
        ctx.trade(instrument, amount)

        # make the assertions work for floats
        assert np.isclose(leverage_adj, ctx.account.get_net_leverage())
        assert np.isclose(leverage_adj, ctx.account.get_gross_leverage())
        assert np.isclose(
            initial_capital - amount * quote.ask,
            ctx.account.get_capital(),
        )
        assert np.isclose(initial_capital, ctx.account.get_total_capital())

        # 3. Exit trade and check account
        ctx.trade(instrument, -amount)

        # get tick size for BTCUSDT
        tick_size = ctx.instruments[0].tick_size
        trade_pnl = -tick_size / quote.ask * leverage
        new_capital = initial_capital * (1 + trade_pnl)

        assert 0 == ctx.account.get_net_leverage()
        assert 0 == ctx.account.get_gross_leverage()
        assert np.isclose(new_capital, ctx.account.get_capital())
        assert ctx.account.get_capital() == ctx.account.get_total_capital()

    def test_commissions(self):
        initial_capital = 10_000

        ctx, logs_writer = run_debug_sim(
            strategy_id="test0",
            strategy=IStrategy(),
            data_reader=CsvStorageDataReader("tests/data/csv"),
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            commissions="vip0_usdt",
            start="2024-01-01",
            stop="2024-01-02",
            initial_capital=initial_capital,
            base_currency="USDT",
        )

        leverage = 0.5
        s = ctx.instruments[0]
        quote = ctx.quote(s)
        assert quote is not None
        capital = ctx.account.get_total_capital()
        amount_in_base = capital * leverage
        amount = ctx.instruments[0].round_size_down(amount_in_base / quote.mid_price())
        ctx.trade(s, amount)
        ctx.trade(s, -amount)

        execs = logs_writer.get_executions()
        commissions = execs.commissions
        assert not any(commissions.isna())
