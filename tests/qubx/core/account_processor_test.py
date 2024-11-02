from qubx.pandaz.utils import *

from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.core.context import StrategyContext
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.data.readers import CsvStorageDataReader, DataReader
from qubx.core.account import AccountProcessor
from qubx.backtester.simulator import simulate, SimulatedTrading, SimulatedExchange, find_instruments_and_exchanges


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
    broker = SimulatedTrading(exchange, commissions, np.datetime64(start, "ns"))
    broker = SimulatedExchange(exchange, broker, data_reader)
    instruments, _ = find_instruments_and_exchanges(symbols, exchange)
    account = AccountProcessor(
        account_id=broker.get_trading_service().get_account_id(),
        base_currency=base_currency,
        initial_capital=initial_capital,
    )
    logs_writer = InMemoryLogsWriter(strategy_id, strategy_id, "0")
    strategy_logging = StrategyLogging(logs_writer)
    ctx = StrategyContext(
        strategy=strategy,
        broker=broker,
        account=account,
        instruments=instruments,
        logging=strategy_logging,
    )
    ctx.start()
    broker.run(start, stop)
    return ctx, logs_writer


class TestAccountProcessorStuff:
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
        tick_size = ctx.instruments[0].min_tick
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
