from qubx.pandaz.utils import *

from qubx.core.strategy import IStrategy, StrategyContext
from qubx.core.context import StrategyContextImpl
from qubx.core.loggers import InMemoryLogsWriter
from qubx.data.readers import CsvStorageDataReader, DataReader
from qubx.backtester.simulator import simulate, SimulatedTrading, SimulatedExchange, find_instruments_and_exchanges


def run_debug_sim(
    strategy_id: str,
    strategy: IStrategy,
    data_reader: DataReader,
    exchange: str,
    instruments: list[str],
    trigger: str,
    subscription: dict,
    commissions: str | None,
    start: str,
    stop: str,
    initial_capital: float,
    base_currency: str,
) -> StrategyContextImpl:
    broker = SimulatedTrading(exchange, commissions, np.datetime64(start, "ns"))
    broker = SimulatedExchange(exchange, broker, data_reader)
    logs_writer = InMemoryLogsWriter("test", strategy_id, "0")
    _instr, _ = find_instruments_and_exchanges(instruments, exchange)
    ctx = StrategyContextImpl(
        strategy=strategy,
        config=None,
        broker_connector=broker,
        initial_capital=initial_capital,
        base_currency=base_currency,
        instruments=_instr,
        md_subscription=subscription,
        trigger_spec=trigger,
        logs_writer=logs_writer,
    )
    ctx.start()
    broker.run(start, stop)
    return ctx


class TestAccountProcessorStuff:
    def test_account_basics(self):
        initial_capital = 10_000

        ctx = run_debug_sim(
            strategy_id="test0",
            strategy=IStrategy(),
            data_reader=CsvStorageDataReader("tests/data/csv"),
            exchange="BINANCE.UM",
            instruments=["BTCUSDT"],
            trigger="1H -1Sec",
            subscription=dict(type="ohlc", timeframe="1h", nback=0),
            commissions=None,
            start="2024-01-01",
            stop="2024-01-02",
            initial_capital=initial_capital,
            base_currency="USDT",
        )

        logs_writer = ctx._logs_writer
        assert isinstance(logs_writer, InMemoryLogsWriter)

        # 1. Check account in the beginning
        assert 0 == ctx.acc.get_net_leverage()
        assert 0 == ctx.acc.get_gross_leverage()
        assert initial_capital == ctx.acc.get_free_capital()
        assert initial_capital == ctx.acc.get_total_capital()

        # 2. Execute a trade and check account
        leverage = 0.5
        s = "BTCUSDT"
        quote = ctx.quote(s)
        capital = ctx.acc.get_total_capital()
        amount_in_base = capital * leverage
        amount = ctx.instruments[0].round_size_down(amount_in_base / quote.mid_price())
        leverage_adj = amount * quote.ask / capital
        ctx.trade("BTCUSDT", amount)

        # make the assertions work for floats
        assert np.isclose(leverage_adj, ctx.acc.get_net_leverage())
        assert np.isclose(leverage_adj, ctx.acc.get_gross_leverage())
        assert np.isclose(
            initial_capital - amount * quote.ask,
            ctx.acc.get_free_capital(),
        )
        assert np.isclose(initial_capital, ctx.acc.get_total_capital())

        # 3. Exit trade and check account
        ctx.trade("BTCUSDT", -amount)

        # get tick size for BTCUSDT
        tick_size = ctx.instruments[0].min_tick
        trade_pnl = -tick_size / quote.ask * leverage
        new_capital = initial_capital * (1 + trade_pnl)

        assert 0 == ctx.acc.get_net_leverage()
        assert 0 == ctx.acc.get_gross_leverage()
        assert np.isclose(new_capital, ctx.acc.get_free_capital())
        assert ctx.acc.get_free_capital() == ctx.acc.get_total_capital()

    def test_commissions(self):
        initial_capital = 10_000

        ctx = run_debug_sim(
            strategy_id="test0",
            strategy=IStrategy(),
            data_reader=CsvStorageDataReader("tests/data/csv"),
            exchange="BINANCE.UM",
            instruments=["BTCUSDT"],
            trigger="1H -1Sec",
            subscription=dict(type="ohlc", timeframe="1h", nback=0),
            commissions="vip0_usdt",
            start="2024-01-01",
            stop="2024-01-02",
            initial_capital=initial_capital,
            base_currency="USDT",
        )

        logs_writer = ctx._logs_writer
        assert isinstance(logs_writer, InMemoryLogsWriter)

        leverage = 0.5
        s = "BTCUSDT"
        quote = ctx.quote(s)
        capital = ctx.acc.get_total_capital()
        amount_in_base = capital * leverage
        amount = ctx.instruments[0].round_size_down(amount_in_base / quote.mid_price())
        ctx.trade("BTCUSDT", amount)
        ctx.trade("BTCUSDT", -amount)

        execs = logs_writer.get_executions()
        commissions = execs.commissions
        assert not any(commissions.isna())
