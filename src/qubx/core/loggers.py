from typing import Any, Dict, List, Tuple
from multiprocessing.pool import ThreadPool
import numpy as np
import csv, os

from qubx import logger
from qubx.core.basics import Deal, Position

from qubx.core.series import time_as_nsec
from qubx.core.utils import time_to_str, time_delta_to_str, recognize_timeframe
from qubx.utils.misc import makedirs, Stopwatch

_SW = Stopwatch()


class LogsWriter:
    account_id: str
    strategy_id: str
    run_id: str

    """
    Log writer interface with default implementation
    """
    def __init__(self, account_id: str, strategy_id: str, run_id: str) -> None:
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.run_id = run_id

    def write_data(self, log_type: str, data: List[Dict[str, Any]]):
        pass
    
    def flush_data(self):
        pass


class CsvFileLogsWriter(LogsWriter):
    """
    Simple CSV strategy log data writer. It does data writing in separate thread.
    """
    def __init__(self, account_id: str, strategy_id: str, run_id: str, log_folder='logs') -> None:
        super().__init__(account_id, strategy_id, run_id)

        path = makedirs(log_folder)
        # - it rewrites positions every time
        self._pos_file_path = f"{path}/{self.strategy_id}_{self.account_id}_positions.csv"
        self._balance_file_path = f"{path}/{self.strategy_id}_{self.account_id}_balance.csv"
        _pfl_path = f"{path}/{strategy_id}_{account_id}_portfolio.csv"
        _exe_path =  f"{path}/{strategy_id}_{account_id}_executions.csv"
        self._hdr_pfl = not os.path.exists(_pfl_path)
        self._hdr_exe = not os.path.exists(_exe_path)

        self._pfl_file_ = open(_pfl_path, "+a", newline='')
        self._execs_file_ = open(_exe_path, "+a", newline='')
        self._pfl_writer = csv.writer(self._pfl_file_)
        self._exe_writer = csv.writer(self._execs_file_)
        self.pool = ThreadPool(3)

    @staticmethod
    def _header(d: dict) -> List[str]:
        return list(d.keys()) + ['run_id']

    def _values(self, data: List[Dict[str, Any]]) -> List[List[str]]:
        # - attach run_id (last column)
        return [list((d | {'run_id': self.run_id}).values()) for d in data]

    def _do_write(self, log_type, data):
        match log_type:

            case 'positions':
                with open(self._pos_file_path, "w", newline='') as f:
                    w = csv.writer(f)
                    w.writerow(self._header(data[0]))
                    w.writerows(self._values(data))

            case 'portfolio':
                if self._hdr_pfl:
                    self._pfl_writer.writerow(self._header(data[0]))
                    self._hdr_pfl = False
                self._pfl_writer.writerows(self._values(data))
                self._pfl_file_.flush()

            case 'executions':
                if self._hdr_exe:
                    self._exe_writer.writerow(self._header(data[0]))
                    self._hdr_exe = False
                self._exe_writer.writerows(self._values(data))
                self._execs_file_.flush()

            case 'balance':
                with open(self._balance_file_path, "w", newline='') as f:
                    w = csv.writer(f)
                    w.writerow(self._header(data[0]))
                    w.writerows(self._values(data))

    def write_data(self, log_type: str, data: List[Dict[str, Any]]):
        if len(data) > 0:
            self.pool.apply_async(self._do_write, (log_type, data))

    def flush_data(self):
        try:
            self._pfl_file_.flush()
            self._execs_file_.flush()
        except Exception as e:
            logger.warning(f"Error flushing log writer: {str(e)}")


class _BaseIntervalDumper:
    """
    Basic functionality for all interval based dumpers
    """
    _last_log_time_ns: int
    _freq: np.timedelta64 | None
    
    def __init__(self, frequency: str | None) -> None:
        self._freq: np.timedelta64 | None = recognize_timeframe(frequency) if frequency else None
        self._last_log_time_ns = 0

    def store(self, timestamp: np.datetime64):
        _t_ns = time_as_nsec(timestamp)
        if self._freq:
            _interval_start_time = int(_t_ns  - _t_ns % self._freq)
            if _t_ns - self._last_log_time_ns >= self._freq:
                self.dump(np.datetime64(_interval_start_time, 'ns'), timestamp)
                self._last_log_time_ns = _interval_start_time
        else:
            self.dump(timestamp, timestamp)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        raise NotImplementedError(f"dump(np.datetime64, np.datetime64) must be implemented in {self.__class__.__name__}")


class PositionsDumper(_BaseIntervalDumper):
    """
    Positions dumper is designed to dump positions once per given interval to storage
    so we could check current situation.
    """
    positions: Dict[str, Position]
    _writer: LogsWriter

    def __init__(
        self, 
        writer: LogsWriter,
        interval: str, 
    ) -> None:
        super().__init__(interval)
        self.positions = dict()
        self._writer = writer

    def attach_positions(self, *positions: Position) -> 'PositionsDumper':
        for p in positions:
            self.positions[p.instrument.symbol] = p
        return self

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        data = []
        for s, p in self.positions.items():
            data.append({
                'timestamp': str(actual_timestamp),
                'instrument_id': s,
                'pnl_quoted': p.total_pnl(),
                'quantity': p.quantity,
                'realized_pnl_quoted': p.r_pnl,
                'avg_position_price': p.position_avg_price if p.quantity != 0.0 else 0.0,
                'current_price': p.last_update_price,
                'market_value_quoted': p.market_value_funds
            })
        self._writer.write_data('positions', data)


class PortfolioLogger(PositionsDumper):
    """
    Portfolio logger - save portfolio records into storage
    """
    def __init__(self, writer: LogsWriter, interval: str) -> None:
        super().__init__(writer, interval)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        data = []
        for s, p in self.positions.items():
            data.append({
                'timestamp': str(interval_start_time),
                'instrument_id': s,
                'pnl_quoted': p.total_pnl(),
                'quantity': p.quantity,
                'realized_pnl_quoted': p.r_pnl,
                'avg_position_price': p.position_avg_price if p.quantity != 0.0 else 0.0,
                'current_price': p.last_update_price,
                'market_value_quoted': p.market_value_funds,
                'exchange_time': str(actual_timestamp),
                'commissions_quoted': p.commissions,
            })
        self._writer.write_data('portfolio', data)

    def close(self):
        self._writer.flush_data()


class ExecutionsLogger(_BaseIntervalDumper):
    """
    Executions logger - save strategy executions into storage
    """
    _writer: LogsWriter
    _deals: List[Tuple[str, Deal]]

    def __init__(self, writer: LogsWriter, max_records=10) -> None:
        super().__init__(None) # no intervals
        self._writer = writer
        self._max_records = max_records
        self._deals: List[Tuple[str, Deal]] = []
        
    def record_deals(self, symbol: str, deals: List[Deal]):
        for d in deals:
            self._deals.append((symbol, d))
            l_time = d.time

        if len(self._deals) >= self._max_records:
            self.dump(l_time, l_time)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        data = []
        for s, d in self._deals:
            data.append({
                'timestamp': d.time,
                'instrument_id': s,
                'side': 'buy' if d.amount > 0 else 'sell',
                'filled_qty': d.amount,
                'price': d.price,
                'commissions': d.fee_amount,
                'commissions_quoted': d.fee_currency,
            })
        self._deals.clear()
        self._writer.write_data('executions', data)

    def store(self, timestamp: np.datetime64):
        pass

    def close(self):
        if self._deals:
            t = self._deals[-1][1].time
            self.dump(t, t)
        self._writer.flush_data()


class BalanceLogger(_BaseIntervalDumper):
    """
    Balance logger - send balance on strategy start
    """
    _writer: LogsWriter

    def __init__(self, writer: LogsWriter) -> None:
        super().__init__(None) # no intervals
        self._writer = writer
        
    def record_balance(self, timestamp: np.datetime64, balance: Dict[str, Tuple[float, float]]):
        if balance:
            data = []
            for s, d in balance.items():
                data.append({
                    'timestamp': timestamp,
                    'instrument_id': s,
                    'total': d[0],
                    'locked': d[1],
                })
            self._writer.write_data('balance', data)

    def store(self, timestamp: np.datetime64):
        pass

    def close(self):
        self._writer.flush_data()


class StrategyLogging:
    """
    Just combined loggers functionality
    """
    positions_dumper: PositionsDumper | None = None
    portfolio_logger: PortfolioLogger | None = None
    executions_logger: ExecutionsLogger | None = None
    balance_logger: BalanceLogger | None

    def __init__(
        self, 
        logs_writer: LogsWriter | None = None,
        positions_log_freq: str = '1Min',
        portfolio_log_freq: str = '5Min',
        num_exec_records_to_write = 1      # in live let's write every execution 
    ) -> None:
        # - instantiate loggers
        if logs_writer:
            if positions_log_freq:
                # - store current positions
                self.positions_dumper = PositionsDumper(logs_writer, positions_log_freq)

            if portfolio_log_freq:
                # - store portfolio log
                self.portfolio_logger = PortfolioLogger(logs_writer, portfolio_log_freq)

            # - store executions
            if num_exec_records_to_write >= 1:
                self.executions_logger = ExecutionsLogger(logs_writer, num_exec_records_to_write)

            # - balance logger
            self.balance_logger = BalanceLogger(logs_writer)
        else:
            logger.warning("Log writer is not defined - strategy activity will not be saved !")

    def initialize(self, 
                   timestamp: np.datetime64, 
                   positions: Dict[str, Position],
                   balances: Dict[str, Tuple[float, float]]
    ) -> None:
        # - attach positions to loggers
        if self.positions_dumper:
            self.positions_dumper.attach_positions(*list(positions.values()))

        if self.portfolio_logger:
            self.portfolio_logger.attach_positions(*list(positions.values()))

        # - send balance on start
        if self.balance_logger:
            self.balance_logger.record_balance(timestamp, balances)

    def close(self):
        if self.portfolio_logger:
            self.portfolio_logger.close()

        if self.executions_logger:
            self.executions_logger.close()

    @_SW.watch('loggers')
    def notify(self, timestamp: np.datetime64):
        # - notify position logger
        if self.positions_dumper:
            self.positions_dumper.store(timestamp)

        # - notify portfolio records logger
        if self.portfolio_logger:
            self.portfolio_logger.store(timestamp)

    def save_deals(self, symbol: str, deals: List[Deal]):
        if self.executions_logger:
            self.executions_logger.record_deals(symbol, deals)
