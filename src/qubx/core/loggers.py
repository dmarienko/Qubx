from typing import Any, Dict, List, Tuple
from enum import Enum
import numpy as np

from qubx.core.basics import Deal, Position

from qubx.core.series import time_as_nsec
from qubx.core.utils import time_to_str, time_delta_to_str, recognize_timeframe


class LogsWriter:
    """
    Log writer interface with default implementation
    """
    def write_data(self, log_type: str, strategy_id: str, account_id: str, data: List[Any]):
        pass


class _BaseIntervalDumper:
    """
    Basic functionality for all interval based dumpers
    """
    _last_log_time_ns: int
    _freq: np.timedelta64
    
    def __init__(self, frequency: str) -> None:
        self._freq: np.timedelta64 = recognize_timeframe(frequency)
        self._last_log_time_ns = 0

    def store(self, timestamp: np.datetime64):
        _t_ns = time_as_nsec(timestamp)
        _interval_start_time = int(_t_ns  - _t_ns % self._freq)
        if _t_ns - self._last_log_time_ns >= self._freq:
            self.dump(np.datetime64(_interval_start_time, 'ns'), timestamp)
            self._last_log_time_ns = _interval_start_time

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        raise NotImplementedError(f"dump(np.datetime64, np.datetime64) must be implemented in {self.__class__.__name__}")


class PositionsDumper(_BaseIntervalDumper):
    """
    Positions dumper is designed to dump positions once per given interval to storage
    so we could check current situation.
    """
    account_id: str
    strategy_id: str
    positions: Dict[str, Position]
    _writer: LogsWriter

    def __init__(
        self, account_id: str, strategy_id: str, interval: str, writer: LogsWriter
    ) -> None:
        super().__init__(interval)
        self.account_id = account_id
        self.strategy_id = strategy_id
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
        self._writer.write_data('positions', self.account_id, self.strategy_id, data)


class PortfolioLogger(PositionsDumper):
    """
    Portfolio logger - save portfolio records into storage
    """
    def __init__(self, account_id: str, strategy_id: str, interval: str, writer: LogsWriter) -> None:
        super().__init__(account_id, strategy_id, interval, writer)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        pass


class ExecutionsLogger(_BaseIntervalDumper):
    """
    Executions logger - save strategy executions into storage
    """
    account_id: str
    strategy_id: str 
    _writer: LogsWriter
    _deals: List[Tuple[str, Deal]]

    def __init__(self, account_id: str, strategy_id: str, interval: str, writer: LogsWriter) -> None:
        super().__init__(interval)
        self.account_id = account_id
        self.strategy_id = strategy_id
        self._writer = writer
        
    def add_deal(self, symbol: str, deal: Deal):
        self._deals.append((symbol, deal))

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
        self._writer.write_data('executions', self.account_id, self.strategy_id, data)
