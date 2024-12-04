from typing import Any, Callable, TypeAlias
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import stackprinter

from qubx import logger, lookup
from qubx.core.basics import CtrlChannel, ITimeProvider, Instrument
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IStrategy, PositionsTracker


StrategyOrSignals: TypeAlias = IStrategy | pd.DataFrame | pd.Series


class _Types(Enum):
    UKNOWN = "unknown"
    LIST = "list"
    TRACKER = "tracker"
    SIGNAL = "signal"
    STRATEGY = "strategy"
    SIGNAL_AND_TRACKER = "signal_and_tracker"
    STRATEGY_AND_TRACKER = "strategy_and_tracker"


def _type(obj: Any) -> _Types:
    if obj is None:
        t = _Types.UKNOWN
    elif isinstance(obj, (list, tuple)):
        t = _Types.LIST
    elif isinstance(obj, PositionsTracker):
        t = _Types.TRACKER
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        t = _Types.SIGNAL
    elif isinstance(obj, IStrategy):
        t = _Types.STRATEGY
    else:
        t = _Types.UKNOWN
    return t


def _is_strategy(obj):
    return _type(obj) == _Types.STRATEGY


def _is_tracker(obj):
    return _type(obj) == _Types.TRACKER


def _is_signal(obj):
    return _type(obj) == _Types.SIGNAL


def _is_signal_or_strategy(obj):
    return _is_signal(obj) or _is_strategy(obj)


@dataclass
class SimulationSetup:
    setup_type: _Types
    name: str
    generator: StrategyOrSignals
    tracker: PositionsTracker | None
    instruments: list[Instrument]
    exchange: str
    capital: float
    leverage: float
    base_currency: str
    commissions: str


class SimulatedLogFormatter:
    def __init__(self, time_provider: ITimeProvider):
        self.time_provider = time_provider

    def formatter(self, record):
        end = record["extra"].get("end", "\n")
        fmt = "<lvl>{message}</lvl>%s" % end
        if record["level"].name in {"WARNING", "SNAKY"}:
            fmt = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - %s" % fmt

        dt = self.time_provider.time()
        if isinstance(dt, int):
            now = pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            now = self.time_provider.time().astype("datetime64[us]").item().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # prefix = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [ <level>%s</level> ] " % record["level"].icon
        prefix = f"<lc>{now}</lc> [<level>{record['level'].icon}</level>] "

        if record["exception"] is not None:
            record["extra"]["stack"] = stackprinter.format(record["exception"], style="darkbg3")
            fmt += "\n{extra[stack]}\n"

        if record["level"].name in {"TEXT"}:
            prefix = ""

        return prefix + fmt


class SimulatedScheduler(BasicScheduler):
    def run(self):
        self._is_started = True
        _has_tasks = False
        _time = self.time_sec()
        for k in self._crons.keys():
            _has_tasks |= self._arm_schedule(k, _time)


class SimulatedCtrlChannel(CtrlChannel):
    """
    Simulated communication channel. Here we don't use queue but it invokes callback directly
    """

    _callback: Callable[[tuple], bool]

    def register(self, callback):
        self._callback = callback

    def send(self, data):
        # - when data is sent, invoke callback
        return self._callback.process_data(*data)

    def receive(self, timeout: int | None = None) -> Any:
        raise ValueError("This method should not be called in a simulated environment.")

    def stop(self):
        self.control.clear()

    def start(self):
        self.control.set()


def find_instruments_and_exchanges(
    instruments: list[str] | dict[str, list[str]], exchange: str | None
) -> tuple[list[Instrument], list[str]]:
    _instrs: list[Instrument] = []
    _exchanges = [] if exchange is None else [exchange.lower()]
    for i in instruments:
        match i:
            case str():
                _e, _s = i.split(":") if ":" in i else (exchange, i)
                assert _e is not None

                if exchange is not None and _e.lower() != exchange.lower():
                    logger.warning("Exchange from symbol's spec ({_e}) is different from requested: {exchange} !")

                if _e is None:
                    logger.warning(
                        "Can't extract exchange name from symbol's spec ({_e}) and exact exchange name is not provided - skip this symbol !"
                    )

                if (ix := lookup.find_symbol(_e, _s)) is not None:
                    _exchanges.append(_e.lower())
                    _instrs.append(ix)
                else:
                    logger.warning(f"Can't find instrument for specified symbol ({i}) - ignoring !")

            case Instrument():
                _exchanges.append(i.exchange)
                _instrs.append(i)

            case _:
                raise ValueError(f"Unsupported instrument type: {i}")
    return _instrs, _exchanges


def recognize_simulation_setups(
    name: str,
    configs: (
        StrategyOrSignals
        | dict[str, StrategyOrSignals | list[StrategyOrSignals | PositionsTracker]]
        | list[StrategyOrSignals | PositionsTracker]
        | tuple[StrategyOrSignals | PositionsTracker]
    ),
    instruments: list[Instrument],
    exchange: str,
    capital: float,
    leverage: float,
    basic_currency: str,
    commissions: str,
) -> list[SimulationSetup]:
    name_in_list = lambda n: any([n == i for i in instruments])

    def _check_signals_structure(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        if isinstance(s, pd.Series):
            if not name_in_list(s.name):
                raise ValueError(f"Can't find instrument for signal's name: '{s.name}'")

        if isinstance(s, pd.DataFrame):
            for col in s.columns:
                if not name_in_list(col):
                    raise ValueError(f"Can't find instrument for signal's name: '{col}'")
        return s

    def _pick_instruments(s: pd.Series | pd.DataFrame) -> list[Instrument]:
        if isinstance(s, pd.Series):
            _instrs = [i for i in instruments if s.name == i]

        elif isinstance(s, pd.DataFrame):
            _instrs = [i for i in instruments if i in list(s.columns)]

        else:
            raise ValueError("Invalid signals or strategy configuration")

        return list(_instrs)

    r = list()
    # fmt: off
    if isinstance(configs, dict):
        for n, v in configs.items():
            r.extend(
                recognize_simulation_setups(
                    name + "/" + n, v, instruments, exchange, capital, leverage, basic_currency, commissions
                )
            )

    elif isinstance(configs, (list, tuple)):
        if len(configs) == 2 and _is_signal_or_strategy(configs[0]) and _is_tracker(configs[1]):
            c0, c1 = configs[0], configs[1]
            _s = _check_signals_structure(c0)   # type: ignore

            if _is_signal(c0):
                _t = _Types.SIGNAL_AND_TRACKER

            if _is_strategy(c0):
                _t = _Types.STRATEGY_AND_TRACKER

            # - extract actual symbols that have signals
            r.append(
                SimulationSetup(
                    _t, name, _s, c1,   # type: ignore
                    _pick_instruments(_s) if _is_signal(c0) else instruments,
                    exchange, capital, leverage, basic_currency, commissions,
                )
            )
        else:
            for j, s in enumerate(configs):
                r.extend(
                    recognize_simulation_setups(
                        # name + "/" + str(j), s, instruments, exchange, capital, leverage, basic_currency, commissions
                        name, s, instruments, exchange, capital, leverage, basic_currency, commissions, # type: ignore
                    )
                )

    elif _is_strategy(configs):
        r.append(
            SimulationSetup(
                _Types.STRATEGY,
                name, configs, None, instruments,
                exchange, capital, leverage, basic_currency, commissions,
            )
        )

    elif _is_signal(configs):
        # - check structure of signals
        c1 = _check_signals_structure(configs)  # type: ignore
        r.append(
            SimulationSetup(
                _Types.SIGNAL,
                name, c1, None, _pick_instruments(c1),
                exchange, capital, leverage, basic_currency, commissions,
            )
        )

    # fmt: on
    return r
