from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeAlias

import pandas as pd
import stackprinter

from qubx import logger, lookup
from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider, TimestampedDict
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IStrategy, PositionsTracker
from qubx.core.series import OHLCV, Bar, Quote, Trade
from qubx.core.utils import time_delta_to_str
from qubx.data.readers import AsDict, DataReader, InMemoryDataFrameReader
from qubx.utils.time import infer_series_frequency

StrategyOrSignals: TypeAlias = IStrategy | pd.DataFrame | pd.Series
DictOfStrats: TypeAlias = dict[str, StrategyOrSignals]
VariableStrategyConfig: TypeAlias = (
    StrategyOrSignals
    | DictOfStrats
    | dict[str, DictOfStrats]
    | dict[str, StrategyOrSignals | list[StrategyOrSignals | PositionsTracker]]
    | list[StrategyOrSignals | PositionsTracker]
    | tuple[StrategyOrSignals | PositionsTracker]
)


class SetupTypes(Enum):
    UKNOWN = "unknown"
    LIST = "list"
    TRACKER = "tracker"
    SIGNAL = "signal"
    STRATEGY = "strategy"
    SIGNAL_AND_TRACKER = "signal_and_tracker"
    STRATEGY_AND_TRACKER = "strategy_and_tracker"


def _type(obj: Any) -> SetupTypes:
    if obj is None:
        t = SetupTypes.UKNOWN
    elif isinstance(obj, (list, tuple)):
        t = SetupTypes.LIST
    elif isinstance(obj, PositionsTracker):
        t = SetupTypes.TRACKER
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        t = SetupTypes.SIGNAL
    elif isinstance(obj, IStrategy):
        t = SetupTypes.STRATEGY
    else:
        t = SetupTypes.UKNOWN
    return t


def _is_strategy(obj):
    return _type(obj) == SetupTypes.STRATEGY


def _is_tracker(obj):
    return _type(obj) == SetupTypes.TRACKER


def _is_signal(obj):
    return _type(obj) == SetupTypes.SIGNAL


def _is_signal_or_strategy(obj):
    return _is_signal(obj) or _is_strategy(obj)


@dataclass
class SimulationSetup:
    setup_type: SetupTypes
    name: str
    generator: StrategyOrSignals
    tracker: PositionsTracker | None
    instruments: list[Instrument]
    exchange: str
    capital: float
    leverage: float
    base_currency: str
    commissions: str

    def __str__(self) -> str:
        return f"{self.name} {self.setup_type} capital {self.capital} {self.base_currency} for [{','.join(map(lambda x: x.symbol, self.instruments))}] @ {self.exchange}[{self.commissions}]"


@dataclass
class SimulationDataInfo:
    # TODO: ...
    timeframe: str | None
    pass


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


def recognize_simulation_configuration(
    name: str,
    configs: VariableStrategyConfig,
    instruments: list[Instrument],
    exchange: str,
    capital: float,
    leverage: float,
    basic_currency: str,
    commissions: str,
) -> list[SimulationSetup]:
    """
    Recognize and create setups based on the provided simulation configuration.

    This function processes the given configuration and creates a list of SimulationSetup
    objects that represent different simulation scenarios. It handles various types of
    configurations including dictionaries, lists, signals, and strategies.

    Parameters:
    name (str): The name of the simulation setup.
    configs (VariableStrategyConfig): The configuration for the simulation. Can be a
        strategy, signals, or a nested structure of these.
    instruments (list[Instrument]): List of available instruments for the simulation.
    exchange (str): The name of the exchange to be used.
    capital (float): The initial capital for the simulation.
    leverage (float): The leverage to be used in the simulation.
    basic_currency (str): The base currency for the simulation.
    commissions (str): The commission structure to be applied.

    Returns:
    list[SimulationSetup]: A list of SimulationSetup objects, each representing a
        distinct simulation configuration based on the input parameters.

    Raises:
    ValueError: If the signal structure is invalid or if an instrument cannot be found
        for a given signal.
    """

    def _possible_instruments_ids(i: Instrument) -> set[str]:
        return set((i.symbol, str(i), f"{i.exchange}:{i.symbol}"))

    def _name_in_instruments(n, instrs: list[Instrument]) -> bool:
        return any([n in _possible_instruments_ids(i) for i in instrs])

    def _check_signals_structure(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        if isinstance(s, pd.Series):
            # - it's possible to put anything to series name, so we convert it to string
            s.name = str(s.name)
            if not _name_in_instruments(s.name, instruments):
                raise ValueError(f"Can't find instrument for signal's name: '{s.name}'")

        if isinstance(s, pd.DataFrame):
            s.columns = s.columns.map(lambda x: str(x))
            for col in s.columns:
                if not _name_in_instruments(col, instruments):
                    raise ValueError(f"Can't find instrument for signal's name: '{col}'")
        return s

    def _pick_instruments(s: pd.Series | pd.DataFrame) -> list[Instrument]:
        if isinstance(s, pd.Series):
            _instrs = [i for i in instruments if s.name in _possible_instruments_ids(i)]

        elif isinstance(s, pd.DataFrame):
            _s_cols = set(s.columns)
            _instrs = [i for i in instruments if _possible_instruments_ids(i) & _s_cols]

        else:
            raise ValueError("Invalid signals or strategy configuration")

        return list(set(_instrs))

    r = list()

    # fmt: off
    if isinstance(configs, dict):
        for n, v in configs.items():
            _n = (name + "/") if name else ""
            r.extend(
                recognize_simulation_configuration(
                    _n + n, v, instruments, exchange, capital, leverage, basic_currency, commissions
                )
            )

    elif isinstance(configs, (list, tuple)):
        if len(configs) == 2 and _is_signal_or_strategy(configs[0]) and _is_tracker(configs[1]):
            c0, c1 = configs[0], configs[1]
            _s = _check_signals_structure(c0)   # type: ignore

            if _is_signal(c0):
                _t = SetupTypes.SIGNAL_AND_TRACKER

            if _is_strategy(c0):
                _t = SetupTypes.STRATEGY_AND_TRACKER

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
                    recognize_simulation_configuration(
                        # name + "/" + str(j), s, instruments, exchange, capital, leverage, basic_currency, commissions
                        name, s, instruments, exchange, capital, leverage, basic_currency, commissions, # type: ignore
                    )
                )

    elif _is_strategy(configs):
        r.append(
            SimulationSetup(
                SetupTypes.STRATEGY,
                name, configs, None, instruments,
                exchange, capital, leverage, basic_currency, commissions,
            )
        )

    elif _is_signal(configs):
        # - check structure of signals
        c1 = _check_signals_structure(configs)  # type: ignore
        r.append(
            SimulationSetup(
                SetupTypes.SIGNAL,
                name, c1, None, _pick_instruments(c1),
                exchange, capital, leverage, basic_currency, commissions,
            )
        )

    # fmt: on
    return r


class DataSniffer:
    _probe_size: int

    def __init__(self, _probe_size: int = 50) -> None:
        self._probe_size = _probe_size

    def _has_columns(self, v: pd.DataFrame, columns: list[str]):
        return all([c in v.columns for c in columns])

    def _has_keys(self, v: dict[str, Any], keys: list[str]):
        return all([c in v.keys() for c in keys])

    def _sniff_list(self, v: list[Any]) -> str:
        match v[0]:
            case Bar():
                _tf = time_delta_to_str(infer_series_frequency([x.time for x in v[: self._probe_size]]).item())
                return DataType.OHLC[_tf]

            case dict():
                return self._sniff_dicts(v)

            case Quote():
                return DataType.QUOTE

            case TimestampedDict():
                return DataType.RECORD

            case Trade():
                return DataType.TRADE

        return DataType.NONE

    def _sniff_dicts(self, v: dict[str, Any] | list[dict[str, Any]]) -> str:
        v, vs = (v[0], v) if isinstance(v, list) else (v, None)

        if self._has_keys(v, ["open", "high", "low", "close"]):
            if vs:
                _tf = time_delta_to_str(infer_series_frequency([x.get("time") for x in vs[: self._probe_size]]).item())
                return DataType.OHLC[_tf]
            return DataType.OHLC

        if self._has_keys(v, ["bid", "ask"]):
            return DataType.QUOTE

        if self._has_keys(v, ["price", "size"]):
            return DataType.TRADE

        return DataType.NONE

    def _sniff_pandas(self, v: pd.DataFrame) -> str:
        if self._has_columns(v, ["open", "high", "low", "close"]):
            _tf = time_delta_to_str(infer_series_frequency(v[: self._probe_size]).item())
            return DataType.OHLC[_tf]

        if self._has_columns(v, ["bid", "ask"]):
            return DataType.QUOTE

        if self._has_columns(v, ["price", "size"]):
            return DataType.TRADE

        return DataType.NONE

    def _pre_read(self, symbol: str, reader: DataReader, time: str) -> list[Any]:
        for dt in ["2h", "12h", "2d", "28d", "60d", "720d"]:
            try:
                _it = reader.read(
                    symbol,
                    transform=AsDict(),
                    start=time,
                    stop=pd.Timestamp(time) + pd.Timedelta(dt),  # type: ignore
                    timeframe=None,
                    chunksize=self._probe_size,
                )
                if len(data := next(_it)) >= 2:  # type: ignore
                    return data
            except Exception:
                pass
        return []

    def _sniff_reader(self, symbol: str, reader: DataReader, time: str | None) -> str:
        if time is None:
            for _type in [DataType.OHLC, DataType.QUOTE, DataType.TRADE]:
                _t1, _t2 = reader.get_time_ranges(symbol, str(_type))
                if _t1 is not None:
                    time = str(_t1 + (_t2 - _t1) / 2)
                    break
            else:
                logger.warning(f"Failed to find data start time for symbol: {symbol}")
                return DataType.NONE

        data = self._pre_read(symbol, reader, time)
        if data:
            return self._sniff_list(data)

        logger.warning(f"Failed to read probe data for symbol: {symbol}")
        return DataType.NONE

    def extract_types(
        self,
        data: dict[str, Any],
        time: str | None = None,
    ) -> dict[str, str]:
        """
        Tries to infer data types from provided data and instruments.
        """
        _types = {}
        for k, v in data.items():
            match v:
                case DataReader():
                    _types[k] = (self._sniff_reader(k, v, time), "reader")

                case pd.DataFrame():
                    _types[k] = (self._sniff_pandas(v), "dataframe")

                case OHLCV():
                    _types[k] = (self._sniff_pandas(v.pd()), "dataframe")

                case list():
                    _types[k] = (self._sniff_list(v), "list")

                case dict():
                    _types[k] = (self._sniff_dicts(v), "dict")

                case _:
                    logger.warning(f"Unsupported data type: {type(v)} for symbol: {k}")
                    _types[k] = (DataType.NONE, None)

        return _types

    def extract_types_from_reader(
        self,
        reader: DataReader,
        instruments: list[str],
        time: str | None = None,
    ) -> dict[str, str]:
        return self.extract_types(dict(zip(instruments, [reader] * len(instruments))), time)


def recognize_simulation_data(
    data: dict[str, pd.DataFrame] | DataReader,
    instruments: list[Instrument],
    # aux_data: DataReader | None = None,
) -> SimulationDataInfo:
    if isinstance(data, dict):
        data_reader = InMemoryDataFrameReader(data)  # type: ignore

        if not instruments:
            instruments = list(data_reader.get_names())

    elif isinstance(data, DataReader):
        data_reader = data
        if not instruments:
            raise ValueError("Symbol list must be provided for generic data reader !")

    else:
        raise ValueError(f"Unsupported data type: {type(data).__name__}")
    pass

    return None
