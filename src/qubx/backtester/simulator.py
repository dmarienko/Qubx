import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Callable, Literal
from enum import Enum
from tqdm.auto import tqdm
from itertools import chain

from qubx import lookup, logger, QubxLogConfig
from qubx.core.account import AccountProcessor
from qubx.core.helpers import BasicScheduler
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.core.series import Quote, time_as_nsec
from qubx.core.basics import (
    ITimeProvider,
    Instrument,
    Deal,
    Order,
    Signal,
    SimulatedCtrlChannel,
    Position,
    TradingSessionResult,
    TransactionCostsCalculator,
    dt_64,
    Subtype,
)
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV
from qubx.core.interfaces import (
    IStrategy,
    IBrokerServiceProvider,
    ITradingServiceProvider,
    PositionsTracker,
    IStrategyContext,
    TriggerEvent,
)

from qubx.core.context import StrategyContext
from qubx.backtester.ome import OrdersManagementEngine, OmeReport

from qubx.data.helpers import InMemoryCachedReader, TimeGuardedWrapper
from qubx.data.readers import (
    AsTrades,
    DataReader,
    DataTransformer,
    RestoreTicksFromOHLC,
    AsQuotes,
    AsTimestampedRecords,
    InMemoryDataFrameReader,
)
from qubx.pandaz.utils import scols
from qubx.utils.misc import ProgressParallel
from qubx.utils.time import infer_series_frequency
from joblib import delayed
from .queue import DataLoader, SimulatedDataQueue, EventBatcher

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
    instruments: List[Instrument]
    exchange: str
    capital: float
    leverage: float
    base_currency: str
    commissions: str


import stackprinter


class _SimulatedLogFormatter:
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


class SimulatedTrading(ITradingServiceProvider):
    """
    First implementation of a simulated broker.
    TODO:
        1. Add margin control
        2. Need to solve problem with _get_ohlcv_data_sync (actually this method must be removed from here) [DONE]
        3. Add support for stop orders (not urgent) [DONE]
    """

    _current_time: dt_64
    _name: str
    _ome: Dict[Instrument, OrdersManagementEngine]
    _fees_calculator: TransactionCostsCalculator | None
    _order_to_instrument: Dict[str, Instrument]
    _half_tick_size: Dict[Instrument, float]
    _fill_stop_order_at_price: bool

    def __init__(
        self,
        name: str,
        commissions: str | None = None,
        simulation_initial_time: dt_64 | str = np.datetime64(0, "ns"),
        accurate_stop_orders_execution: bool = False,
    ) -> None:
        """
        This function sets up a simulated trading environment with following parameters.

        Parameters:
        -----------
        name : str
            The name of the simulated trading environment.
        commissions : str | None, optional
            The commission structure to be used. If None, no commissions will be applied.
            Default is None.
        simulation_initial_time : dt_64 | str, optional
            The initial time for the simulation. Can be a dt_64 object or a string.
            Default is np.datetime64(0, "ns").
        accurate_stop_orders_execution : bool, optional
            If True, stop orders will be executed at the exact stop order's price.
            If False, they may be executed at the next quote that could lead to
            significant slippage especially if simuation run on OHLC data.
            Default is False.

        Raises:
        -------
        ValueError
            If the fees configuration is not found for the given name.

        """
        self._current_time = (
            np.datetime64(simulation_initial_time, "ns")
            if isinstance(simulation_initial_time, str)
            else simulation_initial_time
        )
        self._name = name
        self._ome = {}
        self._fees_calculator = lookup.fees.find(name.lower(), commissions)
        self._half_tick_size = {}
        self._fill_stop_order_at_price = accurate_stop_orders_execution

        self._order_to_instrument = {}
        if self._fees_calculator is None:
            raise ValueError(
                f"SimulatedExchangeService :: Fees configuration '{commissions}' is not found for '{name}' !"
            )

        # - we want to see simulate time in log messages
        QubxLogConfig.setup_logger(QubxLogConfig.get_log_level(), _SimulatedLogFormatter(self).formatter)
        if self._fill_stop_order_at_price:
            logger.info(f"SimulatedExchangeService emulates stop orders executions at exact price")

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> Order:
        ome = self._ome.get(instrument)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument.symbol}'!")

        # - try to place order in OME
        report = ome.place_order(
            order_side.upper(),  # type: ignore
            order_type.upper(),  # type: ignore
            amount,
            price,
            client_id,
            time_in_force,
            **options,
        )
        order = report.order
        self._order_to_instrument[order.id] = instrument

        if report.exec is not None:
            self.process_execution_report(instrument, {"order": order, "deals": [report.exec]})
        else:
            self.acc.add_active_orders({order.id: order})

        # - send reports to channel
        self.send_execution_report(instrument, report)

        return report.order

    def send_execution_report(self, instrument: Instrument, report: OmeReport):
        self.get_communication_channel().send((instrument, "order", report.order))
        if report.exec is not None:
            self.get_communication_channel().send((instrument, "deals", [report.exec]))

    def cancel_order(self, order_id: str) -> Order | None:
        instrument = self._order_to_instrument.get(order_id)
        if instrument is None:
            raise ValueError(f"ExchangeService:cancel_order :: can't find order with id = '{order_id}'!")

        ome = self._ome.get(instrument)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument}'!")

        # - cancel order in OME and remove from the map to free memory
        self._order_to_instrument.pop(order_id)
        order_update = ome.cancel_order(order_id)
        self.acc.process_order(order_update.order)

        # - notify channel about order cancellation
        self.send_execution_report(instrument, order_update)

        return order_update.order

    def get_orders(self, instrument: Instrument | None = None) -> List[Order]:
        if instrument is not None:
            ome = self._ome.get(instrument)
            if ome is None:
                raise ValueError(f"ExchangeService:get_orders :: No OME configured for '{instrument}'!")
            return ome.get_open_orders()

        return [o for ome in self._ome.values() for o in ome.get_open_orders()]

    def get_position(self, instrument: Instrument) -> Position:
        if instrument in self.acc.positions:
            return self.acc.positions[instrument]

        # - initiolize OME for this instrument
        self._ome[instrument] = OrdersManagementEngine(
            instrument=instrument,
            time_provider=self,
            tcc=self._fees_calculator,  # type: ignore
            fill_stop_order_at_price=self._fill_stop_order_at_price,
        )

        # - initiolize empty position
        position = Position(instrument)  # type: ignore
        self._half_tick_size[instrument] = instrument.min_tick / 2  # type: ignore
        self.acc.attach_positions(position)
        return self.acc._positions[instrument]

    def time(self) -> dt_64:
        return self._current_time

    def get_base_currency(self) -> str:
        return self.acc.base_currency

    def get_name(self) -> str:
        return self._name

    def get_account_id(self) -> str:
        return "Simulated0"

    def process_execution_report(self, instrument: Instrument, report: Dict[str, Any]) -> Tuple[Order, List[Deal]]:
        order = report["order"]
        deals = report.get("deals", [])
        self.acc.process_deals(instrument, deals)
        self.acc.process_order(order)
        return order, deals

    def emulate_quote_from_data(
        self, instrument: Instrument, timestamp: dt_64, data: float | Trade | Bar
    ) -> Quote | None:
        _ts2 = self._half_tick_size[instrument]
        if isinstance(data, Quote):
            return data
        elif isinstance(data, Trade):
            if data.taker:  # type: ignore
                return Quote(timestamp, data.price - _ts2 * 2, data.price, 0, 0)  # type: ignore
            else:
                return Quote(timestamp, data.price, data.price + _ts2 * 2, 0, 0)  # type: ignore
        elif isinstance(data, Bar):
            return Quote(timestamp, data.close - _ts2, data.close + _ts2, 0, 0)  # type: ignore
        elif isinstance(data, float):
            return Quote(timestamp, data - _ts2, data + _ts2, 0, 0)
        else:
            return None

    def update_position_price(self, instrument: Instrument, timestamp: dt_64, update: float | Trade | Quote | Bar):
        # logger.info(f"{symbol} -> {timestamp} -> {update}")
        # - set current time from update
        self._current_time = timestamp

        # - first we need to update OME with new quote.
        # - if update is not a quote we need 'emulate' it.
        # - actually if SimulatedExchangeService is used in backtesting mode it will recieve only quotes
        # - case when we need that - SimulatedExchangeService is used for paper trading and data provider configured to listen to OHLC or TAS.
        # - probably we need to subscribe to quotes in real data provider in any case and then this emulation won't be needed.
        quote = update if isinstance(update, Quote) else self.emulate_quote_from_data(instrument, timestamp, update)
        if quote is None:
            return

        # - process new quote
        self._process_new_quote(instrument, quote)

        # - update positions data
        super().update_position_price(instrument, timestamp, update)

    def _process_new_quote(self, instrument: Instrument, data: Quote) -> None:
        ome = self._ome.get(instrument)
        if ome is None:
            logger.warning("ExchangeService:update :: No OME configured for '{symbol}' yet !")
            return
        for r in ome.update_bbo(data):
            if r.exec is not None:
                self._order_to_instrument.pop(r.order.id)
                self.process_execution_report(instrument, {"order": r.order, "deals": [r.exec]})

                # - notify channel about order cancellation
                self.send_execution_report(instrument, r)


class _SimulatedScheduler(BasicScheduler):
    def run(self):
        self._is_started = True
        _has_tasks = False
        _time = self.time_sec()
        for k in self._crons.keys():
            _has_tasks |= self._arm_schedule(k, _time)


class SimulatedExchange(IBrokerServiceProvider):
    trading_service: SimulatedTrading
    _last_quotes: Dict[Instrument, Optional[Quote]]
    _scheduler: BasicScheduler
    _current_time: dt_64
    _hist_data_type: str
    _loaders: dict[Instrument, dict[str, DataLoader]]
    _pregenerated_signals: Dict[Instrument, pd.Series]

    def __init__(
        self,
        exchange_id: str,
        trading_service: SimulatedTrading,
        reader: DataReader,
        hist_data_type: str = "ohlc",
    ):
        super().__init__(exchange_id, trading_service)
        self._reader = reader
        self._hist_data_type = hist_data_type
        exchange_id = exchange_id.lower()

        # - create exchange's instance
        self._last_quotes = defaultdict(lambda: None)
        self._current_time = self.trading_service.time()
        self._loaders = defaultdict(dict)
        self._symbol_to_instrument: dict[str, Instrument] = {}

        # - setup communication bus
        self.set_communication_channel(bus := SimulatedCtrlChannel("databus", sentinel=(None, None, None)))
        self.trading_service.set_communication_channel(bus)

        # - simulated scheduler
        self._scheduler = _SimulatedScheduler(bus, lambda: self.trading_service.time().item())

        # - pregenerated signals storage
        self._pregenerated_signals = dict()
        self._to_process = {}

        # - data queue
        self._data_queue = SimulatedDataQueue()

        logger.info(f"SimulatedData.{exchange_id} initialized")

    def subscribe(
        self,
        instruments: list[Instrument],
        subscription_type: str,
        warmup_period: str | None = None,
        timeframe: str | None = None,
        **kwargs,
    ) -> bool:
        units = kwargs.get("timestamp_units", "ns")

        for instr in instruments:
            logger.debug(
                f"SimulatedExchangeService :: subscribe :: {instr.symbol}({subscription_type}, {warmup_period}, {timeframe})"
            )
            self._symbol_to_instrument[instr.symbol] = instr

            _params: Dict[str, Any] = dict(
                reader=self._reader,
                instrument=instr,
                warmup_period=warmup_period,
                timeframe=timeframe,
            )

            # - for ohlc data we need to restore ticks from OHLC bars
            if subscription_type == Subtype.OHLC:
                _params["transformer"] = RestoreTicksFromOHLC(
                    trades="trades" in subscription_type,
                    spread=instr.min_tick,
                    timestamp_units=units,
                )
                _params["output_type"] = Subtype.QUOTE
            elif subscription_type == Subtype.QUOTE:
                _params["transformer"] = AsQuotes()
            elif subscription_type == Subtype.TRADE:
                _params["transformer"] = AsTrades()
            else:
                raise ValueError(f"Unknown subscription type: {subscription_type}")

            _params["data_type"] = subscription_type

            # - add loader for this instrument
            ldr = DataLoader(**_params)
            self._loaders[instr][subscription_type] = ldr
            self._data_queue += ldr

        return True

    def unsubscribe(self, instruments: List[Instrument], subscription_type: str | None) -> bool:
        for instr in instruments:
            if instr.symbol in self._loaders:
                if subscription_type:
                    logger.debug(f"SimulatedExchangeService :: unsubscribe :: {instr.symbol} :: {subscription_type}")
                    self._data_queue -= self._loaders[instr].pop(subscription_type)
                    if not self._loaders[instr]:
                        self._loaders.pop(instr)
                else:
                    logger.debug(f"SimulatedExchangeService :: unsubscribe :: {instr.symbol}")
                    for ldr in self._loaders[instr].values():
                        self._data_queue -= ldr
                    self._loaders.pop(instr)
        return True

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        return instrument in self._loaders and subscription_type in self._loaders[instrument]

    def get_subscriptions(self, instrument: Instrument) -> Dict[str, Dict[str, Any]]:
        # TODO: implement
        # return {k: v.params for k, v in self._loaders[instrument].items()}
        return {}

    def _try_add_process_signals(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> None:
        if self._pregenerated_signals:
            for s, v in self._pregenerated_signals.items():
                sel = v[pd.Timestamp(start) : pd.Timestamp(end)]
                self._to_process[s] = list(zip(sel.index, sel.values))

    def run(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        silent: bool = False,
        enable_event_batching: bool = True,
    ) -> None:
        logger.info(f"SimulatedExchangeService :: run :: Simulation started at {start}")
        self._try_add_process_signals(start, end)

        _run = self._run_generated_signals if self._pregenerated_signals else self._run_as_strategy

        qiter = EventBatcher(self._data_queue.create_iterable(start, end), passthrough=not enable_event_batching)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        total_duration = end - start
        update_delta = total_duration / 100
        prev_dt = pd.Timestamp(start)

        if silent:
            for instrument, data_type, event in qiter:
                if not _run(instrument, data_type, event):
                    break
        else:
            with tqdm(total=total_duration.total_seconds(), desc="Simulating", unit="s", leave=False) as pbar:
                for instrument, data_type, event in qiter:
                    if not _run(instrument, data_type, event):
                        break
                    dt = pd.Timestamp(event.time)
                    # update only if date has changed
                    if dt - prev_dt > update_delta:
                        pbar.n = (dt - start).total_seconds()
                        pbar.refresh()
                        prev_dt = dt
                pbar.n = total_duration.total_seconds()
                pbar.refresh()

        logger.info(f"SimulatedExchangeService :: run :: Simulation finished at {end}")

    def _run_generated_signals(self, instrument: Instrument, data_type: str, data: Any) -> bool:
        is_hist = data_type.startswith("hist")
        if is_hist:
            raise ValueError("Historical data is not supported for pre-generated signals !")
        cc = self.get_communication_channel()
        t = data.time  # type: ignore
        self._current_time = max(np.datetime64(t, "ns"), self._current_time)
        q = self.trading_service.emulate_quote_from_data(instrument, np.datetime64(t, "ns"), data)
        self._last_quotes[instrument] = q
        self.trading_service.update_position_price(instrument, self._current_time, data)

        # - we need to send quotes for invoking portfolio logging etc
        # match event type
        cc.send((instrument, data_type, data))
        sigs = self._to_process[instrument]
        while sigs and sigs[0][0].as_unit("ns").asm8 <= self._current_time:
            cc.send((instrument, "event", {"order": sigs[0][1]}))
            sigs.pop(0)

        return cc.control.is_set()

    def _run_as_strategy(self, instrument: Instrument, data_type: str, data: Any) -> bool:
        cc = self.get_communication_channel()
        t = data.time  # type: ignore
        self._current_time = max(np.datetime64(t, "ns"), self._current_time)
        q = self.trading_service.emulate_quote_from_data(instrument, np.datetime64(t, "ns"), data)
        is_hist = data_type.startswith("hist")

        if not is_hist and q is not None:
            self._last_quotes[instrument] = q
            self.trading_service.update_position_price(instrument, self._current_time, q)

            # we have to schedule possible crons before sending the data event itself
            if self._scheduler.check_and_run_tasks():
                # - push nothing - it will force to process last event
                cc.send((None, "service_time", None))

        cc.send((instrument, data_type, data))

        if not is_hist:
            if q is not None and data_type != "quote":
                cc.send((instrument, "quote", q))

        return cc.control.is_set()

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument]

    def close(self):
        pass

    def time(self) -> dt_64:
        return self._current_time

    def get_scheduler(self) -> BasicScheduler:
        return self._scheduler

    def is_simulated_trading(self) -> bool:
        return True

    def get_historical_ohlcs(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]:
        start = pd.Timestamp(self.time())
        end = start - nbarsback * (_timeframe := pd.Timedelta(timeframe))
        _spec = f"{instrument.exchange}:{instrument.symbol}"
        return self._convert_records_to_bars(
            self._reader.read(data_id=_spec, start=start, stop=end, transform=AsTimestampedRecords()), 
            time_as_nsec(self.time()), 
            _timeframe.asm8.item()
        )

    def _convert_records_to_bars(self, records: List[Dict[str, Any]], cut_time_ns: int, timeframe_ns: int) -> List[Bar]:
        """
        Convert records to bars and we need to cut last bar up to the cut_time_ns
        """
        bars = []

        _data_tf = infer_series_frequency([r['timestamp_ns'] for r in records[:100]])
        timeframe_ns = _data_tf.item()

        if records is not None:
            for r in records:
                _bts_0 = np.datetime64(r["timestamp_ns"], "ns").item()
                o, h, l, c, v = r["open"], r["high"], r["low"], r["close"], r["volume"]

                if _bts_0 <= cut_time_ns and cut_time_ns < _bts_0 + timeframe_ns:
                    break

                bars.append(Bar(_bts_0, o, h, l, c, v))

        return bars

    def set_generated_signals(self, signals: pd.Series | pd.DataFrame):
        logger.debug(f"Using pre-generated signals:\n {str(signals.count()).strip('ndtype: int64')}")
        # - sanity check
        signals.index = pd.DatetimeIndex(signals.index)

        if isinstance(signals, pd.Series):
            self._pregenerated_signals[signals.name] = signals

        elif isinstance(signals, pd.DataFrame):
            for col in signals.columns:
                self._pregenerated_signals[col] = signals[col]
        else:
            raise ValueError("Invalid signals or strategy configuration")


def _recognize_simulation_setups(
    name: str,
    configs: (
        StrategyOrSignals
        | Dict[str, StrategyOrSignals | List[StrategyOrSignals | PositionsTracker]]
        | List[StrategyOrSignals | PositionsTracker]
        | Tuple[StrategyOrSignals | PositionsTracker]
    ),
    instruments: List[Instrument],
    exchange: str,
    capital: float,
    leverage: float,
    basic_currency: str,
    commissions: str,
):
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
                _recognize_simulation_setups(
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
                    _recognize_simulation_setups(
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


def simulate(
    config: StrategyOrSignals | Dict | List[StrategyOrSignals | PositionsTracker],
    data: Dict[str, pd.DataFrame] | DataReader,
    capital: float,
    instruments: List[str] | Dict[str, List[str]] | None,
    commissions: str,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp | None = None,
    exchange: str | None = None,  # in case if exchange is not specified in symbols list
    base_currency: str = "USDT",
    signal_timeframe: str = "1Min",
    leverage: float = 1.0,  # TODO: we need to add support for leverage
    n_jobs: int = 1,
    silent: bool = False,
    enable_event_batching: bool = True,
    accurate_stop_orders_execution: bool = False,
    aux_data: DataReader | None = None,
    debug: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = "WARNING",
) -> list[TradingSessionResult]:
    """
    Backtest utility for trading strategies or signals using historical data.

    Parameters:
    ----------

    config (StrategyOrSignals | Dict | List[StrategyOrSignals | PositionsTracker]):
        Trading strategy or signals configuration.
    data (Dict[str, pd.DataFrame] | DataReader):
        Historical data for simulation, either as a dictionary of DataFrames or a DataReader object.
    capital (float):
        Initial capital for the simulation.
    instruments (List[str] | Dict[str, List[str]] | None):
        List of trading instruments or a dictionary mapping exchanges to instrument lists.
    subscription (Dict[str, Any]):
        Subscription details for market data.
    trigger (str | list[str]):
        Trigger specification for strategy execution.
    commissions (str):
        Commission structure for trades.
    start (str | pd.Timestamp):
        Start time of the simulation.
    stop (str | pd.Timestamp | None):
        End time of the simulation. If None, simulates until the last accessible data.
    fit (str | None):
        Specification for strategy fitting, if applicable.
    exchange (str | None):
        Exchange name if not specified in the instruments list.
    base_currency (str):
        Base currency for the simulation, default is "USDT".
    signal_timeframe (str):
        Timeframe for signals, default is "1Min".
    leverage (float):
        Leverage factor for trading, default is 1.0.
    n_jobs (int):
        Number of parallel jobs for simulation, default is 1.
    silent (bool):
        If True, suppresses output during simulation.
    enable_event_batching (bool):
        If True, enables event batching for optimization.
    accurate_stop_orders_execution (bool):
        If True, enables more accurate stop order execution simulation.
    aux_data (DataReader | None):
        Auxiliary data provider (default is None).
    debug (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None):
        Logging level for debugging.

    Returns:
    --------
    list[TradingSessionResult]:
        A list of TradingSessionResult objects containing the results of each simulation setup.
    """

    # - setup logging
    QubxLogConfig.set_log_level(debug.upper() if debug else "WARNING")

    # - recognize provided data
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

    # - process instruments:
    #    check if instruments are from the same exchange (mmulti-exchanges is not supported yet)
    _instrs, _exchanges = find_instruments_and_exchanges(instruments, exchange)

    if not _exchanges:
        logger.error(
            f"No exchange iformation provided - you can specify it by exchange parameter or use <yellow>EXCHANGE:SYMBOL</yellow> format for symbols"
        )
        # - TODO: probably we need to raise exceptions here ?
        return []

    # - check exchanges
    if len(set(_exchanges)) > 1:
        logger.error(f"Multiple exchanges found: {', '.join(_exchanges)} - this mode is not supported yet in Qubx !")
        # - TODO: probably we need to raise exceptions here ?
        return []

    exchange = list(set(_exchanges))[0]

    # - recognize setup: it can be either a strategy or set of signals
    setups = _recognize_simulation_setups("", config, _instrs, exchange, capital, leverage, base_currency, commissions)
    if not setups:
        logger.error(
            f"Can't recognize setup - it should be a strategy, a set of signals or list of signals/strategies + tracker !"
        )
        # - TODO: probably we need to raise exceptions here ?
        return []

    # - check stop time : here we try to backtest till now (may be we need to get max available time from data reader ?)
    if stop is None:
        stop = pd.Timestamp.now(tz="UTC").astimezone(None)

    # - run simulations
    return _run_setups(
        setups,
        start,
        stop,
        data_reader,
        n_jobs=n_jobs,
        silent=silent,
        enable_event_batching=enable_event_batching,
        accurate_stop_orders_execution=accurate_stop_orders_execution,
        aux_data=aux_data,
        signal_timeframe=signal_timeframe,
    )


def find_instruments_and_exchanges(
    instruments: List[str] | Dict[str, List[str]], exchange: str | None
) -> Tuple[List[Instrument], List[str]]:
    _instrs: List[Instrument] = []
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


class SignalsProxy(IStrategy):
    timeframe: str = "1m"

    def on_init(self, ctx: IStrategyContext):
        ctx.set_base_subscription(Subtype.OHLC, timeframe=self.timeframe)

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> Optional[List[Signal]]:
        if event.data and event.type == "event":
            signal = event.data.get("order")
            # - TODO: also need to think about how to pass stop/take here
            if signal is not None and event.instrument:
                return [event.instrument.signal(signal)]
        return None


def _run_setups(
    setups: List[SimulationSetup],
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    data_reader: DataReader,
    n_jobs: int = -1,
    silent: bool = False,
    enable_event_batching: bool = True,
    accurate_stop_orders_execution: bool = False,
    aux_data: DataReader | None = None,
    **kwargs,
) -> List[TradingSessionResult]:
    # loggers don't work well with joblib and multiprocessing in general because they contain
    # open file handlers that cannot be pickled. I found a solution which requires the usage of enqueue=True
    # in the logger configuration and specifying backtest "multiprocessing" instead of the default "loky"
    # for joblib. But it works now.
    # See: https://stackoverflow.com/questions/59433146/multiprocessing-logging-how-to-use-loguru-with-joblib-parallel
    _main_loop_silent = len(setups) == 1
    n_jobs = 1 if _main_loop_silent else n_jobs

    reports = ProgressParallel(n_jobs=n_jobs, total=len(setups), silent=_main_loop_silent, backend="multiprocessing")(
        delayed(_run_setup)(
            id,
            s,
            start,
            stop,
            data_reader,
            silent=silent,
            enable_event_batching=enable_event_batching,
            accurate_stop_orders_execution=accurate_stop_orders_execution,
            aux_data_provider=aux_data,
            **kwargs,
        )
        for id, s in enumerate(setups)
    )
    return reports  # type: ignore


def _run_setup(
    setup_id: int,
    setup: SimulationSetup,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    data_reader: DataReader,
    silent: bool = False,
    enable_event_batching: bool = True,
    accurate_stop_orders_execution: bool = False,
    aux_data_provider: InMemoryCachedReader | None = None,
    signal_timeframe: str = "1Min",
) -> TradingSessionResult:
    _stop = stop
    logger.debug(
        f"<red>{pd.Timestamp(start)}</red> Initiating simulated trading for {setup.exchange} for {setup.capital} x {setup.leverage} in {setup.base_currency}..."
    )
    trading_service = SimulatedTrading(
        setup.exchange,
        setup.commissions,
        np.datetime64(start, "ns"),
        accurate_stop_orders_execution=accurate_stop_orders_execution,
    )
    broker = SimulatedExchange(setup.exchange, trading_service, data_reader)

    # - it will store simulation results into memory
    logs_writer = InMemoryLogsWriter("test", setup.name, "0")
    strat: IStrategy | None = None

    match setup.setup_type:
        case _Types.STRATEGY:
            strat = setup.generator  # type: ignore

        case _Types.STRATEGY_AND_TRACKER:
            strat = setup.generator  # type: ignore
            strat.tracker = lambda ctx: setup.tracker  # type: ignore

        case _Types.SIGNAL:
            strat = SignalsProxy(timeframe=signal_timeframe)
            broker.set_generated_signals(setup.generator)  # type: ignore
            # - we don't need any unexpected triggerings
            _stop = setup.generator.index[-1]  # type: ignore

            # - no historical data for generated signals, so disable it
            enable_event_batching = False

        case _Types.SIGNAL_AND_TRACKER:
            strat = SignalsProxy(timeframe=signal_timeframe)
            strat.tracker = lambda ctx: setup.tracker
            broker.set_generated_signals(setup.generator)  # type: ignore
            # - we don't need any unexpected triggerings
            _stop = setup.generator.index[-1]  # type: ignore

            # - no historical data for generated signals, so disable it
            enable_event_batching = False

        case _:
            raise ValueError(f"Unsupported setup type: {setup.setup_type} !")

    # - check aux data provider
    _aux_data = None
    if aux_data_provider is not None:
        if not isinstance(aux_data_provider, InMemoryCachedReader):
            logger.error("Aux data provider should be an instance of InMemoryCachedReader! Skipping it.")
        _aux_data = TimeGuardedWrapper(aux_data_provider, trading_service)

    account = AccountProcessor(
        account_id=trading_service.get_account_id(),
        base_currency=setup.base_currency,
        initial_capital=setup.capital,
    )

    ctx = StrategyContext(
        strategy=strat,  # type: ignore
        config=None,  # TODO: need to think how we could pass altered parameters here (from variating etc)
        broker=broker,
        account=account,
        instruments=setup.instruments,
        logging=StrategyLogging(logs_writer),
        aux_data_provider=_aux_data,
    )
    ctx.start()

    try:
        broker.run(start, _stop, silent=silent, enable_event_batching=enable_event_batching)  # type: ignore
    except KeyboardInterrupt:
        logger.error("Simulated trading interrupted by user !")

    return TradingSessionResult(
        setup_id,
        setup.name,
        start,
        stop,
        setup.exchange,
        setup.instruments,
        setup.capital,
        setup.leverage,
        setup.base_currency,
        setup.commissions,
        logs_writer.get_portfolio(as_plain_dataframe=True),
        logs_writer.get_executions(),
        logs_writer.get_signals(),
        True,
    )
