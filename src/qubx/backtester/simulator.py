from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeAlias
import numpy as np
import pandas as pd
from enum import Enum
from tqdm.auto import tqdm

from qubx import lookup, logger
from qubx.core.helpers import BasicScheduler
from qubx.core.loggers import InMemoryLogsWriter
from qubx.core.series import Quote
from qubx.core.account import AccountProcessor
from qubx.core.basics import (
    Instrument,
    Deal,
    Order,
    Signal,
    SimulatedCtrlChannel,
    Position,
    TradingSessionResult,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV
from qubx.core.strategy import (
    IStrategy,
    IBrokerServiceProvider,
    ITradingServiceProvider,
    PositionsTracker,
    StrategyContext,
    TriggerEvent,
)
from qubx.backtester.ome import OrdersManagementEngine, OmeReport

from qubx.data.readers import (
    DataReader,
    DataTransformer,
    RestoreTicksFromOHLC,
    AsQuotes,
    AsTimestampedRecords,
    InMemoryDataFrameReader,
)
from qubx.pandaz.utils import scols

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


class SimulatedTrading(ITradingServiceProvider):
    """
    First implementation of a simulated broker.
    TODO:
        1. Add margin control
        2. Need to solve problem with _get_ohlcv_data_sync (actually this method must be removed from here)
        3. Add support for stop orders (not urgent)
    """

    _current_time: dt_64
    _name: str
    _ome: Dict[str, OrdersManagementEngine]
    _fees_calculator: TransactionCostsCalculator | None
    _order_to_symbol: Dict[str, str]
    _half_tick_size: Dict[str, float]

    def __init__(
        self,
        name: str,
        capital: float,
        commissions: str,
        base_currency: str,
        simulation_initial_time: dt_64 | str = np.datetime64(0, "ns"),
    ) -> None:
        self._current_time = (
            np.datetime64(simulation_initial_time, "ns")
            if isinstance(simulation_initial_time, str)
            else simulation_initial_time
        )
        self._name = name
        self._ome = {}
        self.acc = AccountProcessor("Simulated0", base_currency, None, capital, 0)
        self._fees_calculator = lookup.fees.find(name.lower(), commissions)
        self._half_tick_size = {}

        self._order_to_symbol = {}
        if self._fees_calculator is None:
            raise ValueError(
                f"SimulatedExchangeService :: Fees configuration '{commissions}' is not found for '{name}' !"
            )

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
    ) -> Order | None:
        ome = self._ome.get(instrument.symbol)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument.symbol}'!")

        # - try to place order in OME
        report = ome.place_order(order_side.upper(), order_type.upper(), amount, price, client_id, time_in_force)
        order = report.order
        self._order_to_symbol[order.id] = instrument.symbol

        if report.exec is not None:
            self.process_execution_report(instrument.symbol, {"order": order, "deals": [report.exec]})
        else:
            self.acc.add_active_orders({order.id: order})

        # - send reports to channel
        self.send_execution_report(instrument.symbol, report)

        return report.order

    def send_execution_report(self, symbol: str, report: OmeReport):
        self.get_communication_channel().send((symbol, "order", report.order))
        if report.exec is not None:
            self.get_communication_channel().send((symbol, "deals", [report.exec]))

    def cancel_order(self, order_id: str) -> Order | None:
        symb = self._order_to_symbol.get(order_id)
        if symb is None:
            raise ValueError(f"ExchangeService:cancel_order :: can't find order with id = '{order_id}'!")

        ome = self._ome.get(symb)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{symb}'!")

        # - cancel order in OME and remove from the map to free memory
        self._order_to_symbol.pop(order_id)
        order_update = ome.cancel_order(order_id)
        self.acc.process_order(order_update.order)

        # - notify channel about order cancellation
        self.send_execution_report(symb, order_update)

        return order_update.order

    def get_orders(self, symbol: str | None = None) -> List[Order]:
        if symbol is not None:
            ome = self._ome.get(symbol)
            if ome is None:
                raise ValueError(f"ExchangeService:get_orders :: No OME configured for '{symbol}'!")
            return ome.get_open_orders()

        return [o for ome in self._ome.values() for o in ome.get_open_orders()]

    def get_position(self, instrument: Instrument) -> Position:
        symbol = instrument.symbol

        if symbol not in self.acc._positions:
            # - initiolize OME for this instrument
            self._ome[instrument.symbol] = OrdersManagementEngine(instrument, self)  # type: ignore

            # - initiolize empty position
            position = Position(instrument, self._fees_calculator)  # type: ignore
            self._half_tick_size[instrument.symbol] = instrument.min_tick / 2  # type: ignore
            self.acc.attach_positions(position)

        return self.acc._positions[symbol]

    def time(self) -> dt_64:
        return self._current_time

    def get_base_currency(self) -> str:
        return self.acc.base_currency

    def get_name(self) -> str:
        return self._name

    def process_execution_report(self, symbol: str, report: Dict[str, Any]) -> Tuple[Order, List[Deal]]:
        order = report["order"]
        deals = report.get("deals", [])
        self.acc.process_deals(symbol, deals)
        self.acc.process_order(order)
        return order, deals

    def _emulate_quote_from_data(self, symbol: str, timestamp: dt_64, data: float | Trade | Bar) -> Quote:
        _ts2 = self._half_tick_size[symbol]
        if isinstance(data, Trade):
            if data.taker:  # type: ignore
                return Quote(timestamp, data.price - _ts2 * 2, data.price, 0, 0)  # type: ignore
            else:
                return Quote(timestamp, data.price, data.price + _ts2 * 2, 0, 0)  # type: ignore
        elif isinstance(data, Bar):
            return Quote(timestamp, data.close - _ts2, data.close + _ts2, 0, 0)  # type: ignore
        elif isinstance(data, float):
            return Quote(timestamp, data - _ts2, data + _ts2, 0, 0)
        else:
            raise ValueError(f"Unknown update type: {type(data)}")

    def update_position_price(self, symbol: str, timestamp: dt_64, update: float | Trade | Quote | Bar):
        # logger.info(f"{symbol} -> {timestamp} -> {update}")
        # - set current time from update
        self._current_time = timestamp

        # - first we need to update OME with new quote.
        # - if update is not a quote we need 'emulate' it.
        # - actually if SimulatedExchangeService is used in backtesting mode it will recieve only quotes
        # - case when we need that - SimulatedExchangeService is used for paper trading and data provider configured to listen to OHLC or TAS.
        # - probably we need to subscribe to quotes in real data provider in any case and then this emulation won't be needed.
        quote = update if isinstance(update, Quote) else self._emulate_quote_from_data(symbol, timestamp, update)

        # - process new quote
        self._process_new_quote(symbol, quote)

        # - update positions data
        super().update_position_price(symbol, timestamp, update)

    def _process_new_quote(self, symbol: str, data: Quote) -> None:
        ome = self._ome.get(symbol)
        if ome is None:
            logger.warning("ExchangeService:update :: No OME configured for '{symbol}' yet !")
            return
        for r in ome.update_bbo(data):
            if r.exec is not None:
                self._order_to_symbol.pop(r.order.id)
                self.process_execution_report(symbol, {"order": r.order, "deals": [r.exec]})

                # - notify channel about order cancellation
                self.send_execution_report(symbol, r)


class DataLoader:
    def __init__(
        self,
        transformer: DataTransformer,
        reader: DataReader,
        instrument: Instrument,
        timeframe: str | None,
        preload_bars: int = 0,
    ) -> None:
        self._instrument = instrument
        self._spec = f"{instrument.exchange}:{instrument.symbol}"
        self._reader = reader
        self._transformer = transformer
        self._init_bars_required = preload_bars
        self._timeframe = timeframe
        self._first_load = True

    def load(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> List[Quote]:
        if self._first_load:
            if self._init_bars_required > 0 and self._timeframe:
                start = pd.Timestamp(start) - self._init_bars_required * pd.Timedelta(self._timeframe)
            self._first_load = False

        args = dict(
            data_id=self._spec,
            start=start,
            stop=end,
            transform=self._transformer,
        )

        if self._timeframe:
            args["timeframe"] = self._timeframe

        return self._reader.read(**args)  # type: ignore

    def get_historical_ohlc(self, timeframe: str, start_time: str, nbarsback: int) -> List[Bar]:
        start = pd.Timestamp(start_time)
        end = start - nbarsback * pd.Timedelta(timeframe)
        records = self._reader.read(
            data_id=self._spec, start=start, stop=end, transform=AsTimestampedRecords()  # type: ignore
        )
        return [
            Bar(np.datetime64(r["timestamp_ns"], "ns").item(), r["open"], r["high"], r["low"], r["close"], r["volume"])
            for r in records
        ]


class _SimulatedScheduler(BasicScheduler):
    def run(self):
        self._is_started = True
        _has_tasks = False
        _time = self.time_sec()
        for k in self._crons.keys():
            _has_tasks |= self._arm_schedule(k, _time)


class SimulatedExchange(IBrokerServiceProvider):
    _last_quotes: Dict[str, Optional[Quote]]
    _scheduler: BasicScheduler
    _current_time: dt_64
    _hist_data_type: str
    _loader: Dict[str, DataLoader]
    _pregenerated_signals: Dict[str, pd.Series]

    def __init__(
        self,
        exchange_id: str,
        trading_service: ITradingServiceProvider,
        reader: DataReader,
        hist_data_type: str = "ohlc",
    ):
        super().__init__(exchange_id, trading_service)
        self._reader = reader
        self._hist_data_type = hist_data_type
        exchange_id = exchange_id.lower()

        # - create exchange's instance
        self._last_quotes = defaultdict(lambda: None)
        self._current_time = np.datetime64(0, "ns")
        self._loader = {}

        # - setup communication bus
        self.set_communication_channel(bus := SimulatedCtrlChannel("databus", sentinel=(None, None, None)))
        self.trading_service.set_communication_channel(bus)

        # - simulated scheduler
        self._scheduler = _SimulatedScheduler(bus, lambda: self.trading_service.time().item())

        # - pregenerated signals storage
        self._pregenerated_signals = dict()

        logger.info(f"SimulatedData.{exchange_id} initialized")

    def subscribe(
        self,
        subscription_type: str,
        instruments: List[Instrument],
        timeframe: str | None = None,
        nback: int = 0,
        **kwargs,
    ) -> bool:
        units = kwargs.get("timestamp_units", "ns")

        for instr in instruments:
            _params: Dict[str, Any] = dict(
                reader=self._reader,
                instrument=instr,
                preload_bars=nback,
                timeframe=timeframe,
            )

            # - for ohlc data we need to restore ticks from OHLC bars
            if "ohlc" in subscription_type:
                _params["transformer"] = RestoreTicksFromOHLC(
                    trades="trades" in subscription_type, spread=instr.min_tick, timestamp_units=units
                )
            elif "quote" in subscription_type:
                _params["transformer"] = AsQuotes()

            # - create loader for this instrument
            self._loader[instr.symbol] = DataLoader(**_params)

        return True

    def run(self, start: str | pd.Timestamp, end: str | pd.Timestamp):
        ds = []
        for s, ld in self._loader.items():
            data = ld.load(start, end)
            ds.append(pd.Series({q.time: q for q in data}, name=s) if data else pd.Series(name=s))

        merged = scols(*ds)
        if self._pregenerated_signals:
            self._run_generated_signals(merged)
        else:
            self._run_as_strategy(merged)
        return merged

    def _run_generated_signals(self, data: pd.DataFrame) -> None:
        cc = self.get_communication_channel()
        s0, e0 = pd.Timestamp(data.index[0]), pd.Timestamp(data.index[-1])

        to_process = {}
        for s, v in self._pregenerated_signals.items():
            sel = v[s0:e0]
            to_process[s] = list(zip(sel.index, sel.values))

        # - send initial quotes - this will invoke calling of on_fit method
        # for s in data.columns:
        # cc.send((s, "quote", data[s].values[0]))

        for t, u in tqdm(zip(data.index, data.values), total=len(data), leave=False):
            for i, s in enumerate(data.columns):
                q = u[i]
                if q:
                    self._current_time = max(np.datetime64(t, "ns"), self._current_time)
                    self.trading_service.update_position_price(s, self._current_time, q)
                    # - we need to send quotes for invoking portfolio logginf etc
                    cc.send((s, "quote", q))
                    sigs = to_process[s]
                    if sigs and sigs[0][0].as_unit("ns").asm8 <= self._current_time:
                        cc.send((s, "event", {"order": sigs[0][1]}))
                        sigs.pop(0)

    def _run_as_strategy(self, data: pd.DataFrame) -> None:
        cc = self.get_communication_channel()
        for t, u in tqdm(zip(data.index, data.values), total=len(data), leave=False):
            for i, s in enumerate(data.columns):
                q = u[i]
                if q:
                    self._last_quotes[s] = q
                    self._current_time = max(np.datetime64(t, "ns"), self._current_time)
                    self.trading_service.update_position_price(s, self._current_time, q)
                    cc.send((s, "quote", q))

            if self._scheduler.check_and_run_tasks():
                # - push nothing - it will force to process last event
                cc.send((None, "time", None))

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quotes[symbol]

    def close(self):
        pass

    def time(self) -> dt_64:
        return self._current_time

    def get_scheduler(self) -> BasicScheduler:
        return self._scheduler

    def is_simulated_trading(self) -> bool:
        return True

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> List[Bar]:
        return self._loader[symbol].get_historical_ohlc(timeframe, self.time(), nbarsback)

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
    name_in_list = lambda n: any([n == i.symbol for i in instruments])

    def _check_signals_structure(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        if isinstance(s, pd.Series):
            if not name_in_list(s.name):
                raise ValueError(f"Can't find instrument for signal's name: '{s.name}'")

        if isinstance(s, pd.DataFrame):
            for col in s.columns:
                if not name_in_list(col):
                    raise ValueError(f"Can't find instrument for signal's name: '{col}'")
        return s

    def _pick_instruments(s: pd.Series | pd.DataFrame) -> List[Instrument]:
        if isinstance(s, pd.Series):
            _instrs = [i for i in instruments if s.name == i.symbol]

        elif isinstance(s, pd.DataFrame):
            _instrs = [i for i in instruments if i.symbol in list(s.columns)]

        else:
            raise ValueError("Invalid signals or strategy configuration")

        return _instrs

    r = list()
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
            _s = _check_signals_structure(c0)

            if _is_signal(c0):
                _t = _Types.SIGNAL_AND_TRACKER

            if _is_strategy(c0):
                _t = _Types.STRATEGY_AND_TRACKER

            # - extract actual symbols that have signals
            r.append(
                SimulationSetup(
                    _t,
                    name,
                    _s,
                    c1,
                    _pick_instruments(_s),
                    exchange,
                    capital,
                    leverage,
                    basic_currency,
                    commissions,
                )
            )
        else:
            for j, s in enumerate(configs):
                r.extend(
                    _recognize_simulation_setups(
                        name + "/" + str(j), s, instruments, exchange, capital, leverage, basic_currency, commissions
                    )
                )

    elif _is_strategy(configs):
        r.append(
            SimulationSetup(
                _Types.STRATEGY,
                name,
                configs,
                None,
                instruments,
                exchange,
                capital,
                leverage,
                basic_currency,
                commissions,
            )
        )

    elif _is_signal(configs):
        # - check structure of signals
        c1 = _check_signals_structure(configs)
        r.append(
            SimulationSetup(
                _Types.SIGNAL,
                name,
                c1,
                None,
                _pick_instruments(c1),
                exchange,
                capital,
                leverage,
                basic_currency,
                commissions,
            )
        )

    return r


def simulate(
    config: StrategyOrSignals | Dict | List[StrategyOrSignals | PositionsTracker],
    data: Dict[str, pd.DataFrame] | DataReader,
    capital: float,
    instruments: List[str] | Dict[str, List[str]] | None,
    subscription: Dict[str, Any],
    trigger: str,
    commissions: str,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp | None = None,
    exchange: str | None = None,  # in case if exchange is not specified in symbols list
    base_currency: str = "USDT",
    leverage: float = 1.0,  # TODO: we need to add support for leverage
    n_jobs: int = -1,  # TODO: if need to run simulation in parallel
) -> TradingSessionResult | List[TradingSessionResult]:
    # - recognize provided data
    if isinstance(data, dict):
        data_reader = InMemoryDataFrameReader(data)
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
    _instrs: List[Instrument] = []
    _exchanges = [] if exchange is None else [exchange.lower()]
    for i in instruments:
        match i:
            case str():
                _e, _s = i.split(":") if ":" in i else (exchange, i)

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

    if not _exchanges:
        logger.error(
            f"No exchange iformation provided - you can specify it by exchange parameter or use <yellow>EXCHANGE:SYMBOL</yellow> format for symbols"
        )
        # - TODO: probably we need to raise exceptions here ?
        return None

    # - check exchanges
    if len(set(_exchanges)) > 1:
        logger.error(f"Multiple exchanges found: {', '.join(_exchanges)} - this mode is not supported yet in Qubx !")
        # - TODO: probably we need to raise exceptions here ?
        return None

    exchange = list(set(_exchanges))[0]

    # - recognize setup: it can be either a strategy or set of signals
    setups = _recognize_simulation_setups("", config, _instrs, exchange, capital, leverage, base_currency, commissions)
    if not setups:
        logger.error(
            f"Can't recognize setup - it should be a strategy, a set of signals or list of signals/strategies + tracker !"
        )
        # - TODO: probably we need to raise exceptions here ?
        return None

    # - check stop time : here we try to backtest till now (may be we need to get max available time from data reader ?)
    if stop is None:
        stop = pd.Timestamp.now(tz="UTC").astimezone(None)

    # - run simulations
    return _run_setups(setups, start, stop, data_reader, subscription, trigger, n_jobs=n_jobs)


class _GeneratedSignalsStrategy(IStrategy):

    def on_fit(
        self, ctx: StrategyContext, fit_time: str | pd.Timestamp, previous_fit_time: str | pd.Timestamp | None = None
    ):
        return None

    def on_event(self, ctx: StrategyContext, event: TriggerEvent) -> Optional[List[Signal]]:
        if event.data and event.type == "event":
            signal = event.data.get("order")
            if signal:
                ctx.trade(event.instrument, signal)
        return None


def _run_setups(
    setups: List[SimulationSetup],
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    data_reader: DataReader,
    subscription: Dict[str, Any],
    trigger: str,
    n_jobs: int = -1,
) -> List[TradingSessionResult]:
    reports = []

    # - TODO: we need to run this in multiprocessing environment if n_jobs > 1
    for s in tqdm(setups, total=len(setups)):
        _trigger = trigger
        logger.debug(
            f"<red>{pd.Timestamp(start)}</red> Initiating simulated trading for {s.exchange} for {s.capital} x {s.leverage} in {s.base_currency}..."
        )
        broker = SimulatedTrading(s.exchange, s.capital, s.commissions, s.base_currency, np.datetime64(start, "ns"))
        exchange = SimulatedExchange(s.exchange, broker, data_reader)

        # - it will store simulation results into memory
        logs_writer = InMemoryLogsWriter("test", s.name, "0")
        strat: IStrategy | None = None

        match s.setup_type:
            case _Types.STRATEGY:
                strat = s.generator

            case _Types.STRATEGY_AND_TRACKER:
                strat = s.generator
                strat.tracker = lambda ctx: s.tracker

            case _Types.SIGNAL:
                strat = _GeneratedSignalsStrategy()
                exchange.set_generated_signals(s.generator)
                # - we don't need any unexpected triggerings
                _trigger = "bar: 0s"

            case _Types.SIGNAL_AND_TRACKER:
                strat = _GeneratedSignalsStrategy()
                strat.tracker = lambda ctx: s.tracker
                exchange.set_generated_signals(s.generator)
                # - we don't need any unexpected triggerings
                _trigger = "bar: 0s"

            case _:
                raise ValueError(f"Unsupported setup type: {s.setup_type} !")

        ctx = StrategyContext(
            strat,
            None,  # TODO: need to think how we could pass altered parameters here (from variating etc)
            exchange,
            instruments=s.instruments,
            md_subscription=subscription,
            trigger_spec=_trigger,
            logs_writer=logs_writer,
        )
        ctx.start()

        _r = exchange.run(start, stop)
        reports.append(
            TradingSessionResult(
                s.name,
                start,
                stop,
                s.exchange,
                s.instruments,
                s.capital,
                s.leverage,
                s.base_currency,
                s.commissions,
                logs_writer.get_portfolio(as_plain_dataframe=True),
                logs_writer.get_executions(),
                logs_writer.get_signals(),
                True,
            )
        )

    return reports
