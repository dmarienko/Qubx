from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import delayed
from tqdm.auto import tqdm

from qubx import QubxLogConfig, logger, lookup
from qubx.backtester.ome import OmeReport, OrdersManagementEngine
from qubx.backtester.simulated_data import EventBatcher, IterableSimulationData
from qubx.backtester.utils import (
    SetupTypes,
    SimulatedCtrlChannel,
    SimulatedLogFormatter,
    SimulatedScheduler,
    SimulationSetup,
    VariableStrategyConfig,
    find_instruments_and_exchanges,
    recognize_simulation_configuration,
)
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    Deal,
    Instrument,
    Order,
    Position,
    Signal,
    Subtype,
    TradingSessionResult,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.context import StrategyContext
from qubx.core.helpers import BasicScheduler, extract_parameters_from_object, full_qualified_class_name
from qubx.core.interfaces import (
    IAccountProcessor,
    IBrokerServiceProvider,
    IStrategy,
    IStrategyContext,
    ITradingServiceProvider,
    TriggerEvent,
)
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.core.series import OHLCV, Bar, Quote, TimeSeries, Trade, time_as_nsec
from qubx.data.helpers import InMemoryCachedReader, TimeGuardedWrapper
from qubx.data.readers import (
    AsTimestampedRecords,
    DataReader,
    InMemoryDataFrameReader,
)
from qubx.utils.misc import ProgressParallel
from qubx.utils.time import infer_series_frequency


class SimulatedTrading(ITradingServiceProvider):
    """
    First implementation of a simulated broker.
    TODO:
        1. Add margin control (in account processor)
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
        account_processor: IAccountProcessor,
        exchange_name: str,
        commissions: str | None = None,
        simulation_initial_time: dt_64 | str = np.datetime64(0, "ns"),
        accurate_stop_orders_execution: bool = False,
    ) -> None:
        """
        This function sets up a simulated trading environment with following parameters.

        Parameters:
        -----------
        account_processor: IAccountProcessor
            The account processor to be used for the simulation.
        exchange_name : str
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
        self.acc = account_processor
        self._current_time = (
            np.datetime64(simulation_initial_time, "ns")
            if isinstance(simulation_initial_time, str)
            else simulation_initial_time
        )
        self._name = exchange_name
        self._ome = {}
        self._fees_calculator = lookup.fees.find(exchange_name.lower(), commissions)
        self._half_tick_size = {}
        self._fill_stop_order_at_price = accurate_stop_orders_execution

        self._order_to_instrument = {}
        if self._fees_calculator is None:
            raise ValueError(
                f"SimulatedExchangeService :: Fees configuration '{commissions}' is not found for '{exchange_name}' !"
            )

        # - we want to see simulate time in log messages
        QubxLogConfig.setup_logger(QubxLogConfig.get_log_level(), SimulatedLogFormatter(self).formatter)
        if self._fill_stop_order_at_price:
            logger.info(f"{self.__class__.__name__} emulates stop orders executions at exact price")

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
            self.acc.process_order(order)

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
        self._half_tick_size[instrument] = instrument.tick_size / 2  # type: ignore
        self.acc.attach_positions(position)
        return self.acc.positions[instrument]

    def time(self) -> dt_64:
        return self._current_time

    def get_base_currency(self) -> str:
        return self.acc.get_base_currency()

    def get_name(self) -> str:
        return self._name

    def get_account_id(self) -> str:
        return self.acc.account_id

    def process_execution_report(self, instrument: Instrument, report: Dict[str, Any]) -> Tuple[Order, List[Deal]]:
        order = report["order"]
        deals = report.get("deals", [])
        self.acc.process_deals(instrument, deals)
        self.acc.process_order(order)
        return order, deals

    def emulate_quote_from_data(
        self, instrument: Instrument, timestamp: dt_64, data: float | Trade | Bar
    ) -> Quote | None:
        if instrument not in self._half_tick_size:
            _ = self.get_position(instrument)

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


class SimulatedExchange(IBrokerServiceProvider):
    trading_service: SimulatedTrading
    _last_quotes: Dict[Instrument, Optional[Quote]]
    _scheduler: BasicScheduler
    _current_time: dt_64
    _pregenerated_signals: dict[Instrument, pd.Series | pd.DataFrame]
    _to_process: dict[Instrument, list]
    _data_source: IterableSimulationData

    def __init__(
        self, exchange_id: str, trading_service: SimulatedTrading, reader: DataReader, open_close_time_indent_secs=1
    ):
        super().__init__(exchange_id, trading_service)
        self._reader = reader
        exchange_id = exchange_id.lower()

        # - create exchange's instance
        self._last_quotes = defaultdict(lambda: None)
        self._current_time = self.trading_service.time()

        # - setup communication bus
        self.set_communication_channel(bus := SimulatedCtrlChannel("databus", sentinel=(None, None, None)))
        self.trading_service.set_communication_channel(bus)

        # - simulated scheduler
        self._scheduler = SimulatedScheduler(bus, lambda: self.trading_service.time().item())

        # - pregenerated signals storage
        self._pregenerated_signals = dict()
        self._to_process = {}

        # - simulation data source
        self._data_source = IterableSimulationData(
            self._reader, open_close_time_indent_secs=open_close_time_indent_secs
        )

        logger.info(f"{self.__class__.__name__}.{exchange_id} is initialized")

    def subscribe(self, subscription_type: str, instruments: set[Instrument], reset: bool) -> None:
        logger.debug(f" | subscribe: {subscription_type} -> {instruments}")
        self._data_source.add_instruments_for_subscription(subscription_type, list(instruments))

    def unsubscribe(self, subscription_type: str, instruments: set[Instrument] | Instrument | None = None) -> None:
        logger.debug(f" | unsubscribe: {subscription_type} -> {instruments}")
        if instruments is not None:
            self._data_source.remove_instruments_from_subscription(
                subscription_type, [instruments] if isinstance(instruments, Instrument) else list(instruments)
            )

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        return self._data_source.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument) -> list[str]:
        _s_lst = self._data_source.get_subscriptions_for_instrument(instrument)
        logger.debug(f" | get_subscriptions {instrument} -> {_s_lst}")
        return _s_lst

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        _in_lst = self._data_source.get_instruments_for_subscription(subscription_type or Subtype.ALL)
        logger.debug(f" | get_subscribed_instruments {subscription_type} -> {_in_lst}")
        return _in_lst

    def warmup(self, configs: dict[tuple[str, Instrument], str]) -> None:
        for si, warm_period in configs.items():
            logger.debug(f" | Warming up {si} -> {warm_period}")
            self._data_source.set_warmup_period(si[0], warm_period)

    def _prepare_generated_signals(self, start: str | pd.Timestamp, end: str | pd.Timestamp):
        for s, v in self._pregenerated_signals.items():
            _s_inst = None

            for i in self.get_subscribed_instruments():
                # - we can process series with variable id's if we can find some similar instrument
                if s == i.symbol or s == str(i) or s == f"{i.exchange}:{i.symbol}" or str(s) == str(i):
                    sel = v[pd.Timestamp(start) : pd.Timestamp(end)]
                    self._to_process[i] = list(zip(sel.index, sel.values))
                    _s_inst = i
                    break

            if _s_inst is None:
                logger.error(f"Can't find instrument for pregenerated signals with id {s}")
                raise ValueError(f"Can't find instrument for pregenerated signals with id {s}")

    def run(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        silent: bool = False,
        enable_event_batching: bool = True,
    ) -> None:
        logger.info(f"{self.__class__.__name__} ::: Simulation started at {start} :::")

        if self._pregenerated_signals:
            self._prepare_generated_signals(start, end)
            _run = self._run_generated_signals
            enable_event_batching = False  # no batching for pre-generated signals
        else:
            _run = self._run_as_strategy

        qiter = EventBatcher(self._data_source.create_iterable(start, end), passthrough=not enable_event_batching)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        total_duration = end - start
        update_delta = total_duration / 100
        prev_dt = pd.Timestamp(start)

        if silent:
            for instrument, data_type, event in qiter:
                if not _run(instrument, data_type, event):
                    break
        else:
            _p = 0
            with tqdm(total=100, desc="Simulating", unit="%", leave=False) as pbar:
                for instrument, data_type, event in qiter:
                    if not _run(instrument, data_type, event):
                        break
                    dt = pd.Timestamp(event.time)
                    # update only if date has changed
                    if dt - prev_dt > update_delta:
                        _p += 1
                        pbar.n = _p
                        pbar.refresh()
                        prev_dt = dt
                pbar.n = 100
                pbar.refresh()

        logger.info(f"{self.__class__.__name__} ::: Simulation finished at {end} :::")

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

        # - TODO: not sure why we need it here ???
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
            self._reader.read(data_id=_spec, start=start, stop=end, transform=AsTimestampedRecords()),  # type: ignore
            time_as_nsec(self.time()),
            _timeframe.asm8.item(),
        )

    def _convert_records_to_bars(self, records: List[Dict[str, Any]], cut_time_ns: int, timeframe_ns: int) -> List[Bar]:
        """
        Convert records to bars and we need to cut last bar up to the cut_time_ns
        """
        bars = []

        _data_tf = infer_series_frequency([r["timestamp_ns"] for r in records[:100]])
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
            self._pregenerated_signals[str(signals.name)] = signals  # type: ignore

        elif isinstance(signals, pd.DataFrame):
            for col in signals.columns:
                self._pregenerated_signals[col] = signals[col]  # type: ignore
        else:
            raise ValueError("Invalid signals or strategy configuration")


def simulate(
    config: VariableStrategyConfig,
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
    open_close_time_indent_secs=1,
    debug: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = "WARNING",
) -> list[TradingSessionResult]:
    """
    Backtest utility for trading strategies or signals using historical data.

    Parameters:
    ----------

    config (VariableStrategyConfig):
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
            "No exchange information provided - you can specify it by exchange parameter or use <yellow>EXCHANGE:SYMBOL</yellow> format for symbols"
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
    setups = recognize_simulation_configuration(
        "", config, _instrs, exchange, capital, leverage, base_currency, commissions
    )
    if not setups:
        logger.error(
            "Can't recognize setup - it should be a strategy, a set of signals or list of signals/strategies + tracker !"
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
        open_close_time_indent_secs=open_close_time_indent_secs,
    )


class SignalsProxy(IStrategy):
    timeframe: str = "1m"

    def on_init(self, ctx: IStrategyContext):
        ctx.set_base_subscription(Subtype.OHLC[self.timeframe])

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
    open_close_time_indent_secs=1,
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
            open_close_time_indent_secs=open_close_time_indent_secs,
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
    open_close_time_indent_secs=1,
    account_id: str = "Simulated0",
) -> TradingSessionResult:
    _stop = stop
    logger.debug(
        f"<red>{pd.Timestamp(start)}</red> Initiating simulated trading for {setup.exchange} for {setup.capital} x {setup.leverage} in {setup.base_currency}..."
    )
    account = BasicAccountProcessor(
        account_id=account_id,
        base_currency=setup.base_currency,
        initial_capital=setup.capital,
    )
    trading_service = SimulatedTrading(
        account,
        setup.exchange,
        setup.commissions,
        np.datetime64(start, "ns"),
        accurate_stop_orders_execution=accurate_stop_orders_execution,
    )
    broker = SimulatedExchange(
        setup.exchange, trading_service, data_reader, open_close_time_indent_secs=open_close_time_indent_secs
    )

    # - it will store simulation results into memory
    logs_writer = InMemoryLogsWriter("test", setup.name, "0")
    strat: IStrategy | None = None

    match setup.setup_type:
        case SetupTypes.STRATEGY:
            strat = setup.generator  # type: ignore

        case SetupTypes.STRATEGY_AND_TRACKER:
            strat = setup.generator  # type: ignore
            strat.tracker = lambda ctx: setup.tracker  # type: ignore

        case SetupTypes.SIGNAL:
            strat = SignalsProxy(timeframe=signal_timeframe)
            broker.set_generated_signals(setup.generator)  # type: ignore
            # - we don't need any unexpected triggerings
            _stop = setup.generator.index[-1]  # type: ignore

            # - no historical data for generated signals, so disable it
            enable_event_batching = False

        case SetupTypes.SIGNAL_AND_TRACKER:
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

    # - get strategy parameters for this run
    _s_class, _s_params = "", None
    if setup.setup_type in [SetupTypes.STRATEGY, SetupTypes.STRATEGY_AND_TRACKER]:
        _s_params = extract_parameters_from_object(setup.generator)
        _s_class = full_qualified_class_name(setup.generator)

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
        strategy_class=_s_class,
        parameters=_s_params,
        is_simulation=True,
    )
