from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from qubx import lookup, logger
from qubx.core.series import Quote
from qubx.core.account import AccountProcessor
from qubx.core.basics import Instrument, Deal, Order, Position, TransactionCostsCalculator, dt_64
from qubx.core.series import TimeSeries, Trade, Quote, Bar, OHLCV
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.backtester.ome import OrdersManagementEngine, OmeReport

from qubx.data.readers import DataReader, DataTransformer, RestoreTicksFromOHLC, AsQuotes
from qubx.pandaz.utils import scols


class SimulatedExchangeService(IExchangeServiceProvider):
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
        self.get_communication_channel().queue.put((symbol, "order", report.order))
        if report.exec is not None:
            self.get_communication_channel().queue.put((symbol, "deals", [report.exec]))

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
        self._symbol = f"{instrument.exchange}:{instrument.symbol}"
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
            data_id=self._symbol,
            start=start,
            stop=end,
            transform=self._transformer,
        )

        if self._timeframe:
            args["timeframe"] = self._timeframe

        return self._reader.read(**args)  # type: ignore


class SimulatedDataProvider(IDataProvider):
    _service_provider: IExchangeServiceProvider
    _last_quotes: Dict[str, Optional[Quote]]
    _current_time: dt_64
    _hist_data_type: str
    _loader: Dict[str, DataLoader]

    def __init__(
        self,
        exchange_id: str,
        service_provider: IExchangeServiceProvider,
        reader: DataReader,
        hist_data_type: str = "ohlc",
    ):
        self._service_provider = service_provider
        self._reader = reader
        self._hist_data_type = hist_data_type
        exchange_id = exchange_id.lower()

        # - create exchange's instance
        self._last_quotes = defaultdict(lambda: None)
        self._current_time = np.datetime64(0, "ns")
        self._loader = {}

        logger.info(f"SimulatedData.{exchange_id} initialized")

    @property
    def is_simulated(self) -> bool:
        return True

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
        for t, u in zip(merged.index, merged.values):
            for i, s in enumerate(merged.columns):
                q = u[i]
                if q:
                    self.get_communication_channel().queue.put((s, "quote", q))
                    self._current_time = max(np.datetime64(t, "ns"), self._current_time)
                    self._service_provider.update_position_price(s, self._current_time, q)
        return merged

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quotes[symbol]

    def close(self):
        pass

    def time(self) -> dt_64:
        return self._current_time
