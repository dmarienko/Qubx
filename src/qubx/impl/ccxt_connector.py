from typing import Any, Dict, List, Optional

import asyncio
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from collections import defaultdict

import ccxt.pro as cxp
from ccxt.base.decimal_to_precision import ROUND_UP
from ccxt.base.exchange import Exchange
from ccxt import NetworkError
import re
import numpy as np
import pandas as pd

from qubx import logger, lookup
from qubx.core.basics import Instrument, Position, Order, TransactionCostsCalculator, dt_64
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote


_aliases = {
    'binance': 'binanceqv',
    'binance.um': 'binanceusdm',
    'binance.cm': 'binancecoinm',
    'kraken.f': 'krakenfutures'
}

# - register custom wrappers
from .exchange_customizations import BinanceQV
cxp.binanceqv = BinanceQV            # type: ignore
cxp.exchanges.append('binanceqv')

ORDERS_HISTORY_LOOKBACK_DAYS = 30


class CCXTConnector(IDataProvider, IExchangeServiceProvider):
    exchange: Exchange
    subsriptions: Dict[str, List[str]]

    _positions: Dict[str, Position]
    _base_currency: str
    _fees_calculator: Optional[TransactionCostsCalculator] = None    # type: ignore
    _total_capital_in_base: float = 0.0
    _locked_capital_in_base: float = 0.0

    _ch_market_data: CtrlChannel
    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop 
    _active_orders: Dict[str, Order]                            # active orders

    def __init__(self, exchange_id: str, base_currency: str, commissions: str|None = None, **exchange_auth):
        super().__init__()
        exchange_id = exchange_id.lower()
        exch = _aliases.get(exchange_id, exchange_id)
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange_id} -> {exch} is not supported by CCXT!")

        self.exchange = getattr(cxp, exch)(exchange_auth)
        self.subsriptions: Dict[str, List[str]] = defaultdict(list)
        self._base_currency = base_currency
        self._ch_market_data = CtrlChannel(exch + '.marketdata')
        self._last_quotes = defaultdict(lambda: None)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # - positions
        self._positions = {}

        # - load all needed information
        self._sync_account_info(commissions)

        logger.info(f"{self.get_name().upper()} initialized - current time {self.time()}")

    def subscribe(self, subscription_type: str, symbols: List[str], timeframe:Optional[str]=None, nback:int=0) -> bool:
        to_process = self._check_existing_subscription(subscription_type.lower(), symbols)
        if not to_process:
            logger.info(f"Symbols {symbols} already subscribed on {subscription_type} data")
            return False

        match sbscr := subscription_type.lower():
            case 'ohlc':
                if timeframe is None:
                    raise ValueError("timeframe must not be None for OHLC data subscription")

                # convert to exchange format
                tframe = self._get_exch_timeframe(timeframe)
                for s in to_process:
                    self._task_a(self._listen_to_ohlcv(self.get_communication_channel(), s, tframe, nback))
                    self.subsriptions[sbscr].append(s.lower())
                logger.info(f'Subscribed on {sbscr} updates for {len(to_process)} symbols: \n\t\t{to_process}')
                return True

            case 'trades':
                raise ValueError("TODO")

            case 'quotes':
                raise ValueError("TODO")

            case _:
                raise ValueError("TODO")

        return False

    def get_communication_channel(self) -> CtrlChannel:
        return self._ch_market_data

    def _check_existing_subscription(self, subscription_type, symbols: List[str]) -> List[str]:
        subscribed = self.subsriptions[subscription_type]
        to_subscribe = []
        for s in symbols: 
            if s not in subscribed:
                to_subscribe.append(s)
        return to_subscribe

    async def _fetch_ohlcs_a(self, symbol: str, timeframe: str, nbarsback: int):
        assert nbarsback > 1
        start = ((pd.Timestamp('now', tz='UTC') - nbarsback * pd.Timedelta(timeframe)).asm8.item()//1000000) if nbarsback > 1 else None 
        return await self.exchange.fetch_ohlcv(symbol, timeframe, since=start, limit=nbarsback + 1)        # type: ignore

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> Optional[List[Bar]]:
        assert nbarsback > 1
        # we want to wait until initial snapshot is arrived so run it in sync mode
        r = self._task_s(self._fetch_ohlcs_a(symbol, self._get_exch_timeframe(timeframe), nbarsback))
        if len(r) > 0:
            return [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in r]

    async def _listen_to_ohlcv(self, channel: CtrlChannel, symbol: str, timeframe: str, nbarsback: int):
        # - check if we need to load initial 'snapshot'
        if nbarsback > 1:
            # ohlcv = asyncio.run(self._fetch_ohlcs_a(symbol, timeframe, nbarsback))
            ohlcv = self._task_s(self._fetch_ohlcs_a(symbol, timeframe, nbarsback))
            for oh in ohlcv:
                channel.queue.put((symbol, Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

        while channel.control.is_set():
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe)        # type: ignore
                # update positions by actual close price
                self._update_position_price_for_symbol(symbol, ohlcv[-1][4])
                for oh in ohlcv:
                    channel.queue.put((symbol, Bar(oh[0] * 1000000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

            except NetworkError as e:
                logger.error(str(e))
                await asyncio.sleep(1)
                continue

            except Exception as e:
                logger.error(str(e))
                await self.exchange.close()        # type: ignore
                raise e

    def _update_position_price_for_symbol(self, symbol: str, price: float):
        p = self._positions[symbol]
        p.update_market_price(self.time(), price, 1)

    def time(self) -> dt_64:
        """
        Returns current time in nanoseconds
        """
        return np.datetime64(self.exchange.microseconds() * 1000, 'ns')

    def get_name(self) -> str:
        return self.exchange.name  # type: ignore

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quotes[symbol]

    def _get_exch_timeframe(self, timeframe: str):
        if timeframe is not None:
            _t = re.match('(\d+)(\w+)', timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self.exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self.get_name()}")

        return tframe

    def _task_a(self, coro) -> Task:
        return self._loop.create_task(coro)

    def _task_s(self, coro) -> Any:
        return self._loop.run_until_complete(coro)

    def close(self):
        try:
            self._task_s(self.exchange.close())
        except Exception as e:
            logger.error(e)

    def _sync_account_info(self, default_commissions: str | None):
        logger.info(f'Loading account data for {self.get_name()}')
        self._balance = self._task_s(self.exchange.fetch_balance())
        self._active_orders = dict()
        _info = self._balance.get('info')

        # - check what we have on balance
        for k, vol in self._balance['total'].items():
            if k.lower() == self._base_currency.lower():
                self._total_capital_in_base = vol
                _free = self._balance['free'][self._base_currency]
                self._locked_capital_in_base = self._total_capital_in_base - _free 

        # - try to get account's commissions calculator or set default one
        if _info:
           _fees = _info.get('commissionRates')
           if _fees:
               self._fees_calculator = TransactionCostsCalculator('account', 100*float(_fees['maker']), 100*float(_fees['taker'])) 

        if self._fees_calculator is None:
            if default_commissions:
                self._fees_calculator = lookup.fees.find(self.get_name().lower(), default_commissions)
            else: 
                raise ValueError("Can't get commissions level from account, but default commissions is not defined !")

    def get_position(self, instrument: Instrument) -> Position:
        symbol = instrument.symbol

        if symbol not in self._positions:
            position = Position(instrument, self._fees_calculator)  # type: ignore
            position = self.sync_position_and_orders(position)
            self._positions[symbol] = position

        return self._positions[symbol] 

    def get_capital(self) -> float:
        # TODO: need to take in account leverage and funds currently locked 
        return self._total_capital_in_base - self._locked_capital_in_base 

    def _get_orders_from_exchange(self, symbol: str, days_before: int = 60):
        t_orders_start_ms = ((self.time() - days_before * pd.Timedelta('1d')).asm8.item() // 1000000)
        orders_data = self._task_s(self.exchange.fetch_orders(symbol, since=t_orders_start_ms))
        orders: Dict[str, Order] = {}
        for o in orders_data:
            oi = o['info']
            avg = o.get('average')
            orders[oi['orderId']] = Order(
                id=oi['orderId'],
                type=oi['type'],
                symbol=symbol,
                time = pd.Timestamp(o['timestamp'], unit='ms'), # type: ignore
                quantity = float(oi['origQty']), 
                price= float(oi['price']), 
                side = oi['side'],
                status = oi['status'],
                time_in_force = oi['timeInForce'],
                client_id = oi['clientOrderId'],
                cost = float(o['cost']),
                executed_quantity = float(oi['executedQty']), 
                executed_price = float(avg) if avg is not None else None, 
            )
        return dict(sorted(orders.items(), key=lambda x: x[1].time, reverse=False))

    def sync_position_and_orders(self, position: Position) -> Position:
        asset = position.instrument.base
        symbol = position.instrument.symbol
        total_amnts = self._balance['total']
        vol_from_exch = total_amnts.get(asset, total_amnts.get(symbol, 0))

        # - get orders from exchange
        orders = self._get_orders_from_exchange(position.instrument.symbol, ORDERS_HISTORY_LOOKBACK_DAYS)

        # - get orders from exchange
        last_pos_orders = []
        for oid, od in reversed(orders.items()):
            if od.status == 'FILLED' or od.status == 'PARTIALLY_FILLED':
                signed_amount = +od.executed_quantity if od.side == 'BUY' else -od.executed_quantity
                vol_from_exch -= signed_amount
                if vol_from_exch >= 0:
                    last_pos_orders.append(od)

            # - store active order into local cache
            if od.status != 'FILLED' and od.status != 'CANCELED':
                self._active_orders[oid] = od

        # - actualize position
        position.reset()

        if vol_from_exch > 0:
            logger.warning(f"Couldn't restore full execution history for {symbol} symbol. Qubx will use zero position !")
        else:
            # - restore position
            for o in reversed(last_pos_orders):
                signed_amount = +o.executed_quantity if o.side == 'BUY' else -o.executed_quantity
                position.change_position_by(o.time.as_unit('ns').asm8, signed_amount, o.executed_price, aggressive=o.type=='MARKET')

        return position

    def get_base_currency(self) -> str:
        return self._base_currency