from typing import Any, Dict, List, Optional

import asyncio
from asyncio.tasks import Task
from asyncio.events import AbstractEventLoop
from collections import defaultdict
import stackprinter
import traceback

import ccxt.pro as cxp
from ccxt.base.decimal_to_precision import ROUND_UP
from ccxt.base.exchange import Exchange
from ccxt import NetworkError

import re
import numpy as np
import pandas as pd

from qubx import logger, lookup
from qubx.core.account import AccountProcessor
from qubx.core.basics import Instrument, Position, Order, TransactionCostsCalculator, dt_64, Deal
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.impl.utils import ccxt_convert_order_info, ccxt_convert_deal_info, ccxt_extract_deals_from_exec, ccxt_restore_position_from_deals


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
    acc: AccountProcessor

    _positions: Dict[str, Position]
    _fees_calculator: Optional[TransactionCostsCalculator] = None    # type: ignore

    _ch_market_data: CtrlChannel
    _last_quotes: Dict[str, Optional[Quote]]
    _loop: AbstractEventLoop 

    def __init__(self, exchange_id: str, base_currency: str, commissions: str|None = None, **exchange_auth):
        super().__init__()
        
        exchange_id = exchange_id.lower()
        exch = _aliases.get(exchange_id, exchange_id)
        if exch not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange_id} -> {exch} is not supported by CCXT!")

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.info("started new event loop")
            # exchange_auth |= {'asyncio_loop':self._loop}

        self.exchange = getattr(cxp, exch)(exchange_auth)
        self.subsriptions: Dict[str, List[str]] = defaultdict(list)
        # self._base_currency = base_currency
        self._ch_market_data = CtrlChannel(exch + '.marketdata')
        self._last_quotes = defaultdict(lambda: None)

        # - load all needed information
        self.acc = AccountProcessor(base_currency)
        self._sync_account_info(commissions)

        # - positions
        self._positions = self.acc._positions

        logger.info(f"{self.get_name().upper()} initialized - current time {self.time()}")

    def subscribe(self, subscription_type: str, symbols: List[str], timeframe:Optional[str]=None, nback:int=0) -> bool:
        to_process = self._check_existing_subscription(subscription_type.lower(), symbols)
        if not to_process:
            logger.info(f"Symbols {symbols} already subscribed on {subscription_type} data")
            return False

        # - subscribe to market data updates
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

            case 'trades':
                raise ValueError("TODO")

            case 'quotes':
                raise ValueError("TODO")

            case _:
                raise ValueError("TODO")

        # - subscibe to executions reports
        for s in to_process:
            self._task_a(self._listen_to_execution_reports(self.get_communication_channel(), s))

        return True

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
        start = ((self.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item()//1000000) if nbarsback > 1 else None 
        return await self.exchange.fetch_ohlcv(symbol, timeframe, since=start, limit=nbarsback + 1)        # type: ignore

    def get_historical_ohlcs(self, symbol: str, timeframe: str, nbarsback: int) -> Optional[List[Bar]]:
        assert nbarsback > 1
        # we want to wait until initial snapshot is arrived so run it in sync mode
        r = self._task_s(self._fetch_ohlcs_a(symbol, self._get_exch_timeframe(timeframe), nbarsback))
        if len(r) > 0:
            return [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in r]

    def send_order(
        self, instrument: Instrument, order_side: str, order_type: str, amount: float, price: float | None = None, 
        client_id: str | None = None, time_in_force: str='gtc'
    ) -> Optional[Order]:
        params={}
        symbol = instrument.symbol

        if order_type == 'limit':
            params['timeInForce'] = time_in_force.upper()
            if price is None:
                raise ValueError('Price must be specified for limit order')

        if client_id:
            params['newClientOrderId'] = client_id

        try:
            r: Dict[str, Any] | None = self._task_s(self.exchange.create_order(
                symbol, order_type, order_side, amount, price, # type: ignore
                params=params)
            )
        except Exception as err:
            logger.error(f"(CCXTConnector) send_order exception : {err}")
            logger.error(traceback.format_exc())
            raise err

        if r is not None:
            order = ccxt_convert_order_info(symbol, r) 
            logger.info(f"(CCXTConnector) New order {order}")
            return order

        return None

    def cancel_order(self, order_id: str) -> Order | None:
        order = None
        if order_id in self.acc._active_orders:
            order = self.acc._active_orders[order_id]
            try:
                logger.info(f"Canceling order {order_id} ...")
                r = self._task_s(self.exchange.cancel_order(order_id, symbol=order.symbol))
            except Exception as err:
                logger.error(f"(CCXTConnector) cancel_order exception : {err}")
                logger.error(traceback.format_exc())
                raise err
        return order

    async def _listen_to_execution_reports(self, channel: CtrlChannel, symbol: str):
        while channel.control.is_set():
            try:
                exec = await self.exchange.watch_orders(symbol)        # type: ignore
                _msg = f"\nexecs_{symbol} = [\n"
                for report in exec:
                    _msg += '\t' + str(report) + ',\n'
                    order = self._process_execution_report(symbol, report)
                    # - send update to client 
                    channel.queue.put((symbol, order))
                logger.info(_msg + "]\n")
            except NetworkError as e:
                logger.error(f"(CCXTConnector) NetworkError in _listen_to_execution_reports : {e}")
                await asyncio.sleep(1)
                continue

            except Exception as err:
                logger.error(f"(CCXTConnector) exception in _listen_to_execution_reports : {err}")
                logger.error(stackprinter.format(err))

    def _process_execution_report(self, symbol: str, report: Dict[str, Any]) -> Order:
        order = ccxt_convert_order_info(symbol, report)
        deals = ccxt_extract_deals_from_exec(report)
        self.acc.process_deals(symbol, deals)
        self.acc.process_order(order)
        return order

    async def _listen_to_ohlcv(self, channel: CtrlChannel, symbol: str, timeframe: str, nbarsback: int):
        # - check if we need to load initial 'snapshot'
        if nbarsback > 1:
            # ohlcv = asyncio.run(self._fetch_ohlcs_a(symbol, timeframe, nbarsback))
            ohlcv = self._task_s(self._fetch_ohlcs_a(symbol, timeframe, nbarsback))
            for oh in ohlcv:
                channel.queue.put((symbol, Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))
            logger.info(f"{symbol}: loaded {len(ohlcv)} {timeframe} bars")

        while channel.control.is_set():
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe)        # type: ignore

                # - update positions by actual close price
                self.acc.update_position_price(self.time(), symbol, ohlcv[-1][4])

                for oh in ohlcv:
                    channel.queue.put((symbol, Bar(oh[0] * 1000000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])))

            except NetworkError as e:
                logger.error(f"(CCXTConnector) NetworkError in _listen_to_ohlcv : {e}")
                await asyncio.sleep(1)
                continue

            except Exception as e:
                # logger.error(str(e))
                logger.error(f"(CCXTConnector) exception in _listen_to_ohlcv : {e}")
                logger.error(stackprinter.format(e))
                await self.exchange.close()        # type: ignore
                raise e

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
            _t = re.match(r'(\d+)(\w+)', timeframe)
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
            self._task_s(self.exchange.close()) # type: ignore
        except Exception as e:
            logger.error(e)

    def _sync_account_info(self, default_commissions: str | None):
        logger.info(f'Loading account data for {self.get_name()}')
        self._balance = self._task_s(self.exchange.fetch_balance())
        _info = self._balance.get('info')

        # - check what we have on balance
        for k, vol in self._balance['total'].items():
            if k.lower() == self.acc.base_currency.lower():
                _free = self._balance['free'][self.acc.base_currency]
                self.acc.update_balance(vol, vol - _free)

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
            self.acc.attach_positions(position)

        return self._positions[symbol] 

    def get_capital(self) -> float:
        return self.acc.get_capital()

    def _get_open_orders_from_exchange(self, symbol: str, days_before: int = 60) -> Dict[str, Order]:
        """
        We need only open orders to restore list of active ones in connector 
        method returns open orders sorted by creation time in ascending order
        """
        t_orders_start_ms = ((self.time() - days_before * pd.Timedelta('1d')).asm8.item() // 1000000)
        # orders_data = self._task_s(self.exchange.fetch_orders(symbol, since=t_orders_start_ms))
        orders_data = self._task_s(self.exchange.fetch_open_orders(symbol, since=t_orders_start_ms))
        orders: Dict[str, Order] = {}
        for o in orders_data:
            order = ccxt_convert_order_info(symbol, o) 
            orders[order.id] = order
        if orders: 
            logger.info(f"{symbol} - loaded {len(orders)} open orders")
        return dict(sorted(orders.items(), key=lambda x: x[1].time, reverse=False))

    def _get_deals_from_exchange(self, symbol: str, days_before: int = 60) -> List[Deal]:
        """
        Load trades for given symbol
        method returns account's trades sorted by creation time in reversed order (latest - first)
        """
        t_orders_start_ms = ((self.time() - days_before * pd.Timedelta('1d')).asm8.item() // 1000000)
        deals_data = self._task_s(self.exchange.fetch_my_trades(symbol, since=t_orders_start_ms))
        deals: List[Deal] = [ccxt_convert_deal_info(o) for o in deals_data]
        if deals:
            return list(sorted(deals, key=lambda x: x.time, reverse=False))
        return list()

    def sync_position_and_orders(self, position: Position) -> Position:
        asset = position.instrument.base
        symbol = position.instrument.symbol
        total_amnts = self._balance['total']
        vol_from_exch = total_amnts.get(asset, total_amnts.get(symbol, 0))

        # - get orders from exchange
        orders = self._get_open_orders_from_exchange(position.instrument.symbol, ORDERS_HISTORY_LOOKBACK_DAYS)
        self.acc.add_active_orders(orders)

        # - get deals from exchange if position is not zero
        if vol_from_exch != 0:
            deals = self._get_deals_from_exchange(symbol)

            # - actualize position
            position = ccxt_restore_position_from_deals(position, vol_from_exch, deals);

        return position

    def get_orders(self, symbol: str | None = None) -> List[Order]:
        return self.acc.get_orders(symbol)

    def get_base_currency(self) -> str:
        return self.acc.base_currency