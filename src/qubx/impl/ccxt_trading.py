from typing import Any, Dict, List, Optional, Tuple

from collections import defaultdict
import stackprinter
import traceback

import ccxt
import ccxt.pro as cxp
from ccxt.base.decimal_to_precision import ROUND_UP
from ccxt.base.exchange import Exchange

import numpy as np
import pandas as pd

from qubx import logger, lookup
from qubx.core.account import AccountProcessor
from qubx.core.basics import Instrument, Position, Order, TransactionCostsCalculator, dt_64, Deal
from qubx.core.strategy import IDataProvider, CtrlChannel, IExchangeServiceProvider
from qubx.core.series import TimeSeries, Bar, Trade, Quote
from qubx.impl.ccxt_utils import EXCHANGE_ALIASES, ccxt_convert_order_info, ccxt_convert_deal_info, ccxt_extract_deals_from_exec, ccxt_restore_position_from_deals


ORDERS_HISTORY_LOOKBACK_DAYS = 30


class CCXTSyncTradingConnector(IExchangeServiceProvider):
    """
    Synchronous instance of trading API
    """
    sync: Exchange

    _fees_calculator: Optional[TransactionCostsCalculator] = None    # type: ignore
    _positions: Dict[str, Position]

    def __init__(self, 
                 exchange_id: str, 
                 account_id: str,
                 base_currency: str | None, commissions: str|None = None, 
                 reserves: Dict[str, float] | None = None,
                 **exchange_auth):
        if base_currency is None:
            raise ValueError("Base currency is not specified !")

        exchange_id = exchange_id.lower()
        exch = EXCHANGE_ALIASES.get(exchange_id, exchange_id)

        if exch not in ccxt.exchanges:
            raise ValueError(f"Exchange {exchange_id} -> {exch} is not supported by CCXT!")

        # - sync exchange
        self.sync: Exchange = getattr(ccxt, exchange_id.lower())(exchange_auth)
        self.acc = AccountProcessor(account_id, base_currency, reserves)

        logger.info(f"{exch.upper()} loading ...")
        self.sync.load_markets()        
        self._sync_account_info(commissions)
        self._positions = self.acc._positions

        # - show reserves info
        for s, v in self.acc.reserved.items():
            logger.info(f" > {v} of {s} is reserved from trading")

    def _sync_account_info(self, default_commissions: str | None):
        logger.info(f'Loading account data for {self.get_name()}')
        self._balance = self.sync.fetch_balance()
        _info = self._balance.get('info')

        # - check what we have on balance: TODO test on futures account
        for k, vol in self._balance['total'].items(): # type: ignore
            if k.lower() == self.acc.base_currency.lower():
                _free = self._balance['free'][self.acc.base_currency]
                self.acc.update_base_balance(vol, vol - _free)

            if vol != 0.0: # - get all non zero balances 
                self.acc.update_balance(k, vol, vol - self._balance['free'].get(k, 0))

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

    def _get_open_orders_from_exchange(self, symbol: str, days_before: int = 60) -> Dict[str, Order]:
        """
        We need only open orders to restore list of active ones in connector 
        method returns open orders sorted by creation time in ascending order
        """
        t_orders_start_ms = ((self.time() - days_before * pd.Timedelta('1d')).asm8.item() // 1000000)
        orders_data = self.sync.fetch_open_orders(symbol, since=t_orders_start_ms)
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
        deals_data = self.sync.fetch_my_trades(symbol, since=t_orders_start_ms)
        deals: List[Deal] = [ccxt_convert_deal_info(o) for o in deals_data] # type: ignore
        if deals:
            return list(sorted(deals, key=lambda x: x.time, reverse=False))
        return list()

    def _sync_position_and_orders(self, position: Position) -> Position:
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
            position = ccxt_restore_position_from_deals(position, vol_from_exch, deals, reserved_amount=self.acc.get_reserved(position.instrument));

        return position

    def get_position(self, instrument: Instrument) -> Position:
        symbol = instrument.symbol

        if symbol not in self._positions:
            position = Position(instrument, self._fees_calculator)  # type: ignore
            position = self._sync_position_and_orders(position)
            self.acc.attach_positions(position)

        return self._positions[symbol] 

    def update_position_price(self, symbol: str, price: float):
        self.acc.update_position_price(self.time(), symbol, price)

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
            r: Dict[str, Any] | None = self.sync.create_order(
                symbol, order_type, order_side, amount, price, # type: ignore
                params=params)
        except ccxt.BadRequest as exc:
            logger.error(f"(CCXTSyncTradingConnector::send_order) BAD REQUEST for {order_side} {amount} {order_type} for {symbol} : {exc}")
            raise exc
        except Exception as err:
            logger.error(f"(CCXTSyncTradingConnector::send_order) {order_side} {amount} {order_type} for {symbol} exception : {err}")
            logger.error(traceback.format_exc())
            raise err

        if r is not None:
            order = ccxt_convert_order_info(symbol, r) 
            logger.info(f"(CCXTSyncTradingConnector) New order {order}")
            return order

        return None

    def cancel_order(self, order_id: str) -> Order | None:
        order = None
        if order_id in self.acc._active_orders:
            order = self.acc._active_orders[order_id]
            try:
                logger.info(f"Canceling order {order_id} ...")
                # r = self._task_s(self.exchange.cancel_order(order_id, symbol=order.symbol))
                r = self.sync.cancel_order(order_id, symbol=order.symbol)
            except Exception as err:
                logger.error(f"(CCXTSyncTradingConnector) canceling [{order}] exception : {err}")
                logger.error(traceback.format_exc())
                raise err
        return order

    def time(self) -> dt_64:
        """
        Returns current time in nanoseconds
        """
        return np.datetime64(self.sync.microseconds() * 1000, 'ns')

    def get_name(self) -> str:
        return self.sync.name  # type: ignore

    def _process_execution_report(self, symbol: str, report: Dict[str, Any]) -> Tuple[Order, List[Deal]]:
        order = ccxt_convert_order_info(symbol, report)
        deals = ccxt_extract_deals_from_exec(report)
        self.acc.process_deals(symbol, deals)
        self.acc.process_order(order)
        return order, deals

    def get_orders(self, symbol: str | None = None) -> List[Order]:
        return self.acc.get_orders(symbol)

    def get_base_currency(self) -> str:
        return self.acc.base_currency