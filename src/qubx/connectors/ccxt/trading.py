import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import ccxt
import ccxt.pro as cxp
from ccxt.base.errors import ExchangeError, NotSupported
from qubx import logger, lookup
from qubx.core.basics import (
    CtrlChannel,
    Deal,
    Instrument,
    Order,
    Position,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.interfaces import (
    IAccountProcessor,
    ITradingServiceProvider,
)
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.ntp import time_now

from .utils import (
    ccxt_build_qubx_exchange_name,
    ccxt_convert_order_info,
    ccxt_extract_deals_from_exec,
    instrument_to_ccxt_symbol,
)


class CcxtTradingConnector(ITradingServiceProvider):
    exchange: cxp.Exchange

    _fees_calculator: TransactionCostsCalculator | None = None
    _positions: Dict[Instrument, Position]
    _loop: AsyncThreadLoop

    def __init__(
        self,
        exchange: cxp.Exchange,
        account_processor: IAccountProcessor,
        commissions: str | None = None,
    ):
        self.exchange = exchange
        self.ccxt_exchange_id = str(exchange.name)
        self.acc = account_processor
        self.account_id = account_processor.account_id
        self.commissions = commissions
        self._loop = AsyncThreadLoop(exchange.asyncio_loop)
        self._positions = self.acc.get_positions()

    def set_communication_channel(self, channel: CtrlChannel):
        super().set_communication_channel(channel)
        self.acc.set_communication_channel(self.get_communication_channel())

    def get_position(self, instrument: Instrument) -> Position:
        return self.acc.get_position(instrument)

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
    ) -> Order:
        params = {}

        if order_type == "limit":
            params["timeInForce"] = time_in_force.upper()
            if price is None:
                raise ValueError("Price must be specified for limit order")

        if client_id:
            params["newClientOrderId"] = client_id

        if instrument.is_futures():
            params["type"] = "swap"

        ccxt_symbol = instrument_to_ccxt_symbol(instrument)

        r: Dict[str, Any] | None = None
        try:
            r = self._loop.submit(
                self.exchange.create_order(
                    symbol=ccxt_symbol,
                    type=order_type,  # type: ignore
                    side=order_side,  # type: ignore
                    amount=amount,
                    price=price,
                    params=params,
                )
            ).result()
        except ccxt.BadRequest as exc:
            logger.error(
                f"(::send_order) BAD REQUEST for {order_side} {amount} {order_type} for {instrument.symbol} : {exc}"
            )
            raise exc
        except Exception as err:
            logger.error(f"(::send_order) {order_side} {amount} {order_type} for {instrument.symbol} exception : {err}")
            logger.error(traceback.format_exc())
            raise err

        if r is None:
            logger.error(f"(::send_order) No response from exchange")
            raise ExchangeError("(::send_order) No response from exchange")

        order = ccxt_convert_order_info(instrument, r)
        logger.info(f"New order {order}")
        return order

    def cancel_order(self, order_id: str) -> Order | None:
        order = None
        orders = self.acc.get_orders()
        if order_id in orders:
            order = orders[order_id]
            try:
                logger.info(f"Canceling order {order_id} ...")
                r = self._loop.submit(
                    self.exchange.cancel_order(order_id, symbol=instrument_to_ccxt_symbol(order.instrument))
                ).result()
            except Exception as err:
                logger.error(f"Canceling [{order}] exception : {err}")
                logger.error(traceback.format_exc())
                raise err
        return order

    def time(self) -> dt_64:
        """
        Returns current time as dt64
        """
        return time_now()

    def get_name(self) -> str:
        return self.exchange.name  # type: ignore

    def get_account_id(self) -> str:
        return self.account_id

    def get_orders(self, instrument: Instrument | None = None) -> list[Order]:
        return list(self.acc.get_orders(instrument).values())

    def get_base_currency(self) -> str:
        return self.acc.get_base_currency()

    def process_execution_report(self, instrument: Instrument, report: dict[str, Any]) -> Tuple[Order, List[Deal]]:
        order = ccxt_convert_order_info(instrument, report)
        deals = ccxt_extract_deals_from_exec(report)
        self._fill_missing_fee_info(instrument, deals)
        self.acc.process_deals(instrument, deals)
        self.acc.process_order(order)
        return order, deals

    def _fill_missing_fee_info(self, instrument: Instrument, deals: List[Deal]) -> None:
        for d in deals:
            if d.fee_amount is None:
                assert self._fees_calculator is not None
                d.fee_amount = self._fees_calculator.get_execution_fees(
                    instrument=instrument, exec_price=d.price, amount=d.amount, crossed_market=d.aggressive
                )
                # this is only true for linear contracts
                d.fee_currency = instrument.quote
