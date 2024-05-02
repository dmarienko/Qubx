from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from qubx import logger
from qubx.core.basics import Order, Deal, Position


EXCHANGE_ALIASES = {
    "binance.um": "binanceusdm",
    "binance.cm": "binancecoinm",
    "kraken.f": "krakenfutures"
}

DATA_PROVIDERS_ALIASES = EXCHANGE_ALIASES | { "binance": "binanceqv" }


def ccxt_convert_order_info(symbol: str, raw: Dict[str,Any]) -> Order:
    """
    Convert CCXT excution record to Order object
    """
    ri = raw['info']
    amnt = float(ri.get('origQty', raw.get('amount')))
    price = raw['price']
    status = raw['status']
    side = raw['side'].upper()
    _type = ri.get('type', raw.get('type')).upper()
    if status == 'open':
        status = ri.get('status', status)  # for filled / part_filled ?

    return Order(
        id=raw['id'],
        type=_type,
        symbol=symbol,
        time = pd.Timestamp(raw['timestamp'], unit='ms'), # type: ignore
        quantity = amnt, 
        price= float(price) if price is not None else 0.0,  
        side = side,
        status = status.upper(),
        time_in_force = raw['timeInForce'],
        client_id = raw['clientOrderId'],
        cost = float(raw['cost']),
    )


def ccxt_convert_deal_info(raw: Dict[str,Any]) -> Deal:
    fee_amount = None
    fee_currency = None
    if 'fee' in raw:
        fee_amount = float(raw['fee']['cost'])
        fee_currency = raw['fee']['currency']
    return Deal(
        id=raw['id'],
        order_id=raw['order'],
        time = pd.Timestamp(raw['timestamp'], unit='ms'), # type: ignore
        amount=float(raw['amount']) * (-1 if raw['side'] == 'sell' else +1),
        price=float(raw['price']),
        aggressive=raw['takerOrMaker'] == 'taker',
        fee_amount=fee_amount,
        fee_currency=fee_currency,
    )


def ccxt_extract_deals_from_exec(report: Dict[str,Any]) -> List[Deal]:
    """
    Small helper for extracting deals (trades) from CCXT execution report
    """
    deals = list()
    if (trades := report.get('trades')):
        for t in trades:
            deals.append(ccxt_convert_deal_info(t))
    return deals


def ccxt_restore_position_from_deals(
    pos: Position, current_volume: float, deals: List[Deal], reserved_amount:float=0.0
) -> Position:
    if  current_volume != 0:
        instr = pos.instrument
        _last_deals = []

        # - try to find last deals that led to this position
        for d in sorted(deals, key=lambda x: x.time, reverse=True):
            current_volume -= d.amount
            # - spot case when fees may be deducted from the base coin
            #   that may decrease total amount
            if d.fee_amount is not None:
                if instr.base == d.fee_currency:
                    current_volume += d.fee_amount
            # print(d.amount, current_volume)
            _last_deals.insert(0, d)

            # - take in account reserves
            if abs(current_volume) - abs(reserved_amount) < instr.min_size_step:
                break
            
        # - reset to 0
        pos.reset()

        if abs(current_volume) - abs(reserved_amount) > instr.min_size_step:
            # - - - TODO - - - !!!!
            logger.warning(f"Couldn't restore full deals history for {instr.symbol} symbol. Qubx will use zero position !")
        else:
            for d in _last_deals:
                pos.update_position_by_deal(d)
                if d.fee_amount is not None:
                    if instr.base == d.fee_currency:
                        pos.quantity -= d.fee_amount
    return pos