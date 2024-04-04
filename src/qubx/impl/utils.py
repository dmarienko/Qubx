from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from qubx.core.basics import Order, Deal


def ccxt_convert_order_info(symbol: str, raw: Dict[str,Any]) -> Order:
    """
    Convert CCXT excution record to Order object
    """
    deal = None
    ri = raw['info']
    amnt = float(ri.get('origQty', raw.get('amount')))
    price = raw['price']
    status = raw['status']
    side = raw['side'].upper()
    _type = ri.get('type', raw.get('type')).upper()
    if status == 'open':
        status = ri.get('status', status)  # for filled / part_filled ?

    avg = raw.get('average')
    exec_amount = float(ri.get('executedQty', raw.get('filled')))
    exec_price = float(avg) if avg is not None else None

    if exec_amount > 0 and exec_price is not None:
        aggressive = _type == 'MARKET'
        _S = -1 if side == 'SELL' else +1
        trade_time = pd.Timestamp(raw['lastTradeTimestamp'], unit='ms')
        deal = Deal(trade_time, _S * exec_amount, exec_price, aggressive)

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
        execution=deal
    )