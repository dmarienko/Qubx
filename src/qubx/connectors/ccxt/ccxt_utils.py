import pandas as pd
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from qubx import logger
from qubx.core.basics import Order, Deal, Position, Instrument
from qubx.core.series import TimeSeries, Bar, Trade, Quote, OrderBook, time_as_nsec
from qubx.utils.orderbook import build_orderbook_snapshots
from .ccxt_exceptions import CcxtOrderBookParsingError


EXCHANGE_ALIASES = {"binance.um": "binanceusdm", "binance.cm": "binancecoinm", "kraken.f": "krakenfutures"}

DATA_PROVIDERS_ALIASES = EXCHANGE_ALIASES | {"binance": "binanceqv", "binance.um": "binanceqv_usdm"}


def ccxt_convert_order_info(symbol: str, raw: Dict[str, Any]) -> Order:
    """
    Convert CCXT excution record to Order object
    """
    ri = raw["info"]
    amnt = float(ri.get("origQty", raw.get("amount")))
    price = raw["price"]
    status = raw["status"]
    side = raw["side"].upper()
    _type = ri.get("type", raw.get("type")).upper()
    if status == "open":
        status = ri.get("status", status)  # for filled / part_filled ?

    return Order(
        id=raw["id"],
        type=_type,
        symbol=symbol,
        time=pd.Timestamp(raw["timestamp"], unit="ms"),  # type: ignore
        quantity=amnt,
        price=float(price) if price is not None else 0.0,
        side=side,
        status=status.upper(),
        time_in_force=raw["timeInForce"],
        client_id=raw["clientOrderId"],
        cost=float(raw["cost"] or 0),  # cost can be None
    )


def ccxt_convert_deal_info(raw: Dict[str, Any]) -> Deal:
    fee_amount = None
    fee_currency = None
    if "fee" in raw:
        fee_amount = float(raw["fee"]["cost"])
        fee_currency = raw["fee"]["currency"]
    return Deal(
        id=raw["id"],
        order_id=raw["order"],
        time=pd.Timestamp(raw["timestamp"], unit="ms"),  # type: ignore
        amount=float(raw["amount"]) * (-1 if raw["side"] == "sell" else +1),
        price=float(raw["price"]),
        aggressive=raw["takerOrMaker"] == "taker",
        fee_amount=fee_amount,
        fee_currency=fee_currency,
    )


def ccxt_extract_deals_from_exec(report: Dict[str, Any]) -> List[Deal]:
    """
    Small helper for extracting deals (trades) from CCXT execution report
    """
    deals = list()
    if trades := report.get("trades"):
        for t in trades:
            deals.append(ccxt_convert_deal_info(t))
    return deals


def ccxt_restore_position_from_deals(
    pos: Position, current_volume: float, deals: List[Deal], reserved_amount: float = 0.0
) -> Position:
    if current_volume != 0:
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
            logger.warning(
                f"Couldn't restore full deals history for {instr.symbol} symbol. Qubx will use zero position !"
            )
        else:
            fees_in_base = 0.0
            for d in _last_deals:
                pos.update_position_by_deal(d)
                if d.fee_amount is not None:
                    if instr.base == d.fee_currency:
                        fees_in_base += d.fee_amount
            # - we round fees up in case of fees in base currency
            pos.quantity -= pos.instrument.round_size_up(fees_in_base)
    return pos


def ccxt_convert_trade(trade: dict[str, Any]) -> Trade:
    t_ns = trade["timestamp"] * 1_000_000  # this is trade time
    s, info, price, amnt = trade["symbol"], trade["info"], trade["price"], trade["amount"]
    m = info["m"]
    return Trade(t_ns, price, amnt, int(not m), int(trade["id"]))


def ccxt_convert_orderbook(
    ob: dict, instr: Instrument, levels: int = 50, tick_size_pct: float = 0.01, size_in_quoted: bool = False
) -> OrderBook:
    """
    Convert a ccxt order book to an OrderBook object with a fixed tick size percentage.
    Parameters:
        ob (dict): The order book dictionary from ccxt.
        instr (Instrument): The instrument object containing market-specific details.
        levels (int, optional): The number of levels to include in the order book. Default is 50.
        tick_size_pct (float, optional): The tick size percentage. Default is 0.01%.
        size_in_quoted (bool, optional): Whether the size is in the quoted currency. Default is False.
    Returns:
        OrderBook: The converted OrderBook object.
    """
    _dt = pd.Timestamp(ob["datetime"]).replace(tzinfo=None).asm8
    _prev_dt = _dt - pd.Timedelta("1ms").asm8

    updates = [
        *[(_prev_dt, update[0], update[1], True) for update in ob["bids"]],
        *[(_prev_dt, update[0], update[1], False) for update in ob["asks"]],
    ]
    # add an artificial update to trigger the snapshot building
    updates.append((_dt, 0, 0, True))

    snapshots = build_orderbook_snapshots(
        updates,
        levels=levels,
        tick_size_pct=tick_size_pct,
        min_tick_size=instr.min_tick,
        min_size_step=instr.min_size_step,
        size_in_quoted=size_in_quoted,
    )

    if not snapshots:
        raise CcxtOrderBookParsingError("Failed to build order book snapshots")

    (dt, _bids, _asks, top_bid, top_ask, tick_size) = snapshots[-1]
    bids = np.array([s for _, s in _bids[::-1]])
    asks = np.array([s for _, s in _asks])

    return OrderBook(
        time=time_as_nsec(dt),
        top_bid=top_bid,
        top_ask=top_ask,
        tick_size=tick_size,
        bids=bids,
        asks=asks,
    )
