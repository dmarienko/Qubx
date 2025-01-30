from typing import Any

import pandas as pd

from qubx.core.basics import AssetType, Instrument, MarketType


def ccxt_build_qubx_exchange_name(ccxt_exchange: str, market_type: str | None = None, is_linear: bool = True) -> str:
    """
    Build a Qubx exchange name from a ccxt exchange name and a market dictionary.
    Parameters:
        ccxt_exchange (str): The ccxt exchange name.
        market (dict): The market dictionary.
    Returns:
        str: The Qubx exchange name.
    """
    if ccxt_exchange == "BINANCE.PM":
        if market_type in ["spot", "margin"]:
            return "BINANCE"
        elif market_type == "swap" and is_linear:
            return "BINANCE.UM"
        elif market_type == "swap" and not is_linear:
            return "BINANCE.CM"
        else:
            return "BINANCE.UM"
    else:
        # for not just return the input exchange and extend later if needed
        return ccxt_exchange


def ccxt_symbol_to_instrument(ccxt_exchange_name: str, market: dict[str, Any]) -> Instrument:
    exchange = ccxt_build_qubx_exchange_name(ccxt_exchange_name, market["type"], market.get("linear", True))
    inner_info = market["info"]
    maint_margin = 0.0
    required_margin = 0.0
    liquidation_fee = 0.0
    if "marginLevels" in inner_info:
        margins = inner_info["marginLevels"][0]
        maint_margin = float(margins["maintenanceMargin"])
        required_margin = float(margins["initialMargin"])
    if "maintMarginPercent" in inner_info:
        maint_margin = float(inner_info["maintMarginPercent"]) / 100
    if "requiredMarginPercent" in inner_info:
        required_margin = float(inner_info["requiredMarginPercent"]) / 100
    if "liquidationFee" in inner_info:
        liquidation_fee = float(inner_info["liquidationFee"])

    # symbol = market["id"]
    # let's use unified symbol format across all exchanges / types: BASEQUOTE
    symbol = market["base"] + market["quote"]

    # add some exchange specific formatting of symbol name
    tick_size = float(market["precision"]["price"] or 0.0)
    lot_size = float(market["precision"]["amount"] or 0.0)
    min_size = float(market["limits"]["amount"]["min"] or 0.0)
    min_notional = float(market["limits"]["cost"]["min"] or 0.0)

    if exchange in ["BITFINEX", "BITFINEX.F"]:
        if symbol.startswith("t"):
            symbol = symbol[1:]
        symbol = symbol.replace(":", "-")
        tick_size = 10**-tick_size
        lot_size = 10**-lot_size
        min_size = 10**-min_size
        min_notional = 10**-min_notional

    return Instrument(
        symbol=symbol,
        asset_type=AssetType.CRYPTO,
        market_type=MarketType[market["type"].upper()],
        exchange=exchange,
        base=market["base"],
        quote=market["quote"],
        settle=market["settle"],
        exchange_symbol=market["id"],
        tick_size=tick_size,
        lot_size=lot_size,
        min_size=min_size,
        min_notional=min_notional,
        initial_margin=required_margin,
        maint_margin=maint_margin,
        liquidation_fee=liquidation_fee,
        contract_size=float(market.get("contractSize", 1.0) or 1.0),
        onboard_date=pd.Timestamp(int(inner_info["onboardDate"]), unit="ms") if "onboardDate" in inner_info else None,
        delivery_date=(
            pd.Timestamp(int(market["expiryDatetime"]), unit="ms")
            if "expiryDatetime" in inner_info
            else pd.Timestamp("2100-01-01T00:00:00")
        ),
    )
