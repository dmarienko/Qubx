import ccxt
import asyncio
import ccxt.pro as cxp
from threading import Thread
from typing import Any
from .customizations import BinanceQV, BinanceQVUSDM, BinancePortfolioMargin


EXCHANGE_ALIASES = {
    "binance": "binanceqv",
    "binance.um": "binanceqv_usdm",
    "binance.cm": "binancecoinm",
    "binance.pm": "binancepm",
    "kraken.f": "krakenfutures",
}

cxp.binanceqv = BinanceQV  # type: ignore
cxp.binanceqv_usdm = BinanceQVUSDM  # type: ignore
cxp.binancepm = BinancePortfolioMargin  # type: ignore

cxp.exchanges.append("binanceqv")
cxp.exchanges.append("binanceqv_usdm")
cxp.exchanges.append("binancepm")
cxp.exchanges.append("binancepm_usdm")


def get_ccxt_exchange(
    exchange: str,
    api_key: str | None = None,
    secret: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    use_testnet: bool = False,
    **kwargs,
) -> cxp.Exchange:
    """
    Get a ccxt exchange object with the given api_key and api_secret.
    Parameters:
        exchange (str): The exchange name.
        api_key (str, optional): The API key. Default is None.
        api_secret (str, optional): The API secret. Default is None.
    Returns:
        ccxt.Exchange: The ccxt exchange object.
    """
    _exchange = exchange.lower()
    _exchange = EXCHANGE_ALIASES.get(_exchange, _exchange)

    if _exchange not in cxp.exchanges:
        raise ValueError(f"Exchange {exchange} is not supported by ccxt.")

    options: dict[str, Any] = {"name": exchange}

    if loop is not None:
        options["asyncio_loop"] = loop
    else:
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        options["thread_asyncio_loop"] = thread
        options["asyncio_loop"] = loop

    if api_key and secret:
        options["apiKey"] = api_key
        options["secret"] = secret

    ccxt_exchange = getattr(cxp, _exchange)(options | kwargs)

    if use_testnet:
        ccxt_exchange.set_sandbox_mode(True)

    return ccxt_exchange
