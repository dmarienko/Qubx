import ccxt.pro as cxp
from ccxt.async_support.base.ws.client import Client
from ccxt.async_support.base.ws.cache import ArrayCacheByTimestamp, ArrayCache


class BinanceQV(cxp.binance):
    """
    Extended binance exchange to provide quote asset volumes support
    """

    def describe(self):
        """
        Overriding watchTrades to use aggTrade instead of trade.
        """
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "watchTrades": {
                        "name": "aggTrade",
                    }
                }
            },
        )

    def parse_ohlcv(self, ohlcv, market=None):
        """
        [
            1499040000000,      // Kline open time                   0
            "0.01634790",       // Open price                        1
            "0.80000000",       // High price                        2
            "0.01575800",       // Low price                         3
            "0.01577100",       // Close price                       4
            "148976.11427815",  // Volume                            5
            1499644799999,      // Kline Close time                  6
            "2434.19055334",    // Quote asset volume                7
            308,                // Number of trades                  8
            "1756.87402397",    // Taker buy base asset volume       9
            "28.46694368",      // Taker buy quote asset volume     10
            "0"                 // Unused field, ignore.
        ]
        """
        return [
            self.safe_integer(ohlcv, 0),
            self.safe_number(ohlcv, 1),
            self.safe_number(ohlcv, 2),
            self.safe_number(ohlcv, 3),
            self.safe_number(ohlcv, 4),
            self.safe_number(ohlcv, 5),
            self.safe_number(ohlcv, 7),  # Quote asset volume
            self.safe_number(ohlcv, 10),  # Taker buy quote asset volume
        ]

    def handle_ohlcv(self, client: Client, message):
        event = self.safe_string(message, "e")
        eventMap = {
            "indexPrice_kline": "indexPriceKline",
            "markPrice_kline": "markPriceKline",
        }
        event = self.safe_string(eventMap, event, event)
        kline = self.safe_value(message, "k")
        marketId = self.safe_string_2(kline, "s", "ps")
        if event == "indexPriceKline":
            # indexPriceKline doesn't have the _PERP suffix
            marketId = self.safe_string(message, "ps")
        lowercaseMarketId = marketId.lower()
        interval = self.safe_string(kline, "i")
        # use a reverse lookup in a static map instead
        timeframe = self.find_timeframe(interval)
        messageHash = lowercaseMarketId + "@" + event + "_" + interval
        parsed = [
            self.safe_integer(kline, "t"),
            self.safe_float(kline, "o"),
            self.safe_float(kline, "h"),
            self.safe_float(kline, "l"),
            self.safe_float(kline, "c"),
            self.safe_float(kline, "v"),
            # - additional fields
            self.safe_float(kline, "q"),  # - quote asset volume
            self.safe_float(kline, "Q"),  # - taker buy quote asset volume
            # self.safe_integer(message, "E")
        ]
        isSpot = (client.url.find("/stream") > -1) or (client.url.find("/testnet.binance") > -1)
        marketType = "spot" if (isSpot) else "contract"
        symbol = self.safe_symbol(marketId, None, None, marketType)
        self.ohlcvs[symbol] = self.safe_value(self.ohlcvs, symbol, {})
        stored = self.safe_value(self.ohlcvs[symbol], timeframe)
        if stored is None:
            limit = self.safe_integer(self.options, "OHLCVLimit", 2)
            stored = ArrayCacheByTimestamp(limit)
            # self.ohlcvs[symbol][timeframe] = stored
        stored.append(parsed)
        client.resolve(stored, messageHash)

    def handle_trade(self, client: Client, message):
        """
        There is a custom trade handler implementation, because Binance sends
        some trades marked with "X" field, which is "MARKET" for market trades
        and "INSURANCE_FUND" for insurance fund trades. We are interested only
        in market trades, so we filter the rest out.

        Update 07072024: Apparently insurance fund trades not aggregated so
        we don't need to filter via "X" field, but let's keep it just in case.
        """
        # the trade streams push raw trade information in real-time
        # each trade has a unique buyer and seller
        isSpot = (client.url.find("wss://stream.binance.com") > -1) or (client.url.find("/testnet.binance") > -1)
        marketType = "spot" if (isSpot) else "contract"
        marketId = self.safe_string(message, "s")
        market = self.safe_market(marketId, None, None, marketType)
        symbol = market["symbol"]
        lowerCaseId = self.safe_string_lower(message, "s")
        event = self.safe_string(message, "e")
        messageHash = lowerCaseId + "@" + event
        executionType = self.safe_string(message, "X")
        if executionType == "INSURANCE_FUND":
            return
        trade = self.parse_ws_trade(message, market)
        tradesArray = self.safe_value(self.trades, symbol)
        if tradesArray is None:
            limit = self.safe_integer(self.options, "tradesLimit", 1000)
            tradesArray = ArrayCache(limit)
        tradesArray.append(trade)
        self.trades[symbol] = tradesArray
        client.resolve(tradesArray, messageHash)


class BinanceQVUSDM(cxp.binanceusdm, BinanceQV):
    """
    The order of inheritance is important here, because we want
    binanceusdm to take precedence over binanceqv. And this is how MRO is defined
    in Python.

    Describe method needs to be overriden, because of the way super is called in binanceusdm.
    """

    def describe(self):
        """
        Overriding watchTrades to use aggTrade instead of trade.
        """
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "watchTrades": {
                        "name": "aggTrade",
                    }
                }
            },
        )
