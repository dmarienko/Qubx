from typing import Any
from qubx.core.basics import Instrument
from qubx.core.interfaces import IBrokerServiceProvider, ISubscriptionManager, SubscriptionType


class SubscriptionManager(ISubscriptionManager):
    __broker: IBrokerServiceProvider
    __base_subscription: SubscriptionType
    __base_subscription_params: dict[str, Any]
    __is_simulation: bool
    __subscription_to_warmup: dict[str, str]

    def __init__(self, broker: IBrokerServiceProvider):
        self.__broker = broker
        self.__is_simulation = broker.is_simulated_trading
        self.__base_subscription = SubscriptionType.OHLC if self.__is_simulation else SubscriptionType.ORDERBOOK
        self.__base_subscription_params = {
            SubscriptionType.OHLC: {"timeframe": "1h"},
            SubscriptionType.ORDERBOOK: {},
        }[self.__base_subscription]
        self.__subscription_to_warmup = {
            SubscriptionType.OHLC: "7d",
            SubscriptionType.ORDERBOOK: "1m",
            SubscriptionType.QUOTE: "1m",
            SubscriptionType.TRADE: "1m",
        }

    def subscribe(self, instrument: Instrument, subscription_type: str, **kwargs) -> bool:
        __subscription_to_warmup = self.__subscription_to_warmup.copy()
        # - take default warmup period for current subscription if None is given
        kwargs["warmup_period"] = kwargs.get("warmup_period", __subscription_to_warmup.get(subscription_type))
        # - if this is the base subscription, we also need to fetch historical OHLC data for warmup
        if subscription_type == self.__base_subscription:
            kwargs["ohlc_warmup_period"] = kwargs.get(
                "ohlc_warmup_period", __subscription_to_warmup.get(subscription_type)
            )
        return self.__broker.subscribe([instrument], subscription_type, **kwargs)

    def unsubscribe(self, instrument: Instrument, subscription_type: str | None = None) -> bool:
        return self.__broker.unsubscribe([instrument], subscription_type)

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        return self.__broker.has_subscription(instrument, subscription_type)

    def get_base_subscription(self) -> tuple[SubscriptionType, dict]:
        """
        Get the main subscription which should be used for the simulation.
        This data is used for updating the internal OHLCV data series.
        By default, simulation uses 1h OHLCV bars and live trading uses orderbook data.
        """
        return self.__base_subscription, self.__base_subscription_params

    def set_base_subscription(self, subscription_type: SubscriptionType, **kwargs) -> None:
        """
        Set the main subscription which should be used for the simulation.
        """
        self.__base_subscription = subscription_type
        self.__base_subscription_params = kwargs

    def get_warmup(self, subscription_type: str) -> str:
        return self.__subscription_to_warmup[subscription_type]

    def set_warmup(self, subscription_type: str, period: str) -> None:
        self.__subscription_to_warmup[subscription_type] = period
