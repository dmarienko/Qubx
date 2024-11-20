from typing import Any, List, Dict
from qubx.core.basics import Instrument, Subtype
from qubx.core.interfaces import IBrokerServiceProvider, ISubscriptionManager


class SubscriptionManager(ISubscriptionManager):
    __broker: IBrokerServiceProvider
    __base_subscription: Subtype
    __base_subscription_params: dict[str, Any]
    __is_simulation: bool
    __subscription_to_warmup: dict[str, str]

    def __init__(self, broker: IBrokerServiceProvider):
        self.__broker = broker
        self.__is_simulation = broker.is_simulated_trading
        self.__base_subscription = Subtype.OHLC if self.__is_simulation else Subtype.ORDERBOOK
        self.__base_subscription_params = {
            Subtype.OHLC: {"timeframe": "1m"},
            Subtype.ORDERBOOK: {},
        }[self.__base_subscription]
        self.__subscription_to_warmup = {
            Subtype.OHLC: "1h",
            Subtype.ORDERBOOK: "1m",
            Subtype.QUOTE: "1m",
            Subtype.TRADE: "1m",
            Subtype.LIQUIDATION: "1m",
            Subtype.FUNDING_RATE: "1m",
        }

    def subscribe(
        self, instruments: List[Instrument] | Instrument, subscription_type: str | None = None, **kwargs
    ) -> None:
        if subscription_type is None:
            subscription_type = self.__base_subscription

        __subscription_to_warmup = self.__subscription_to_warmup.copy()

        # - take default warmup period for current subscription if None is given
        kwargs["warmup_period"] = kwargs.get("warmup_period", __subscription_to_warmup.get(subscription_type))

        # - if this is the base subscription, we also need to fetch historical OHLC data for warmup
        if subscription_type == self.__base_subscription and subscription_type != Subtype.OHLC:
            kwargs["ohlc_warmup_period"] = kwargs.get(
                "ohlc_warmup_period", __subscription_to_warmup.get(subscription_type)
            )
            kwargs |= self.__base_subscription_params

        instruments = [instruments] if isinstance(instruments, Instrument) else instruments
        self.__broker.subscribe(instruments, subscription_type, **kwargs)

    def unsubscribe(self, instruments: List[Instrument] | Instrument, subscription_type: str | None = None) -> None:
        instruments = instruments if isinstance(instruments, list) else [instruments]
        self.__broker.unsubscribe(instruments, subscription_type)

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        return self.__broker.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument) -> Dict[str, Dict[str, Any]]:
        return self.__broker.get_subscriptions(instrument)

    def get_base_subscription(self) -> tuple[Subtype, dict]:
        """
        Get the main subscription which should be used for the simulation.
        This data is used for updating the internal OHLCV data series.
        By default, simulation uses 1h OHLCV bars and live trading uses orderbook data.
        """
        return self.__base_subscription, self.__base_subscription_params

    def set_base_subscription(self, subscription_type: Subtype, **kwargs) -> None:
        """
        Set the main subscription which should be used for the simulation.
        """
        self.__base_subscription = subscription_type
        self.__base_subscription_params = kwargs

    def get_warmup(self, subscription_type: str) -> str:
        return self.__subscription_to_warmup[subscription_type]

    def set_warmup(self, subscription_type: str, period: str) -> None:
        self.__subscription_to_warmup[subscription_type] = period
