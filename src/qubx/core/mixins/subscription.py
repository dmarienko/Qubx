from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

from qubx.core.basics import Instrument, Subtype
from qubx.core.interfaces import IBrokerServiceProvider, ISubscriptionManager
from qubx.utils.helpers import synchronized


class SubscriptionManager(ISubscriptionManager):
    _broker: IBrokerServiceProvider
    _base_sub: str
    _is_simulation: bool
    _sub_to_warmup: dict[str, str]
    _auto_subscribe: bool

    _pending_global_subscriptions: Set[str]
    _pending_global_unsubscriptions: Set[str]

    _pending_stream_subscriptions: Dict[str, Set[Instrument]]
    _pending_stream_unsubscriptions: Dict[str, Set[Instrument]]
    _pending_warmups: Dict[Tuple[str, Instrument], str]

    def __init__(self, broker: IBrokerServiceProvider, auto_subscribe: bool = True) -> None:
        self._broker = broker
        self._is_simulation = broker.is_simulated_trading
        self._base_sub = Subtype.OHLC["1Min"] if self._is_simulation else Subtype.ORDERBOOK
        self._sub_to_warmup = {}
        self._pending_warmups = {}
        self._pending_global_subscriptions = set()
        self._pending_global_unsubscriptions = set()
        self._pending_stream_subscriptions = defaultdict(set)
        self._pending_stream_unsubscriptions = defaultdict(set)
        self._auto_subscribe = auto_subscribe

    def subscribe(self, subscription_type: str, instruments: List[Instrument] | Instrument | None = None) -> None:
        # - figure out which instruments to subscribe to (all or specific)
        if instruments is None:
            self._pending_global_subscriptions.add(subscription_type)
            return

        if isinstance(instruments, Instrument):
            instruments = [instruments]

        # - get instruments that are not already subscribed to
        _current_instruments = self._broker.get_subscribed_instruments(subscription_type)
        instruments = list(set(instruments).difference(_current_instruments))

        # - subscribe to all existing subscriptions if subscription_type is ALL
        if subscription_type == Subtype.ALL:
            subscriptions = self.get_subscriptions()
            for sub in subscriptions:
                self.subscribe(sub, instruments)
            return

        self._pending_stream_subscriptions[subscription_type].update(instruments)
        self._update_pending_warmups(subscription_type, instruments)

    def unsubscribe(self, subscription_type: str, instruments: List[Instrument] | Instrument | None = None) -> None:
        if instruments is None:
            self._pending_global_unsubscriptions.add(subscription_type)
            return

        if isinstance(instruments, Instrument):
            instruments = [instruments]

        # - subscribe to all existing subscriptions if subscription_type is ALL
        if subscription_type == Subtype.ALL:
            subscriptions = self.get_subscriptions()
            for sub in subscriptions:
                self.unsubscribe(sub, instruments)
            return

        self._pending_stream_unsubscriptions[subscription_type].update(instruments)

    @synchronized
    def commit(self) -> None:
        _subs = self._get_updated_subs()
        if not _subs:
            return

        # - warm up subscriptions
        self._run_warmup()

        # - update subscriptions
        for _sub in _subs:
            _current_sub_instruments = set(self._broker.get_subscribed_instruments(_sub))
            _removed_instruments = self._pending_stream_unsubscriptions.get(_sub, set())
            _added_instruments = self._pending_stream_subscriptions.get(_sub, set())
            if _sub in self._pending_global_unsubscriptions:
                _removed_instruments.update(_current_sub_instruments)
            if _sub in self._pending_global_subscriptions:
                _added_instruments.update(self._broker.get_subscribed_instruments())
            _updated_instruments = _current_sub_instruments.union(_added_instruments).difference(_removed_instruments)
            if _updated_instruments != _current_sub_instruments:
                self._broker.subscribe(_sub, _updated_instruments, reset=True)

        # - clean up pending subs and unsubs
        self._pending_stream_subscriptions.clear()
        self._pending_stream_unsubscriptions.clear()
        self._pending_global_subscriptions.clear()
        self._pending_global_unsubscriptions.clear()

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        return self._broker.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        return list(
            set(self._broker.get_subscriptions(instrument))
            | {self.get_base_subscription()}
            | self._pending_global_subscriptions
        )

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> List[Instrument]:
        return self._broker.get_subscribed_instruments(subscription_type)

    def get_base_subscription(self) -> str:
        return self._base_sub

    def set_base_subscription(self, subscription_type: str) -> None:
        self._base_sub = subscription_type

    def get_warmup(self, subscription_type: str) -> str:
        return self._sub_to_warmup[subscription_type]

    def set_warmup(self, configs: Dict[Any, str]) -> None:
        for subscription_type, period in configs.items():
            self._sub_to_warmup[subscription_type] = period

    @property
    def auto_subscribe(self) -> bool:
        return self._auto_subscribe

    @auto_subscribe.setter
    def auto_subscribe(self, value: bool) -> None:
        self._auto_subscribe = value

    def _get_updated_subs(self) -> list[str]:
        return list(
            set(self._pending_stream_unsubscriptions.keys())
            | set(self._pending_stream_subscriptions.keys())
            | self._pending_global_subscriptions
            | self._pending_global_unsubscriptions
        )

    def _update_pending_warmups(self, subscription_type: str, instruments: List[Instrument]) -> None:
        # TODO: refactor pending warmups in a way that would allow to subscribe and then call set_warmup in the same iteration
        # - ohlc is handled separately
        if Subtype.from_str(subscription_type) != Subtype.OHLC:
            _warmup_period = self._sub_to_warmup.get(subscription_type)
            if _warmup_period is not None:
                for instrument in instruments:
                    self._pending_warmups[(subscription_type, instrument)] = _warmup_period

        # - if base subscription, then we need to fetch historical OHLC data for warmup
        if subscription_type == self._base_sub:
            self._pending_warmups.update(
                {
                    (sub, instrument): period
                    for sub, period in self._sub_to_warmup.items()
                    for instrument in instruments
                    if Subtype.OHLC == sub
                }
            )

    def _run_warmup(self) -> None:
        # - handle warmup for global subscriptions
        _subscribed_instruments = set(self._broker.get_subscribed_instruments())
        _new_instruments = (
            set.union(*self._pending_stream_subscriptions.values()) if self._pending_stream_subscriptions else set()
        )

        for sub in self._pending_global_subscriptions:
            _warmup_period = self._sub_to_warmup.get(sub)
            if _warmup_period is None:
                continue
            _sub_instruments = self._broker.get_subscribed_instruments(sub)
            _add_instruments = _subscribed_instruments.union(_new_instruments).difference(_sub_instruments)
            for instr in _add_instruments:
                self._pending_warmups[(sub, instr)] = _warmup_period

        # TODO: think about appropriate handling of timeouts
        self._broker.warmup(self._pending_warmups.copy())
        self._pending_warmups.clear()
