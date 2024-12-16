class BaseError(Exception):
    pass


class ExchangeError(BaseError):
    pass


class BadRequest(ExchangeError):
    pass


class InvalidOrder(ExchangeError):
    pass


class OrderNotFound(InvalidOrder):
    pass


class NotSupported(ExchangeError):
    pass


class SimulatorError(BaseError):
    pass


class QueueTimeout(BaseError):
    pass


class SimulationError(Exception):
    pass


class SimulationConfigError(Exception):
    pass
