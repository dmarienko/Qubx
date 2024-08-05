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
