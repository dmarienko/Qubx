from qubx.core.exceptions import BaseError


class CcxtOrderBookParsingError(BaseError):
    pass


class CcxtSymbolNotRecognized(BaseError):
    pass


class CcxtLiquidationParsingError(BaseError):
    pass


class CcxtPositionRestoreError(BaseError):
    pass
