from dataclasses import dataclass
from datetime import datetime


@dataclass
class Liquidation:
    """Liquidation data class"""

    timestamp: datetime
    side: str
    price: float
    average_price: float
    size: float
    filled_size: float
    quote_filled_size: float
    unfilled_size: float
    quote_unfilled_size: float
    price_diff: float
