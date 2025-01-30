import pandas as pd

from qubx.core.basics import ITimeProvider, dt_64
from qubx.core.utils import prec_ceil, prec_floor


def test_prec_floor():
    a = 608.8135
    precision = 2
    assert prec_floor(a, precision) == 608.81
    assert prec_floor(prec_floor(a, precision), precision) == prec_floor(a, precision)

    assert prec_floor(608.16, 1) == 608.1


def test_prec_ceil():
    a = 608.8135
    precision = 2
    assert prec_ceil(a, precision) == 608.82
    assert prec_ceil(prec_ceil(a, precision), precision) == prec_ceil(a, precision)


class DummyTimeProvider(ITimeProvider):
    def time(self) -> dt_64:
        return pd.Timestamp("2024-04-07 13:48:37.611000").asm8
