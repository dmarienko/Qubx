from qubx.core.utils import prec_floor, prec_ceil


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
