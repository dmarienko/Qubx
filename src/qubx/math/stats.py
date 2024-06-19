import numpy as np
from qubx.utils import sbp


def percentile_rank(x: np.ndarray, v, pctls=np.arange(1, 101)):
    """
    Find percentile rank of value v
    :param x: values array
    :param v: vakue to be ranked
    :param pctls: percentiles
    :return: rank

    >>> percentile_rank(np.random.randn(1000), 1.69)
    >>> 95
    >>> percentile_rank(np.random.randn(1000), 1.69, [10,50,100])
    >>> 2
    """
    return np.argmax(np.sign(np.append(np.percentile(x, pctls), np.inf) - v))


def compare_to_norm(xs, xranges=None):
    """
    Compare distribution from xs against normal using estimated mean and std
    """
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    _m, _s = np.mean(xs), np.std(xs)
    fit = stats.norm.pdf(sorted(xs), _m, _s)

    sbp(12, 1)
    plt.plot(sorted(xs), fit, "r--", lw=2, label="N(%.2f, %.2f)" % (_m, _s))
    plt.legend(loc="upper right")

    sns.kdeplot(xs, color="g", label="Data", fill=True)
    if xranges is not None and len(xranges) > 1:
        plt.xlim(xranges)
    plt.legend(loc="upper right")

    sbp(12, 2)
    stats.probplot(xs, dist="norm", sparams=(_m, _s), plot=plt)


def kde(array, cut_down=True, bw_method="scott"):
    """
    Kernel dense estimation
    """
    from scipy.stats import gaussian_kde

    if cut_down:
        bins, counts = np.unique(array, return_counts=True)
        f_mean = counts.mean()
        f_above_mean = bins[counts > f_mean]
        if len(f_above_mean) > 0:
            bounds = [f_above_mean.min(), f_above_mean.max()]
            array = array[np.bitwise_and(bounds[0] < array, array < bounds[1])]

    return gaussian_kde(array, bw_method=bw_method)
