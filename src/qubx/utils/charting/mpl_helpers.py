"""
   Misc graphics handy utilitites to be used in interactive analysis
"""
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as st
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler


DARK_MATLPLOT_THEME = [
    ('backend', 'module://matplotlib_inline.backend_inline'),
    ('interactive', True),
    ('lines.color', '#5050f0'),
    ('text.color', '#d0d0d0'),
    ('axes.facecolor', '#000000'),
    ('axes.edgecolor', '#404040'),
    ('axes.grid', True),
    ('axes.labelsize', 'large'),
    ('axes.labelcolor', 'green'),
    ('axes.prop_cycle', cycler('color', ['#08F7FE', '#00ff41', '#FE53BB', '#F5D300', '#449AcD', 'g',
                                         '#f62841', 'y', '#088487', '#E24A33', '#f01010'])),
    ('legend.fontsize', 'small'),
    ('legend.fancybox', False),
    ('legend.edgecolor', '#305030'),
    ('legend.shadow', False),
    ('lines.antialiased', True),
    ('lines.linewidth', 0.8),  # reduced line width
    ('patch.linewidth', 0.5),
    ('patch.antialiased', True),
    ('xtick.color', '#909090'),
    ('ytick.color', '#909090'),
    ('xtick.labelsize', 'large'),
    ('ytick.labelsize', 'large'),
    ('grid.color', '#404040'),
    ('grid.linestyle', '--'),
    ('grid.linewidth', 0.5),
    ('grid.alpha', 0.8),
    ('figure.figsize', [12.0, 5.0]),
    ('figure.dpi', 80.0),
    ('figure.facecolor', '#050505'),
    ('figure.edgecolor', (1, 1, 1, 0)),
    ('figure.subplot.bottom', 0.125),
    ('savefig.facecolor', '#000000'),
]

LIGHT_MATPLOT_THEME = [
    ('backend', 'module://matplotlib_inline.backend_inline'),
    ('interactive', True),
    ('lines.color', '#101010'),
    ('text.color', '#303030'),
    ('lines.antialiased', True),
    ('lines.linewidth', 1),
    ('patch.linewidth', 0.5),
    ('patch.facecolor', '#348ABD'),
    ('patch.edgecolor', '#eeeeee'),
    ('patch.antialiased', True),
    ('axes.facecolor', '#fafafa'),
    ('axes.edgecolor', '#d0d0d0'),
    ('axes.linewidth', 1),
    ('axes.titlesize', 'x-large'),
    ('axes.labelsize', 'large'),
    ('axes.labelcolor', '#555555'),
    ('axes.axisbelow', True),
    ('axes.grid', True),
    ('axes.prop_cycle', cycler('color', ['#6792E0', '#27ae60', '#c44e52', '#975CC3', '#ff914d', '#77BEDB',
                                         '#303030', '#4168B7', '#93B851', '#e74c3c', '#bc89e0', '#ff711a',
                                         '#3498db', '#6C7A89'])),
    ('legend.fontsize', 'small'),
    ('legend.fancybox', False),
    ('xtick.color', '#707070'),
    ('ytick.color', '#707070'),
    ('grid.color', '#606060'),
    ('grid.linestyle', '--'),
    ('grid.linewidth', 0.5),
    ('grid.alpha', 0.3),
    ('figure.figsize', [8.0, 5.0]),
    ('figure.dpi', 80.0),
    ('figure.facecolor', '#ffffff'),
    ('figure.edgecolor', '#ffffff'),
    ('figure.subplot.bottom', 0.1)
]


def fig(w=16, h=5, dpi=96, facecolor=None, edgecolor=None, num=None):
    """
    Simple helper for creating figure
    """
    return plt.figure(num=num, figsize=(w, h), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)


def subplot(shape, loc, rowspan=2, colspan=1):
    """
    Some handy grid splitting for plots. Example for 2x2:
    
    >>> subplot(22, 1); plt.plot([-1,2,-3])
    >>> subplot(22, 2); plt.plot([1,2,3])
    >>> subplot(22, 3); plt.plot([1,2,3])
    >>> subplot(22, 4); plt.plot([3,-2,1])

    same as following

    >>> subplot((2,2), (0,0)); plt.plot([-1,2,-3])
    >>> subplot((2,2), (0,1)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,0)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,1)); plt.plot([3,-2,1])

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param rowspan: rows spanned
    :param colspan: columns spanned
    """
    isscalar = lambda x: not isinstance(x, (list, tuple, dict, np.ndarray))

    if isscalar(shape):
        if 0 < shape < 100:
            shape = (max(shape // 10, 1), max(shape % 10, 1))
        else:
            raise ValueError("Wrong scalar value for shape. It should be in range (1...99)")

    if isscalar(loc):
        nm = max(shape[0], 1) * max(shape[1], 1)
        if 0 < loc <= nm:
            x = (loc - 1) // shape[1]
            y = loc - 1 - shape[1] * x
            loc = (x, y)
        else:
            raise ValueError("Wrong scalar value for location. It should be in range (1...%d)" % nm)

    return plt.subplot2grid(shape, loc=loc, rowspan=rowspan, colspan=colspan)


def sbp(shape, loc, r=1, c=1):
    """
    Just shortcut for subplot(...) function

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param r: rows spanned
    :param c: columns spanned
    :return:
    """
    return subplot(shape, loc, rowspan=r, colspan=c)


def vline(ax, x, c, lw=1, ls='--'):
    x = pd.to_datetime(x) if isinstance(x, str) else x
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        a.axvline(x, 0, 1, c=c, lw=1, linestyle=ls)


def hline(*zs, mirror=True):
    [plt.axhline(z, ls='--', c='r', lw=0.5) for z in zs]
    if mirror:
        [plt.axhline(-z, ls='--', c='r', lw=0.5) for z in zs]


def ellips(ax, x, y, c='r', r=2.5, lw=2, ls='-'):
    """
    Draw ellips annotation on specified plot at (x,y) point
    """
    from matplotlib.patches import Ellipse
    x = pd.to_datetime(x) if isinstance(x, str) else x
    w, h = (r, r) if np.isscalar(r) else (r[0], r[1])
    ax.add_artist(Ellipse(xy=[x, y], width=w, height=h, angle=0, fill=False, color=c, lw=lw, ls=ls))


def set_mpl_theme(theme: str):
    import plotly.io as pio

    if 'dark' in theme.lower():
        pio.templates.default = "plotly_dark"
        for (k, v) in DARK_MATLPLOT_THEME:
            matplotlib.rcParams[k] = v

    elif 'light' in theme.lower():
        pio.templates.default = "plotly_white"
        for (k, v) in LIGHT_MATPLOT_THEME:
            matplotlib.rcParams[k] = v