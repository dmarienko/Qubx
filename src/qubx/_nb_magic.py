""""
Here stuff we want to have in every Jupyter notebook after calling %qube magic
"""
import importlib_metadata

import qubx
from qubx.utils import runtime_env
from qubx.utils.misc import add_project_to_system_path


def np_fmt_short():
    # default np output is 75 columns so extend it a bit and suppress scientific fmt for small floats
    np.set_printoptions(linewidth=240, suppress=True)


def np_fmt_reset():
    # reset default np printing options
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)


if runtime_env() in ['notebook', 'shell']:

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # -- all imports below will appear in notebook after calling %%alphalab magic ---
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - Common stuff - - - -
    import numpy as np
    import pandas as pd
    from datetime import time, timedelta
    from tqdm.auto import tqdm

    # - - - - TA stuff and indicators - - - -
    # - - - - Portfolio analysis - - - -
    # - - - - Simulator stuff - - - -
    # - - - - Learn stuff - - - -
    # - - - - Charting stuff - - - -
    from matplotlib import pyplot as plt
    from qubx.utils.charting.mpl_helpers import fig, subplot, sbp
    # - - - - Utils - - - -

    # - setup short numpy output format
    np_fmt_short()
    
    # - add project home to system path
    add_project_to_system_path()

    # - check current version
    try: 
        version = importlib_metadata.version('qube2')
    except:
        version = 'Dev'

    # some new logo
    if not hasattr(qubx.QubxMagics, '__already_initialized__'):
        from qubx.utils.misc import (green, yellow, cyan, magenta, white, blue, red)

        print(
        f"""
                   {red("╻")}
   {green("┏┓      ╻     ")}  {red("┃")}  {yellow("┏┓")}       {cyan("Quantitative Backtesting Environment")} 
   {green("┃┃  ┓┏  ┣┓  ┏┓")}  {red("┃")}  {yellow("┏┛")}       
   {green("┗┻  ┗┻  ┗┛  ┗ ")}  {red("┃")}  {yellow("┗━")}       (c) 2024,  ver. {magenta(version.rstrip())}
                   {red("╹")}       
"""
        )
        qubx.QubxMagics.__already_initialized__ = True

