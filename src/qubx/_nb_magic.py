""""
Here stuff we want to have in every Jupyter notebook after calling %qubx magic
"""

import qubx
from qubx.utils import runtime_env
from qubx.utils.misc import add_project_to_system_path, logo


def np_fmt_short():
    # default np output is 75 columns so extend it a bit and suppress scientific fmt for small floats
    np.set_printoptions(linewidth=240, suppress=True)


def np_fmt_reset():
    # reset default np printing options
    np.set_printoptions(
        edgeitems=3,
        infstr="inf",
        linewidth=75,
        nanstr="nan",
        precision=8,
        suppress=False,
        threshold=1000,
        formatter=None,
    )


if runtime_env() in ["notebook", "shell"]:

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # -- all imports below will appear in notebook after calling %%qubx magic ---
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - Common stuff - - - -
    import numpy as np
    import pandas as pd
    from datetime import time, timedelta
    from tqdm.auto import tqdm

    # - - - - TA stuff and indicators - - - -
    import qubx.pandaz.ta as pta
    import qubx.ta.indicators as ta

    # - - - - Portfolio analysis - - - -
    from qubx.core.metrics import (
        tearsheet,
        chart_signals,
        get_symbol_pnls,
        get_equity,
        portfolio_metrics,
        pnl,
        drop_symbols,
        pick_symbols,
    )

    # - - - - Data reading - - - -
    from qubx.data.readers import (
        CsvStorageDataReader,
        MultiQdbConnector,
        QuestDBConnector,
        AsOhlcvSeries,
        AsPandasFrame,
        AsQuotes,
        AsTimestampedRecords,
        RestoreTicksFromOHLC,
    )

    # - - - - Simulator stuff - - - -
    from qubx.backtester.simulator import simulate
    from qubx.backtester.optimization import variate

    # - - - - Charting stuff - - - -
    from matplotlib import pyplot as plt
    from qubx.utils.charting.mpl_helpers import fig, subplot, sbp, plot_trends, ohlc_plot
    from qubx.utils.charting.lookinglass import LookingGlass

    # - - - - Utils - - - -
    from qubx.pandaz.utils import (
        scols,
        srows,
        ohlc_resample,
        continuous_periods,
        generate_equal_date_ranges,
        drop_duplicated_indexes,
        retain_columns_and_join,
        rolling_forward_test_split,
    )

    # - setup short numpy output format
    np_fmt_short()

    # - add project home to system path
    add_project_to_system_path()

    # show logo first time
    if not hasattr(qubx.QubxMagics, "__already_initialized__"):
        setattr(qubx.QubxMagics, "__already_initialized__", True)
        logo()
