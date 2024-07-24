"""
    Temporary implementation of portfolio performance metrics
    Probably need to consider some third-party library for this purpose
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

import re
from scipy import stats
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS

import plotly.graph_objects as go

from qubx.core.basics import TradingSessionResult
from qubx.utils.charting.lookinglass import LookingGlass
from qubx.utils.time import infer_series_frequency


YEARLY = 1
MONTHLY = 12
WEEKLY = 52
DAILY = 252
DAILY_365 = 365
HOURLY = DAILY * 6.5
MINUTELY = HOURLY * 60
HOURLY_FX = DAILY * 24
MINUTELY_FX = HOURLY_FX * 60


def absmaxdd(data: List | Tuple | pd.Series | np.ndarray) -> Tuple[float, int, int, int, pd.Series]:
    """

    Calculates the maximum absolute drawdown of series data.

    Args:
        data: vector of doubles. Data may be presented as list,
        tuple, numpy array or pandas series object.

    Returns:
        (max_abs_dd, d_start, d_peak, d_recovered, dd_data)

    Where:
        - max_abs_dd: absolute maximal drawdown value
        - d_start: index from data array where drawdown starts
        - d_peak: index when drawdown reach it's maximal value
        - d_recovered: index when DD is fully recovered
        - dd_data: drawdown series

    Example:

    mdd, ds, dp, dr, dd_data = absmaxdd(np.random.randn(1,100).cumsum())
    """

    if not isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError("Unknown type of input series")

    datatype = type(data)

    if datatype is pd.Series:
        indexes = data.index
        data = data.values
    elif datatype is not np.ndarray:
        data = np.array(data)

    dd = np.maximum.accumulate(data) - data
    mdd = dd.max()
    d_peak = dd.argmax()

    if mdd == 0:
        return 0, 0, 0, 0, [0]

    zeros_ixs = np.where(dd == 0)[0]
    zeros_ixs = np.insert(zeros_ixs, 0, 0)
    zeros_ixs = np.append(zeros_ixs, dd.size)

    d_start = zeros_ixs[zeros_ixs < d_peak][-1]
    d_recover = zeros_ixs[zeros_ixs > d_peak][0]

    if d_recover >= data.__len__():
        d_recover = data.__len__() - 1

    if datatype is pd.Series:
        dd = pd.Series(dd, index=indexes)

    return mdd, d_start, d_peak, d_recover, dd


def max_drawdown_pct(returns):
    """
    Finds the maximum drawdown of a strategy returns in percents

    :param returns: pd.Series or np.ndarray daily returns of the strategy, noncumulative
    :return: maximum drawdown in percents
    """
    if len(returns) < 1:
        return np.nan

    if isinstance(returns, pd.Series):
        returns = returns.values

    # drop nans
    returns[np.isnan(returns) | np.isinf(returns)] = 0.0

    cumrets = 100 * (returns + 1).cumprod(axis=0)
    max_return = np.fmax.accumulate(cumrets)
    return np.nanmin((cumrets - max_return) / max_return)


def portfolio_returns(portfolio_log: pd.DataFrame, method="pct", init_cash=0) -> pd.Series:
    """
    Calculates returns based on specified method.

    :param pfl_log: portfolio log frame
    :param method: method to calculate, there are 3 main methods:
                    - percentage on equity ('pct', 'equity', 'on equity')
                    - percentage on previous portfolio value ('gmv', 'gross')
                    - percentage on fixed deposit amount ('fixed')

    :param init_cash: must be > 0 if used method is 'depo'
    :return: returns series
    """
    if "Total_PnL" not in portfolio_log.columns:
        portfolio_log = calculate_total_pnl(portfolio_log, split_cumulative=True)

    if method in ["pct", "equity", "on equity"]:
        # 'standard' percent of changes. It also takes initial deposit
        rets = (portfolio_log["Total_PnL"] + init_cash).pct_change()
    elif method in ["gmv", "gross"]:
        # today return is pct of yesterday's portfolio value (is USD)
        rets = (
            portfolio_log["Total_PnL"].diff() / (portfolio_log.filter(regex=".*_Value").abs().sum(axis=1).shift(1))
        ).fillna(0)
    elif method in ["fixed"]:
        # return is pct of PL changes to initial deposit (for fixed BP)
        if init_cash <= 0:
            raise ValueError("You must specify exact initial cash value when using 'fixed' method")
        rets = portfolio_log["Total_PnL"].diff() / init_cash
    else:
        raise ValueError("Unknown returns calculation method '%s'" % method)

    # cleanup returns
    rets.name = "Returns"
    rets[np.isinf(abs(rets))] = 0
    rets[np.isnan(rets)] = 0

    return rets


def cagr(returns, periods=DAILY):
    """
    Calculates the Compound Annual Growth Rate (CAGR) for the portfolio, by determining the number of years
    and then creating a compound annualised rate based on the total return.

    :param returns: A pandas Series or np.ndarray representing the returns
    :param periods: Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: CAGR's value
    """
    if len(returns) < 1:
        return np.nan

    cumrets = (returns + 1).cumprod(axis=0)
    years = len(cumrets) / float(periods)
    return (cumrets.iloc[-1] ** (1.0 / years)) - 1.0


def calmar_ratio(returns, periods=DAILY):
    """
    Calculates the Calmar ratio, or drawdown ratio, of a strategy.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: Calmar ratio (drawdown ratio) as float
    """
    max_dd = max_drawdown_pct(returns)
    if max_dd < 0:
        temp = cagr(returns, periods) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def sharpe_ratio(returns, risk_free=0.0, periods=DAILY) -> float:
    """
    Calculates the Sharpe ratio.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param risk_free: constant risk-free return throughout the period
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: Sharpe ratio
    """
    if len(returns) < 2:
        return np.nan

    returns_risk_adj = returns - risk_free
    returns_risk_adj = returns_risk_adj[~np.isnan(returns_risk_adj)]

    if np.std(returns_risk_adj, ddof=1) == 0:
        return np.nan

    return np.mean(returns_risk_adj) / np.std(returns_risk_adj, ddof=1) * np.sqrt(periods)


def rolling_sharpe_ratio(returns, risk_free=0.0, periods=DAILY) -> pd.Series:
    """
    Rolling Sharpe ratio.
    :param returns: pd.Series periodic returns of the strategy, noncumulative
    :param risk_free: constant risk-free return throughout the period
    :param periods: rolling window length
    :return:
    """
    returns_risk_adj = returns - risk_free
    returns_risk_adj = returns_risk_adj[~np.isnan(returns_risk_adj)]
    rolling = returns_risk_adj.rolling(window=periods)
    return pd.Series(np.sqrt(periods) * (rolling.mean() / rolling.std()), name="RollingSharpe")


def sortino_ratio(returns: pd.Series, required_return=0, periods=DAILY, _downside_risk=None) -> float:
    """
    Calculates the Sortino ratio of a strategy.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param required_return: minimum acceptable return
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :param _downside_risk: the downside risk of the given inputs, if known. Will be calculated if not provided
    :return: annualized Sortino ratio
    """
    if len(returns) < 2:
        return np.nan

    mu = np.nanmean(returns - required_return, axis=0)
    dsr = _downside_risk if _downside_risk is not None else downside_risk(returns, required_return)
    if dsr == 0.0:
        return np.nan if mu == 0 else np.inf
    return periods * mu / dsr


def information_ratio(returns, factor_returns) -> float:
    """
    Calculates the Information ratio of a strategy (see https://en.wikipedia.org/wiki/information_ratio)
    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param factor_returns: benchmark return to compare returns against
    :return: information ratio
    """
    if len(returns) < 2:
        return np.nan

    active_return = returns - factor_returns
    tracking_error = np.nanstd(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.nanmean(active_return) / tracking_error


def downside_risk(returns, required_return=0.0, periods=DAILY):
    """
    Calculates the downside deviation below a threshold

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param required_return: minimum acceptable return
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: annualized downside deviation
    """
    if len(returns) < 1:
        return np.nan

    downside_diff = (returns - required_return).copy()
    downside_diff[downside_diff > 0] = 0.0
    mean_squares = np.nanmean(np.square(downside_diff), axis=0)
    ds_risk = np.sqrt(mean_squares) * np.sqrt(periods)

    if len(returns.shape) == 2 and isinstance(returns, pd.DataFrame):
        ds_risk = pd.Series(ds_risk, index=returns.columns)

    return ds_risk


def omega_ratio(returns, risk_free=0.0, required_return=0.0, periods=DAILY):
    """
    Omega ratio (see https://en.wikipedia.org/wiki/Omega_ratio for more details)

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param risk_free: constant risk-free return throughout the period
    :param required_return: Minimum acceptance return of the investor. Threshold over which to
                             consider positive vs negative returns. It will be converted to a
                             value appropriate for the period of the returns. E.g. An annual minimum
                             acceptable return of 100 will translate to a minimum acceptable
                             return of 0.018.
    :param periods: Factor used to convert the required_return into a daily
                     value. Enter 1 if no time period conversion is necessary.
    :return: Omega ratio
    """
    if len(returns) < 2:
        return np.nan

    if periods == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1.0 / periods) - 1

    returns_less_thresh = returns - risk_free - return_threshold
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    return (numer / denom) if denom > 0.0 else np.nan


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by specified time period
    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param convert_to: 'D', 'W', 'M', 'Y' (and any supported in pandas.resample method)
    :return: aggregated returns
    """

    def cumulate_returns(x):
        return ((x + 1).cumprod(axis=0) - 1).iloc[-1] if len(x) > 0 else 0.0

    str_check = convert_to.lower()
    resample_mod = None
    if str_check in ["a", "annual", "y", "yearly"]:
        resample_mod = "A"
    elif str_check in ["m", "monthly", "mon"]:
        resample_mod = "ME"
    elif str_check in ["w", "weekly"]:
        resample_mod = "W"
    elif str_check in ["d", "daily"]:
        resample_mod = "D"
    else:
        resample_mod = convert_to

    return returns.resample(resample_mod).apply(cumulate_returns)


def annual_volatility(returns, periods=DAILY, alpha=2.0):
    """
    Calculates annual volatility of a strategy

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                    Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :param alpha: scaling relation (Levy stability exponent).
    :return:
    """
    if len(returns) < 2:
        return np.nan

    return np.nanstd(returns, ddof=1) * (periods ** (1.0 / alpha))


def stability_of_returns(returns):
    """
    Calculates R-squared of a linear fit to the cumulative log returns.
    Computes an ordinary least squares linear fit, and returns R-squared.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :return: R-squared
    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]
    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]
    return rhat**2


def tail_ratio(returns):
    """
    Calculates the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times as bad as profits.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :return: tail ratio
    """
    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    pc5 = np.abs(np.percentile(returns, 5))

    return (np.abs(np.percentile(returns, 95)) / pc5) if pc5 != 0 else np.nan


def split_cumulative_pnl(pfl_log: pd.DataFrame) -> pd.DataFrame:
    """
    Position.pnl tracks cumulative PnL (realized+unrealized) but if we want to operate with PnL for every bar
    we need to find diff from these cumulative series

    :param pfl_log: position manager log (portfolio log)
    :return: frame with splitted PL
    """
    # take in account commissions (now we cumsum it)
    pl = pfl_log.filter(regex=r".*_PnL|.*_Commissions")
    if pl.shape[1] == 0:
        raise ValueError("PnL columns not found. Input frame must contain at least 1 column with '_PnL' suffix")

    pl_diff = pl.diff()

    # at first row we use first value of PnL
    pl_diff.loc[pl.index[0]] = pl.iloc[0]

    # substitute new diff PL
    pfl_splitted = pfl_log.copy()
    pfl_splitted.loc[:, pfl_log.columns.isin(pl_diff.columns)] = pl_diff
    return pfl_splitted


def calculate_total_pnl(pfl_log: pd.DataFrame, split_cumulative=True) -> pd.DataFrame:
    """
    Finds summary of all P&L column (should have '_PnL' suffix) in given portfolio log dataframe.
    Attaches additional Total_PnL column with result.

    :param pfl_log: position manager log (portfolio log)
    :param split_cumulative: set true if we need to split cumulative PnL [default is True]
    :return:
    """
    n_pfl = pfl_log.copy()
    if "Total_PnL" not in n_pfl.columns:
        if split_cumulative:
            n_pfl = split_cumulative_pnl(n_pfl)

        n_pfl["Total_PnL"] = n_pfl.filter(regex=r".*_PnL").sum(axis=1)
        n_pfl["Total_Commissions"] = n_pfl.filter(regex=r".*_Commissions").sum(axis=1)

    return n_pfl


def alpha(returns, factor_returns, risk_free=0.0, period=DAILY, _beta=None):
    """
    Calculates annualized alpha of portfolio.

    :param returns: Daily returns of the strategy, noncumulative.
    :param factor_returns: Daily noncumulative returns of the factor to which beta is
           computed. Usually a benchmark such as the market
    :param risk_free: Constant risk-free return throughout the period. For example, the
                      interest rate on a three month us treasury bill
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                    Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :param _beta: The beta for the given inputs, if already known. Will be calculated
                internally if not provided.
    :return: alpha
    """
    if len(returns) < 2:
        return np.nan

    if _beta is None:
        _beta = beta(returns, factor_returns, risk_free)

    adj_returns = returns - risk_free
    adj_factor_returns = factor_returns - risk_free
    alpha_series = adj_returns - (_beta * adj_factor_returns)

    return np.nanmean(alpha_series) * period


def beta(returns, benchmark_returns, risk_free=0.0):
    """
    Calculates beta of portfolio.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    :param returns: pd.Series or np.ndarray. Daily returns of the strategy, noncumulative.
    :param benchmark_returns: pd.Series or np.ndarray. Daily noncumulative returns of the factor to which beta is
           computed. Usually a benchmark such as the market.
    :param risk_free: Constant risk-free return throughout the period. For example, the interest rate
                      on a three month us treasury bill.
    :return: beta
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return np.nan

    # Filter out dates with np.nan as a return value

    if len(returns) != len(benchmark_returns):
        if len(returns) > len(benchmark_returns):
            returns = returns.drop(returns.index.difference(benchmark_returns.index))
        else:
            benchmark_returns = benchmark_returns.drop(benchmark_returns.index.difference(returns.index))

    joint = np.vstack([returns - risk_free, benchmark_returns])
    joint = joint[:, ~np.isnan(joint).any(axis=0)]
    if joint.shape[1] < 2:
        return np.nan

    cov = np.cov(joint, ddof=0)

    if np.absolute(cov[1, 1]) < 1.0e-30:
        return np.nan

    return cov[0, 1] / cov[1, 1]


def var_cov_var(P_usd, mu, sigma, c=0.95):
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma, on a portfolio of value P.

    https://www.quantstart.com/articles/Value-at-Risk-VaR-for-Algorithmic-Trading-Risk-Management-Part-I

    also here:
    http://stackoverflow.com/questions/30878265/calculating-value-at-risk-or-most-probable-loss-for-a-given-distribution-of-r#30895548

    :param P_usd: portfolio value
    :param c: confidence level
    :param mu: mean of returns
    :param sigma: standard deviation of returns
    :return: value at risk
    """
    alpha = norm.ppf(1 - c, mu, sigma)
    return P_usd - P_usd * (alpha + 1)


def qr(equity):
    """

    QR = R2 * B / S

    Where:
     B - slope (roughly in the ranges of average trade PnL, higher is better)
     R2 - r squared metric (proportion of variance explained by linear model, straight line has r2 == 1)
     S - standard error represents volatility of equity curve (lower is better)

    :param equity: equity (cumulative)
    :return: QR measure or NaN if not enough data for calculaitons
    """
    if len(equity) < 1 or all(equity == 0.0):
        return np.nan

    rgr = OLS(equity, np.vander(np.linspace(-1, 1, len(equity)), 2)).fit()
    b = rgr.params.iloc[0] if isinstance(rgr.params, pd.Series) else rgr.params[0]
    return rgr.rsquared * b / np.std(rgr.resid)


def monthly_returns(
    portfolio, init_cash, period="monthly", daily="pct", monthly="pct", weekly="pct", performace_period=DAILY
):
    """
    Calculate monthly or weekly returns table along with account balance
    """
    pft_total = calculate_total_pnl(portfolio, split_cumulative=False)
    pft_total["Total_PnL"] = pft_total["Total_PnL"].cumsum()
    returns = portfolio_returns(pft_total, init_cash=init_cash, method=daily)
    r_daily = aggregate_returns(returns, "daily")
    print("CAGR: %.2f%%" % (100 * cagr(r_daily, performace_period)))

    if period == "weekly":
        returns = portfolio_returns(pft_total, init_cash=init_cash, method=weekly)
        r_month = aggregate_returns(returns, "weekly")
        acc_balance = init_cash + pft_total.Total_PnL.groupby(pd.Grouper(freq="1W")).last()
    else:
        returns = portfolio_returns(pft_total, init_cash=init_cash, method=monthly)
        r_month = aggregate_returns(returns, "monthly")
        acc_balance = init_cash + pft_total.Total_PnL.groupby(pd.Grouper(freq="1M")).last()

    return pd.concat((100 * r_month, acc_balance), axis=1, keys=["Returns", "Balance"])


def portfolio_symbols(df: pd.DataFrame) -> List[str]:
    """
    Get list of symbols from portfolio log
    """
    return list(df.columns[::5].str.split("_").str.get(0).values)


def pnl(x: pd.DataFrame, c=1, cum=False, total=False, resample=None):
    """
    Extract PnL from portfolio log
    """
    pl = x.filter(regex=".*_PnL").rename(lambda x: x.split("_")[0], axis=1)
    comms = x.filter(regex=".*_Commissions").rename(lambda x: x.split("_")[0], axis=1)
    r = pl - c * comms
    if resample:
        r = r.resample(resample).sum()
    r = r.cumsum() if cum else r
    return r.sum(axis=1) if total else r


def drop_symbols(df: pd.DataFrame, *args, quoted="USDT"):
    """
    Drop symbols (is quoted currency) from portfolio log
    """
    s = "|".join([f"{a}{quoted}" if not a.endswith(quoted) else a for a in args])
    return df.filter(filter(lambda si: not re.match(f"^{s}_.*", si), df.columns))


def pick_symbols(df: pd.DataFrame, *args, quoted="USDT"):
    """
    Select symbols (is quoted currency) from portfolio log
    """
    # - pick up from execution report
    if "instrument" in df.columns and "quantity" in df.columns:
        rx = "|".join([f"{a}.*" for a in args])
        return df[df["instrument"].str.match(rx)]

    # - pick up from PnL log report
    s = "|".join([f"{a}{quoted}" if not a.endswith(quoted) else a for a in args])
    return df.filter(filter(lambda si: re.match(f"^{s}_.*", si), df.columns))


def portfolio_metrics(
    portfolio_log: pd.DataFrame,
    executions_log: pd.DataFrame,
    init_cash: float,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    risk_free: float = 0.0,
    rolling_sharpe_window=12,
    account_transactions=True,
    performance_statistics_period=DAILY_365,
    **kwargs,
) -> dict:
    if len(portfolio_log) == 0:
        raise ValueError("Can't calculate statistcis on empty portfolio")

    sheet = dict()

    pft_total = calculate_total_pnl(portfolio_log, split_cumulative=False)
    pft_total["Total_PnL"] = pft_total["Total_PnL"].cumsum()
    pft_total["Total_Commissions"] = pft_total["Total_Commissions"].cumsum()

    # if it's asked to account transactions into equ
    if account_transactions:
        pft_total["Total_PnL"] -= pft_total["Total_Commissions"]

    # calculate returns
    returns = portfolio_returns(pft_total, init_cash=init_cash, method="pct")
    returns_on_init_bp = portfolio_returns(pft_total, init_cash=init_cash, method="fixed")

    if start:
        returns = returns[start:]
        returns_on_init_bp = returns_on_init_bp[start:]

    if end:
        returns = returns[:end]
        returns_on_init_bp = returns_on_init_bp[:end]

    # aggregate them to daily (if we have intraday portfolio)
    if infer_series_frequency(returns) < pd.Timedelta("1D").to_timedelta64():
        returns_daily = aggregate_returns(returns, "daily")
        returns_on_init_bp = aggregate_returns(returns_on_init_bp, "daily")

    # todo: add transaction_cost calculations
    equity = init_cash + pft_total["Total_PnL"]
    mdd, ddstart, ddpeak, ddrecover, dd_data = absmaxdd(equity)
    sheet["equity"] = equity
    sheet["gain"] = sheet["equity"].iloc[-1] - sheet["equity"].iloc[0]
    sheet["cagr"] = cagr(returns_daily, performance_statistics_period)
    sheet["sharpe"] = sharpe_ratio(returns_daily, risk_free, performance_statistics_period)
    sheet["qr"] = qr(equity)
    sheet["drawdown_usd"] = dd_data
    sheet["drawdown_pct"] = 100 * dd_data / init_cash
    # 25-May-2019: MDE fixed Max DD pct calculations
    sheet["max_dd_pct"] = 100 * mdd / equity.iloc[ddstart]  # max_drawdown_pct(returns)
    # sheet["max_dd_pct_on_init"] = 100 * mdd / init_cash
    sheet["mdd_usd"] = mdd
    sheet["mdd_start"] = equity.index[ddstart]
    sheet["mdd_peak"] = equity.index[ddpeak]
    sheet["mdd_recover"] = equity.index[ddrecover]
    sheet["returns"] = returns
    sheet["returns_daily"] = returns_daily
    sheet["compound_returns"] = (returns + 1).cumprod(axis=0) - 1
    sheet["rolling_sharpe"] = rolling_sharpe_ratio(returns_daily, risk_free, periods=rolling_sharpe_window)
    sheet["sortino"] = sortino_ratio(
        returns_daily, risk_free, performance_statistics_period, _downside_risk=kwargs.pop("downside_risk", None)
    )
    sheet["calmar"] = calmar_ratio(returns_daily, performance_statistics_period)
    # sheet["ann_vol"] = annual_volatility(returns_daily)
    sheet["tail_ratio"] = tail_ratio(returns_daily)
    sheet["stability"] = stability_of_returns(returns_daily)
    sheet["monthly_returns"] = aggregate_returns(returns_daily, convert_to="mon")
    r_m = np.mean(returns_daily)
    r_s = np.std(returns_daily)
    sheet["var"] = var_cov_var(init_cash, r_m, r_s)
    sheet["avg_return"] = 100 * r_m

    # portfolio market values
    mkt_value = pft_total.filter(regex=".*_Value")
    sheet["long_value"] = mkt_value[mkt_value > 0].sum(axis=1).fillna(0)
    sheet["short_value"] = mkt_value[mkt_value < 0].sum(axis=1).fillna(0)

    # total commissions
    sheet["fees"] = pft_total["Total_Commissions"].iloc[-1]

    # executions metrics
    sheet["execs"] = len(executions_log)

    return sheet


def tearsheet(
    session: TradingSessionResult | List[TradingSessionResult],
    compound: bool = True,
    account_transactions=True,
    performance_statistics_period=365,
):
    if isinstance(session, list):
        if len(session) == 1:
            return _tearsheet_single(session[0], compound, account_transactions, performance_statistics_period)
        else:
            # multiple sessions - just show table
            _rs = []
            _eq = []
            for s in session:
                report, mtrx = _pfl_metrics_prepare(s, account_transactions, performance_statistics_period)
                _rs.append(report)
                if compound:
                    _eq.append(pd.Series(100 * mtrx["compound_returns"], name=s.trading_id))
                else:
                    _eq.append(pd.Series(mtrx["equity"], name=s.trading_id))

            pd.concat(_eq, axis=1).plot()
            return pd.concat(_rs, axis=1).T

    else:
        return _tearsheet_single(session, compound, account_transactions, performance_statistics_period)


def _pfl_metrics_prepare(session: TradingSessionResult, account_transactions: bool, performance_statistics_period: int):
    mtrx = portfolio_metrics(
        session.portfolio_log,
        session.executions_log,
        session.capital,
        performance_statistics_period=performance_statistics_period,
        account_transactions=account_transactions,
    )
    rpt = {}
    for k, v in mtrx.items():
        if isinstance(v, (float, int, str)):
            n = (k[0].upper() + k[1:]).replace("_", " ")
            rpt[n] = v
    return pd.Series(rpt, name=session.trading_id), mtrx


def _tearsheet_single(
    session: TradingSessionResult,
    compound: bool = True,
    account_transactions=True,
    performance_statistics_period=365,
):
    report, mtrx = _pfl_metrics_prepare(session, account_transactions, performance_statistics_period)
    tbl = go.Table(
        columnwidth=[130, 130, 130, 130, 200],
        header=dict(
            values=report.index,
            line_color="darkslategray",
            fill_color="#303030",
            font=dict(color="white", size=11),
        ),
        cells=dict(
            values=round(report, 3).values.tolist(),
            line_color="darkslategray",
            fill_color="#101010",
            align=["center", "left"],
            font=dict(size=10),
        ),
    )

    _eqty = (
        ["area", "green", 100 * mtrx["compound_returns"]]
        if compound
        else ["area", "green", mtrx["equity"] - mtrx["equity"].iloc[0]]
    )
    _dd = (
        [
            "area",
            -mtrx["drawdown_pct"],
            "lim",
            [-mtrx["max_dd_pct"], 0],
        ]
        if compound
        else [
            "area",
            -mtrx["drawdown_usd"],
            "lim",
            [-mtrx["mdd_usd"], 0],
        ]
    )
    chart = (
        LookingGlass(
            _eqty,
            {
                "Drawdown": _dd,
            },
            study_plot_height=75,
        )
        .look(title=("Simulation: " if session.is_simulation else "") + session.trading_id)
        .hover(h=500)
    )
    table = go.FigureWidget(tbl).update_layout(margin=dict(r=5, l=5, t=0, b=1), height=80)
    chart.show()
    table.show()