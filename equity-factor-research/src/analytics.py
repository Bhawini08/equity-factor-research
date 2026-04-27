"""
Performance and diagnostic analytics.

These are the numbers a quant researcher actually looks at to decide
whether a factor is real:

    IC      Information Coefficient. Spearman rank correlation between the
            factor score at time t and the realized return at t+1, computed
            cross-sectionally and reported as a time series.
    ICIR    Information Coefficient Information Ratio. mean(IC) / std(IC),
            optionally annualized. The cross-sectional analogue of a Sharpe
            ratio: it tells you how reliably the factor predicts returns,
            not how big the bet is.
    Decile spread     The realized return spread between top and bottom
                      deciles. The portfolio-level translation of IC.
    Sharpe, MaxDD, hit rate, monthly stats — standard portfolio metrics
    on the long-short return stream.

We use Spearman (rank) correlation rather than Pearson because:
  - It is robust to factor outliers without requiring winsorization.
  - The economic claim is "high-ranked stocks beat low-ranked stocks,"
    which is a rank statement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def information_coefficient(
    panel: pd.DataFrame,
    factor_col: str,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-sectional Spearman IC between factor_col at date t and the
    realized return over [t, t+1].

    Returns
    -------
    DataFrame [date, ic, n] where n is the number of stocks in the
    cross-section that month (a sanity check: if n drops to ~50 in 2008,
    that's a sample issue worth flagging).
    """
    fwd = returns.sort_values(["permno", "date"]).copy()
    fwd["fwd_ret"] = fwd.groupby("permno")["total_ret"].shift(-1)
    merged = panel.merge(
        fwd[["permno", "date", "fwd_ret"]],
        on=["permno", "date"],
        how="inner",
    ).dropna(subset=[factor_col, "fwd_ret"])

    out = []
    for d, g in merged.groupby("date"):
        if len(g) < 20:
            continue
        rho, _ = stats.spearmanr(g[factor_col], g["fwd_ret"])
        out.append({"date": d, "ic": rho, "n": len(g)})
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


def ic_summary(ic_series: pd.DataFrame, periods_per_year: int = 12) -> dict:
    """
    Summary statistics for an IC time series.
        mean_ic     average cross-sectional rank correlation
        std_ic      time-series volatility of IC
        icir        mean / std (per period)
        ann_icir    icir * sqrt(periods_per_year)
        hit_rate    fraction of months with IC > 0
        t_stat      t-statistic on mean(IC) being > 0
    """
    ic = ic_series["ic"].dropna()
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "ann_icir": icir * np.sqrt(periods_per_year),
        "hit_rate": (ic > 0).mean(),
        "t_stat": mean_ic / (std_ic / np.sqrt(len(ic))) if std_ic > 0 else np.nan,
        "n_periods": len(ic),
    }


# ---------------------------------------------------------------------------
# Portfolio performance metrics
# ---------------------------------------------------------------------------

def annualize_return(monthly_returns: pd.Series) -> float:
    """Geometric annualization of a monthly return series."""
    r = monthly_returns.dropna()
    if r.empty:
        return np.nan
    return (1 + r).prod() ** (12 / len(r)) - 1


def annualize_vol(monthly_returns: pd.Series) -> float:
    return monthly_returns.dropna().std(ddof=1) * np.sqrt(12)


def sharpe(monthly_returns: pd.Series, rf_monthly: float = 0.0) -> float:
    r = monthly_returns.dropna() - rf_monthly
    if r.empty or r.std(ddof=1) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=1)) * np.sqrt(12)


def max_drawdown(monthly_returns: pd.Series) -> float:
    """Max drawdown of the cumulative return path (in decimal, e.g. -0.35)."""
    r = monthly_returns.dropna()
    if r.empty:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.min()


def hit_rate(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if r.empty:
        return np.nan
    return (r > 0).mean()


def performance_summary(monthly_returns: pd.Series) -> dict:
    return {
        "ann_return": annualize_return(monthly_returns),
        "ann_vol": annualize_vol(monthly_returns),
        "sharpe": sharpe(monthly_returns),
        "max_drawdown": max_drawdown(monthly_returns),
        "hit_rate": hit_rate(monthly_returns),
        "n_months": int(monthly_returns.dropna().shape[0]),
    }


# ---------------------------------------------------------------------------
# Decile spread analysis
# ---------------------------------------------------------------------------

def decile_summary(decile_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Per-decile annualized return, vol, and Sharpe. Useful for showing the
    monotonicity of the factor: in a working factor, mean return rises
    roughly monotonically from D1 to D10.
    """
    rows = []
    for col in decile_ret.columns:
        s = decile_ret[col].dropna()
        rows.append({
            "portfolio": col,
            "ann_return": annualize_return(s),
            "ann_vol": annualize_vol(s),
            "sharpe": sharpe(s),
            "n_months": len(s),
        })
    return pd.DataFrame(rows)


def cumulative_returns(monthly_returns: pd.Series) -> pd.Series:
    """Cumulative wealth path starting at 1.0."""
    return (1 + monthly_returns.fillna(0)).cumprod()


def rolling_sharpe(
    monthly_returns: pd.Series,
    window: int = 36,
) -> pd.Series:
    """3-year rolling Sharpe. Useful for the dashboard regime view."""
    r = monthly_returns.dropna()
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    return (mu / sd) * np.sqrt(12)
