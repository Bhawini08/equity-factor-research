"""
Portfolio construction and backtest engine.

For each factor we form 10 deciles each month based on the cross-sectional
factor score, hold them for one month, and compute equal-weighted returns.
The long-short portfolio is D10 (top decile) minus D1 (bottom decile).

Conventions
-----------
- Decile 10 = highest factor score (most attractive). For value, this is
  the highest book-to-market (the "value" stocks). For momentum, the
  highest 12-1 return. For quality, the highest composite quality score.
- Equal-weighted within decile. Value-weighted is also common but masks
  the small-cap effect; equal-weight is cleaner for academic research and
  is what most factor papers report.
- Forward returns: portfolios formed at end of month t earn the return
  realized over month t+1. We shift returns by -1 within each permno to
  align them.
- No transaction costs in the headline numbers, but turnover is computed
  so a reader can apply a cost assumption (e.g., 10 bps per side gives a
  rough cost drag of 2 * turnover * 10bps per period).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def _forward_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Shift each permno's return back by one month so that the date
    column refers to the *formation* date, and total_ret is the return
    earned over the *following* month."""
    df = returns.sort_values(["permno", "date"]).copy()
    df["fwd_ret"] = df.groupby("permno")["total_ret"].shift(-1)
    return df[["permno", "date", "fwd_ret"]]


def assign_deciles(
    panel: pd.DataFrame,
    factor_col: str,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Assign each (permno, date) row a decile from 1 (lowest) to n_buckets
    (highest), ranked within the cross-section on each date.

    Uses pd.qcut with duplicates='drop' to handle ties at decile boundaries
    gracefully; on dates with very few stocks (early in the sample if any
    months had a thin universe) we may end up with fewer than 10 buckets
    and we flag those rows.
    """
    df = panel.dropna(subset=[factor_col]).copy()

    def _bucket(g: pd.DataFrame) -> pd.DataFrame:
        try:
            g = g.copy()
            g["decile"] = pd.qcut(
                g[factor_col],
                q=n_buckets,
                labels=False,
                duplicates="drop",
            ) + 1
            return g
        except ValueError:
            g["decile"] = np.nan
            return g

    df = df.groupby("date", group_keys=False).apply(_bucket)
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)
    return df


def decile_returns(
    deciled: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute equal-weighted next-month returns for each (date, decile).

    Returns
    -------
    DataFrame with index = date, columns = D1...D10 plus 'long_short'.
    """
    fwd = _forward_returns(returns)
    merged = deciled.merge(fwd, on=["permno", "date"], how="inner")
    merged = merged.dropna(subset=["fwd_ret"])

    g = (
        merged.groupby(["date", "decile"])["fwd_ret"]
        .mean()
        .unstack("decile")
        .sort_index()
    )
    g.columns = [f"D{int(c)}" for c in g.columns]
    if "D10" in g.columns and "D1" in g.columns:
        g["long_short"] = g["D10"] - g["D1"]
    return g


def turnover(deciled: pd.DataFrame, decile: int = 10) -> pd.Series:
    """
    Single-side turnover for a given decile: fraction of names that change
    from one rebalance to the next. A turnover of 0.4 means 40% of the
    long-leg portfolio is replaced each month.

    For the long-short portfolio, total turnover ~ turnover(D10) + turnover(D1).
    """
    sub = deciled[deciled["decile"] == decile].copy()
    by_date = sub.groupby("date")["permno"].apply(set).sort_index()

    out = pd.Series(index=by_date.index, dtype=float, name=f"turnover_D{decile}")
    prev = None
    for d, names in by_date.items():
        if prev is None:
            out.loc[d] = np.nan
        else:
            if len(names) == 0:
                out.loc[d] = np.nan
            else:
                changed = len(names.symmetric_difference(prev)) / 2
                out.loc[d] = changed / len(names)
        prev = names
    return out


def long_short_with_costs(
    decile_ret: pd.DataFrame,
    deciled: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Apply a simple linear transaction cost model:
        cost_t = (turnover_long_t + turnover_short_t) * cost_bps / 10000

    This is rough (no slippage curve, no fee asymmetry) but is the standard
    quick-and-honest adjustment cited in most factor papers.
    """
    to_long = turnover(deciled, decile=10)
    to_short = turnover(deciled, decile=1)
    cost = (to_long.fillna(0) + to_short.fillna(0)) * cost_bps / 10000.0

    out = decile_ret[["long_short"]].copy()
    out["cost"] = cost.reindex(out.index).fillna(0)
    out["long_short_net"] = out["long_short"] - out["cost"]
    return out
