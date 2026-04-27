"""
Factor construction: momentum, value, and quality.

Three factors, each with a clear economic story and a clean implementation:

    momentum (MOM)        12-1 month return. Skip the most recent month to
                          avoid the short-term reversal effect documented by
                          Jegadeesh and Titman (1993). Constructed from CRSP
                          monthly total returns.

    value (BM)            Book-to-Market ratio. Book equity from Compustat
                          (ceq, common equity), market equity from CRSP
                          (|prc| * shrout). Following Fama and French (1992).

    quality (ROE, GP)     Return on Equity (ib / ceq) and Gross Profitability
                          (gp / at), the latter from Novy-Marx (2013) which
                          shows GP/A is one of the cleanest quality signals
                          in the literature.

Point-in-time discipline
------------------------
The single most important thing in this file is the fundamentals lag. A
firm's fiscal-year-end-December balance sheet (datadate = 2010-12-31) is not
in the public 10-K until ~March 2011. Using it to rank stocks in January
2011 is look-ahead bias and will inflate IC by 30-50% in our experience.

Standard practice, and what we do here:
  - Quarterly fundamentals: lag 4 months from datadate before they become
    available for ranking.
  - Annual fundamentals: lag 6 months (the conservative academic choice;
    Fama-French use a June-of-year-t formation date for fiscal-year-end-
    December data, which is a 6-month lag).

We use the CCM linking table (crsp.ccmxpf_linktable) to map Compustat's
GVKEY to CRSP's PERMNO. Linktype is restricted to LU/LC (the two strongest
link types) and linkprim is restricted to P/C (primary links only) to avoid
duplicate joins on multi-share-class firms.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .universe import get_wrds_connection


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def compute_momentum(returns: pd.DataFrame) -> pd.DataFrame:
    """
    12-1 month momentum: cumulative return from t-12 through t-2, skipping
    the most recent month (t-1). The skip-month convention is standard and
    avoids short-term reversal contaminating the signal.

    Parameters
    ----------
    returns : DataFrame
        Long format with columns [permno, date, total_ret].

    Returns
    -------
    DataFrame [permno, date, momentum]
        date is the formation date. The momentum value is known at end of
        date and is used to rank stocks for the period beginning the next
        trading day.
    """
    df = returns.sort_values(["permno", "date"]).copy()
    df["log_ret"] = np.log1p(df["total_ret"])

    # Sum of log returns from t-12 to t-2 inclusive = 11-month window ending
    # one month before the formation date. Implement as: rolling 12-month sum,
    # then subtract the most recent month.
    grp = df.groupby("permno")["log_ret"]
    df["cum12"] = grp.transform(lambda s: s.rolling(12, min_periods=12).sum())
    df["last1"] = grp.shift(0)  # the t-0 month return; we want to remove it
    df["log_mom"] = df["cum12"] - df["last1"]
    df["momentum"] = np.expm1(df["log_mom"])

    return df[["permno", "date", "momentum"]].dropna(subset=["momentum"])


# ---------------------------------------------------------------------------
# Compustat fundamentals
# ---------------------------------------------------------------------------

def get_compustat_annual(
    start_date: str = "1999-01-01",
    end_date: str = "2025-12-31",
    db=None,
) -> pd.DataFrame:
    """
    Pull annual fundamentals needed for value and quality factors.

    We pull from comp.funda with the standard population filters:
      indfmt = 'INDL'      industrial (vs financial-services format)
      datafmt = 'STD'      standardized
      popsrc = 'D'         domestic
      consol = 'C'         consolidated

    Columns:
      gvkey, datadate    primary key
      ceq                common equity (book value)
      ib                 income before extraordinary items (for ROE)
      gp                 gross profit
      at                 total assets
      revt               revenue (sanity check)
    """
    own = db is None
    if own:
        db = get_wrds_connection()
    try:
        query = f"""
            SELECT gvkey, datadate, ceq, ib, gp, at, revt
            FROM   comp.funda
            WHERE  indfmt  = 'INDL'
              AND  datafmt = 'STD'
              AND  popsrc  = 'D'
              AND  consol  = 'C'
              AND  datadate BETWEEN '{start_date}' AND '{end_date}'
        """
        df = db.raw_sql(query, date_cols=["datadate"])
    finally:
        if own:
            db.close()

    # Drop rows where the fields we need are all missing
    df = df.dropna(subset=["ceq"], how="any")
    return df


def get_ccm_link(db=None) -> pd.DataFrame:
    """
    CRSP-Compustat Merged linking table. Maps gvkey -> permno with a
    validity window [linkdt, linkenddt]. We restrict to:
      linktype in ('LU','LC')   strongest link types only
      linkprim in ('P','C')     primary or co-primary security
    Following standard WRDS research practice.
    """
    own = db is None
    if own:
        db = get_wrds_connection()
    try:
        query = """
            SELECT gvkey, lpermno AS permno, linktype, linkprim,
                   linkdt, linkenddt
            FROM   crsp.ccmxpf_linktable
            WHERE  linktype IN ('LU','LC')
              AND  linkprim IN ('P','C')
        """
        df = db.raw_sql(query, date_cols=["linkdt", "linkenddt"])
    finally:
        if own:
            db.close()
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    return df


def link_fundamentals_to_permno(
    funda: pd.DataFrame,
    ccm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join annual fundamentals to PERMNO using the CCM linktable,
    keeping only rows where datadate falls inside the link's validity
    window. This is the canonical CCM merge.
    """
    merged = funda.merge(ccm, on="gvkey", how="inner")
    valid = (merged["datadate"] >= merged["linkdt"]) & \
            (merged["datadate"] <= merged["linkenddt"])
    merged = merged.loc[valid].copy()
    merged["permno"] = merged["permno"].astype(int)
    return merged


# ---------------------------------------------------------------------------
# Point-in-time alignment
# ---------------------------------------------------------------------------

def make_pit_fundamentals(
    funda_linked: pd.DataFrame,
    monthly_dates: pd.DatetimeIndex,
    lag_months: int = 6,
) -> pd.DataFrame:
    """
    Convert annual fundamentals (one row per gvkey-datadate) into a monthly
    point-in-time panel where each (permno, month) row contains the most
    recent fundamentals that would have been *publicly known* on that date.

    Parameters
    ----------
    funda_linked : DataFrame
        Output of link_fundamentals_to_permno.
    monthly_dates : DatetimeIndex
        The rebalance dates we care about (month-ends).
    lag_months : int
        Months to add to datadate before we treat the fundamental as known.
        6 is the Fama-French convention for fiscal-year-end-December data.

    Returns
    -------
    DataFrame [permno, date, ceq, ib, gp, at, datadate]
        date is the rebalance date; datadate is the source fiscal year-end.
    """
    f = funda_linked.copy()
    # The date at which this fundamental becomes available for ranking
    f["available_date"] = f["datadate"] + pd.DateOffset(months=lag_months)
    f["available_date"] = f["available_date"] + pd.offsets.MonthEnd(0)

    # For each (permno, rebalance_date), find the most recent fundamental
    # whose available_date <= rebalance_date. asof-merge handles this cleanly.
    f = f.sort_values(["permno", "available_date"])
    rebal = pd.DataFrame({"date": monthly_dates}).sort_values("date")

    # Build the cross product, then asof within each permno
    permnos = f["permno"].unique()
    pieces = []
    for p in permnos:
        sub = f[f["permno"] == p][["available_date", "datadate", "ceq",
                                    "ib", "gp", "at"]]
        if sub.empty:
            continue
        sub = sub.sort_values("available_date")
        out = pd.merge_asof(
            rebal.rename(columns={"date": "available_date"})
                 .sort_values("available_date"),
            sub,
            on="available_date",
            direction="backward",
        )
        out["permno"] = p
        out = out.rename(columns={"available_date": "date"})
        pieces.append(out)

    pit = pd.concat(pieces, ignore_index=True)
    return pit.dropna(subset=["ceq"])


# ---------------------------------------------------------------------------
# Value and quality factors
# ---------------------------------------------------------------------------

def compute_value(
    pit_funda: pd.DataFrame,
    mktcap_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Book-to-Market ratio. Numerator is book equity (ceq) at the most recent
    fiscal year-end available point-in-time. Denominator is current market
    cap from CRSP. Drops negative book equity firms (standard, otherwise
    B/M flips sign and the ranking becomes nonsensical).
    """
    df = pit_funda.merge(mktcap_panel, on=["permno", "date"], how="inner")
    df = df[df["ceq"] > 0].copy()
    df["book_to_market"] = df["ceq"] / df["mktcap"]
    return df[["permno", "date", "book_to_market"]]


def compute_quality(pit_funda: pd.DataFrame) -> pd.DataFrame:
    """
    Two quality signals, equally weighted into a composite:

        ROE = ib / ceq                 income before extraordinary / book equity
        GP_A = gp / at                 gross profitability over assets
                                        (Novy-Marx 2013)

    Each is z-scored cross-sectionally and averaged. We z-score *before*
    averaging because the two signals are on different scales.
    """
    df = pit_funda.copy()
    df = df[(df["ceq"] > 0) & (df["at"] > 0)]
    df["roe"] = df["ib"] / df["ceq"]
    df["gp_a"] = df["gp"] / df["at"]

    # Cross-sectional z-score within each rebalance date, with winsorization
    # at 1/99% to keep extreme outliers from dominating the score.
    def _zscore(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        return (s - s.mean()) / s.std(ddof=0)

    df["roe_z"] = df.groupby("date")["roe"].transform(_zscore)
    df["gp_z"] = df.groupby("date")["gp_a"].transform(_zscore)
    df["quality"] = df[["roe_z", "gp_z"]].mean(axis=1)
    return df[["permno", "date", "quality"]]


# ---------------------------------------------------------------------------
# Cross-sectional ranking and standardization
# ---------------------------------------------------------------------------

def cross_sectional_zscore(
    panel: pd.DataFrame,
    factor_col: str,
    winsor: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Cross-sectionally winsorize and z-score a factor within each date.
    This is what we feed into decile formation and the IC calculation.
    """
    df = panel.copy()

    def _z(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(winsor[0]), s.quantile(winsor[1])
        s = s.clip(lo, hi)
        return (s - s.mean()) / s.std(ddof=0)

    df[f"{factor_col}_z"] = df.groupby("date")[factor_col].transform(_z)
    return df


def build_factor_panel(
    returns: pd.DataFrame,
    pit_funda: pd.DataFrame,
    mktcap_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the final wide-format factor panel:
        index: (date, permno)
        cols : momentum, book_to_market, quality, plus their z-scores
               and a composite score (equal-weight average of z-scores).
    """
    mom = compute_momentum(returns)
    val = compute_value(pit_funda, mktcap_panel)
    qual = compute_quality(pit_funda)

    panel = (
        mom.merge(val, on=["permno", "date"], how="outer")
           .merge(qual, on=["permno", "date"], how="outer")
    )

    panel = cross_sectional_zscore(panel, "momentum")
    panel = cross_sectional_zscore(panel, "book_to_market")
    panel = cross_sectional_zscore(panel, "quality")

    panel["composite"] = panel[
        ["momentum_z", "book_to_market_z", "quality_z"]
    ].mean(axis=1)

    return panel
