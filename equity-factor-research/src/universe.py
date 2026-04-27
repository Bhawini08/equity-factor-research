"""
Universe construction: historical S&P 500 membership and price data.

Key design decisions
--------------------
1. Point-in-time membership.
   We pull S&P 500 constituents from CRSP (msp500list), which gives the start
   and end date of each stock's index membership. At each rebalance date we
   take only the names that were in the index on that date. This eliminates
   survivorship bias and look-ahead bias from using a static current list.

2. PERMNO as the primary key.
   We carry CRSP's PERMNO through the whole pipeline rather than ticker.
   Tickers are not stable over time (mergers, ticker changes, share class
   reshuffles); PERMNO is. We map PERMNO -> ticker only at the end for
   display and for joining yfinance price history if a user wants to extend
   the universe beyond CRSP.

3. Monthly returns from CRSP, not yfinance.
   CRSP returns are dividend-adjusted, delisting-return-adjusted, and
   point-in-time. yfinance is fine for current data but has known issues
   with delisted tickers and historical splits. We use yfinance only as an
   optional fallback for users without WRDS access.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

WRDS_USERNAME = "bhawini"


def get_wrds_connection(username: str = WRDS_USERNAME):
    """
    Open a WRDS connection. Imported lazily so the module can be imported
    on machines without the wrds package installed (e.g., a CI runner that
    only touches the yfinance path).
    """
    import wrds  # noqa: WPS433
    return wrds.Connection(wrds_username=username)


def get_sp500_membership(
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
    db=None,
) -> pd.DataFrame:
    """
    Pull historical S&P 500 membership from CRSP.

    CRSP's msp500list table has one row per (permno, start, end) membership
    spell. A stock can have multiple spells if it was added, dropped, then
    re-added.

    Returns
    -------
    DataFrame with columns: permno, start, ending
        start and ending are pandas Timestamps. ending is NaT for stocks
        currently in the index.
    """
    own = db is None
    if own:
        db = get_wrds_connection()
    try:
        query = f"""
            SELECT permno, start, ending
            FROM crsp.msp500list
            WHERE ending >= '{start_date}'
              AND start  <= '{end_date}'
            ORDER BY permno, start
        """
        df = db.raw_sql(query, date_cols=["start", "ending"])
    finally:
        if own:
            db.close()

    # ending is NaT for current members; fill with end_date so the membership
    # check below treats them as still active through the sample.
    df["ending"] = df["ending"].fillna(pd.Timestamp(end_date))
    return df


def members_on_date(membership: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Return the set of PERMNOs that were S&P 500 members on a given date."""
    mask = (membership["start"] <= asof) & (membership["ending"] >= asof)
    return membership.loc[mask, "permno"].unique()


def build_monthly_universe(
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
    db=None,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame of (date, permno) pairs for every
    month-end in the sample, restricted to S&P 500 members on that date.

    Returns
    -------
    DataFrame with columns: date, permno
        date is the rebalance date (month-end business day).
    """
    membership = get_sp500_membership(start_date, end_date, db=db)

    # Month-end business days. We use ME (month-end) frequency; CRSP returns
    # are reported through the last trading day of the month so this aligns.
    rebal_dates = pd.date_range(start=start_date, end=end_date, freq="ME")

    rows = []
    for d in rebal_dates:
        permnos = members_on_date(membership, d)
        rows.append(pd.DataFrame({"date": d, "permno": permnos}))

    universe = pd.concat(rows, ignore_index=True)
    universe["permno"] = universe["permno"].astype(int)
    return universe


def get_monthly_returns(
    permnos: list[int],
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
    db=None,
) -> pd.DataFrame:
    """
    Pull CRSP monthly returns for a list of PERMNOs.

    We use msf (monthly stock file) joined to msenames for the ticker
    mapping. The ret column is total return (price + dividends), already
    adjusted for splits. We also pull dlret (delisting return) from
    msedelist and chain them so a stock's last observation includes the
    delisting return rather than dropping silently.
    """
    own = db is None
    if own:
        db = get_wrds_connection()
    try:
        permno_str = ",".join(str(p) for p in permnos)
        query = f"""
            SELECT  msf.permno,
                    msf.date,
                    msf.ret,
                    msf.prc,
                    msf.shrout,
                    msd.dlret
            FROM    crsp.msf AS msf
            LEFT JOIN crsp.msedelist AS msd
                   ON msf.permno = msd.permno
                  AND date_trunc('month', msf.date) = date_trunc('month', msd.dlstdt)
            WHERE   msf.permno IN ({permno_str})
              AND   msf.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY msf.permno, msf.date
        """
        df = db.raw_sql(query, date_cols=["date"])
    finally:
        if own:
            db.close()

    # Chain delisting returns where present. CRSP convention: total return for
    # a delisting month = (1 + ret) * (1 + dlret) - 1.
    df["dlret"] = df["dlret"].fillna(0.0)
    df["ret"] = df["ret"].fillna(0.0)
    df["total_ret"] = (1.0 + df["ret"]) * (1.0 + df["dlret"]) - 1.0

    # Market cap in $ millions. CRSP prc is negative when it represents a
    # bid/ask midpoint rather than a trade; absolute value is the right number.
    df["mktcap"] = df["prc"].abs() * df["shrout"] / 1000.0  # shrout in thousands

    # Align dates to month-end so they join cleanly to the universe and
    # factor tables.
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    return df[["permno", "date", "total_ret", "mktcap"]]


def get_ticker_map(permnos: list[int], db=None) -> pd.DataFrame:
    """
    Map PERMNO -> most recent ticker for display purposes only.
    Do not use ticker as a join key in the pipeline.
    """
    own = db is None
    if own:
        db = get_wrds_connection()
    try:
        permno_str = ",".join(str(p) for p in permnos)
        query = f"""
            SELECT permno, ticker, comnam, namedt, nameendt
            FROM   crsp.msenames
            WHERE  permno IN ({permno_str})
            ORDER BY permno, namedt DESC
        """
        df = db.raw_sql(query, date_cols=["namedt", "nameendt"])
    finally:
        if own:
            db.close()
    # Most recent ticker per permno
    return df.drop_duplicates(subset=["permno"], keep="first")


# ---------------------------------------------------------------------------
# yfinance fallback. Only used if a reader of this repo does not have WRDS.
# Carries survivorship bias because yfinance has no historical membership.
# Documented as such in README and Analysis.md.
# ---------------------------------------------------------------------------

def get_current_sp500_tickers_yf() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia. Survivorship-biased."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def get_yf_monthly_returns(
    tickers: list[str],
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """yfinance fallback: monthly total returns. No delisted names available."""
    import yfinance as yf  # noqa: WPS433
    px = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )["Close"]
    rets = px.pct_change().dropna(how="all")
    long = rets.stack().reset_index()
    long.columns = ["date", "ticker", "total_ret"]
    long["date"] = long["date"] + pd.offsets.MonthEnd(0)
    return long


if __name__ == "__main__":
    # Smoke test. Requires WRDS access.
    db = get_wrds_connection()
    universe = build_monthly_universe("2020-01-01", "2020-12-31", db=db)
    print(f"Universe rows: {len(universe):,}")
    print(f"Unique permnos: {universe['permno'].nunique()}")
    print(f"Date range: {universe['date'].min()} to {universe['date'].max()}")
    sample_permnos = universe["permno"].unique()[:10].tolist()
    rets = get_monthly_returns(sample_permnos, "2020-01-01", "2020-12-31", db=db)
    print(rets.head())
    db.close()
