"""
End-to-end data build script.

Run once locally with WRDS access. Produces:
    data/factor_panel.parquet     (date, permno, momentum, value, quality, composite, z-scores)
    data/monthly_returns.parquet  (date, permno, total_ret, mktcap)
    data/sp500_returns.parquet    (date, sp500_ret)  - benchmark series
    data/decile_returns_<factor>.parquet  - cached portfolio returns per factor

After this runs, dashboard.py can be deployed to Streamlit Cloud with
just the parquet files, no WRDS connection needed.

Usage
-----
    cd equity-factor-research
    python -m src.build_data --start 2000-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .universe import (
    get_wrds_connection,
    build_monthly_universe,
    get_monthly_returns,
)
from .factors import (
    get_compustat_annual,
    get_ccm_link,
    link_fundamentals_to_permno,
    make_pit_fundamentals,
    build_factor_panel,
)


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_sp500_index_returns(start: str, end: str, db) -> pd.DataFrame:
    """
    S&P 500 total return index from CRSP (sprtrn from msi or msp500).
    msp500 has the canonical S&P 500 monthly total return.
    """
    query = f"""
        SELECT caldt AS date, sprtrn AS sp500_ret
        FROM   crsp.msp500
        WHERE  caldt BETWEEN '{start}' AND '{end}'
        ORDER BY caldt
    """
    df = db.raw_sql(query, date_cols=["date"])
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    return df


def main(start: str, end: str) -> None:
    print(f"[build_data] Connecting to WRDS as bhawini...")
    db = get_wrds_connection()

    try:
        # ----- 1. Universe -------------------------------------------------
        print(f"[build_data] Building monthly S&P 500 universe {start} -> {end}...")
        universe = build_monthly_universe(start, end, db=db)
        print(f"[build_data]   {len(universe):,} (date, permno) rows, "
              f"{universe['permno'].nunique():,} unique permnos")

        all_permnos = universe["permno"].unique().tolist()

        # ----- 2. Monthly returns -----------------------------------------
        print(f"[build_data] Pulling CRSP monthly returns for {len(all_permnos):,} permnos...")
        # Pull 1 year before start so momentum has a 12-month lookback ready
        ret_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        rets = get_monthly_returns(all_permnos, ret_start, end, db=db)
        rets = rets.rename(columns={"total_ret": "total_ret"})
        # Restrict returns to permnos when they were actually in the universe.
        # Forward returns will be matched to formation dates downstream.
        rets.to_parquet(DATA_DIR / "monthly_returns.parquet", index=False)
        print(f"[build_data]   wrote monthly_returns.parquet ({len(rets):,} rows)")

        # ----- 3. Compustat fundamentals ----------------------------------
        print("[build_data] Pulling Compustat annual fundamentals...")
        funda_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        funda = get_compustat_annual(funda_start, end, db=db)
        print(f"[build_data]   {len(funda):,} funda rows")

        print("[build_data] Pulling CCM linking table...")
        ccm = get_ccm_link(db=db)
        funda_linked = link_fundamentals_to_permno(funda, ccm)
        # Restrict to permnos that ever appear in our S&P 500 universe
        funda_linked = funda_linked[funda_linked["permno"].isin(all_permnos)]
        print(f"[build_data]   {len(funda_linked):,} linked funda rows "
              f"covering {funda_linked['permno'].nunique():,} permnos")

        # ----- 4. Point-in-time fundamentals ------------------------------
        print("[build_data] Building point-in-time fundamentals panel (6m lag)...")
        rebal_dates = pd.DatetimeIndex(sorted(universe["date"].unique()))
        pit = make_pit_fundamentals(funda_linked, rebal_dates, lag_months=6)
        # Restrict PIT panel to (date, permno) pairs that were in the index
        # on that date. This is the survivorship-bias-corrected universe.
        pit = pit.merge(universe, on=["permno", "date"], how="inner")
        print(f"[build_data]   {len(pit):,} pit-funda rows after universe filter")

        # Market cap panel for the value calc
        mktcap_panel = rets[["permno", "date", "mktcap"]]

        # ----- 5. Factor panel --------------------------------------------
        print("[build_data] Computing factor panel...")
        panel = build_factor_panel(rets, pit, mktcap_panel)
        # Restrict factor panel to in-index (date, permno)
        panel = panel.merge(universe, on=["permno", "date"], how="inner")
        panel.to_parquet(DATA_DIR / "factor_panel.parquet", index=False)
        print(f"[build_data]   wrote factor_panel.parquet ({len(panel):,} rows)")

        # ----- 6. Benchmark -----------------------------------------------
        print("[build_data] Pulling S&P 500 benchmark returns...")
        bench = get_sp500_index_returns(start, end, db)
        bench.to_parquet(DATA_DIR / "sp500_returns.parquet", index=False)
        print(f"[build_data]   wrote sp500_returns.parquet ({len(bench):,} rows)")

    finally:
        db.close()

    print("[build_data] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    main(args.start, args.end)
