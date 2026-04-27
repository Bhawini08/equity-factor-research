"""
Streamlit dashboard for the cross-sectional factor model.

Layout
------
Sidebar:
    - Factor selector (momentum, value, quality, composite)
    - Date range slider
    - Cost assumption (bps per side)
    - Toggle: gross vs net of cost
Main:
    Tab 1 — Performance
        cumulative return chart (long-short, S&P 500 benchmark)
        rolling 3-year Sharpe
        summary statistics table
    Tab 2 — Decile analysis
        per-decile annualized return bar chart (monotonicity check)
        decile summary table
    Tab 3 — IC diagnostics
        IC time series with 12-month rolling mean
        IC summary statistics
        IC by year heatmap
    Tab 4 — Turnover and capacity
        D10 and D1 turnover time series
        cost-adjusted return path
    Tab 5 — Methodology
        plain-language description of universe, factors, lag, etc.

Data is loaded from preprocessed parquet files in ./data/. The full WRDS
pull is run once in a separate script (build_data.py) and cached. This
keeps the dashboard fast and lets it be deployed without WRDS credentials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from analytics import (
    information_coefficient,
    ic_summary,
    performance_summary,
    decile_summary,
    cumulative_returns,
    rolling_sharpe,
)
from portfolio import (
    assign_deciles,
    decile_returns,
    turnover,
    long_short_with_costs,
)


DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "factor_panel.parquet")


@st.cache_data(show_spinner=False)
def load_returns() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "monthly_returns.parquet")


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.Series:
    """S&P 500 monthly total return as a benchmark line on the cumulative plot."""
    bench = pd.read_parquet(DATA_DIR / "sp500_returns.parquet")
    return bench.set_index("date")["sp500_ret"]


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cross-Sectional Equity Factor Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Cross-Sectional Equity Factor Research")
st.caption(
    "S&P 500 universe, point-in-time membership and fundamentals, "
    "monthly rebalance, 2000-2025."
)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")

    factor_label = st.selectbox(
        "Factor",
        options=["Momentum", "Value (B/M)", "Quality", "Composite"],
        index=0,
    )
    factor_map = {
        "Momentum": "momentum",
        "Value (B/M)": "book_to_market",
        "Quality": "quality",
        "Composite": "composite",
    }
    factor_col = factor_map[factor_label]

    cost_bps = st.slider(
        "Transaction cost (bps per side)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
    )

    show_net = st.checkbox("Show net of cost", value=False)

    st.markdown("---")
    st.markdown(
        "**Repo:** "
        "[github.com/Bhawini08/equity-factor-research]"
        "(https://github.com/Bhawini08/equity-factor-research)"
    )


# ---------------------------------------------------------------------------
# Build portfolios on the fly given the user-selected factor
# ---------------------------------------------------------------------------

panel = load_panel()
returns = load_returns()
benchmark = load_benchmark()

date_min = panel["date"].min().to_pydatetime()
date_max = panel["date"].max().to_pydatetime()
with st.sidebar:
    start, end = st.slider(
        "Sample period",
        min_value=date_min,
        max_value=date_max,
        value=(date_min, date_max),
    )

panel_w = panel[(panel["date"] >= start) & (panel["date"] <= end)]
returns_w = returns[(returns["date"] >= start) & (returns["date"] <= end)]

deciled = assign_deciles(panel_w, factor_col=factor_col, n_buckets=10)
dec_ret = decile_returns(deciled, returns_w)
ls_costed = long_short_with_costs(dec_ret, deciled, cost_bps=cost_bps)

ls_series = (
    ls_costed["long_short_net"] if show_net else ls_costed["long_short"]
)
ls_series.name = f"{factor_label} L/S"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_perf, tab_decile, tab_ic, tab_turn, tab_method = st.tabs([
    "Performance",
    "Decile Analysis",
    "IC Diagnostics",
    "Turnover",
    "Methodology",
])


with tab_perf:
    col1, col2 = st.columns([2, 1])

    with col1:
        cum_ls = cumulative_returns(ls_series)
        cum_bench = cumulative_returns(benchmark.reindex(cum_ls.index).fillna(0))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_ls.index, y=cum_ls.values,
            mode="lines", name=f"{factor_label} L/S",
        ))
        fig.add_trace(go.Scatter(
            x=cum_bench.index, y=cum_bench.values,
            mode="lines", name="S&P 500", line=dict(dash="dot"),
        ))
        fig.update_layout(
            title=f"Cumulative Return — {factor_label} Long/Short vs S&P 500",
            yaxis_title="Wealth ($1 invested)",
            xaxis_title="",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        rs = rolling_sharpe(ls_series, window=36)
        fig2 = px.line(
            rs, title="36-Month Rolling Sharpe (L/S)",
            labels={"value": "Sharpe", "index": ""},
        )
        fig2.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        stats = performance_summary(ls_series)
        st.subheader("Long/Short Performance")
        st.metric("Annualized Return", f"{stats['ann_return']:.2%}")
        st.metric("Annualized Vol", f"{stats['ann_vol']:.2%}")
        st.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}")
        st.metric("Max Drawdown", f"{stats['max_drawdown']:.2%}")
        st.metric("Hit Rate (monthly)", f"{stats['hit_rate']:.1%}")
        st.caption(f"{stats['n_months']} monthly observations")


with tab_decile:
    summary = decile_summary(dec_ret)
    summary_long = summary[summary["portfolio"].str.startswith("D")]

    fig = px.bar(
        summary_long, x="portfolio", y="ann_return",
        title=f"Annualized Return by Decile — {factor_label}",
        labels={"ann_return": "Annualized Return", "portfolio": ""},
    )
    fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        summary.style.format({
            "ann_return": "{:.2%}",
            "ann_vol": "{:.2%}",
            "sharpe": "{:.2f}",
        }),
        use_container_width=True,
    )


with tab_ic:
    ic_df = information_coefficient(panel_w, factor_col, returns_w)
    ic_df["ic_12m"] = ic_df["ic"].rolling(12).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ic_df["date"], y=ic_df["ic"],
        name="Monthly IC", marker_color="lightgray",
    ))
    fig.add_trace(go.Scatter(
        x=ic_df["date"], y=ic_df["ic_12m"],
        name="12-month rolling mean", line=dict(color="navy"),
    ))
    fig.update_layout(
        title=f"Information Coefficient — {factor_label}",
        yaxis_title="Spearman IC", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    s = ic_summary(ic_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean IC", f"{s['mean_ic']:.3f}")
    c2.metric("ICIR (annualized)", f"{s['ann_icir']:.2f}")
    c3.metric("IC Hit Rate", f"{s['hit_rate']:.1%}")
    c4.metric("t-stat", f"{s['t_stat']:.2f}")

    # IC by calendar year
    ic_df["year"] = ic_df["date"].dt.year
    by_year = ic_df.groupby("year")["ic"].mean().reset_index()
    fig_yr = px.bar(
        by_year, x="year", y="ic",
        title="Mean IC by Calendar Year",
        color="ic", color_continuous_scale="RdBu", color_continuous_midpoint=0,
    )
    fig_yr.update_layout(height=320)
    st.plotly_chart(fig_yr, use_container_width=True)


with tab_turn:
    to_d10 = turnover(deciled, decile=10)
    to_d1 = turnover(deciled, decile=1)
    to_df = pd.DataFrame({"D10 (long)": to_d10, "D1 (short)": to_d1})

    fig = px.line(
        to_df, title=f"Monthly Turnover by Leg — {factor_label}",
        labels={"value": "Turnover (fraction of names changed)", "index": ""},
    )
    fig.update_layout(yaxis_tickformat=".0%", height=380)
    st.plotly_chart(fig, use_container_width=True)

    avg_long = to_d10.mean()
    avg_short = to_d1.mean()
    cost_drag_bps = (avg_long + avg_short) * cost_bps * 12
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg long turnover", f"{avg_long:.1%}")
    c2.metric("Avg short turnover", f"{avg_short:.1%}")
    c3.metric(
        "Implied annual cost drag",
        f"{cost_drag_bps:.0f} bps",
        help=f"At {cost_bps} bps per side, applied to both legs.",
    )

    st.subheader("Cost-adjusted return path")
    cum_gross = cumulative_returns(ls_costed["long_short"])
    cum_net = cumulative_returns(ls_costed["long_short_net"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cum_gross.index, y=cum_gross.values,
                              name="Gross", line=dict(color="navy")))
    fig2.add_trace(go.Scatter(x=cum_net.index, y=cum_net.values,
                              name=f"Net ({cost_bps} bps/side)",
                              line=dict(color="firebrick")))
    fig2.update_layout(height=380, yaxis_title="Wealth ($1 invested)")
    st.plotly_chart(fig2, use_container_width=True)


with tab_method:
    st.markdown(
        """
### Universe
S&P 500 constituents at each month-end, drawn from the CRSP membership
table (`crsp.msp500list`). Stocks enter and leave the universe based on
their actual index membership history, eliminating survivorship bias.

### Factors
**Momentum.** 12-1 month return: cumulative log-return from t-12 through
t-2, skipping the most recent month to avoid short-term reversal.
Constructed from CRSP monthly total returns.

**Value.** Book-to-Market ratio. Book equity from Compustat (`ceq`),
market equity from CRSP. Following Fama and French (1992).

**Quality.** Equal-weighted z-score average of:
  - Return on Equity: `ib / ceq`
  - Gross Profitability: `gp / at` (Novy-Marx 2013)

**Composite.** Equal-weighted average of the three z-scored factor scores.

### Point-in-time discipline
Compustat fundamentals are lagged 6 months from `datadate` before they
become available for ranking. This matches the Fama-French convention
and ensures we never use information that was not yet public.

PERMNO is the primary key throughout. Compustat's GVKEY is mapped to
PERMNO through the CCM linking table (`crsp.ccmxpf_linktable`),
restricted to `linktype IN ('LU','LC')` and `linkprim IN ('P','C')`.

### Portfolio construction
Each month, stocks are sorted into deciles on the chosen factor. Decile
portfolios are equal-weighted, held for one month, then rebalanced. The
long-short portfolio is D10 minus D1.

### Diagnostics
- Information Coefficient: Spearman rank correlation between factor and
  next-period return, computed cross-sectionally each month.
- ICIR: mean(IC) divided by std(IC), annualized by sqrt(12).
- Turnover: fraction of names changing in each leg per rebalance.
- Cost adjustment: linear, applied to both legs at the user-chosen rate.
        """
    )
