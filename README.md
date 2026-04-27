# Equity Factor Research

A cross-sectional factor model on the S&P 500, built with proper survivorship-bias and look-ahead-bias controls. Constructs momentum, value, and quality factor scores from CRSP and Compustat (via WRDS), forms long-short decile portfolios, and reports the diagnostics quant researchers actually use: information coefficient, ICIR, decile-spread monotonicity, turnover, and cost-adjusted Sharpe.

**Live dashboard:** [equity-factor-research.streamlit.app](https://equity-factor-research.streamlit.app)
**Sample period:** January 2000 to December 2025 (300 monthly observations across the dot-com bust, GFC, COVID, the 2022 rate shock, and the 2023-25 AI rally).

## What it does

For every month-end from 2000 to 2025, the pipeline:

1. Reconstructs the S&P 500 membership as it existed on that date, using CRSP's point-in-time index file.
2. Computes three factor scores for each member stock:
   - **Momentum** as the 12-1 month total return (skipping the most recent month to avoid short-term reversal).
   - **Value** as Book-to-Market, with book equity from Compustat lagged 6 months and market equity from CRSP.
   - **Quality** as an equal-weighted z-score of ROE and Gross Profitability/Assets (Novy-Marx 2013).
3. Cross-sectionally winsorizes (1/99%) and z-scores each factor, sorts stocks into deciles, and forms equal-weighted long-short portfolios (D10 minus D1).
4. Computes the IC, ICIR, hit rate, t-statistic, decile spread, turnover, Sharpe, max drawdown, and cost-adjusted return path.
5. Surfaces all of it in an interactive Streamlit dashboard.

## Why the design choices matter

**Survivorship bias.** Backtests using today's S&P 500 list silently drop every stock that was kicked out (Lehman, Enron, Sears, Bear Stearns) and include current members before they were added. This biases returns up by roughly 1-2% per year and is the single most common error in factor backtests written from public data. The pipeline uses CRSP's `msp500list` to get a point-in-time membership snapshot at every rebalance.

**Look-ahead bias on fundamentals.** A firm's fiscal-year-end-December balance sheet is not in the public 10-K until March of the following year. Using it to rank stocks in January overstates the IC by a meaningful amount. All Compustat fundamentals are lagged 6 months from `datadate` before they enter the ranking universe, matching the Fama-French convention.

**Delisting returns.** When a stock is delisted (bankruptcy, merger), CRSP records the final return in `msedelist.dlret` rather than the regular monthly return file. Ignoring it makes failures silently disappear from the backtest. The pipeline chains delisting returns into total returns so losses are properly accounted for.

**PERMNO as primary key.** Tickers change (Google to Alphabet, Facebook to Meta), get reused, and break time-series joins. CRSP's PERMNO is permanent and is the join key throughout. The CRSP-Compustat link uses the CCM linking table with `linktype IN ('LU','LC')` and `linkprim IN ('P','C')`, the standard restrictions.

## Repo structure

```
equity-factor-research/
├── src/
│   ├── universe.py        Historical S&P 500 membership and CRSP returns
│   ├── factors.py         Momentum, value, quality construction with PIT lag
│   ├── portfolio.py       Decile formation, monthly rebalance, turnover
│   ├── analytics.py       IC, ICIR, Sharpe, drawdown, decile diagnostics
│   ├── build_data.py      One-shot data pull, writes parquet for the dashboard
│   └── dashboard.py       Streamlit app
├── data/                  Cached parquet files written by build_data.py
├── notebooks/
│   └── factor_research.ipynb   Walkthrough of the methodology and results
├── Analysis.md            Results writeup with interpretation
├── requirements.txt
└── README.md
```

## Running it

The data build needs WRDS credentials. The dashboard does not.

```bash
# 1. Pull data (requires WRDS access, takes 3-5 minutes)
python -m src.build_data --start 2000-01-01 --end 2025-12-31

# 2. Launch the dashboard locally
streamlit run src/dashboard.py
```

For deployment, the parquet files in `data/` are committed to the repo and Streamlit Community Cloud serves the dashboard directly.

## Headline results

See `Analysis.md` for the full writeup. The short version:

| Factor    | Mean IC | ICIR (ann) | L/S Sharpe | L/S Max DD |
|-----------|---------|------------|------------|------------|
| Momentum  |  TBD    |  TBD       |  TBD       |  TBD       |
| Value     |  TBD    |  TBD       |  TBD       |  TBD       |
| Quality   |  TBD    |  TBD       |  TBD       |  TBD       |
| Composite |  TBD    |  TBD       |  TBD       |  TBD       |

Numbers are filled in after the first full data run.

## References

- Fama, E. F., and French, K. R. (1992). The Cross-Section of Expected Stock Returns. *Journal of Finance*.
- Jegadeesh, N., and Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*.
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium. *Journal of Financial Economics*.
- Asness, C., Frazzini, A., and Pedersen, L. (2019). Quality Minus Junk. *Review of Accounting Studies*.
