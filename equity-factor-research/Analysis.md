# Analysis: Cross-Sectional Equity Factor Model

This document interprets the results from the factor pipeline. Numerical values are populated after running `build_data.py`; the structure and the questions the analysis is organized around are stable.

## Sample and methodology recap

- Universe: S&P 500 members at each month-end, point-in-time from CRSP `msp500list`. 2000-01 to 2025-12, 312 rebalance dates.
- Returns: CRSP monthly total returns with delisting returns chained in.
- Fundamentals: Compustat annual, lagged 6 months from fiscal year-end before becoming available for ranking. Linked to CRSP via the CCM linktable.
- Factors: 12-1 momentum, Book-to-Market value, equal-weighted z-score of ROE and GP/A for quality, equal-weighted composite of all three.
- Portfolios: cross-sectional decile sorts, equal-weighted, monthly rebalance. Long-short = D10 minus D1.

## What we're looking for

A factor is real if (1) the IC is positive on average and stable enough that the t-stat clears 2, (2) the decile spread is roughly monotonic from D1 to D10 rather than driven by one tail, and (3) the long-short Sharpe survives a reasonable transaction cost assumption. We evaluate each factor against all three.

## Single-factor results

### Momentum (12-1)

*To be populated.* Expected based on prior literature: positive mean IC around 0.03-0.05, decile monotonicity that breaks during sharp reversals (March 2009, April 2020), large drawdowns during momentum crashes. The 2009 momentum crash and the value-rotation episode of late 2020 should both be visible in the rolling Sharpe.

Discussion points:
- Is the long leg or the short leg doing more of the work? In US large-cap, the short leg of momentum has historically been more profitable than the long leg, which is relevant for capacity (shorting is harder).
- Turnover is mechanically high for momentum (~80-100% per month per leg). The cost-adjusted Sharpe is the honest number.

### Value (Book-to-Market)

*To be populated.* Expected: weaker IC than the academic literature suggests because the value premium has compressed sharply since 2010. The 2017-2020 period should show a sustained drawdown; 2021-22 should show a partial recovery.

Discussion points:
- B/M is the simplest value definition. Composite value (B/M plus E/P plus CF/P plus sales/P) typically has a higher IC; we kept B/M for clarity and to match Fama-French (1992).
- Sector tilts: B/M structurally tilts long financials and energy and short tech. Sector-neutralization would change the result; we report the unconstrained version.

### Quality (ROE + GP/A composite)

*To be populated.* Expected: lower mean return than momentum but higher Sharpe (lower vol), and shallower drawdowns. Quality typically holds up best in risk-off regimes; the GFC and 2022 should be visible as outperformance windows.

Discussion points:
- ROE alone has a known issue: it loads on leverage. GP/A from Novy-Marx is cleaner because the denominator is total assets. Combining them via z-score average smooths out the bias.
- Quality's correlation to value is typically negative: high-quality firms tend to be expensive. The composite factor benefits from this diversification.

## Composite factor

*To be populated.* The equal-weighted composite of momentum, value, and quality z-scores should produce a higher Sharpe than any single factor, due to low pairwise correlations. We report:
- Composite IC and ICIR vs the three single-factor ICIRs.
- Pairwise correlation matrix of the three long-short return series.
- Diversification ratio: composite vol divided by weighted-average single-factor vol.

## Robustness

Three checks worth running and reporting:

1. **Lag sensitivity.** Re-run the value and quality factors with 3-month, 6-month, and 12-month lags. The 6-month result should be the most defensible; the 3-month result will look better but is partially look-ahead.
2. **Sub-period stability.** Split the sample into 2000-2010, 2010-2020, 2020-2025. A factor that only works in one regime is much weaker evidence than one that works across all three.
3. **Transaction cost sensitivity.** Show the Sharpe at 0, 5, 10, 20, and 50 bps per side. Momentum will degrade fastest; quality will be most robust.

## Limitations

- Equal-weighted within decile. Value-weighted is more realistic for institutional implementation but understates the small-cap effect inside the S&P 500. Both are defensible.
- No sector neutralization. Real factor portfolios at quant funds typically neutralize to GICS sectors. This pipeline reports the unconstrained version; a sector-neutral extension is straightforward (subtract sector-mean factor score before ranking).
- Long-short, no leverage cap. In practice, the short leg has hard-to-borrow constraints, especially for small-cap names in D1. Our backtest assumes free shorting.
- US large-cap only. The factors are well-documented to work in international and small-cap universes too; a multi-region extension is the natural next step.

## What this project demonstrates

The point of this repo is not to discover a new factor. The point is to show that I can build a research pipeline the way it would be built at a quant fund: PIT data, proper linking, defensible benchmarks, and an honest reporting of IC, ICIR, and turnover-adjusted Sharpe. Anyone reviewing this should be able to swap in a new factor in one file (`factors.py`) and have the rest of the pipeline run unchanged.
