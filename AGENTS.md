# AGENTS.md

## Purpose

This repository is a production-oriented quantitative trading system.
Optimize for:
1. data integrity,
2. leak-free evaluation,
3. realistic trading assumptions,
4. reproducibility,
5. maintainability,
6. economic usefulness over offline model beauty.

Prefer robust and auditable solutions over clever but fragile ones.

---

## Default mindset

Act like a senior quant engineer and quant researcher building a live trading pipeline.

This repo is primarily for medium/low-frequency trading enhanced by minute-level data.
Do not optimize for ultra-high-frequency strategies unless explicitly requested and the required data exists.

When in doubt, prefer:
- simpler models,
- stronger baselines,
- clearer assumptions,
- lower turnover,
- easier auditability.

---

## Non-negotiable rules

### 1) No leakage
Never:
- use future data in features,
- normalize across train/test boundaries,
- use revised data when point-in-time data is required,
- mix train and test windows in feature generation,
- tune on final test results,
- use survivorship-biased universes unless explicitly intended.

Assume any unusually strong result may be leakage until proven otherwise.

### 2) Economic realism beats predictive metrics
Do not judge success by ML metrics alone.
Any meaningful result should be checked against:
- strict chronological splits,
- transaction costs,
- slippage or execution penalty,
- turnover,
- liquidity and capacity sanity.

If offline metrics improve but economic performance degrades, prefer the economically robust version.

### 3) Baselines first
Use this escalation path unless explicitly told otherwise:
1. deterministic rules / heuristics,
2. linear or regularized models,
3. tree-based models,
4. regime-conditioned models,
5. representation learning only if justified,
6. larger deep models only with clear incremental value.

Do not start with large deep models, RL, or generative pipelines.

### 4) Research and production stay separate
Do not mix exploratory notebook logic with production pipeline logic.
Research code must not silently become live logic.

### 5) Everything important must be auditable
For any material transformation, keep clear:
- input data source,
- timestamp convention,
- lookback window,
- target horizon,
- execution assumption,
- cost assumption.

---

## What this repo should produce

The pipeline should remain modular and staged:

1. data ingestion
2. data cleaning and alignment
3. feature generation
4. label / target construction
5. train / validation / test split
6. model training
7. evaluation
8. backtest and portfolio construction
9. live inference and execution
10. monitoring and diagnostics

Do not collapse these into one opaque script.

---

## Quant-specific rules

### Data
- Never mutate raw data in place.
- Keep raw, processed, and derived data separate.
- Minute data must be timezone-consistent and session-aware.
- Missingness is potentially informative.
- Point-in-time joins are required whenever timestamps matter.

### Features
- Prefer interpretable feature families first:
  - momentum / reversal
  - volatility / semivolatility
  - liquidity proxies
  - intraday state summaries
  - market-relative behavior
  - regime/state indicators
- Minute data should mainly improve state estimation, volatility measurement, and execution-aware context.
- Avoid uncontrolled feature explosions.

### Labels
Targets must clearly define:
- prediction time,
- decision time,
- execution time,
- holding horizon.

Prefer tradable targets when appropriate:
- cross-sectional ranking,
- bucketed labels,
- meta-labeling,
- state classification.

### Evaluation
Separate:
- predictive metrics,
- ranking metrics,
- economic metrics,
- stability metrics.

When relevant, prefer:
- IC / rank IC
- Sharpe / Sortino
- turnover
- drawdown
- exposure analysis
- regime-wise stability

Do not present a model as successful based only on loss, MSE, or AUC.

### Backtesting
Backtests must be:
- chronological,
- cost-aware,
- constraint-aware,
- reproducible.

Never present cost-free backtests as deployable.

---

## Validation defaults

Unless explicitly told otherwise:
- use chronological splits,
- never shuffle time series,
- prefer walk-forward evaluation,
- use purged / embargoed validation when horizons overlap,
- keep a final untouched test window.

Any exception must be explained.

---

## Coding expectations

- Write clear, boring, testable code.
- Prefer small modular functions.
- Avoid hidden global state.
- Keep business logic out of notebooks.
- Use precise finance-specific names.
- Add comments for assumptions and intent, not for obvious syntax.

---

## Before considering a task complete

Check:
- Is timestamp logic correct?
- Is there any leakage path?
- Are train/validation/test separated correctly?
- Are execution timing assumptions explicit?
- Are costs and turnover considered if strategy behavior changed?
- Is the result reproducible?
- Is the implementation auditable?
- Would a live operator trust this change?

If not, it is not done.

---

## Decision order

When unsure, follow this order:
1. protect data integrity
2. protect evaluation integrity
3. preserve trading realism
4. prefer simpler implementation
5. preserve existing repo conventions
6. document assumptions
7. only then increase model sophistication

If there is a trade-off between elegance and auditability, choose auditability.
If there is a trade-off between complexity and robustness, choose robustness.

---

## Local overrides

If a subdirectory contains its own `AGENTS.md`, follow the more specific file for local rules.
Keep this root file stable and concise.
Put specialized instructions closer to the relevant code.