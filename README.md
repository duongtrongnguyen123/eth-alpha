# ETHUSDT ML Trading Strategy

Machine learning strategy for ETHUSDT on 30-minute bars. Predicts 2-hour (4-bar) forward returns using walk-forward validation across 2019–2026 data.

Designed for **live trading with realistic transaction costs** (2–5 bps/side). All reported results are out-of-sample.

---

## Improvement Story

![Equity Curve](assets/equity_curve.png)

Starting from the theoretical no-fee ceiling, each step brings the strategy closer to live-trading reality:

| Step | Sharpe | What changed |
|---|---|---|
| No-fee ceiling (fixed hold) | 2.04 | Raw signal quality, ~2,000 trades/yr |
| → Dynamic hold · 5 bps | 1.11 | Exit when model conviction gone, ~100 trades/yr |
| → + ER > 0.6 filter · 5 bps | **1.29** | Only trade trending regimes, ~60 trades/yr |
| Buy & Hold | 0.73 | Benchmark |

---

## Validation

Three independent test layers, all out-of-sample:

### 1. Walk-forward (2021–2024)
Params selected on first 80% of data via walk-forward. 8 folds, sliding window, 20% train → 10% validate.

| | Sharpe | Ann Return | Max DD | RT/yr |
|---|---|---|---|---|
| Dynamic hold · 5 bps | 1.02 | 43.0% | -38.3% | 114 |
| **+ ER > 0.6 filter · 5 bps** | **1.13** | **35.2%** | **-21.9%** | **58** |
| Buy & Hold | 0.73 | — | — | — |

**Fold stability (ER filter, 5 bps):**

| Mean Sharpe | Std | Min (worst fold) | All positive? |
|---|---|---|---|
| 1.37 | 0.61 | 0.35 | Yes |

### 2. Holdout test (Dec 2024 – Mar 2026)
Last 20% of full dataset, never touched during param selection.

| | Sharpe | Ann Return | Max DD |
|---|---|---|---|
| **+ ER > 0.6 filter · 5 bps** | **1.64** | **56.5%** | **-31.2%** |
| Buy & Hold | -0.34 | — | — |

### 3. Live evaluation (Aug 2025 – Mar 2026)
![Live Eval](assets/live_eval.png)

New data fetched after model development. ETH fell ~60% from Aug 2025 peak during this period.

| | Sharpe | Ann Return | Max DD |
|---|---|---|---|
| **+ ER > 0.6 filter · 5 bps** | **2.03** | **+68.8%** | **-8.6%** |
| Dynamic hold (no filter) · 5 bps | 0.54 | +13.8% | -24.3% |
| Buy & Hold | -1.89 | -79.0% | -63.8% |

> ER filter sat out most of the crash — only 37 round-trips over 7 months while ETH lost 79% annualised.

---

## No-fee baseline (theoretical ceiling)

| Strategy | Sharpe | Annual Return | Max Drawdown |
|---|---|---|---|
| **Ensemble fixed hold — 80th pctile** | **2.07** | **158.6%** | **-30.2%** |
| ElasticNet standalone | 1.94 | 154.2% | -42.9% |
| LightGBM standalone | 1.86 | 141.5% | -46.4% |

> Fixed hold makes ~2,000 position changes/year. Sharpe collapses below 0 at 5 bps/side.
> Dynamic hold + ER filter reduces this to ~60 round-trips/year.

---

## Strategy Design

### 1. Prediction
LightGBM + ElasticNet ensemble predicting 4-bar (2h) forward return.
Predictions are z-score normalized per model using training statistics, then averaged.

### 2. Entry signal
- Threshold = **95th percentile of |z-score|** on training set
- Enter **long** when `mean_z > threshold`
- Enter **short** when `mean_z < -threshold + drift_bias`
- **Efficiency Ratio filter**: only enter when `ER > 0.6` (trending regime)

### 3. Dynamic hold exit
Instead of exiting after a fixed N bars:
- Hold for **at least 4 bars** (minimum)
- Stay in position while the 4-bar prediction remains in the same direction
- **Exit when prediction crosses zero** — model conviction gone
- Average hold: ~20 bars (10 hours)

### 4. Why ER filter matters
The Kaufman Efficiency Ratio measures trending (≈1) vs choppy (≈0) markets:

```
ER = |net price move over 10 bars| / sum(|each bar's move| over 10 bars)
```

ER > 0.6 fires on only 12% of bars — the strongest trending periods.
This filter sat out the entire 2021–2022 crypto bear market (choppy, mean-reverting)
while the baseline strategy bled through 385 days of drawdown.

```
Baseline MDD:  -42.5%  peak 2021-05-23 → trough 2022-06-13  (385 days)
ER>0.6 MDD:    -25.6%  peak 2025-03-02 → trough 2025-03-11  (8 days)
```

### 5. Walk-forward validation
Sliding window: 20% train → 10% validate → step 10% forward, 8 folds.

```
Fold 1:  |████████████████████|░░░░░░░░░░|
Fold 2:           |████████████████████|░░░░░░░░░░|
...
Fold 8:                                |████████████████████|░░░░░░░░░░|
         ▓ = Train (20%)   ░ = Validation (10%)
```

---

## Features

### Tree models (`strategy_lgb.yaml`) — 18 features
| Category | Features |
|---|---|
| Returns | `return_1`, `return_4`, `return_48`, `return_96` |
| Volatility | `volatility_48`, `max_volatility_480` |
| Momentum | `rsi`, `sma_cross`, `roc_20` |
| Price/SMA | `price_to_sma20`, `price_to_sma50`, `price_to_ema20` |
| Price-to-max | `close_to_max_240`, `close_to_max_2400` |
| Volume | `volume_ratio`, `volume_to_max_240`, `volume_to_max_480`, `force` |

### Linear models (`strategy_linear.yaml`) — 19 features
Same as above minus `return_96` and `close_to_max_2400`, plus lagged returns:
| Feature | Description |
|---|---|
| `return_4_lag48` | 2h return seen 24h ago |
| `return_4_lag96` | 2h return seen 48h ago |

---

## Key Findings

**Feature importance (LightGBM, avg across 8 folds):**
```
volatility_48       16.6%  ████████
close_to_max_2400   11.4%  █████
close_to_max_240    10.3%  █████
sma_cross            9.8%  ████
return_96            9.4%  ████
max_volatility_480   8.7%  ████   ← top 6 = 66% of total gain
```

**Why the ensemble works:**
LightGBM and ElasticNet have low signal correlation (0.19) — genuinely independent models.
Averaging their z-scored predictions smooths out each model's bad folds.

**Prediction quality is weak but consistent:**
- LightGBM directional accuracy on 4-bar target: ~50.1% (barely above random)
- Alpha comes from consistent application over thousands of bars, not individual accuracy
- The 4-bar model has 2.5× better signal/noise than a 1-bar model — don't use short-horizon for exits

---

## Project Structure

```
eth-alpha/
├── data/
│   └── ETHUSDT.csv                    # Raw 30-min OHLCV (2019–2026, ~110k bars)
├── src/
│   ├── features.py                    # generate_features(), get_data()
│   ├── signals.py                     # build_position_dynamic(), build_position_filtered()
│   ├── backtest.py                    # evaluate_holdN(cost_bps=)
│   ├── models.py                      # All model definitions
│   └── walk_forward.py                # run_walk_forward(), eva_full_result()
├── configs/
│   ├── strategy_lgb.yaml              # Tree model features (18)
│   └── strategy_linear.yaml           # Linear model features (19, +lags)
├── assets/
│   ├── equity_curve.png               # Improvement story plot
│   └── live_eval.png                  # Live evaluation plot
│
├── run_er_filter.py                   # ★ Final strategy: ER filter + dynamic hold
├── run_live_eval.py                   # ★ Live evaluation Aug 2025 → Mar 2026
├── run_clean_split.py                 # ★ Clean 80/20 split validation
├── run_holdout_test.py                # Holdout test: confirm no overfit
├── fetch_new_data.py                  # Fetch latest OHLCV from Binance
├── run_plot_equity.py                 # Generate improvement story chart
├── run_dynamic_threshold_sweep.py     # Threshold sweep for dynamic hold
├── run_costs.py                       # Cost sweep across all models
├── run_best.py                        # Best standalone models
├── run_feature_importance.py          # LightGBM feature importance
├── run_signal_corr.py                 # Signal correlation between models
├── run_stability.py                   # Per-fold Sharpe stability
└── run_pred_quality.py                # Prediction quality diagnostic
```

---

## Quick Start

```bash
pip install pandas numpy scikit-learn lightgbm xgboost pyyaml scipy matplotlib requests

# Fetch latest data (extends dataset to current date)
python fetch_new_data.py

# ★ Final strategy with transaction costs
python run_er_filter.py

# ★ Live evaluation on Aug 2025 → Mar 2026
python run_live_eval.py

# Clean 80/20 split (params from first 80%, test on last 20%)
python run_clean_split.py

# Holdout test (confirm no overfit to dev period)
python run_holdout_test.py

# Cost sweep to see fee sensitivity
python run_costs.py
```

---

## Notes

- **No slippage modeled** — live performance will be lower for large sizes
- **No position sizing** — raw signal evaluation; vol-targeting would reduce MDD further
- ER filter based on 10-bar lookback — longer lookback (e.g. 20) could be more stable
- 2019–2020 data is training-only; all reported metrics are 2021+ OOS
- The live eval period (Aug 2025–Mar 2026) was not used in any parameter selection
