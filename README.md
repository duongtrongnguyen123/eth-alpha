# ETHUSDT ML Trading Strategy

Machine learning strategy for ETHUSDT on 30-minute bars. Predicts 2-hour (4-bar) forward returns using walk-forward validation across 2019–2025 data.

---

## Results

![Equity Curve](assets/equity_curve.png)

Best strategy: **LightGBM + ElasticNet ensemble** (z-scored mean of predictions)

| Strategy | Sharpe | Annual Return | Max Drawdown |
|----------|--------|--------------|-------------|
| **Ensemble (LGB + ElasticNet)** | **2.07** | **158.6%** | **-30.2%** |
| ElasticNet (standalone) | 1.94 | 154.2% | -42.9% |
| Ridge (standalone) | 1.90 | 146.2% | -40.7% |
| LightGBM (standalone) | 1.86 | 141.5% | -46.4% |
| Buy & Hold | 0.73 | 29.9% | — |

> All metrics are out-of-sample across 8 walk-forward folds (2021–2025).

---

## Project Structure

```
qproj/
├── data/
│   └── ETHUSDT.csv              # Raw 30-min OHLCV data (2019–2025, ~100k bars)
│
├── src/
│   ├── features.py              # generate_features(), get_data()
│   ├── signals.py               # threshold_signals(), build_position_holdN()
│   ├── backtest.py              # evaluate_holdN(), decile_analysis()
│   ├── models.py                # get_models() — all 6 model definitions
│   └── walk_forward.py          # run_walk_forward(), eva_full_result()
│
├── configs/
│   ├── strategy_lgb.yaml        # LightGBM/XGBoost feature set (18 features)
│   └── strategy_linear.yaml     # Ridge/ElasticNet/Lasso feature set (20 features, +lags)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Raw data exploration
│   ├── 02_features.ipynb        # Feature distributions & correlations
│   ├── 03_research.ipynb        # Walk-forward training
│   └── 04_results.ipynb         # Equity curves & evaluation
│
├── run_best.py                  # Main runner: best config per model family
├── run_ensemble_lgb_elastic.py  # Best ensemble: LightGBM + ElasticNet
├── run_stability.py             # Per-fold Sharpe/return std analysis
├── run_signal_corr.py           # Signal correlation between models
└── run_feature_importance.py    # LightGBM feature importance across folds
```

---

## Strategy Design

### Prediction target
4-bar forward return (2 hours ahead) on 30-min bars.

### Signal generation
1. Train model on a 20% rolling window
2. Compute threshold = 80th percentile of `|pred|` on training set
3. Go **long** if `pred > threshold`, **short** if `pred < -threshold + drift_bias`
4. Hold position for 4 bars, no overlapping trades

### Walk-forward validation
Sliding window: 20% train → 10% validate → step 10% forward, 8 folds total.

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
|----------|----------|
| Returns | `return_1`, `return_4`, `return_48`, `return_96` |
| Volatility | `volatility_48`, `max_volatility_480` |
| Momentum | `rsi`, `sma_cross`, `roc_20` |
| Price/SMA | `price_to_sma20`, `price_to_sma50`, `price_to_ema20` |
| Price-to-max | `close_to_max_240`, `close_to_max_2400` |
| Volume | `volume_ratio`, `volume_to_max_240`, `volume_to_max_480`, `force` |

### Linear models (`strategy_linear.yaml`) — 20 features
Same as above, plus:
| Feature | Description |
|---------|-------------|
| `return_4_lag48` | 2h return seen 24h ago |
| `return_4_lag96` | 2h return seen 48h ago |

> Lagged returns are computed on-demand only when listed in `feature_cols` to avoid polluting the tree model's dataset with NaN rows.

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

**Signal correlation between models:**
- Trees vs linear: ~0.15–0.19 (genuinely independent)
- Linear vs linear: 0.75–0.80 (nearly identical — pick one)
- LightGBM vs XGBoost: 0.48

**Why the ensemble works:**
LightGBM and ElasticNet have low signal correlation (0.19) but both have Sharpe > 1.86. Averaging their z-score-normalized predictions smooths out each model's bad folds. The ensemble MDD drops from -46% (LGB alone) to -30%.

**Stability (Sharpe std across 8 folds):**
| Model | Mean Sharpe | Std | Min (worst fold) |
|-------|------------|-----|-----------------|
| ElasticNet | 2.01 | 0.85 | 0.49 |
| Ridge | 1.70 | **0.78** | **0.64** — never lost money |
| LightGBM | 2.05 | 1.38 | 0.23 — high variance |

---

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost plotly pyyaml

# Run best standalone models (LightGBM + linear)
python run_best.py

# Run best ensemble
python run_ensemble_lgb_elastic.py

# Feature importance analysis
python run_feature_importance.py

# Stability analysis (per-fold Sharpe std)
python run_stability.py

# Signal correlation between models
python run_signal_corr.py
```

---

## Notes

- **No transaction costs or slippage** modeled — live performance will be lower
- **No position sizing or stop-loss** — raw signal evaluation only
- The -30% MDD of the ensemble is still significant for live trading; vol-targeting position sizing would reduce it further
