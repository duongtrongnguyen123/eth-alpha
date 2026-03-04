"""
run_plot_equity.py
Generate equity curve plot: Ensemble vs LightGBM vs ElasticNet vs Buy & Hold.
Saves to assets/equity_curve.png for README embedding.
"""
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from pathlib import Path

from src.features import generate_features, get_data
from src.signals import threshold_signals, build_position_holdN

BARS_PER_YEAR = 365 * 24 * 2

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml') as f:    cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb['horizon_bars']
N_FOLDS      = cfg_lgb['walk_forward']['n_folds']

df_lgb, target_lgb = generate_features(df_raw, feature_cols=cfg_lgb['features'], HORIZON_BARS=HORIZON_BARS)
df_lin, target_lin = generate_features(df_raw, feature_cols=cfg_lin['features'], HORIZON_BARS=HORIZON_BARS)

lgb_model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1)

en_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))
])

lgb_preds, en_preds = [], []
lgb_signals_list, en_signals_list = [], []
X_folds = []

for k in range(N_FOLDS):
    i = k * 0.1
    X_train_lgb, X_valid_lgb, y_train_lgb, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_train_lin, X_valid_lin, y_train_lin, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_train_lgb, y_train_lgb)
    en_model.fit(X_train_lin, y_train_lin)

    lgb_pred_train = lgb_model.predict(X_train_lgb)
    lgb_pred_valid = lgb_model.predict(X_valid_lgb)
    en_pred_train  = en_model.predict(X_train_lin)
    en_pred_valid  = en_model.predict(X_valid_lin)

    # standalone signals
    lgb_thr = np.quantile(np.abs(lgb_pred_train), 0.8)
    lgb_sig = threshold_signals(X_valid_lgb, lgb_pred_valid, threshold=lgb_thr, bias=lgb_pred_train.mean())
    en_thr  = np.quantile(np.abs(en_pred_train), 0.8)
    en_sig  = threshold_signals(X_valid_lin, en_pred_valid, threshold=en_thr, bias=en_pred_train.mean())

    lgb_signals_list.append(lgb_sig)
    en_signals_list.append(en_sig)

    # z-scored for ensemble
    lgb_z = pd.Series((lgb_pred_valid - lgb_pred_train.mean()) / (lgb_pred_train.std() + 1e-9),
                       index=X_valid_lgb.index)
    en_z  = pd.Series((en_pred_valid  - en_pred_train.mean())  / (en_pred_train.std()  + 1e-9),
                       index=X_valid_lin.index)

    shared = X_valid_lgb.index.intersection(X_valid_lin.index)
    lgb_preds.append(lgb_z.loc[shared])
    en_preds.append(en_z.loc[shared])
    X_folds.append(X_valid_lgb.loc[shared])

# full OOS series
X_all    = pd.concat(X_folds)
lgb_full = pd.concat(lgb_preds)
en_full  = pd.concat(en_preds)
r1       = df_lgb.loc[X_all.index, 'return_1'].astype(float)

def equity_from_signals(signals, r1):
    pos = build_position_holdN(signals, HORIZON_BARS)
    sr  = (pos * r1).dropna()
    return (1 + sr).cumprod()

# ensemble signals
mean_z    = (lgb_full + en_full) / 2
ens_thr   = np.quantile(np.abs(mean_z), 0.8)
ens_sig   = threshold_signals(X_all, mean_z.values, threshold=ens_thr, bias=mean_z.mean())
ens_eq    = equity_from_signals(ens_sig, r1)

lgb_sig_full = pd.concat(lgb_signals_list)
lgb_eq       = equity_from_signals(lgb_sig_full, df_lgb.loc[lgb_sig_full.index, 'return_1'].astype(float))

en_sig_full  = pd.concat(en_signals_list)
en_r1        = df_lin.loc[en_sig_full.index, 'return_1'].astype(float)
en_eq        = equity_from_signals(en_sig_full, en_r1)

bh_eq = (1 + r1).cumprod()

# ── plot ───────────────────────────────────────────────────────────────────
Path('assets').mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

ax.semilogy(ens_eq.index, ens_eq.values,   color='#00e5ff', linewidth=2.0, label='Ensemble (LGB + ElasticNet)  Sharpe 2.07')
ax.semilogy(lgb_eq.index, lgb_eq.values,   color='#4fc3f7', linewidth=1.4, label='LightGBM  Sharpe 1.86', alpha=0.85)
ax.semilogy(en_eq.index,  en_eq.values,    color='#81c784', linewidth=1.4, label='ElasticNet  Sharpe 1.94', alpha=0.85)
ax.semilogy(bh_eq.index,  bh_eq.values,    color='#888888', linewidth=1.2, label='Buy & Hold  Sharpe 0.73', linestyle='--', alpha=0.7)

ax.set_title('ETHUSDT Strategy — Out-of-Sample Equity Curve (2021–2025)',
             color='white', fontsize=14, pad=14)
ax.set_xlabel('Date', color='#aaaaaa', fontsize=11)
ax.set_ylabel('Cumulative Return (log scale)', color='#aaaaaa', fontsize=11)

ax.tick_params(colors='#aaaaaa')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')

ax.grid(True, color='#222222', linewidth=0.6)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}x'))

legend = ax.legend(loc='upper left', framealpha=0.2, fontsize=10,
                   labelcolor='white', facecolor='#111111', edgecolor='#333333')

plt.tight_layout()
plt.savefig('assets/equity_curve.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: assets/equity_curve.png")
