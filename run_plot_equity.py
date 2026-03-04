"""
run_plot_equity.py
Strategy improvement story plotted as equity curves:
  No-fee ceiling → Dynamic hold (5bps) → ER > 0.6 filter (5bps) vs Buy & Hold
Saves to assets/equity_curve.png for README embedding.
"""
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from pathlib import Path

from src.features import generate_features, get_data
from src.signals import (threshold_signals, build_position_holdN,
                         build_position_dynamic, build_position_filtered)

BARS_PER_YEAR = 365 * 24 * 2
COST_BPS      = 5
ER_N          = 10
BEST_ER       = 0.6

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb['horizon_bars']
N_FOLDS      = cfg_lgb['walk_forward']['n_folds']

# Efficiency Ratio on full price series
direction = df_raw['close'].diff(ER_N).abs()
noise     = df_raw['close'].diff().abs().rolling(ER_N).sum().replace(0, np.nan)
er_full   = (direction / noise).clip(0, 1)

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

lgb_z_list, en_z_list, X_folds = [], [], []
thr_list, bias_list             = [], []
nofee_signals_list              = []

print("Training 8 folds...")
for k in range(N_FOLDS):
    i = k * 0.1
    X_tl, X_vl, y_tl, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_tn, X_vn, y_tn, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_tl, y_tl)
    en_model.fit(X_tn, y_tn)

    lgb_ptr = lgb_model.predict(X_tl);  lgb_pv = lgb_model.predict(X_vl)
    en_ptr  = en_model.predict(X_tn);   en_pv  = en_model.predict(X_vn)

    # z-score per model using training stats
    lgb_z_tr = (lgb_ptr - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9)
    en_z_tr  = (en_ptr  - en_ptr.mean())  / (en_ptr.std()  + 1e-9)

    thr_list.append((np.quantile(np.abs(lgb_z_tr), 0.95) +
                     np.quantile(np.abs(en_z_tr),  0.95)) / 2)
    bias_list.append((lgb_z_tr.mean() + en_z_tr.mean()) / 2)

    shared = X_vl.index.intersection(X_vn.index)
    lgb_z_v = pd.Series((lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_vl.index).loc[shared]
    en_z_v  = pd.Series((en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_vn.index).loc[shared]
    lgb_z_list.append(lgb_z_v)
    en_z_list.append(en_z_v)
    X_folds.append(X_vl.loc[shared])

    # no-fee fixed-hold signals (80th pctile threshold on training z)
    mean_z_v   = (lgb_z_v + en_z_v) / 2
    nofee_thr  = (np.quantile(np.abs(lgb_z_tr), 0.8) + np.quantile(np.abs(en_z_tr), 0.8)) / 2
    nofee_bias = (lgb_z_tr.mean() + en_z_tr.mean()) / 2
    nofee_sig  = threshold_signals(X_vl.loc[shared], mean_z_v.values,
                                   threshold=nofee_thr, bias=nofee_bias)
    nofee_signals_list.append(nofee_sig)

X_all  = pd.concat(X_folds)
mean_z = (pd.concat(lgb_z_list) + pd.concat(en_z_list)) / 2
r1     = df_lgb.loc[X_all.index, 'return_1'].astype(float)
thr    = float(np.mean(thr_list))
bias   = float(np.mean(bias_list))
er     = er_full.loc[X_all.index]


# ── equity helpers ─────────────────────────────────────────────────────────────
def equity_nofee(signals):
    pos = build_position_holdN(signals, HORIZON_BARS)
    ret = (pos * r1).dropna()
    return (1 + ret).cumprod()

def equity_fee(pos, cost_bps=COST_BPS):
    r1p     = r1.loc[pos.index]
    changes = pos.diff().fillna(pos.abs()).abs()
    ret     = (pos * r1p - changes * (cost_bps / 10_000)).dropna()
    return (1 + ret).cumprod()

def sharpe_from_pos(pos, cost_bps=COST_BPS):
    r1p     = r1.loc[pos.index]
    changes = pos.diff().fillna(pos.abs()).abs()
    ret     = (pos * r1p - changes * (cost_bps / 10_000)).dropna()
    std     = ret.std()
    return ret.mean() / std * np.sqrt(BARS_PER_YEAR) if std > 0 else np.nan


# ── build all curves ───────────────────────────────────────────────────────────
# 1. No-fee fixed hold — theoretical ceiling
nofee_sig_full = pd.concat(nofee_signals_list)
pos_nofee      = build_position_holdN(nofee_sig_full, HORIZON_BARS)
nofee_eq       = equity_nofee(nofee_sig_full)
sh_nofee       = sharpe_from_pos(pos_nofee, cost_bps=0)

# 2. Dynamic hold, no filter, 5 bps
pos_dyn  = build_position_dynamic(mean_z, entry_thr=thr, min_hold=HORIZON_BARS, bias=bias)
dyn_eq   = equity_fee(pos_dyn)
sh_dyn   = sharpe_from_pos(pos_dyn)

# 3. Dynamic hold + ER > 0.6, 5 bps
pos_er   = build_position_filtered(mean_z, entry_thr=thr, min_hold=HORIZON_BARS,
                                   filter_series=er, filter_thr=BEST_ER, bias=bias)
er_eq    = equity_fee(pos_er)
sh_er    = sharpe_from_pos(pos_er)

# 4. Buy & Hold
bh_eq    = (1 + r1).cumprod()
sh_bh    = r1.mean() / r1.std() * np.sqrt(BARS_PER_YEAR)

print(f"\nSharpe summary:")
print(f"  No-fee ceiling:      {sh_nofee:.2f}")
print(f"  Dynamic hold 5bps:   {sh_dyn:.2f}")
print(f"  + ER>0.6 filter 5bps:{sh_er:.2f}")
print(f"  Buy & Hold:          {sh_bh:.2f}")


# ── plot ───────────────────────────────────────────────────────────────────────
Path('assets').mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

ax.semilogy(nofee_eq.index, nofee_eq.values,
            color='#607d8b', linewidth=1.2, linestyle=':',  alpha=0.75,
            label=f'No-fee ceiling (fixed hold)  Sharpe {sh_nofee:.2f}')
ax.semilogy(dyn_eq.index,   dyn_eq.values,
            color='#4fc3f7', linewidth=1.5,
            label=f'→ Dynamic hold · {COST_BPS} bps  Sharpe {sh_dyn:.2f}')
ax.semilogy(er_eq.index,    er_eq.values,
            color='#00e5ff', linewidth=2.0,
            label=f'→ + ER > {BEST_ER} filter · {COST_BPS} bps  Sharpe {sh_er:.2f}')
ax.semilogy(bh_eq.index,    bh_eq.values,
            color='#888888', linewidth=1.2, linestyle='--', alpha=0.7,
            label=f'Buy & Hold  Sharpe {sh_bh:.2f}')

ax.set_title('ETHUSDT Strategy — Improvement Story (2021–2025, Out-of-Sample)',
             color='white', fontsize=14, pad=14)
ax.set_xlabel('Date',                            color='#aaaaaa', fontsize=11)
ax.set_ylabel('Cumulative Return (log scale)',   color='#aaaaaa', fontsize=11)

ax.tick_params(colors='#aaaaaa')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')

ax.grid(True, color='#222222', linewidth=0.6)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}x'))

ax.legend(loc='upper left', framealpha=0.2, fontsize=10,
          labelcolor='white', facecolor='#111111', edgecolor='#333333')

plt.tight_layout()
plt.savefig('assets/equity_curve.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("\nSaved: assets/equity_curve.png")
