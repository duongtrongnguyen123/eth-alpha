"""
run_costs.py
Transaction cost sweep: 0, 2, 5, 10, 20 bps per side.
Tests Ensemble (LGB + ElasticNet), LightGBM, ElasticNet, and Ridge.
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features import generate_features, get_data
from src.signals import threshold_signals, build_position_holdN
from src.backtest import BARS_PER_YEAR

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
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
    ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))])

ridge_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))])

# collect per-fold signals and z-scores
lgb_z_list, en_z_list  = [], []
lgb_sig_list, en_sig_list, ridge_sig_list = [], [], []
X_folds = []

for k in range(N_FOLDS):
    i = k * 0.1
    X_train_lgb, X_valid_lgb, y_train_lgb, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_train_lin, X_valid_lin, y_train_lin, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_train_lgb, y_train_lgb)
    en_model.fit(X_train_lin, y_train_lin)
    ridge_model.fit(X_train_lin, y_train_lin)

    lgb_ptr = lgb_model.predict(X_train_lgb)
    lgb_pv  = lgb_model.predict(X_valid_lgb)
    en_ptr  = en_model.predict(X_train_lin)
    en_pv   = en_model.predict(X_valid_lin)
    rg_ptr  = ridge_model.predict(X_train_lin)
    rg_pv   = ridge_model.predict(X_valid_lin)

    # standalone signals
    lgb_sig_list.append(threshold_signals(X_valid_lgb, lgb_pv,
        threshold=np.quantile(np.abs(lgb_ptr), 0.8), bias=lgb_ptr.mean()))
    en_sig_list.append(threshold_signals(X_valid_lin, en_pv,
        threshold=np.quantile(np.abs(en_ptr), 0.8), bias=en_ptr.mean()))
    ridge_sig_list.append(threshold_signals(X_valid_lin, rg_pv,
        threshold=np.quantile(np.abs(rg_ptr), 0.8), bias=rg_ptr.mean()))

    # z-scores for ensemble
    shared = X_valid_lgb.index.intersection(X_valid_lin.index)
    lgb_z_list.append(pd.Series(
        (lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_valid_lgb.index).loc[shared])
    en_z_list.append(pd.Series(
        (en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_valid_lin.index).loc[shared])
    X_folds.append(X_valid_lgb.loc[shared])

X_ens     = pd.concat(X_folds)
lgb_z     = pd.concat(lgb_z_list)
en_z      = pd.concat(en_z_list)
mean_z    = (lgb_z + en_z) / 2
ens_thr   = np.quantile(np.abs(mean_z), 0.8)
ens_sig   = threshold_signals(X_ens, mean_z.values, threshold=ens_thr, bias=mean_z.mean())

lgb_sig_full   = pd.concat(lgb_sig_list)
en_sig_full    = pd.concat(en_sig_list)
ridge_sig_full = pd.concat(ridge_sig_list)


def metrics(signals, r1_series, cost_bps):
    position = build_position_holdN(signals, HORIZON_BARS)
    r1 = r1_series.loc[position.index]
    cost_per_side = cost_bps / 10_000
    pos_changes   = position.diff().fillna(position.abs()).abs()
    ret = (position * r1 - pos_changes * cost_per_side).dropna()
    if len(ret) == 0:
        return np.nan, np.nan, np.nan
    eq  = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    sharpe  = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR) if ret.std() > 0 else np.nan
    cagr    = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    mdd     = ((eq / eq.cummax()) - 1).min()
    return sharpe, cagr, mdd


r1_lgb   = df_lgb.loc[lgb_sig_full.index,   'return_1'].astype(float)
r1_en    = df_lin.loc[en_sig_full.index,     'return_1'].astype(float)
r1_ridge = df_lin.loc[ridge_sig_full.index,  'return_1'].astype(float)
r1_ens   = df_lgb.loc[ens_sig.index,         'return_1'].astype(float)

COST_LEVELS = [0, 2, 5, 10, 20]

print("=" * 70)
print("TRANSACTION COST SWEEP  (bps per side)")
print("=" * 70)

for label, sig, r1 in [
    ("Ensemble (LGB+EN)", ens_sig,        r1_ens),
    ("LightGBM",          lgb_sig_full,   r1_lgb),
    ("ElasticNet",        en_sig_full,    r1_en),
    ("Ridge",             ridge_sig_full, r1_ridge),
]:
    print(f"\n{label}")
    print(f"  {'Cost (bps)':>12}  {'Sharpe':>8}  {'Ann Return':>11}  {'Max DD':>9}")
    print("  " + "-" * 46)
    for c in COST_LEVELS:
        sh, cagr, mdd = metrics(sig, r1, c)
        print(f"  {c:>12}  {sh:>8.2f}  {cagr:>10.1%}  {mdd:>8.1%}")

# trades per year for ensemble (to show cost impact)
pos_ens = build_position_holdN(ens_sig, HORIZON_BARS)
n_trades = pos_ens.diff().abs().gt(0).sum()
n_years  = len(pos_ens) / BARS_PER_YEAR
print(f"\nEnsemble: ~{n_trades/n_years:.0f} position changes/year  "
      f"({n_trades} total over {n_years:.1f} years)")
print("Round-trip cost = 2 × bps per side")
