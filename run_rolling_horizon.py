"""
run_rolling_horizon.py
Compare three exit strategies for the LGB+ElasticNet ensemble:
  1. Fixed hold       — always exit after N bars
  2. Dynamic hold     — exit when 4-bar pred crosses zero (after min N bars)
  3. Rolling horizon  — extend deadline by N each bar pred stays aligned; exit when deadline expires

Threshold: 85th percentile. Costs: 0, 2, 5, 10 bps/side.
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features import generate_features, get_data
from src.signals import (threshold_signals, build_position_holdN,
                          build_position_dynamic, build_position_rolling_horizon)
from src.backtest import BARS_PER_YEAR

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS  = cfg_lgb['horizon_bars']   # 4 bars
N_FOLDS       = cfg_lgb['walk_forward']['n_folds']
THRESHOLD_PCT = 85
COSTS         = [0, 2, 5, 10]

df_lgb, target_lgb = generate_features(df_raw, feature_cols=cfg_lgb['features'], HORIZON_BARS=HORIZON_BARS)
df_lin, target_lin = generate_features(df_raw, feature_cols=cfg_lin['features'], HORIZON_BARS=HORIZON_BARS)

lgb_model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1)
en_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))])

lgb_z_list, en_z_list, X_folds = [], [], []

print("Training 8 folds...")
for k in range(N_FOLDS):
    i = k * 0.1
    X_tl, X_vl, y_tl, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_tn, X_vn, y_tn, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_tl, y_tl);  en_model.fit(X_tn, y_tn)

    lgb_ptr = lgb_model.predict(X_tl);  lgb_pv = lgb_model.predict(X_vl)
    en_ptr  = en_model.predict(X_tn);   en_pv  = en_model.predict(X_vn)

    shared = X_vl.index.intersection(X_vn.index)
    lgb_z_list.append(pd.Series(
        (lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_vl.index).loc[shared])
    en_z_list.append(pd.Series(
        (en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_vn.index).loc[shared])
    X_folds.append(X_vl.loc[shared])

X_all  = pd.concat(X_folds)
mean_z = (pd.concat(lgb_z_list) + pd.concat(en_z_list)) / 2
r1     = df_lgb.loc[X_all.index, 'return_1'].astype(float)
bias   = mean_z.mean()
thr    = np.quantile(np.abs(mean_z), THRESHOLD_PCT / 100)

# pre-thresholded signals for fixed hold
signals_fixed = pd.Series(0, index=X_all.index)
signals_fixed[mean_z.values >  thr]        =  1
signals_fixed[mean_z.values < -thr + bias] = -1

pos_fixed   = build_position_holdN(signals_fixed, HORIZON_BARS)
pos_dynamic = build_position_dynamic(mean_z, entry_thr=thr, min_hold=HORIZON_BARS, bias=bias)
pos_rolling = build_position_rolling_horizon(mean_z, threshold=thr, horizon=HORIZON_BARS, bias=bias)


def metrics(pos, cost_bps):
    r1p     = r1.loc[pos.index]
    changes = pos.diff().fillna(pos.abs()).abs()
    ret     = (pos * r1p - changes * (cost_bps / 10_000)).dropna()
    if len(ret) == 0 or ret.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, rt_yr=0, avg_hold=0)
    eq      = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    sharpe  = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)
    cagr    = eq.iloc[-1] ** (1 / n_years) - 1
    mdd     = ((eq / eq.cummax()) - 1).min()
    rt_yr   = changes.gt(0).sum() / 2 / n_years
    in_pos  = pos.ne(0).astype(int)
    starts  = (in_pos.diff().fillna(in_pos) > 0).sum()
    avg_hold = in_pos.sum() / starts if starts > 0 else 0
    return dict(sharpe=sharpe, cagr=cagr, mdd=mdd, rt_yr=rt_yr, avg_hold=avg_hold)


# ── results table ─────────────────────────────────────────────────────────────
print(f"\nThreshold: {THRESHOLD_PCT}th percentile  |  Horizon N = {HORIZON_BARS} bars\n")
print("=" * 82)
print("FIXED vs DYNAMIC vs ROLLING HORIZON — Ensemble (LGB + ElasticNet)")
print("=" * 82)
print(f"  {'Strategy':<22}  {'Cost':>6}  {'Sharpe':>8}  {'Ann Ret':>10}  {'Max DD':>9}  {'RT/yr':>7}  {'AvgHold':>9}")
print("  " + "-" * 76)

for label, pos in [
    ("Fixed (4 bars)",    pos_fixed),
    ("Dynamic (0-cross)", pos_dynamic),
    ("Rolling horizon",   pos_rolling),
]:
    for c in COSTS:
        m = metrics(pos, c)
        print(f"  {label:<22}  {c:>4}bps  {m['sharpe']:>8.2f}  {m['cagr']:>9.1%}  "
              f"{m['mdd']:>8.1%}  {m['rt_yr']:>7.0f}  {m['avg_hold']:>7.1f}b")
    print()

# ── trade duration distribution ───────────────────────────────────────────────
print("=" * 82)
print("TRADE DURATION DISTRIBUTION")
print("=" * 82)
for label, pos in [("Dynamic (0-cross)", pos_dynamic), ("Rolling horizon", pos_rolling)]:
    in_pos = pos.ne(0).astype(int)
    runs, dur = [], 0
    for v in in_pos:
        if v == 1:  dur += 1
        elif dur > 0:  runs.append(dur); dur = 0
    if dur > 0: runs.append(dur)
    runs = pd.Series(runs)
    print(f"\n  {label} — {len(runs):,} trades")
    print(f"    Mean {runs.mean():.1f}b  |  "
          f"p25={runs.quantile(.25):.0f}  p50={runs.quantile(.5):.0f}  "
          f"p75={runs.quantile(.75):.0f}  p90={runs.quantile(.9):.0f}  p95={runs.quantile(.95):.0f}")
    print(f"    ≤4b: {(runs<=4).mean():.1%}   ≤12b: {(runs<=12).mean():.1%}   >24b: {(runs>24).mean():.1%}")
