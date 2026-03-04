"""
run_dynamic_threshold_sweep.py
Sweep signal threshold percentile (70–99) for dynamic hold (4-bar exit).
Evaluate at 0, 2, 5, 10 bps/side. Find best threshold per cost level.
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features import generate_features, get_data
from src.signals import build_position_dynamic
from src.backtest import BARS_PER_YEAR

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb['horizon_bars']
N_FOLDS      = cfg_lgb['walk_forward']['n_folds']
COSTS        = [0, 2, 5, 10]
THRESHOLDS   = [70, 75, 80, 85, 88, 90, 92, 95, 97, 99]

lgb_model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1)
en_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))])

df_lgb, target_lgb = generate_features(df_raw, feature_cols=cfg_lgb['features'], HORIZON_BARS=HORIZON_BARS)
df_lin, target_lin = generate_features(df_raw, feature_cols=cfg_lin['features'], HORIZON_BARS=HORIZON_BARS)

lgb_z_list, en_z_list, X_folds = [], [], []

print("Training 8 folds...")
for k in range(N_FOLDS):
    i = k * 0.1
    X_tl, X_vl, y_tl, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_tn, X_vn, y_tn, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_tl, y_tl)
    en_model.fit(X_tn, y_tn)

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


def eval_threshold(pct, cost_bps):
    thr = np.quantile(np.abs(mean_z), pct / 100)
    pos = build_position_dynamic(mean_z, entry_thr=thr, min_hold=HORIZON_BARS, bias=bias)

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

    in_pos       = pos.ne(0).astype(int)
    trade_starts = (in_pos.diff().fillna(in_pos) > 0).sum()
    avg_hold     = in_pos.sum() / trade_starts if trade_starts > 0 else 0

    return dict(sharpe=sharpe, cagr=cagr, mdd=mdd, rt_yr=rt_yr, avg_hold=avg_hold)


# ── sweep ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 82)
print("DYNAMIC HOLD (4-bar exit) — Sharpe by threshold × cost")
print("=" * 82)
print(f"  {'Pctile':>7}  {'RT/yr':>7}  {'AvgHold':>8}", end="")
for c in COSTS:
    print(f"  {'@'+str(c)+'bps':>10}", end="")
print()
print("  " + "-" * 66)

results = {}
for pct in THRESHOLDS:
    row = {c: eval_threshold(pct, c) for c in COSTS}
    results[pct] = row
    rt  = row[0]['rt_yr']
    ah  = row[0]['avg_hold']
    print(f"  {pct:>7}  {rt:>7.0f}  {ah:>7.1f}b", end="")
    for c in COSTS:
        sh = row[c]['sharpe']
        print(f"  {sh:>10.2f}", end="")
    print()

# ── best per cost ─────────────────────────────────────────────────────────────
print("\n" + "=" * 82)
print("BEST THRESHOLD per cost level  (dynamic hold)")
print("=" * 82)
print(f"  {'Cost':>6}  {'Pctile':>7}  {'Sharpe':>8}  {'Ann Ret':>10}  {'Max DD':>9}  {'RT/yr':>7}  {'AvgHold':>9}")
print("  " + "-" * 66)

for c in COSTS:
    best = max(THRESHOLDS, key=lambda p: results[p][c]['sharpe']
               if np.isfinite(results[p][c]['sharpe']) else -999)
    r = results[best][c]
    print(f"  {c:>4}bps  {best:>7}  {r['sharpe']:>8.2f}  {r['cagr']:>9.1%}  "
          f"{r['mdd']:>8.1%}  {r['rt_yr']:>7.0f}  {r['avg_hold']:>7.1f}b")

# ── reference: fixed hold best (80th pctile, 0 cost = 2.07) ──────────────────
print("\n── Reference: fixed hold 80th pctile ──")
print(f"  {'Cost':>6}  {'Pctile':>7}  note")
print(f"  {'0bps':>6}  {'80th':>7}  Sharpe 2.07  Ann 158.6%")
print(f"  {'2bps':>6}  {'85th':>7}  Sharpe 1.40  Ann  76.4%")
print(f"  {'5bps':>6}  {'99th':>7}  Sharpe 0.43  Ann   8.0%")
