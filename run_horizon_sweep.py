"""
run_horizon_sweep.py
Sweep HORIZON_BARS (hold period) for the Ensemble (LGB + ElasticNet).
For each horizon, lag features use return_H (horizon-matched return).
Uses 85th percentile threshold (best cost-adjusted from threshold sweep).
Evaluates at 0, 2, 5, 10 bps per side.
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features import generate_features, get_data
from src.signals import threshold_signals, build_position_holdN
from src.backtest import BARS_PER_YEAR

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

N_FOLDS    = cfg_lgb['walk_forward']['n_folds']
THRESHOLD_PCT = 85   # best cost-adjusted threshold from run_threshold_sweep.py

# horizon-matched feature sets (replace return_4/lags with return_H/lags)
BASE_LGB = [f for f in cfg_lgb['features'] if f != 'return_4'] + ['return_H']
BASE_LIN = [f for f in cfg_lin['features']
            if f not in ('return_4', 'return_4_lag48', 'return_4_lag96')] + \
           ['return_H', 'return_H_lag48', 'return_H_lag96']

HORIZONS = [4, 8, 12, 16, 24, 48]   # bars (30 min each) → 2h, 4h, 6h, 8h, 12h, 24h
COSTS    = [0, 2, 5, 10]


def run_horizon(h):
    feats_lgb = BASE_LGB
    feats_lin = BASE_LIN

    df_lgb, tgt_lgb = generate_features(df_raw, feature_cols=feats_lgb, HORIZON_BARS=h)
    df_lin, tgt_lin = generate_features(df_raw, feature_cols=feats_lin, HORIZON_BARS=h)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1)
    en_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))])

    lgb_z_list, en_z_list, X_folds = [], [], []

    for k in range(N_FOLDS):
        i = k * 0.1
        X_tl, X_vl, y_tl, _ = get_data(df_lgb, tgt_lgb, split_pos=[i, i+0.2, i+0.3])
        X_tn, X_vn, y_tn, _ = get_data(df_lin, tgt_lin, split_pos=[i, i+0.2, i+0.3])

        lgb_model.fit(X_tl, y_tl)
        en_model.fit(X_tn, y_tn)

        lgb_ptr = lgb_model.predict(X_tl)
        lgb_pv  = lgb_model.predict(X_vl)
        en_ptr  = en_model.predict(X_tn)
        en_pv   = en_model.predict(X_vn)

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

    thr = np.quantile(np.abs(mean_z), THRESHOLD_PCT / 100)
    sig = pd.Series(0, index=X_all.index)
    sig[mean_z.values >  thr]        =  1
    sig[mean_z.values < -thr + bias] = -1

    pos = build_position_holdN(sig, h)
    r1p = r1.loc[pos.index]
    changes    = pos.diff().fillna(pos.abs()).abs()
    trades_yr  = changes.gt(0).sum() / (len(pos) / BARS_PER_YEAR)
    in_market  = pos.ne(0).mean()

    metrics = {}
    for c in COSTS:
        cost_per_side = c / 10_000
        ret = (pos * r1p - changes * cost_per_side).dropna()
        if len(ret) == 0 or ret.std() == 0:
            metrics[c] = dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan)
            continue
        eq      = (1 + ret).cumprod()
        n_years = len(ret) / BARS_PER_YEAR
        metrics[c] = dict(
            sharpe = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR),
            cagr   = eq.iloc[-1] ** (1 / n_years) - 1,
            mdd    = ((eq / eq.cummax()) - 1).min(),
        )

    return metrics, trades_yr, in_market


# ── run sweep ────────────────────────────────────────────────────────────────
print(f"Threshold: {THRESHOLD_PCT}th percentile  |  Features: return_H + return_H_lag48/96")
print(f"Horizons tested: {[f'{h*0.5:.0f}h' for h in HORIZONS]}\n")

all_results = {}
for h in HORIZONS:
    print(f"  Running horizon={h} bars ({h*0.5:.0f}h)...", end=" ", flush=True)
    m, tyr, im = run_horizon(h)
    all_results[h] = (m, tyr, im)
    print(f"Sharpe@0={m[0]['sharpe']:.2f}  Sharpe@5={m[5]['sharpe']:.2f}  {tyr:.0f} trades/yr")

# ── Sharpe table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"ENSEMBLE — Sharpe by horizon × cost (threshold={THRESHOLD_PCT}th pctile)")
print("=" * 80)
print(f"  {'Horizon':>9}  {'Hold':>6}  {'Trades/yr':>10}  {'In Mkt':>7}", end="")
for c in COSTS:
    print(f"  {'@'+str(c)+'bps':>10}", end="")
print()
print("  " + "-" * 72)

for h in HORIZONS:
    m, tyr, im = all_results[h]
    print(f"  {h:>6}bars  {h*0.5:>4.0f}h  {tyr:>10.0f}  {im:>6.1%}", end="")
    for c in COSTS:
        sh = m[c]['sharpe']
        marker = " *" if c == 5 and sh == max(all_results[hh][0][5]['sharpe'] for hh in HORIZONS if np.isfinite(all_results[hh][0][5]['sharpe'])) else "  "
        print(f"  {sh:>8.2f}{marker}", end="")
    print()

# ── best per cost ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("BEST HORIZON per cost level")
print("=" * 80)
print(f"  {'Cost':>6}  {'Best horizon':>13}  {'Sharpe':>8}  {'Ann Ret':>10}  {'Max DD':>9}  {'Trades/yr':>10}")
print("  " + "-" * 62)
for c in COSTS:
    best_h = max(HORIZONS, key=lambda h: all_results[h][0][c]['sharpe']
                 if np.isfinite(all_results[h][0][c]['sharpe']) else -999)
    m = all_results[best_h][0][c]
    tyr = all_results[best_h][1]
    print(f"  {c:>4}bps  {best_h:>5}bars ({best_h*0.5:.0f}h)  {m['sharpe']:>8.2f}  {m['cagr']:>9.1%}  {m['mdd']:>8.1%}  {tyr:>10.0f}")

# ── detail for best overall @ 5bps ───────────────────────────────────────────
best5 = max(HORIZONS, key=lambda h: all_results[h][0][5]['sharpe']
            if np.isfinite(all_results[h][0][5]['sharpe']) else -999)
m5, tyr5, im5 = all_results[best5]
print(f"\n── Best horizon at 5 bps/side: {best5} bars ({best5*0.5:.0f}h hold) ──")
for c in COSTS:
    r = m5[c]
    print(f"  {c:>2} bps/side  Sharpe {r['sharpe']:>5.2f}  Ann {r['cagr']:>7.1%}  MDD {r['mdd']:>7.1%}")
print(f"  ~{tyr5:.0f} trades/yr  {im5:.1%} in market")
