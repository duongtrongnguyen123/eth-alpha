"""
run_dynamic_hold.py
Compare fixed hold (4 bars) vs dynamic hold for the LGB+ElasticNet ensemble.
Dynamic hold: stays in position until an opposite signal fires (min_hold = 4 bars).
Threshold: 85th percentile (best cost-adjusted).
Costs: 0, 2, 5, 10 bps/side.
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features import generate_features, get_data
from src.signals import threshold_signals, build_position_holdN, build_position_dynamic
from src.backtest import BARS_PER_YEAR

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS  = cfg_lgb['horizon_bars']
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

# exit model: LightGBM predicting 1-bar forward return (same features as entry)
exit_lgb = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1)

# target for exit model: 1-bar forward return
_, target_r1 = generate_features(df_raw, feature_cols=cfg_lgb['features'], HORIZON_BARS=1)

lgb_z_list, en_z_list, exit_pred_list, X_folds = [], [], [], []
exit_bias_list = []   # per-fold mean of exit model training predictions

print("Training 8 folds (entry + exit model)...")
for k in range(N_FOLDS):
    i = k * 0.1
    X_tl, X_vl, y_tl, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_tn, X_vn, y_tn, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])
    _, _, y_exit_tr, _   = get_data(df_lgb, target_r1,  split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_tl, y_tl)
    en_model.fit(X_tn, y_tn)
    exit_lgb.fit(X_tl, y_exit_tr)

    lgb_ptr  = lgb_model.predict(X_tl);  lgb_pv = lgb_model.predict(X_vl)
    en_ptr   = en_model.predict(X_tn);   en_pv  = en_model.predict(X_vn)
    exit_ptr = exit_lgb.predict(X_tl)    # training predictions → bias
    exit_pv  = exit_lgb.predict(X_vl)

    shared = X_vl.index.intersection(X_vn.index)
    lgb_z_list.append(pd.Series(
        (lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_vl.index).loc[shared])
    en_z_list.append(pd.Series(
        (en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_vn.index).loc[shared])
    exit_pred_list.append(pd.Series(exit_pv, index=X_vl.index).loc[shared])
    exit_bias_list.append(exit_ptr.mean())
    X_folds.append(X_vl.loc[shared])

X_all      = pd.concat(X_folds)
mean_z     = (pd.concat(lgb_z_list) + pd.concat(en_z_list)) / 2
exit_preds = pd.concat(exit_pred_list)
r1         = df_lgb.loc[X_all.index, 'return_1'].astype(float)
bias       = mean_z.mean()
exit_bias  = float(np.mean(exit_bias_list))
print(f"  exit_bias (mean r1 train pred): {exit_bias:.6f}")

thr = np.quantile(np.abs(mean_z), THRESHOLD_PCT / 100)
# fixed-hold: pre-thresholded signals as before
signals_fixed = pd.Series(0, index=X_all.index)
signals_fixed[mean_z.values >  thr]        =  1
signals_fixed[mean_z.values < -thr + bias] = -1


def compute_metrics(pos, r1, cost_bps):
    r1p     = r1.loc[pos.index]
    changes = pos.diff().fillna(pos.abs()).abs()
    ret     = (pos * r1p - changes * (cost_bps / 10_000)).dropna()
    if len(ret) == 0 or ret.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, trades_yr=0, avg_hold=0)
    eq      = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    sharpe  = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)
    cagr    = eq.iloc[-1] ** (1 / n_years) - 1
    mdd     = ((eq / eq.cummax()) - 1).min()
    trades_yr = changes.gt(0).sum() / 2 / n_years   # round-trips per year
    in_pos  = pos.ne(0).astype(int)
    trade_starts = (in_pos.diff().fillna(in_pos) > 0).sum()
    avg_hold = in_pos.sum() / trade_starts if trade_starts > 0 else 0
    return dict(sharpe=sharpe, cagr=cagr, mdd=mdd, trades_yr=trades_yr, avg_hold=avg_hold)


pos_fixed      = build_position_holdN(signals_fixed, HORIZON_BARS)
pos_dyn_4bar   = build_position_dynamic(mean_z, entry_thr=thr, min_hold=HORIZON_BARS, bias=bias)
pos_dyn_r1exit = build_position_dynamic(mean_z, entry_thr=thr, min_hold=HORIZON_BARS, bias=bias,
                                         exit_preds=exit_preds, exit_bias=exit_bias)

# ── summary table ────────────────────────────────────────────────────────────
print(f"\nThreshold: {THRESHOLD_PCT}th percentile  |  Min hold: {HORIZON_BARS} bars")
print("\n" + "=" * 78)
print("FIXED HOLD vs DYNAMIC HOLD — Ensemble (LGB + ElasticNet)")
print("=" * 78)
print(f"  {'Strategy':<20}  {'Cost':>6}  {'Sharpe':>8}  {'Ann Ret':>10}  {'Max DD':>9}  {'RT/yr':>7}  {'Avg hold':>9}")
print("  " + "-" * 74)

for label, pos in [
    ("Fixed (4 bars)",      pos_fixed),
    ("Dynamic (4bar exit)", pos_dyn_4bar),
    ("Dynamic (r1 exit)",   pos_dyn_r1exit),
]:
    for c in COSTS:
        m = compute_metrics(pos, r1, c)
        print(f"  {label:<20}  {c:>4}bps  {m['sharpe']:>8.2f}  {m['cagr']:>9.1%}  "
              f"{m['mdd']:>8.1%}  {m['trades_yr']:>7.0f}  {m['avg_hold']:>7.1f}bars")
    print()

# ── trade duration distribution ──────────────────────────────────────────────
print("=" * 78)
print("TRADE DURATION DISTRIBUTION")
print("=" * 78)
for label, pos in [("Dynamic (4bar exit)", pos_dyn_4bar), ("Dynamic (r1 exit)", pos_dyn_r1exit)]:
    print(f"\n  {label}")
    in_pos = pos.ne(0).astype(int)
# label each trade run
    runs, dur = [], 0
    for v in in_pos:
        if v == 1:
            dur += 1
        elif dur > 0:
            runs.append(dur)
            dur = 0
    if dur > 0:
        runs.append(dur)

    runs = pd.Series(runs)
    pcts = [25, 50, 75, 90, 95]
    print(f"    Total trades : {len(runs):,}")
    print(f"    Mean hold    : {runs.mean():.1f} bars ({runs.mean()*0.5:.1f}h)")
    print(f"    Percentiles  :", end="")
    for p in pcts:
        print(f"  p{p}={runs.quantile(p/100):.0f}", end="")
    print()
    print(f"    ≤4 bars: {(runs<=4).mean():.1%}   "
          f"≤12 bars: {(runs<=12).mean():.1%}   "
          f">24 bars: {(runs>24).mean():.1%}")
