"""
run_er_filter.py
Test Kaufman Efficiency Ratio as an entry filter on the dynamic hold strategy.
ER measures trending (≈1) vs choppy (≈0) regime over the last N bars.
Only enter a trade when ER > filter_thr.
Sweep filter_thr to find the best value. Costs: 0, 2, 5, 10 bps/side.
Baseline: dynamic hold with no filter.
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features import generate_features, get_data
from src.signals import build_position_dynamic, build_position_filtered
from src.backtest import BARS_PER_YEAR

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS  = cfg_lgb['horizon_bars']
N_FOLDS       = cfg_lgb['walk_forward']['n_folds']
THRESHOLD_PCT = 95
COSTS         = [0, 2, 5, 10]
ER_N          = 10   # lookback for efficiency ratio

# ── compute efficiency ratio on full price series ─────────────────────────────
direction = df_raw['close'].diff(ER_N).abs()
noise     = df_raw['close'].diff().abs().rolling(ER_N).sum().replace(0, np.nan)
er_full   = (direction / noise).clip(0, 1)   # bounded [0,1]

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
thr_list, bias_list = [], []
fold_indices = []   # track per-fold OOS index for stability analysis

print("Training 8 folds...")
for k in range(N_FOLDS):
    i = k * 0.1
    X_tl, X_vl, y_tl, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_tn, X_vn, y_tn, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_tl, y_tl);  en_model.fit(X_tn, y_tn)

    lgb_ptr = lgb_model.predict(X_tl);  lgb_pv = lgb_model.predict(X_vl)
    en_ptr  = en_model.predict(X_tn);   en_pv  = en_model.predict(X_vn)

    lgb_z_tr = (lgb_ptr - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9)
    en_z_tr  = (en_ptr  - en_ptr.mean())  / (en_ptr.std()  + 1e-9)
    thr_list.append((np.quantile(np.abs(lgb_z_tr), THRESHOLD_PCT/100) +
                     np.quantile(np.abs(en_z_tr),  THRESHOLD_PCT/100)) / 2)
    bias_list.append((lgb_z_tr.mean() + en_z_tr.mean()) / 2)

    shared = X_vl.index.intersection(X_vn.index)
    lgb_z_list.append(pd.Series(
        (lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_vl.index).loc[shared])
    en_z_list.append(pd.Series(
        (en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_vn.index).loc[shared])
    X_folds.append(X_vl.loc[shared])
    fold_indices.append(shared)

X_all  = pd.concat(X_folds)
mean_z = (pd.concat(lgb_z_list) + pd.concat(en_z_list)) / 2
r1     = df_lgb.loc[X_all.index, 'return_1'].astype(float)
thr    = float(np.mean(thr_list))
bias   = float(np.mean(bias_list))

# align ER to OOS index
er = er_full.loc[X_all.index]

print(f"\nER stats on OOS period — mean {er.mean():.3f}  "
      f"p25={er.quantile(.25):.3f}  p50={er.quantile(.5):.3f}  "
      f"p75={er.quantile(.75):.3f}  p90={er.quantile(.9):.3f}")
print(f"Threshold (from training): {thr:.4f}  |  Bias: {bias:.6f}")


def metrics(pos, cost_bps):
    r1p     = r1.loc[pos.index]
    changes = pos.diff().fillna(pos.abs()).abs()
    ret     = (pos * r1p - changes * (cost_bps / 10_000)).dropna()
    if len(ret) == 0 or ret.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, rt_yr=0, in_mkt=0)
    eq      = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    sharpe  = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)
    cagr    = eq.iloc[-1] ** (1 / n_years) - 1
    mdd     = ((eq / eq.cummax()) - 1).min()
    rt_yr   = changes.gt(0).sum() / 2 / n_years
    in_mkt  = pos.ne(0).mean()
    return dict(sharpe=sharpe, cagr=cagr, mdd=mdd, rt_yr=rt_yr, in_mkt=in_mkt)


# ── baseline: no filter ───────────────────────────────────────────────────────
pos_base = build_position_dynamic(mean_z, entry_thr=thr, min_hold=HORIZON_BARS, bias=bias)

# ── ER filter sweep ───────────────────────────────────────────────────────────
ER_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

print("\n" + "=" * 86)
print("EFFICIENCY RATIO FILTER SWEEP — dynamic hold, 95th pctile threshold")
print("=" * 86)
print(f"  {'ER thr':>7}  {'RT/yr':>7}  {'In Mkt':>7}", end="")
for c in COSTS:
    print(f"  {'@'+str(c)+'bps':>10}", end="")
print()
print("  " + "-" * 70)

# baseline row
m0 = metrics(pos_base, 0)
print(f"  {'No filter':>7}  {m0['rt_yr']:>7.0f}  {m0['in_mkt']:>6.1%}", end="")
for c in COSTS:
    print(f"  {metrics(pos_base, c)['sharpe']:>10.2f}", end="")
print()
print()

results = {}
for er_thr in ER_THRESHOLDS:
    pct_above = (er >= er_thr).mean()
    pos = build_position_filtered(mean_z, entry_thr=thr, min_hold=HORIZON_BARS,
                                   filter_series=er, filter_thr=er_thr, bias=bias)
    results[er_thr] = pos
    m = metrics(pos, 0)
    print(f"  {er_thr:>7.2f}  {m['rt_yr']:>7.0f}  {m['in_mkt']:>6.1%}", end="")
    for c in COSTS:
        sh = metrics(pos, c)['sharpe']
        print(f"  {sh:>10.2f}", end="")
    print(f"   (ER>{er_thr:.1f}: {pct_above:.0%} of bars)")

# ── best per cost ─────────────────────────────────────────────────────────────
print("\n" + "=" * 86)
print("BEST ER THRESHOLD per cost level")
print("=" * 86)
print(f"  {'Cost':>6}  {'ER thr':>7}  {'Sharpe':>8}  {'Ann Ret':>10}  {'Max DD':>9}  {'RT/yr':>7}  {'In Mkt':>7}")
print("  " + "-" * 62)
for c in COSTS:
    best_thr = max(ER_THRESHOLDS,
                   key=lambda t: metrics(results[t], c)['sharpe']
                   if np.isfinite(metrics(results[t], c)['sharpe']) else -999)
    m = metrics(results[best_thr], c)
    mb = metrics(pos_base, c)
    diff = m['sharpe'] - mb['sharpe']
    print(f"  {c:>4}bps  {best_thr:>7.2f}  {m['sharpe']:>8.2f}  {m['cagr']:>9.1%}  "
          f"{m['mdd']:>8.1%}  {m['rt_yr']:>7.0f}  {m['in_mkt']:>6.1%}  "
          f"({'vs baseline: '+f'{diff:+.2f}'})")

# ── per-fold stability: baseline vs ER>0.6 ───────────────────────────────────
BEST_ER = 0.6
EVAL_COST = 5   # realistic taker fee

def fold_metrics(pos, fold_idx, cost_bps):
    pos_f   = pos.loc[pos.index.isin(fold_idx)]
    if pos_f.empty or pos_f.ne(0).sum() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan)
    r1f     = r1.loc[pos_f.index]
    changes = pos_f.diff().fillna(pos_f.abs()).abs()
    ret     = (pos_f * r1f - changes * (cost_bps / 10_000)).dropna()
    if len(ret) == 0 or ret.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan)
    eq      = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    sharpe  = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)
    cagr    = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    mdd     = ((eq / eq.cummax()) - 1).min()
    return dict(sharpe=sharpe, cagr=cagr, mdd=mdd)

pos_er06 = results[BEST_ER]

print(f"\n{'='*86}")
print(f"PER-FOLD STABILITY @ {EVAL_COST} bps/side  —  Baseline vs ER>{BEST_ER}")
print(f"{'='*86}")
print(f"  {'Fold':>5}  {'Base Sharpe':>12}  {'Base Ann':>10}  {'Base MDD':>10}  "
      f"{'ER Sharpe':>10}  {'ER Ann':>8}  {'ER MDD':>8}")
print("  " + "-" * 72)

base_sharpes, er_sharpes = [], []
base_anns,    er_anns    = [], []
base_mdds,    er_mdds    = [], []

for k, fidx in enumerate(fold_indices):
    mb = fold_metrics(pos_base,  fidx, EVAL_COST)
    me = fold_metrics(pos_er06,  fidx, EVAL_COST)
    base_sharpes.append(mb['sharpe']); er_sharpes.append(me['sharpe'])
    base_anns.append(mb['cagr']);      er_anns.append(me['cagr'])
    base_mdds.append(mb['mdd']);       er_mdds.append(me['mdd'])
    print(f"  {k+1:>5}  {mb['sharpe']:>12.2f}  {mb['cagr']:>9.1%}  {mb['mdd']:>9.1%}  "
          f"  {me['sharpe']:>9.2f}  {me['cagr']:>7.1%}  {me['mdd']:>7.1%}")

def safe(arr): return [x for x in arr if x is not None and np.isfinite(x)]

print("  " + "-" * 72)
print(f"  {'Mean':>5}  {np.nanmean(base_sharpes):>12.2f}  {np.nanmean(base_anns):>9.1%}  "
      f"{np.nanmean(base_mdds):>9.1%}  "
      f"  {np.nanmean(er_sharpes):>9.2f}  {np.nanmean(er_anns):>7.1%}  {np.nanmean(er_mdds):>7.1%}")
print(f"  {'Std':>5}  {np.nanstd(base_sharpes):>12.2f}  {np.nanstd(base_anns):>9.1%}  "
      f"{'':>9}  "
      f"  {np.nanstd(er_sharpes):>9.2f}  {np.nanstd(er_anns):>7.1%}")
print(f"  {'Min':>5}  {np.nanmin(base_sharpes):>12.2f}  {np.nanmin(base_anns):>9.1%}  "
      f"{'':>9}  "
      f"  {np.nanmin(er_sharpes):>9.2f}  {np.nanmin(er_anns):>7.1%}")
