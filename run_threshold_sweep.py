"""
run_threshold_sweep.py
Sweep signal threshold percentile (70th–99th) and evaluate at realistic
transaction costs (0, 2, 5, 10 bps/side) for the Ensemble (LGB + ElasticNet).

Goal: find the threshold that maximises Sharpe at 5 bps/side (taker fee).
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

# ── collect per-fold z-scores (train once, sweep thresholds cheaply) ────────
lgb_z_list, en_z_list, X_folds = [], [], []

print("Training models across 8 folds...")
for k in range(N_FOLDS):
    i = k * 0.1
    X_train_lgb, X_valid_lgb, y_train_lgb, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_train_lin, X_valid_lin, y_train_lin, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_train_lgb, y_train_lgb)
    en_model.fit(X_train_lin,  y_train_lin)

    lgb_ptr = lgb_model.predict(X_train_lgb)
    lgb_pv  = lgb_model.predict(X_valid_lgb)
    en_ptr  = en_model.predict(X_train_lin)
    en_pv   = en_model.predict(X_valid_lin)

    shared = X_valid_lgb.index.intersection(X_valid_lin.index)
    lgb_z_list.append(pd.Series(
        (lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_valid_lgb.index).loc[shared])
    en_z_list.append(pd.Series(
        (en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_valid_lin.index).loc[shared])
    X_folds.append(X_valid_lgb.loc[shared])

X_all   = pd.concat(X_folds)
lgb_z   = pd.concat(lgb_z_list)
en_z    = pd.concat(en_z_list)
mean_z  = (lgb_z + en_z) / 2
r1      = df_lgb.loc[X_all.index, 'return_1'].astype(float)
bias    = mean_z.mean()


def eval_threshold(pct, cost_bps):
    thr  = np.quantile(np.abs(mean_z), pct / 100)
    sig  = pd.Series(0, index=X_all.index)
    sig[mean_z.values >  thr]            =  1
    sig[mean_z.values < -thr + bias]     = -1

    pos  = build_position_holdN(sig, HORIZON_BARS)
    r1_p = r1.loc[pos.index]
    cost_per_side = cost_bps / 10_000
    changes = pos.diff().fillna(pos.abs()).abs()
    ret  = (pos * r1_p - changes * cost_per_side).dropna()

    if len(ret) == 0 or ret.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, trades_yr=0, in_market=np.nan)

    eq      = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    sharpe  = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)
    cagr    = eq.iloc[-1] ** (1 / n_years) - 1
    mdd     = ((eq / eq.cummax()) - 1).min()
    n_trades_yr = changes.gt(0).sum() / n_years
    in_market   = pos.ne(0).mean()

    return dict(sharpe=sharpe, cagr=cagr, mdd=mdd, trades_yr=n_trades_yr, in_market=in_market)


THRESHOLDS = [70, 75, 80, 85, 88, 90, 92, 95, 97, 99]
COSTS      = [0, 2, 5, 10]

# ── full table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ENSEMBLE — Sharpe by threshold × cost (bps/side)")
print("=" * 80)
print(f"  {'Pctile':>7}  {'Trades/yr':>10}", end="")
for c in COSTS:
    print(f"  {'Sharpe@'+str(c)+'bps':>12}", end="")
print()
print("  " + "-" * 74)

results = {}
for pct in THRESHOLDS:
    row = {}
    for c in COSTS:
        row[c] = eval_threshold(pct, c)
    results[pct] = row
    trades_yr = row[0]['trades_yr']
    print(f"  {pct:>7}  {trades_yr:>10.0f}", end="")
    for c in COSTS:
        sh = row[c]['sharpe']
        print(f"  {sh:>12.2f}", end="")
    print()

# ── best at each cost level ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("BEST THRESHOLD per cost level")
print("=" * 80)
print(f"  {'Cost (bps)':>12}  {'Best Pctile':>12}  {'Sharpe':>8}  {'Ann Ret':>10}  {'Max DD':>9}  {'Trades/yr':>10}  {'In Market':>10}")
print("  " + "-" * 80)

for c in COSTS:
    best_pct = max(THRESHOLDS, key=lambda p: results[p][c]['sharpe'] if np.isfinite(results[p][c]['sharpe']) else -999)
    r = results[best_pct][c]
    print(f"  {c:>12}  {best_pct:>12}  {r['sharpe']:>8.2f}  {r['cagr']:>9.1%}  {r['mdd']:>8.1%}  {r['trades_yr']:>10.0f}  {r['in_market']:>9.1%}")

# ── detail for best @ 5bps ───────────────────────────────────────────────────
best5 = max(THRESHOLDS, key=lambda p: results[p][5]['sharpe'] if np.isfinite(results[p][5]['sharpe']) else -999)
print(f"\n── Best threshold at 5 bps/side: {best5}th percentile ──")
for c in COSTS:
    r = results[best5][c]
    print(f"  {c:>2} bps/side  Sharpe {r['sharpe']:>5.2f}  Ann {r['cagr']:>7.1%}  MDD {r['mdd']:>7.1%}  {r['trades_yr']:.0f} trades/yr")
