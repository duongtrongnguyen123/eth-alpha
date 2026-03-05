"""
run_holdout_test.py
True holdout evaluation: lock last 20% of data (folds 6-7, OOS 0.8→1.0).

Development  — folds 0-5 (OOS 0.2→0.8): param selection
Holdout      — folds 6-7 (OOS 0.8→1.0): fixed params, never touched during tuning

Params locked from dev: THRESHOLD_PCT=95, ER>0.6, 5 bps
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

COST_BPS      = 5
ER_N          = 10
THRESHOLD_PCT = 95   # locked from run_dynamic_threshold_sweep.py on dev
BEST_ER       = 0.6  # locked from run_er_filter.py on dev
N_DEV_FOLDS   = 6    # folds 0-5: OOS 0.2→0.8
N_ALL_FOLDS   = 8    # folds 6-7: OOS 0.8→1.0 = holdout

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml')    as f: cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb['horizon_bars']

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
    ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))])

all_lgb_z, all_en_z, all_X = [], [], []
thr_list, bias_list = [], []

print("Training 8 folds...")
for k in range(N_ALL_FOLDS):
    i = k * 0.1
    X_tl, X_vl, y_tl, _ = get_data(df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_tn, X_vn, y_tn, _ = get_data(df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_tl, y_tl)
    en_model.fit(X_tn, y_tn)

    lgb_ptr = lgb_model.predict(X_tl);  lgb_pv = lgb_model.predict(X_vl)
    en_ptr  = en_model.predict(X_tn);   en_pv  = en_model.predict(X_vn)

    lgb_z_tr = (lgb_ptr - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9)
    en_z_tr  = (en_ptr  - en_ptr.mean())  / (en_ptr.std()  + 1e-9)

    thr_list.append((np.quantile(np.abs(lgb_z_tr), THRESHOLD_PCT/100) +
                     np.quantile(np.abs(en_z_tr),  THRESHOLD_PCT/100)) / 2)
    bias_list.append((lgb_z_tr.mean() + en_z_tr.mean()) / 2)

    shared = X_vl.index.intersection(X_vn.index)
    all_lgb_z.append(pd.Series((lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_vl.index).loc[shared])
    all_en_z.append(pd.Series((en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_vn.index).loc[shared])
    all_X.append(X_vl.loc[shared])


# ── split dev vs holdout ───────────────────────────────────────────────────────
X_dev  = pd.concat(all_X[:N_DEV_FOLDS])
X_hold = pd.concat(all_X[N_DEV_FOLDS:])

mean_z_dev  = (pd.concat(all_lgb_z[:N_DEV_FOLDS]) + pd.concat(all_en_z[:N_DEV_FOLDS])) / 2
mean_z_hold = (pd.concat(all_lgb_z[N_DEV_FOLDS:]) + pd.concat(all_en_z[N_DEV_FOLDS:])) / 2

r1_dev  = df_lgb.loc[X_dev.index,  'return_1'].astype(float)
r1_hold = df_lgb.loc[X_hold.index, 'return_1'].astype(float)

thr_dev   = float(np.mean(thr_list[:N_DEV_FOLDS]))
bias_dev  = float(np.mean(bias_list[:N_DEV_FOLDS]))
thr_hold  = float(np.mean(thr_list[N_DEV_FOLDS:]))
bias_hold = float(np.mean(bias_list[N_DEV_FOLDS:]))

er_dev  = er_full.loc[X_dev.index]
er_hold = er_full.loc[X_hold.index]

print(f"\nDev  period: {X_dev.index[0].date()} → {X_dev.index[-1].date()}")
print(f"Hold period: {X_hold.index[0].date()} → {X_hold.index[-1].date()}")


# ── metrics ────────────────────────────────────────────────────────────────────
def metrics(pos, r1_series, cost_bps=COST_BPS):
    r1p     = r1_series.loc[pos.index]
    changes = pos.diff().fillna(pos.abs()).abs()
    ret     = (pos * r1p - changes * (cost_bps / 10_000)).dropna()
    if len(ret) == 0 or ret.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, rt_yr=0)
    eq      = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    return dict(
        sharpe = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR),
        cagr   = eq.iloc[-1] ** (1 / n_years) - 1,
        mdd    = ((eq / eq.cummax()) - 1).min(),
        rt_yr  = changes.gt(0).sum() / 2 / n_years,
    )

def row(label, m):
    return (f"  {label:<30}  Sharpe {m['sharpe']:>5.2f}  "
            f"Ann {m['cagr']:>6.1%}  MDD {m['mdd']:>7.1%}  RT/yr {m['rt_yr']:>4.0f}")


# ── dev results ────────────────────────────────────────────────────────────────
pos_dev_base = build_position_dynamic(mean_z_dev, entry_thr=thr_dev,
                                      min_hold=HORIZON_BARS, bias=bias_dev)
pos_dev_er   = build_position_filtered(mean_z_dev, entry_thr=thr_dev,
                                       min_hold=HORIZON_BARS, filter_series=er_dev,
                                       filter_thr=BEST_ER, bias=bias_dev)
bh_dev = r1_dev.mean() / r1_dev.std() * np.sqrt(BARS_PER_YEAR)

print(f"\n{'='*72}")
print(f"DEVELOPMENT  OOS 0.2→0.8  (folds 0-5)  @{COST_BPS}bps  |  B&H Sharpe {bh_dev:.2f}")
print(f"{'='*72}")
print(row("Dynamic hold (no filter)", metrics(pos_dev_base, r1_dev)))
print(row(f"+ ER > {BEST_ER} filter  ← params locked", metrics(pos_dev_er, r1_dev)))


# ── holdout results ────────────────────────────────────────────────────────────
pos_hold_base = build_position_dynamic(mean_z_hold, entry_thr=thr_hold,
                                       min_hold=HORIZON_BARS, bias=bias_hold)
pos_hold_er   = build_position_filtered(mean_z_hold, entry_thr=thr_hold,
                                        min_hold=HORIZON_BARS, filter_series=er_hold,
                                        filter_thr=BEST_ER, bias=bias_hold)
bh_hold = r1_hold.mean() / r1_hold.std() * np.sqrt(BARS_PER_YEAR)

print(f"\n{'='*72}")
print(f"TRUE HOLDOUT  OOS 0.8→1.0  (folds 6-7)  @{COST_BPS}bps  |  B&H Sharpe {bh_hold:.2f}")
print(f"{'='*72}")
print(row("Dynamic hold (no filter)", metrics(pos_hold_base, r1_hold)))
print(row(f"+ ER > {BEST_ER} filter  ← same fixed params", metrics(pos_hold_er, r1_hold)))

print(f"\n  Params: THRESHOLD_PCT={THRESHOLD_PCT}, ER>{BEST_ER}, cost={COST_BPS}bps/side")
print(f"  If holdout Sharpe >> dev Sharpe → got lucky on holdout period")
print(f"  If holdout Sharpe << dev Sharpe → overfit to dev or regime shift")
print(f"  If holdout Sharpe ≈ dev Sharpe  → strategy is robust")
