"""
run_clean_split.py
Clean methodology: full dataset 2019→2026-03, split 80/20.
  Dev  0→0.8: walk-forward param selection (ER sweep + threshold sweep)
  Test 0.8→1.0: evaluate locked params, never touched during tuning
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

COST_BPS     = 5
ER_N         = 10
N_DEV_FOLDS  = 6   # folds 0-5: OOS covers 0.2→0.8 of full data
N_ALL_FOLDS  = 8   # folds 6-7: OOS covers 0.8→1.0 = test
ER_THRESHOLDS   = [0.3, 0.4, 0.5, 0.6, 0.7]
PCT_THRESHOLDS  = [85, 90, 92, 95, 97]

df_raw = pd.read_csv("data/ETHUSDT.csv", parse_dates=["timestamp"], index_col="timestamp")

with open("configs/strategy_lgb.yaml")    as f: cfg_lgb = yaml.safe_load(f)
with open("configs/strategy_linear.yaml") as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb["horizon_bars"]

direction = df_raw["close"].diff(ER_N).abs()
noise     = df_raw["close"].diff().abs().rolling(ER_N).sum().replace(0, np.nan)
er_full   = (direction / noise).clip(0, 1)

df_lgb, target_lgb = generate_features(df_raw, feature_cols=cfg_lgb["features"], HORIZON_BARS=HORIZON_BARS)
df_lin, target_lin = generate_features(df_raw, feature_cols=cfg_lin["features"], HORIZON_BARS=HORIZON_BARS)

lgb_model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1)
en_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))])

all_lgb_z, all_en_z, all_X = [], [], []
thr_lists = {p: [] for p in PCT_THRESHOLDS}
bias_list = []

print("Training 8 folds on full dataset (2019→2026-03)...")
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

    for p in PCT_THRESHOLDS:
        thr_lists[p].append((np.quantile(np.abs(lgb_z_tr), p/100) +
                             np.quantile(np.abs(en_z_tr),  p/100)) / 2)
    bias_list.append((lgb_z_tr.mean() + en_z_tr.mean()) / 2)

    shared = X_vl.index.intersection(X_vn.index)
    all_lgb_z.append(pd.Series((lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_vl.index).loc[shared])
    all_en_z.append(pd.Series((en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_vn.index).loc[shared])
    all_X.append(X_vl.loc[shared])

# ── split dev / test ───────────────────────────────────────────────────────────
X_dev  = pd.concat(all_X[:N_DEV_FOLDS])
X_test = pd.concat(all_X[N_DEV_FOLDS:])

mean_z_dev  = (pd.concat(all_lgb_z[:N_DEV_FOLDS]) + pd.concat(all_en_z[:N_DEV_FOLDS])) / 2
mean_z_test = (pd.concat(all_lgb_z[N_DEV_FOLDS:]) + pd.concat(all_en_z[N_DEV_FOLDS:])) / 2

r1_dev  = df_lgb.loc[X_dev.index,  "return_1"].astype(float)
r1_test = df_lgb.loc[X_test.index, "return_1"].astype(float)

bias_dev  = float(np.mean(bias_list[:N_DEV_FOLDS]))
bias_test = float(np.mean(bias_list[N_DEV_FOLDS:]))

er_dev  = er_full.loc[X_dev.index]
er_test = er_full.loc[X_test.index]

print(f"\nDev  period: {X_dev.index[0].date()} → {X_dev.index[-1].date()}")
print(f"Test period: {X_test.index[0].date()} → {X_test.index[-1].date()}")


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


# ── param sweep on dev only ────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"PARAM SWEEP ON DEV (0→0.8)  @{COST_BPS}bps")
print(f"{'='*78}")
print(f"  {'PCT':>5}  {'ER thr':>6}  {'Sharpe':>7}  {'Ann':>7}  {'MDD':>7}  {'RT/yr':>6}")
print(f"  {'-'*50}")

best_sharpe = -999
best_pct    = None
best_er     = None

for pct in PCT_THRESHOLDS:
    thr_d = float(np.mean(thr_lists[pct][:N_DEV_FOLDS]))
    for er_thr in ER_THRESHOLDS:
        pos = build_position_filtered(mean_z_dev, entry_thr=thr_d, min_hold=HORIZON_BARS,
                                      filter_series=er_dev, filter_thr=er_thr, bias=bias_dev)
        m = metrics(pos, r1_dev)
        if np.isfinite(m["sharpe"]) and m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_pct    = pct
            best_er     = er_thr
        marker = " ←" if (pct == best_pct and er_thr == best_er) else ""
        print(f"  {pct:>5}  {er_thr:>6.1f}  {m['sharpe']:>7.2f}  {m['cagr']:>6.1%}  {m['mdd']:>6.1%}  {m['rt_yr']:>6.0f}{marker}")

print(f"\n  Best on dev: PCT={best_pct}, ER>{best_er}  Sharpe {best_sharpe:.2f}")


# ── lock params, evaluate on test ─────────────────────────────────────────────
thr_test = float(np.mean(thr_lists[best_pct][N_DEV_FOLDS:]))

pos_test_er   = build_position_filtered(mean_z_test, entry_thr=thr_test, min_hold=HORIZON_BARS,
                                        filter_series=er_test, filter_thr=best_er, bias=bias_test)
pos_test_base = build_position_dynamic(mean_z_test, entry_thr=thr_test,
                                       min_hold=HORIZON_BARS, bias=bias_test)

bh_dev_sh  = r1_dev.mean()  / r1_dev.std()  * np.sqrt(BARS_PER_YEAR)
bh_test_sh = r1_test.mean() / r1_test.std() * np.sqrt(BARS_PER_YEAR)

print(f"\n{'='*78}")
print(f"DEV RESULT  (0→0.8)  @{COST_BPS}bps  |  B&H Sharpe {bh_dev_sh:.2f}")
print(f"{'='*78}")
thr_d = float(np.mean(thr_lists[best_pct][:N_DEV_FOLDS]))
pos_dev_er = build_position_filtered(mean_z_dev, entry_thr=thr_d, min_hold=HORIZON_BARS,
                                     filter_series=er_dev, filter_thr=best_er, bias=bias_dev)
pos_dev_base = build_position_dynamic(mean_z_dev, entry_thr=thr_d, min_hold=HORIZON_BARS, bias=bias_dev)
m_dev_base = metrics(pos_dev_base, r1_dev)
m_dev_er   = metrics(pos_dev_er,   r1_dev)
print(f"  Dynamic hold (no filter)    Sharpe {m_dev_base['sharpe']:>5.2f}  Ann {m_dev_base['cagr']:>6.1%}  MDD {m_dev_base['mdd']:>7.1%}  RT/yr {m_dev_base['rt_yr']:>4.0f}")
print(f"  + ER > {best_er} filter          Sharpe {m_dev_er['sharpe']:>5.2f}  Ann {m_dev_er['cagr']:>6.1%}  MDD {m_dev_er['mdd']:>7.1%}  RT/yr {m_dev_er['rt_yr']:>4.0f}")

print(f"\n{'='*78}")
print(f"TEST RESULT  (0.8→1.0)  @{COST_BPS}bps  |  B&H Sharpe {bh_test_sh:.2f}")
print(f"{'='*78}")
m_test_base = metrics(pos_test_base, r1_test)
m_test_er   = metrics(pos_test_er,   r1_test)
print(f"  Dynamic hold (no filter)    Sharpe {m_test_base['sharpe']:>5.2f}  Ann {m_test_base['cagr']:>6.1%}  MDD {m_test_base['mdd']:>7.1%}  RT/yr {m_test_base['rt_yr']:>4.0f}")
print(f"  + ER > {best_er} filter          Sharpe {m_test_er['sharpe']:>5.2f}  Ann {m_test_er['cagr']:>6.1%}  MDD {m_test_er['mdd']:>7.1%}  RT/yr {m_test_er['rt_yr']:>4.0f}")
print(f"\n  Locked params: PCT={best_pct}, ER>{best_er}, cost={COST_BPS}bps")
