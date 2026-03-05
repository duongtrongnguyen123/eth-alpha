"""
run_live_eval.py
Live evaluation: train on last 20% of original data (up to 2025-08-15),
predict on new data (2025-08-15 → 2026-03-03).
Params locked from full development: THRESHOLD_PCT=95, ER>0.6, 5 bps.
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
from src.signals import build_position_filtered, build_position_dynamic
from src.backtest import BARS_PER_YEAR

COST_BPS      = 5
ER_N          = 10
THRESHOLD_PCT = 95
BEST_ER       = 0.6
LIVE_START    = pd.Timestamp("2025-08-15 06:30:00")

df_raw = pd.read_csv("data/ETHUSDT.csv", parse_dates=["timestamp"], index_col="timestamp")

with open("configs/strategy_lgb.yaml")    as f: cfg_lgb = yaml.safe_load(f)
with open("configs/strategy_linear.yaml") as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb["horizon_bars"]

direction = df_raw["close"].diff(ER_N).abs()
noise     = df_raw["close"].diff().abs().rolling(ER_N).sum().replace(0, np.nan)
er_full   = (direction / noise).clip(0, 1)

df_lgb, target_lgb = generate_features(df_raw, feature_cols=cfg_lgb["features"], HORIZON_BARS=HORIZON_BARS)
df_lin, target_lin = generate_features(df_raw, feature_cols=cfg_lin["features"], HORIZON_BARS=HORIZON_BARS)

# Split: train on data before LIVE_START, predict on data from LIVE_START onward
n_total   = len(df_lgb)
live_mask_lgb = df_lgb.index >= LIVE_START
live_mask_lin = df_lin.index >= LIVE_START

X_train_lgb = df_lgb[~live_mask_lgb].iloc[-int(n_total * 0.2):]
y_train_lgb = target_lgb[X_train_lgb.index]
X_live_lgb  = df_lgb[live_mask_lgb]

X_train_lin = df_lin[~live_mask_lin].iloc[-int(n_total * 0.2):]
y_train_lin = target_lin[X_train_lin.index]
X_live_lin  = df_lin[live_mask_lin]

print(f"Train: {X_train_lgb.index[0].date()} → {X_train_lgb.index[-1].date()}  ({len(X_train_lgb):,} bars)")
print(f"Live:  {X_live_lgb.index[0].date()} → {X_live_lgb.index[-1].date()}  ({len(X_live_lgb):,} bars)")

lgb_model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1)
en_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))])

lgb_model.fit(X_train_lgb, y_train_lgb)
en_model.fit(X_train_lin, y_train_lin)

lgb_ptr = lgb_model.predict(X_train_lgb)
en_ptr  = en_model.predict(X_train_lin)

lgb_z_tr = (lgb_ptr - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9)
en_z_tr  = (en_ptr  - en_ptr.mean())  / (en_ptr.std()  + 1e-9)

thr  = (np.quantile(np.abs(lgb_z_tr), THRESHOLD_PCT/100) +
        np.quantile(np.abs(en_z_tr),  THRESHOLD_PCT/100)) / 2
bias = (lgb_z_tr.mean() + en_z_tr.mean()) / 2

lgb_pv = lgb_model.predict(X_live_lgb)
en_pv  = en_model.predict(X_live_lin)

shared = X_live_lgb.index.intersection(X_live_lin.index)
lgb_z_v = pd.Series((lgb_pv - lgb_ptr.mean()) / (lgb_ptr.std() + 1e-9), index=X_live_lgb.index).loc[shared]
en_z_v  = pd.Series((en_pv  - en_ptr.mean())  / (en_ptr.std()  + 1e-9), index=X_live_lin.index).loc[shared]
mean_z  = (lgb_z_v + en_z_v) / 2

er_live = er_full.loc[shared]
r1_live = df_lgb.loc[shared, "return_1"].astype(float)

# ── build positions ────────────────────────────────────────────────────────────
pos_base = build_position_dynamic(mean_z, entry_thr=thr, min_hold=HORIZON_BARS, bias=bias)
pos_er   = build_position_filtered(mean_z, entry_thr=thr, min_hold=HORIZON_BARS,
                                   filter_series=er_live, filter_thr=BEST_ER, bias=bias)

# ── metrics ────────────────────────────────────────────────────────────────────
def metrics(pos, cost_bps=COST_BPS):
    r1p     = r1_live.loc[pos.index]
    changes = pos.diff().fillna(pos.abs()).abs()
    ret     = (pos * r1p - changes * (cost_bps / 10_000)).dropna()
    if len(ret) == 0 or ret.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, rt_yr=0, eq=pd.Series(dtype=float))
    eq      = (1 + ret).cumprod()
    n_years = len(ret) / BARS_PER_YEAR
    return dict(
        sharpe = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR),
        cagr   = eq.iloc[-1] ** (1 / n_years) - 1,
        mdd    = ((eq / eq.cummax()) - 1).min(),
        rt_yr  = changes.gt(0).sum() / 2 / n_years,
        eq     = eq,
    )

m_base = metrics(pos_base)
m_er   = metrics(pos_er)

bh_ret  = r1_live
bh_eq   = (1 + bh_ret).cumprod()
bh_sh   = bh_ret.mean() / bh_ret.std() * np.sqrt(BARS_PER_YEAR)
bh_cagr = bh_eq.iloc[-1] ** (1 / (len(bh_ret) / BARS_PER_YEAR)) - 1
bh_mdd  = ((bh_eq / bh_eq.cummax()) - 1).min()

print(f"\n{'='*72}")
print(f"LIVE EVAL  2025-08-15 → 2026-03-03  @{COST_BPS}bps")
print(f"{'='*72}")
print(f"  {'Strategy':<32}  {'Sharpe':>6}  {'Ann Ret':>8}  {'Max DD':>8}  {'RT/yr':>6}")
print(f"  {'-'*62}")
print(f"  {'Dynamic hold (no filter)':<32}  {m_base['sharpe']:>6.2f}  {m_base['cagr']:>7.1%}  {m_base['mdd']:>7.1%}  {m_base['rt_yr']:>6.0f}")
print(f"  {'+ ER > 0.6 filter':<32}  {m_er['sharpe']:>6.2f}  {m_er['cagr']:>7.1%}  {m_er['mdd']:>7.1%}  {m_er['rt_yr']:>6.0f}")
print(f"  {'Buy & Hold':<32}  {bh_sh:>6.2f}  {bh_cagr:>7.1%}  {bh_mdd:>7.1%}")
print(f"\n  Train period: {X_train_lgb.index[0].date()} → {X_train_lgb.index[-1].date()}")
print(f"  Threshold:    {thr:.4f}  |  Bias: {bias:.6f}")

# ── plot ───────────────────────────────────────────────────────────────────────
Path("assets").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

if len(m_er["eq"]):
    ax.plot(m_er["eq"].index,   m_er["eq"].values,   color="#00e5ff", linewidth=2.0,
            label=f"ER > {BEST_ER} filter · {COST_BPS}bps  Sharpe {m_er['sharpe']:.2f}  Ann {m_er['cagr']:.1%}")
if len(m_base["eq"]):
    ax.plot(m_base["eq"].index, m_base["eq"].values, color="#4fc3f7", linewidth=1.4, alpha=0.8,
            label=f"Dynamic hold · {COST_BPS}bps  Sharpe {m_base['sharpe']:.2f}  Ann {m_base['cagr']:.1%}")
ax.plot(bh_eq.index, bh_eq.values, color="#888888", linewidth=1.2, linestyle="--", alpha=0.7,
        label=f"Buy & Hold  Sharpe {bh_sh:.2f}  Ann {bh_cagr:.1%}")

ax.set_title("ETHUSDT Live Eval — Aug 2025 → Mar 2026 (params locked from 2021–2025 dev)",
             color="white", fontsize=13, pad=12)
ax.set_xlabel("Date", color="#aaaaaa", fontsize=11)
ax.set_ylabel("Cumulative Return", color="#aaaaaa", fontsize=11)
ax.tick_params(colors="#aaaaaa")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
for spine in ax.spines.values(): spine.set_edgecolor("#333333")
ax.grid(True, color="#222222", linewidth=0.6)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}x"))
ax.legend(loc="upper left", framealpha=0.2, fontsize=10,
          labelcolor="white", facecolor="#111111", edgecolor="#333333")

plt.tight_layout()
plt.savefig("assets/live_eval.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nSaved: assets/live_eval.png")
