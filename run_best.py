"""
run_best.py — best config per model family.
  Trees  (LightGBM, XGBoost, RF) : strategy_lgb.yaml    (v1 + return_2, no lags)
  Linear (Ridge, ElasticNet, Lasso): strategy_linear.yaml (+ lagged returns + return_2)
"""
import yaml
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.features import generate_features
from src.walk_forward import run_walk_forward, eva_full_result

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml') as f:
    cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f:
    cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb['horizon_bars']
N_FOLDS      = cfg_lgb['walk_forward']['n_folds']

tree_models = {
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1,
    ),
    'XGBoost': XGBRegressor(
        n_estimators=100, max_depth=5, min_child_weight=50,
        reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    ),
}

linear_models = {
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'ElasticNet': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))]),
    'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.0001, max_iter=5000))]),
}

# ── tree run ───────────────────────────────────────────────────────────────
print("=" * 60)
print(f"TREE MODELS — {len(cfg_lgb['features'])} features (strategy_lgb.yaml)")
print("=" * 60)
df_lgb, target_lgb = generate_features(
    df_raw, feature_cols=cfg_lgb['features'], HORIZON_BARS=HORIZON_BARS
)
print(f"Rows: {len(df_lgb):,}\n")
X_total_lgb, result_lgb = run_walk_forward(
    df_lgb, target_lgb, tree_models, HORIZON_BARS=HORIZON_BARS, n_folds=N_FOLDS
)

# ── linear run ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"LINEAR MODELS — {len(cfg_lin['features'])} features (strategy_linear.yaml) [expanding window]")
print("=" * 60)
df_lin, target_lin = generate_features(
    df_raw, feature_cols=cfg_lin['features'], HORIZON_BARS=HORIZON_BARS
)
print(f"Rows: {len(df_lin):,}\n")
X_total_lin, result_lin = run_walk_forward(
    df_lin, target_lin, linear_models, HORIZON_BARS=HORIZON_BARS,
    n_folds=N_FOLDS, expanding=False
)

# ── final summary ──────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("FULL OOS — TREE MODELS")
print("=" * 60)
eva_full_result(df_lgb, X_total_lgb, result_lgb, HORIZON_BARS=HORIZON_BARS)

print("\n" + "=" * 60)
print("FULL OOS — LINEAR MODELS")
print("=" * 60)
eva_full_result(df_lin, X_total_lin, result_lin, HORIZON_BARS=HORIZON_BARS)
