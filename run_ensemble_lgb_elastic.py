"""
run_ensemble_lgb_elastic.py
Ensemble LightGBM + ElasticNet signals.

Challenge: different feature sets → different prediction scales.
Solution: z-score normalize each model's predictions within each fold
          using the training set distribution before combining.

Strategies tested:
  1. Standalone LightGBM
  2. Standalone ElasticNet
  3. Mean of z-scored predictions
  4. Vote — only trade when both agree on direction, else flat
  5. Boosted vote — use mean pred as size, but zero when they disagree
"""
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features import generate_features, get_data
from src.signals import threshold_signals, build_position_holdN
from src.backtest import evaluate_holdN

BARS_PER_YEAR = 365 * 24 * 2

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml') as f:    cfg_lgb = yaml.safe_load(f)
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
    ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))
])

# ── collect per-fold normalized predictions ────────────────────────────────
lgb_preds, en_preds, actuals = [], [], []
X_folds_lgb, X_folds_lin = [], []

for k in range(N_FOLDS):
    i = k * 0.1

    X_train_lgb, X_valid_lgb, y_train_lgb, y_valid_lgb = get_data(
        df_lgb, target_lgb, split_pos=[i, i+0.2, i+0.3])
    X_train_lin, X_valid_lin, y_train_lin, _ = get_data(
        df_lin, target_lin, split_pos=[i, i+0.2, i+0.3])

    lgb_model.fit(X_train_lgb, y_train_lgb)
    en_model.fit(X_train_lin, y_train_lin)

    # raw predictions
    lgb_pred_train = lgb_model.predict(X_train_lgb)
    lgb_pred_valid = lgb_model.predict(X_valid_lgb)
    en_pred_train  = en_model.predict(X_train_lin)
    en_pred_valid  = en_model.predict(X_valid_lin)

    # z-score normalize using training distribution so scales are comparable
    lgb_z = (lgb_pred_valid - lgb_pred_train.mean()) / (lgb_pred_train.std() + 1e-9)
    en_z  = (en_pred_valid  - en_pred_train.mean())  / (en_pred_train.std()  + 1e-9)

    # align on shared index
    shared_idx = X_valid_lgb.index.intersection(X_valid_lin.index)
    lgb_z = pd.Series(lgb_z, index=X_valid_lgb.index).loc[shared_idx]
    en_z  = pd.Series(en_z,  index=X_valid_lin.index).loc[shared_idx]

    lgb_preds.append(lgb_z)
    en_preds.append(en_z)
    actuals.append(y_valid_lgb.loc[shared_idx])
    X_folds_lgb.append(X_valid_lgb.loc[shared_idx])

lgb_full = pd.concat(lgb_preds)
en_full  = pd.concat(en_preds)
X_all    = pd.concat(X_folds_lgb)

# ── signal generator from z-scored pred ───────────────────────────────────
def signals_from_z(z_pred, X_valid, label):
    threshold = np.quantile(np.abs(z_pred), 0.8)
    bias      = z_pred.mean()
    return threshold_signals(X_valid, z_pred.values, threshold=threshold, bias=bias)

# ── evaluate each strategy ─────────────────────────────────────────────────
def show(label, signals, X_valid, df_features):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    evaluate_holdN(df_features, X_valid, signals, HORIZON_BARS)

print("=" * 55)
print("ENSEMBLE: LightGBM + ElasticNet")
print("=" * 55)

# 1. standalone LightGBM (z-scored)
sig_lgb = signals_from_z(lgb_full, X_all, 'LightGBM')
show("Standalone LightGBM", sig_lgb, X_all, df_lgb)

# 2. standalone ElasticNet (z-scored)
sig_en = signals_from_z(en_full, X_all, 'ElasticNet')
show("Standalone ElasticNet", sig_en, X_all, df_lin)

# 3. mean of z-scored predictions
mean_z = (lgb_full + en_full) / 2
sig_mean = signals_from_z(mean_z, X_all, 'Mean')
show("Mean (LGB z + EN z)", sig_mean, X_all, df_lgb)

# 4. vote: only trade when both agree on direction
lgb_dir = np.sign(lgb_full)
en_dir  = np.sign(en_full)
agree   = (lgb_dir == en_dir)
vote_z  = mean_z.where(agree, 0)   # zero out disagreements
sig_vote = signals_from_z(vote_z, X_all, 'Vote')
show("Vote (only when both agree)", sig_vote, X_all, df_lgb)

# 5. weighted: LightGBM 60%, ElasticNet 40% (LGB has higher mean Sharpe)
weighted_z = 0.6 * lgb_full + 0.4 * en_full
sig_weighted = signals_from_z(weighted_z, X_all, 'Weighted')
show("Weighted (LGB 60% + EN 40%)", sig_weighted, X_all, df_lgb)
