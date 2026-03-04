"""
run_signal_corr.py
Collect signals from all models across walk-forward folds,
then compute signal correlation and agreement stats.
"""
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from xgboost import XGBRegressor

from src.features import generate_features, get_data
from src.signals import threshold_signals

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml') as f:    cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb['horizon_bars']
N_FOLDS      = cfg_lgb['walk_forward']['n_folds']

df_lgb, target_lgb = generate_features(df_raw, feature_cols=cfg_lgb['features'], HORIZON_BARS=HORIZON_BARS)
df_lin, target_lin = generate_features(df_raw, feature_cols=cfg_lin['features'], HORIZON_BARS=HORIZON_BARS)

all_models = {
    'LightGBM':  (lgb.LGBMRegressor(
                     n_estimators=100, learning_rate=0.05, max_depth=5,
                     min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
                     random_state=42, verbose=-1), df_lgb, target_lgb),
    'XGBoost':   (XGBRegressor(
                     n_estimators=100, max_depth=5, min_child_weight=50,
                     reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
                     random_state=42, verbosity=0), df_lgb, target_lgb),
    'Ridge':      (Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
                   df_lin, target_lin),
    'ElasticNet': (Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))]),
                   df_lin, target_lin),
    'Lasso':      (Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.0001, max_iter=5000))]),
                   df_lin, target_lin),
}

signals_all = defaultdict(list)

for k in range(N_FOLDS):
    i = k * 0.1
    for name, (model, df_feat, tgt) in all_models.items():
        X_train, X_valid, y_train, _ = get_data(df_feat, tgt, split_pos=[i, i+0.2, i+0.3])
        model.fit(X_train, y_train)
        y_pred       = model.predict(X_valid)
        y_pred_train = model.predict(X_train)
        threshold    = np.quantile(np.abs(y_pred_train), 0.8)
        sig          = threshold_signals(X_valid, y_pred,
                                         threshold=threshold,
                                         bias=y_pred_train.mean())
        signals_all[name].append(sig)

# concat to full OOS series (use shared index — linear starts slightly later)
sig_df = pd.DataFrame({
    name: pd.concat(signals_all[name])
    for name in all_models
}).dropna()

print("=" * 55)
print("SIGNAL CORRELATION")
print("=" * 55)
print(sig_df.corr().round(3).to_string())

print("\n" + "=" * 55)
print("SIGNAL AGREEMENT STATS")
print("=" * 55)
print(f"\n{'Model':<14} {'Long%':>7} {'Short%':>7} {'Flat%':>7}")
print("-" * 38)
for name in all_models:
    s = sig_df[name]
    print(f"{name:<14} {(s==1).mean():>7.1%} {(s==-1).mean():>7.1%} {(s==0).mean():>7.1%}")

# pairwise: % of bars where both models are in a trade AND agree on direction
names = list(all_models.keys())
print(f"\n{'Pair agreement (both in trade, same direction)':}")
print(f"{'':20}", end="")
for n in names: print(f"{n:>12}", end="")
print()
for a in names:
    print(f"{a:<20}", end="")
    for b in names:
        both_in  = (sig_df[a] != 0) & (sig_df[b] != 0)
        agree    = (sig_df[a] == sig_df[b]) & both_in
        pct      = agree.sum() / both_in.sum() if both_in.sum() > 0 else np.nan
        print(f"{pct:>12.1%}", end="")
    print()
