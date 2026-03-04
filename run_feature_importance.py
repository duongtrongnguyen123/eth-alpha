"""
run_feature_importance.py
Train LightGBM on each walk-forward fold and average feature importances.
"""
import yaml
import numpy as np
import pandas as pd

from src.features import generate_features, get_data
from src.models import get_models

with open('configs/strategy.yaml') as f:
    cfg = yaml.safe_load(f)

HORIZON_BARS = cfg['horizon_bars']
FEATURE_COLS = cfg['features']
N_FOLDS      = cfg['walk_forward']['n_folds']

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')
df_features, target = generate_features(df_raw, feature_cols=FEATURE_COLS, HORIZON_BARS=HORIZON_BARS)

importance_gain  = pd.DataFrame(index=FEATURE_COLS)
importance_split = pd.DataFrame(index=FEATURE_COLS)

for k in range(N_FOLDS):
    i = k * 0.1
    X_train, X_valid, y_train, y_valid = get_data(
        df_features, target, split_pos=[i, i + 0.2, i + 0.3]
    )
    model = get_models()['LightGBM']
    model.fit(X_train, y_train)

    importance_gain[f'fold_{k+1}']  = model.booster_.feature_importance(importance_type='gain')
    importance_split[f'fold_{k+1}'] = model.booster_.feature_importance(importance_type='split')

# average across folds, normalize to % of total
gain_mean  = importance_gain.mean(axis=1)
split_mean = importance_split.mean(axis=1)

result = pd.DataFrame({
    'gain_mean':   gain_mean,
    'gain_pct':    gain_mean  / gain_mean.sum()  * 100,
    'split_mean':  split_mean,
    'split_pct':   split_mean / split_mean.sum() * 100,
}).sort_values('gain_pct', ascending=False)

print("Feature Importance — LightGBM (averaged over 8 folds)\n")
print(f"{'Feature':<22} {'Gain %':>8}  {'Split %':>8}")
print("-" * 42)
for feat, row in result.iterrows():
    bar = '█' * int(row['gain_pct'] / 2)
    print(f"{feat:<22} {row['gain_pct']:>7.1f}%  {row['split_pct']:>7.1f}%  {bar}")

print(f"\nTop 5 carry {result['gain_pct'].head(5).sum():.1f}% of total gain")
print(f"Bottom 5 carry {result['gain_pct'].tail(5).sum():.1f}% of total gain")
