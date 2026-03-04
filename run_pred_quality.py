"""
run_pred_quality.py
Compare LightGBM prediction quality for return_1 vs return_4 targets.
Metrics per fold: directional accuracy, Pearson r, Spearman IC, MAE, RMSE.
"""
import yaml
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

from src.features import generate_features, get_data

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml') as f:
    cfg = yaml.safe_load(f)

HORIZON_BARS = cfg['horizon_bars']
N_FOLDS      = cfg['walk_forward']['n_folds']
FEATURES     = cfg['features']

# build feature df once; swap out targets per horizon
df_feat, target_4 = generate_features(df_raw, feature_cols=FEATURES, HORIZON_BARS=4)
_,       target_1 = generate_features(df_raw, feature_cols=FEATURES, HORIZON_BARS=1)

model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1)


def fold_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    dir_acc = (np.sign(yt) == np.sign(yp)).mean()
    pearson = np.corrcoef(yt, yp)[0, 1]
    spearman, _ = stats.spearmanr(yt, yp)
    mae  = np.abs(yt - yp).mean()
    rmse = np.sqrt(((yt - yp) ** 2).mean())
    return dict(dir_acc=dir_acc, pearson=pearson, spearman=spearman,
                mae=mae, rmse=rmse, std_true=yt.std())


rows = []
for k in range(N_FOLDS):
    i = k * 0.1
    X_tr, X_vl, y4_tr, y4_vl = get_data(df_feat, target_4, split_pos=[i, i+0.2, i+0.3])
    _,    _,    y1_tr, y1_vl = get_data(df_feat, target_1, split_pos=[i, i+0.2, i+0.3])

    # return_4 model
    model.fit(X_tr, y4_tr)
    m4 = fold_metrics(y4_vl, model.predict(X_vl))

    # return_1 model
    model.fit(X_tr, y1_tr)
    m1 = fold_metrics(y1_vl, model.predict(X_vl))

    rows.append(dict(fold=k+1, target='return_4', **m4))
    rows.append(dict(fold=k+1, target='return_1', **m1))

df = pd.DataFrame(rows)

# ── per-fold table ────────────────────────────────────────────────────────────
for target in ['return_4', 'return_1']:
    sub = df[df['target'] == target].set_index('fold')
    print(f"\n{'='*68}")
    print(f"TARGET: {target}")
    print(f"{'='*68}")
    print(f"  {'Fold':>5}  {'DirAcc':>8}  {'Pearson':>8}  {'Spearman':>9}  {'MAE':>10}  {'RMSE':>10}  {'σ(true)':>10}")
    print(f"  {'-'*63}")
    for fold, row in sub.iterrows():
        print(f"  {fold:>5}  {row['dir_acc']:>8.2%}  {row['pearson']:>8.4f}  "
              f"{row['spearman']:>9.4f}  {row['mae']:>10.6f}  {row['rmse']:>10.6f}  {row['std_true']:>10.6f}")
    print(f"  {'MEAN':>5}  {sub['dir_acc'].mean():>8.2%}  {sub['pearson'].mean():>8.4f}  "
          f"{sub['spearman'].mean():>9.4f}  {sub['mae'].mean():>10.6f}  {sub['rmse'].mean():>10.6f}  "
          f"{sub['std_true'].mean():>10.6f}")

# ── side-by-side comparison ───────────────────────────────────────────────────
r4 = df[df['target'] == 'return_4'].set_index('fold')
r1 = df[df['target'] == 'return_1'].set_index('fold')

print(f"\n{'='*68}")
print("COMPARISON — return_4 vs return_1 (mean across 8 folds)")
print(f"{'='*68}")
metrics = ['dir_acc', 'pearson', 'spearman', 'mae', 'rmse']
labels  = ['Dir Accuracy', 'Pearson r', 'Spearman IC', 'MAE', 'RMSE']
for m, lab in zip(metrics, labels):
    v4 = r4[m].mean()
    v1 = r1[m].mean()
    if m == 'dir_acc':
        print(f"  {lab:<15}  return_4: {v4:.2%}   return_1: {v1:.2%}   diff: {v4-v1:+.2%}")
    elif m in ('pearson', 'spearman'):
        print(f"  {lab:<15}  return_4: {v4:.4f}   return_1: {v1:.4f}   diff: {v4-v1:+.4f}")
    else:
        ratio = v1 / v4
        print(f"  {lab:<15}  return_4: {v4:.6f}   return_1: {v1:.6f}   ratio: {ratio:.2f}x")

print(f"\n  σ(true) ratio: return_1 / return_4 = "
      f"{r1['std_true'].mean() / r4['std_true'].mean():.2f}x  "
      f"(return_1 is noisier)")
print(f"  Signal/noise: return_4 Pearson/σ_ratio = "
      f"{r4['pearson'].mean() / (r1['std_true'].mean()/r4['std_true'].mean()):.4f}  "
      f"return_1 = {r1['pearson'].mean():.4f}")
