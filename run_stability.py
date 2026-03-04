"""
run_stability.py
Per-fold Sharpe and Annual Return for each model.
Reports mean ± std across 8 folds to measure consistency.
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
from src.backtest import evaluate_holdN

BARS_PER_YEAR = 365 * 24 * 2

def fold_metrics(df_features, X_valid, signals, HORIZON_BARS):
    """Return (sharpe, annual_return) for a single fold — no printing."""
    r1       = df_features.loc[X_valid.index, 'return_1'].astype(float)
    from src.signals import build_position_holdN
    position = build_position_holdN(signals, HORIZON_BARS)
    sr       = (position * r1).dropna()
    equity   = (1 + sr).cumprod()
    n_years  = len(sr) / BARS_PER_YEAR
    std      = sr.std()
    sharpe   = sr.mean() / std * np.sqrt(BARS_PER_YEAR) if std > 0 else np.nan
    annual   = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    return sharpe, annual

def run_and_collect(df_features, target, models, HORIZON_BARS, n_folds, label):
    stats = defaultdict(lambda: {'sharpe': [], 'annual': []})
    for k in range(n_folds):
        i = k * 0.1
        X_train, X_valid, y_train, y_valid = get_data(
            df_features, target, split_pos=[i, i+0.2, i+0.3]
        )
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred       = model.predict(X_valid)
            y_pred_train = model.predict(X_train)
            threshold    = np.quantile(np.abs(y_pred_train), 0.8)
            signals      = threshold_signals(X_valid, y_pred,
                                             threshold=threshold,
                                             bias=y_pred_train.mean())
            sh, ann = fold_metrics(df_features, X_valid, signals, HORIZON_BARS)
            stats[name]['sharpe'].append(sh)
            stats[name]['annual'].append(ann)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"{'Model':<14} {'Sharpe':>22}  {'Annual Return':>26}")
    print(f"{'':14} {'mean':>8} {'std':>7} {'min':>7}  {'mean':>8} {'std':>9} {'min':>8}")
    print("-" * 65)
    for name, v in stats.items():
        sh  = np.array(v['sharpe'])
        ann = np.array(v['annual'])
        print(f"{name:<14} "
              f"{sh.mean():>8.2f} {sh.std():>7.2f} {sh.min():>7.2f}  "
              f"{ann.mean():>8.1%} {ann.std():>9.1%} {ann.min():>8.1%}")
    return stats

df_raw = pd.read_csv('data/ETHUSDT.csv', parse_dates=['timestamp'], index_col='timestamp')

with open('configs/strategy_lgb.yaml') as f:   cfg_lgb = yaml.safe_load(f)
with open('configs/strategy_linear.yaml') as f: cfg_lin = yaml.safe_load(f)

HORIZON_BARS = cfg_lgb['horizon_bars']
N_FOLDS      = cfg_lgb['walk_forward']['n_folds']

df_lgb, target_lgb = generate_features(df_raw, feature_cols=cfg_lgb['features'], HORIZON_BARS=HORIZON_BARS)
df_lin, target_lin = generate_features(df_raw, feature_cols=cfg_lin['features'], HORIZON_BARS=HORIZON_BARS)

tree_models = {
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1),
    'XGBoost': XGBRegressor(
        n_estimators=100, max_depth=5, min_child_weight=50,
        reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0),
}
linear_models = {
    'Ridge':      Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'ElasticNet': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000))]),
    'Lasso':      Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.0001, max_iter=5000))]),
}

run_and_collect(df_lgb, target_lgb, tree_models,   HORIZON_BARS, N_FOLDS, "TREE MODELS (strategy_lgb)")
run_and_collect(df_lin, target_lin, linear_models, HORIZON_BARS, N_FOLDS, "LINEAR MODELS (strategy_linear)")
