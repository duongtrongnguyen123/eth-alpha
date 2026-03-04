from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb


def get_models():
    """
    Return a dict of model name -> fitted-ready regressor instances.

    Includes both tree-based and linear models. Linear models require
    StandardScaler (wrapped in a Pipeline) since financial features have
    very different scales (e.g. return_1 vs rsi).

    Returns:
        Dict[str, estimator]
    """
    return {
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            min_child_samples=100,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=5,
            min_samples_leaf=100, min_samples_split=100,
            max_features='sqrt', max_samples=0.8,
            random_state=42, n_jobs=-1,
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, max_depth=5, min_child_weight=50,
            reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        ),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=1.0)),
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Lasso(alpha=0.0001, max_iter=5000)),
        ]),
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=5000)),
        ]),
    }
