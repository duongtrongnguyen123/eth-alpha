import numpy as np
import pandas as pd
from collections import defaultdict

from src.features import get_data
from src.signals import threshold_signals
from src.backtest import decile_analysis, evaluate_holdN


def run_walk_forward(df_features, target, models, HORIZON_BARS, n_folds=8,
                     expanding=False):
    """
    Run walk-forward validation.

    Two modes:
      - sliding  (expanding=False): fixed 20% train window slides forward 10% each fold
      - expanding(expanding=True) : train always starts at 0, grows each fold
                                    fold k: train [0, 0.2+k*0.1], valid [0.2+k*0.1, 0.3+k*0.1]

    Args:
        df_features: Full feature DataFrame.
        target: Full target Series.
        models: Dict of model name -> sklearn-compatible regressor.
        HORIZON_BARS: Number of bars to hold each trade.
        n_folds: Number of folds (default 8).
        expanding: If True, use expanding window (default False).

    Returns:
        Tuple of (X_total, result).
    """
    result = defaultdict(lambda: {"signals": [], "strategy_returns": [], "equity": []})
    X_total = []

    for k in range(n_folds):
        if expanding:
            train_start = 0.0
            valid_start = 0.2 + k * 0.1
            valid_end   = 0.3 + k * 0.1
        else:
            train_start = k * 0.1
            valid_start = k * 0.1 + 0.2
            valid_end   = k * 0.1 + 0.3

        X_train, X_valid, y_train, y_valid = get_data(
            df_features, target,
            split_pos=[train_start, valid_start, valid_end]
        )

        print(f"==================== Fold {k+1} ====================")
        print(f"Train: {X_train.index.min()} → {X_train.index.max()}  ({len(X_train)} bars)")
        print(f"Valid: {X_valid.index.min()} → {X_valid.index.max()}  ({len(X_valid)} bars)")
        X_total.append(X_valid)

        for name, model in models.items():
            print(f"\n--- {name} ---")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)

            print(decile_analysis(y_pred, y_valid))

            y_pred_train = model.predict(X_train)
            threshold    = np.quantile(np.abs(y_pred_train), 0.8)
            signals      = threshold_signals(
                X_valid, y_pred,
                threshold=threshold,
                bias=y_pred_train.mean(),
            )

            strategy_returns, equity = evaluate_holdN(
                df_features, X_valid, signals, HORIZON_BARS=HORIZON_BARS
            )

            result[name]['signals'].append(signals)
            result[name]['strategy_returns'].append(strategy_returns)
            result[name]['equity'].append(equity)

        print()

    return X_total, result


def eva_full_result(df_features, X_total, result, HORIZON_BARS):
    """
    Concatenate walk-forward results per model and print full-period performance.

    Args:
        df_features: Full feature DataFrame (must contain 'return_1').
        X_total: List of validation DataFrames from each walk-forward fold.
        result: Dict keyed by model name, each with a 'signals' list.
        HORIZON_BARS: Number of bars to hold each trade.
    """
    X_all = pd.concat(X_total, axis=0)

    for name in result:
        full_signals = pd.concat(result[name]["signals"])
        assert len(X_all) == len(full_signals), "Signal/data length mismatch"

        print(f"================== {name} ==================")
        print(f"Period: {full_signals.index.min()} → {full_signals.index.max()}")
        evaluate_holdN(df_features, X_all, full_signals, HORIZON_BARS=HORIZON_BARS)
        print()
