import numpy as np
import pandas as pd

from src.signals import build_position_holdN


BARS_PER_YEAR = 365 * 24 * 2  # 30-min bars per year


def decile_analysis(pred, y, n_bins=10):
    """
    Rank predictions into deciles and compute mean/std of actual returns per group.

    Args:
        pred: Array-like of predicted values.
        y: Array-like of actual return values.
        n_bins: Number of quantile bins (default 10 = deciles).

    Returns:
        DataFrame with columns ['count', 'mean', 'std'] indexed by decile.
    """
    df = pd.DataFrame({"pred": pred, "y": y}).dropna()
    df["pred_rank"] = df["pred"].rank(method="first")
    df["decile"]    = pd.qcut(df["pred_rank"], n_bins, labels=False) + 1
    return df.groupby("decile")["y"].agg(["count", "mean", "std"])


def evaluate_holdN(df_features, X_test, signals, HORIZON_BARS):
    """
    Builds positions from signals, then computes per-bar strategy returns,
    equity curve, Sharpe ratio, CAGR, max drawdown, and win rate.

    Args:
        df_features: Full feature DataFrame (contain 'return_1').
        X_test: Validation feature DataFrame (defines the evaluation period).
        signals: Series of signals (+1, -1, 0) aligned to X_test index.
        HORIZON_BARS: Number of bars each trade is held.

    Returns:
        Tuple of (strategy_returns, equity_curve) as pandas Series.
    """
    r1       = df_features.loc[X_test.index, 'return_1'].astype(float)
    position = build_position_holdN(signals, HORIZON_BARS)

    strategy_returns = (position * r1).dropna()
    buyhold_returns  = r1.loc[strategy_returns.index]

    equity_curve   = (1 + strategy_returns).cumprod()
    buyhold_curve  = (1 + buyhold_returns).cumprod()

    n_bars  = len(strategy_returns)
    n_years = n_bars / BARS_PER_YEAR if n_bars > 0 else np.nan

    # Sharpe
    sr_std = strategy_returns.std()
    sharpe = np.nan if sr_std == 0 else strategy_returns.mean() / sr_std * np.sqrt(BARS_PER_YEAR)

    # CAGR
    annual_return = (
        np.nan if (not np.isfinite(n_years) or n_years <= 0)
        else equity_curve.iloc[-1] ** (1 / n_years) - 1
    )

    # Max drawdown
    max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()

    # Win rate
    nonzero  = strategy_returns != 0
    win_rate = np.nan if nonzero.sum() == 0 else (strategy_returns[nonzero] > 0).mean()

    # Buy-and-hold metrics
    bh_std    = buyhold_returns.std()
    bh_sharpe = np.nan if bh_std == 0 else buyhold_returns.mean() / bh_std * np.sqrt(BARS_PER_YEAR)
    bh_annual = (
        np.nan if (not np.isfinite(n_years) or n_years <= 0)
        else buyhold_curve.iloc[-1] ** (1 / n_years) - 1
    )

    print(f"{'Metric':<20} {'Strategy':>12} {'Buy-Hold':>12}")
    print("-" * 46)
    print(f"{'Sharpe Ratio':<20} {sharpe:>12.2f} {bh_sharpe:>12.2f}")
    print(f"{'Annual Return':<20} {annual_return:>11.1%} {bh_annual:>11.1%}")
    print(f"{'Max Drawdown':<20} {max_drawdown:>11.1%} {'N/A':>12}")
    print(f"{'Win Rate (bar)':<20} {win_rate:>11.1%} {'N/A':>12}")

    return strategy_returns, equity_curve
