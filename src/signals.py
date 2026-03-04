import pandas as pd


def threshold_signals(X_test, y_pred, threshold, bias):
    """
    Generate long/short/flat signals based on an absolute prediction threshold.

    Since the underlying asset (e.g. ETH) has a slightly positive mean return,
    using a symmetric threshold around zero would make short signals too easy
    to trigger. The bias parameter shifts the short threshold upward to
    compensate for this positive drift, requiring a stronger negative prediction
    before going short.

    Args:
        X_test: DataFrame of test features (used only for its index).
        y_pred: Array-like of predicted returns.
        threshold: Minimum absolute predicted return to trigger a trade.
        bias: Positive drift offset applied to the short threshold.
              Typically set to the mean return of the training set
              (e.g. y_train.mean()) to account for the asset's natural upward drift.

    Returns:
        Series of signals: +1 (long), -1 (short), or 0 (flat).
    """
    signals = pd.Series(0, index=X_test.index)
    signals[y_pred > threshold]          =  1   # Long
    signals[y_pred < -threshold + bias]  = -1   # Short

    print(f"Long signals:  {(signals == 1).sum()}")
    print(f"Short signals: {(signals == -1).sum()}")
    print(f"Flat:          {(signals == 0).sum()}")

    return signals


def build_position_holdN(signals, HORIZON_BARS) -> pd.Series:
    """
    Convert per-bar signals (-1/0/+1) into a position series that:
      - enters at next bar (t+1)
      - holds for HORIZON_BARS bars
      - does not overlap trades (ignores signals while in a trade)

    Args:
        signals: Series of signals (+1, -1, 0).
        HORIZON_BARS: Number of bars to hold each trade.

    Returns:
        Series of positions (+1, -1, 0).
    """
    idx = signals.index
    pos = pd.Series(0, index=idx, dtype=int)

    i = 0
    L = len(idx)
    while i < L:
        s = int(signals.iloc[i])
        if s == 0:
            i += 1
            continue

        entry = i + 1
        exit_ = i + HORIZON_BARS

        if entry >= L:
            break

        end = min(exit_, L - 1)
        pos.iloc[entry:end + 1] = s

        i = end + 1

    return pos
