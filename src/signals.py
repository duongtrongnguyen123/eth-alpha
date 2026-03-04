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


def build_position_dynamic(preds, entry_thr, min_hold, bias=0.0,
                           exit_preds=None, exit_bias=0.0) -> pd.Series:
    """
    Dynamic hold using raw predictions:
      - Enter long  when pred >  entry_thr
      - Enter short when pred < -entry_thr + bias
      - Hold for at least min_hold bars
      - After min_hold, exit when exit signal is opposite AND strong enough:
          long  exit: exit_pred < -exit_bias
          short exit: exit_pred >  exit_bias
        where exit_bias = mean of exit model training predictions (drift offset).
        If exit_preds is None, uses entry preds with exit_bias=0.

    Args:
        preds:      Series of raw entry model predictions (4-bar ensemble).
        entry_thr:  Absolute prediction threshold to enter a trade.
        min_hold:   Minimum bars to hold before checking exit.
        bias:       Drift offset applied to the short entry threshold.
        exit_preds: Optional Series of exit model predictions (e.g. return_1).
        exit_bias:  Mean of exit model training predictions — exit only when
                    the exit signal magnitude exceeds this drift level.

    Returns:
        Series of positions (+1, -1, 0).
    """
    exit_src = exit_preds if exit_preds is not None else preds

    idx = preds.index
    pos = pd.Series(0, index=idx, dtype=int)

    i = 0
    L = len(idx)
    while i < L:
        p = preds.iloc[i]
        if p > entry_thr:
            s = 1
        elif p < -entry_thr + bias:
            s = -1
        else:
            i += 1
            continue

        entry = i + 1
        if entry >= L:
            break

        j = entry
        while j < L:
            pos.iloc[j] = s
            if (j - entry) >= min_hold - 1:
                ep = exit_src.iloc[j]
                # exit only when opposite sign AND magnitude > drift bias
                if (s == 1 and ep < -exit_bias) or (s == -1 and ep > exit_bias):
                    i = j   # re-check bar j for new entry
                    break
            j += 1
        else:
            i = L   # end of data

    return pos


def build_position_filtered(preds, entry_thr, min_hold, filter_series, filter_thr, bias=0.0):
    """
    Enhanced Signal Choosing:
    Only enter if prediction > threshold AND filter_series > filter_thr.
    Exits dynamically when the prediction flips sign.

    Args:
        preds: Series of raw model predictions.
        entry_thr: Absolute threshold for entry.
        min_hold: Minimum bars to hold before checking exit.
        filter_series: Secondary series (e.g. Efficiency Ratio) to filter entries.
        filter_thr: Threshold for the filter_series.
        bias: Drift offset for short entries.
    """
    idx = preds.index
    pos = pd.Series(0, index=idx, dtype=int)

    i = 0
    L = len(idx)
    while i < L:
        p = preds.iloc[i]
        f = filter_series.iloc[i]
        
        # ENTRY: Must pass prediction threshold AND efficiency filter
        if p > entry_thr and f > filter_thr:
            s = 1
        elif p < -entry_thr + bias and f > filter_thr:
            s = -1
        else:
            i += 1
            continue

        entry = i + 1
        if entry >= L: break
        
        j = entry
        while j < L:
            pos.iloc[j] = s
            # DYNAMIC EXIT: after min_hold, exit when prediction flips sign
            if (j - entry) >= min_hold - 1:
                if (s == 1 and preds.iloc[j] < 0) or (s == -1 and preds.iloc[j] > 0):
                    i = j
                    break
            j += 1
        else:
            i = L
    return pos


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
