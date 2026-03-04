import numpy as np
import pandas as pd


def generate_features(df_raw, feature_cols, HORIZON_BARS):
    """
    Generate technical features from raw OHLCV data.

    Computes returns (1/4/48/96 bars), volatility, RSI, SMA/EMA crossovers,
    price-to-rolling-max ratios, volume ratios, force, and ROC.

    Args:
        df_raw: DataFrame with columns ['open','high','low','close','volume']
        feature_cols: List of feature column names to return.
        HORIZON_BARS: Number of bars ahead for the return target.

    Returns:
        Tuple of (df_features, target) — features DataFrame and target Series,
        with NaN rows dropped.
    """
    df = df_raw.copy()

    # RETURN
    df['return_1']  = df['close'].pct_change()
    df['return_2']  = df['close'].pct_change(2)
    df['return_4']  = df['close'].pct_change(4)
    df['return_48'] = df['close'].pct_change(48)
    df['return_96'] = df['close'].pct_change(96)

    # LAGGED 4-BAR RETURN — only computed when requested (shift adds NaNs)
    if 'return_4_lag48'  in feature_cols: df['return_4_lag48']  = df['return_4'].shift(48)
    if 'return_4_lag96'  in feature_cols: df['return_4_lag96']  = df['return_4'].shift(96)
    if 'return_4_lag480' in feature_cols: df['return_4_lag480'] = df['return_4'].shift(480)

    # RETURN 2-BAR — only computed when requested
    if 'return_2' in feature_cols: df['return_2'] = df['close'].pct_change(2)

    # VOLATILITY
    df['volatility_48']     = df['return_1'].rolling(48).std()
    df['max_volatility_480'] = (
        df['return_1']
        .rolling(window=48).std()
        .rolling(window=480, min_periods=1).max()
    )

    # MOMENTUM - RSI
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # SMA
    df['sma_20']         = df['close'].rolling(20).mean()
    df['sma_50']         = df['close'].rolling(50).mean()
    df['price_to_sma20'] = df['close'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['close'] / df['sma_50'] - 1
    df['sma_cross']      = df['sma_20'] / df['sma_50'] - 1

    # EMA
    df['ema_20']          = df['close'].ewm(span=20).mean()
    df['price_to_ema20']  = df['close'] / df['ema_20'] - 1

    # CLOSE-TO-MAX ratios
    df['close_max_2400']    = df['close'].rolling(window=2400, min_periods=1).max()
    df['close_to_max_2400'] = df['close'] / df['close_max_2400'] - 1

    df['close_max_240']    = df['close'].rolling(window=240, min_periods=1).max()
    df['close_to_max_240'] = df['close'] / df['close_max_240'] - 1

    # VOLUME
    df['volume_sma']      = df['volume'].rolling(20).mean()
    df['volume_ratio']    = df['volume'] / df['volume_sma'] - 1

    df['max_volume_240']     = df['volume'].rolling(window=240, min_periods=1).max()
    df['volume_to_max_240']  = df['volume'] / df['max_volume_240'] - 1

    df['max_volume_480']     = df['volume'].rolling(window=480, min_periods=1).max()
    df['volume_to_max_480']  = df['volume'] / df['max_volume_480'] - 1

    # FORCE
    df['force'] = df['return_4'] * df['volume']

    # ROC
    df['roc_20'] = df['close'].pct_change(20)

    # TARGET
    df['target'] = df['close'].pct_change(HORIZON_BARS).shift(-HORIZON_BARS)

    df.dropna(inplace=True)

    return df[feature_cols], df['target']


def get_data(df_features, target, split_pos):
    """
    Split features and target into train/validation sets by percentage positions.

    Args:
        df_features: DataFrame of input features.
        target: Series of target values, same index as df_features.
        split_pos: List of 3 floats [train_start, valid_start, valid_end],
                   each in [0, 1] representing percentage positions in the data.

    Returns:
        Tuple of (X_train, X_valid, y_train, y_valid).
    """
    n = len(df_features)
    train_start = int(n * split_pos[0])
    valid_start = int(n * split_pos[1])
    valid_end   = int(n * split_pos[2])

    X_train = df_features.iloc[train_start:valid_start]
    X_valid = df_features.iloc[valid_start:valid_end]
    y_train = target.iloc[train_start:valid_start]
    y_valid = target.iloc[valid_start:valid_end]

    return X_train, X_valid, y_train, y_valid
