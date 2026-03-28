import pandas as pd
import numpy as np

def add_features(df):
    """
    Add technical and date features used by LSTM model
    """
    df = df.copy()

    # Returns
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)

    # Moving averages
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_20'] = df['close'].rolling(20).mean()

    # Volatility & momentum
    df['volatility_5'] = df['close'].rolling(5).std()
    df['momentum_5'] = df['close'] - df['close'].shift(5)

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26

    # Bollinger Band width
    std20 = df['close'].rolling(20).std()
    df['bb_width'] = 2 * std20

    # Volume ratios
    df['vol_ma_5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma_5'] + 1e-8)

    # Date features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    return df