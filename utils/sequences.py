import numpy as np

def create_sequences(df, feature_cols, window_size=20):
    """
    Convert dataframe into sequences for LSTM
    Returns sequences, corresponding dates, and prices
    """
    X, dates, prices = [], [], []
    for i in range(len(df) - window_size):
        X.append(df[feature_cols].iloc[i:i+window_size].values)
        dates.append(df['date'].iloc[i+window_size])
        prices.append(df['close'].iloc[i+window_size])
    return np.array(X), dates, prices