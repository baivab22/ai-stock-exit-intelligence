from sklearn.preprocessing import StandardScaler

def scale_features(df, feature_cols):
    """
    Standardize feature columns for model input
    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler