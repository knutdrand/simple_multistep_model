import pandas as pd


def one_hot_encode_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded columns for each location."""
    dummies = pd.get_dummies(df["location"], prefix="loc").astype(float)
    return pd.concat([df, dummies], axis=1)

def lag_all_features(df: pd.DataFrame, min_lag: int = 1, max_lag: int = 3) -> pd.DataFrame:
    """Add lagged versions of all non-index columns, computed per location.

    For each feature column and each lag min_lag..max_lag, adds a column
    named ``{col}_lag{k}``. The original feature columns are removed.
    Rows are sorted by location and time_period before shifting so lags
    are chronologically correct.
    """
    index_cols = ["time_period", "location"]
    feature_cols = [c for c in df.columns if c not in index_cols]

    df = df.sort_values(["location", "time_period"]).copy()
    for col in feature_cols:
        for lag in range(min_lag, max_lag + 1):
            df[f"{col}_lag{lag}"] = df.groupby("location")[col].shift(lag)
    df = df.drop(columns=feature_cols)
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This is the transformation function imported from train and predict
    '''
    df = lag_all_features(df)
    df = one_hot_encode_locations(df)
    return df

