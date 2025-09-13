import pandas as pd

def add_calendar_features(df: pd.DataFrame):
    out = df.copy()
    out["dayofweek"] = out["ds"].dt.dayofweek
    out["weekofyear"] = out["ds"].dt.isocalendar().week.astype(int)
    out["month"] = out["ds"].dt.month
    out["year"] = out["ds"].dt.year
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out

def add_lagged_features(df: pd.DataFrame, target="y", lags=(1,7,14), rollings=(7,14,28)):
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target].shift(L)
    for R in rollings:
        out[f"rollmean_{R}"] = out[target].shift(1).rolling(R).mean()
        out[f"rollstd_{R}"] = out[target].shift(1).rolling(R).std()
    out = out.dropna().reset_index(drop=True)
    return out