import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def mape(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def rolling_backtest(df, target, feature_cols, folds=6, horizon=14, model_fn=None, model_kwargs=None):
    """
    Expanding-window backtest: split into folds, each time fit, then forecast `horizon`.
    """
    model_kwargs = model_kwargs or {}
    n = len(df)
    fold_size = (n - horizon) // folds
    rows = []

    for i in range(folds):
        train_end = (i+1)*fold_size
        train = df.iloc[:train_end]
        test = df.iloc[train_end:train_end+horizon]

        if len(test) < horizon: break

        y_train = train[target]
        X_train = train[feature_cols] if feature_cols else None
        X_test = test[feature_cols] if feature_cols else None

        fitted = model_fn(y_train, X_train, **model_kwargs)
        if hasattr(fitted, "get_forecast"):
            preds = fitted.get_forecast(steps=horizon, exog=X_test).predicted_mean.values
        else:
            preds = model_fn(y_train, X_train, **model_kwargs).predict(X_test)  # for ML models with .predict

        rows.append({
            "fold": i+1,
            "start": test.index[0],
            "end": test.index[-1],
            "MAE": mean_absolute_error(test[target], preds),
            "MAPE": mape(test[target], preds),
        })
    return pd.DataFrame(rows)