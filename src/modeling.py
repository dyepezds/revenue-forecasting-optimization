import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def naive_forecast(train_y, horizon):
    return np.repeat(train_y.iloc[-1], horizon)


def seasonal_naive_forecast(train_y, horizon, season=7):
    return train_y.iloc[-season:].tolist() * (horizon // season + 1)


def fit_sarimax(endog, exog=None, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res


def forecast_sarimax(fitted, exog_future, steps):
    return fitted.get_forecast(steps=steps, exog=exog_future).predicted_mean