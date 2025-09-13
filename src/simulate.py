import numpy as np
import pandas as pd

def make_synthetic_series(
    start="2022-01-01", end="2024-12-31", seed=42,
    weekly_seasonality=0.2, yearly_seasonality=0.15,
    trend_per_day=0.02, base=1000.0, promo_lift=0.25
):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # components
    dow = dates.dayofweek  # 0=Mon
    day_of_year = dates.dayofyear

    weekly = weekly_seasonality * np.sin(2*np.pi*(dow/7.0))
    yearly = yearly_seasonality * np.sin(2*np.pi*(day_of_year/365.25))
    trend = trend_per_day * np.arange(n)

    # holidays/promos: boost some weekends and special days
    promo = ((dow >= 5).astype(float))  # weekends
    # add a few event spikes
    event_days = pd.to_datetime(["2022-07-04","2022-12-31",
                                 "2023-07-04","2023-12-31",
                                 "2024-07-04","2024-12-31"])
    promo += dates.isin(event_days).astype(float) * 2.0

    noise = rng.normal(0, 0.1, size=n)

    y = base * (1 + weekly + yearly) * (1 + promo_lift*promo) + trend*5 + noise*base
    df = pd.DataFrame({"ds": dates, "y": y.round(2),
                       "promo": promo.astype(int),
                       "dow": dow})
    return df


import os

if __name__ == "__main__":
    df = make_synthetic_series()

    # Ensure the output folder exists
    os.makedirs("data/raw", exist_ok=True)

    # Save to CSV
    df.to_csv("data/raw/synthetic_sales.csv", index=False)
    print("Saved -> data/raw/synthetic_sales.csv", df.shape)