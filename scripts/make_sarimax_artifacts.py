# scripts/make_sarimax_artifacts.py
import os
import json
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt  # <-- this was missing

# Make repo root importable no matter where we run from
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import etl
from src.modeling import fit_sarimax, forecast_sarimax

HORIZON = 14
FEATURES = ["promo", "is_weekend", "dayofweek", "month"]


def main():
    # Load processed (works regardless of current working dir)
    df = etl.load_processed("data/processed/model_ready.csv").dropna().reset_index(drop=True)

    # Ensure required exogenous features exist (self-healing)
    if "dayofweek" not in df.columns:
        if "dow" in df.columns:
            df["dayofweek"] = df["dow"]
        else:
            df["dayofweek"] = df["ds"].dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    if "month" not in df.columns:
        df["month"] = df["ds"].dt.month
    if "promo" not in df.columns:
        df["promo"] = 0

    y = df["y"]
    X = df[FEATURES]

    # Fit SARIMAX
    fitted = fit_sarimax(y, exog=X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))

    # Build future exog (calendar recomputed; promo=0)
    future_index = pd.date_range(df["ds"].iloc[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    future_exog = pd.DataFrame({
        "dayofweek": future_index.dayofweek,
        "month": future_index.month,
        "is_weekend": (future_index.dayofweek >= 5).astype(int),
        "promo": 0
    })

    pred = forecast_sarimax(fitted, exog_future=future_exog, steps=HORIZON)

    # Root-relative output dirs
    fig_dir = ROOT / "reports" / "figures"
    tab_dir = ROOT / "reports" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save PNG
    plt.figure(figsize=(12, 4))
    plt.plot(df["ds"], y, label="actual")
    plt.plot(future_index, pred.values, label="forecast")
    plt.title("SARIMAX 14-day forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "sarimax_14day_forecast.png", dpi=150, bbox_inches="tight")

    # Save forecast table
    pd.DataFrame({"ds": future_index, "forecast": pred.values}).to_csv(
        tab_dir / "forecast_14day_sarimax.csv", index=False
    )

    print("Artifacts saved in:", fig_dir, "and", tab_dir)


if __name__ == "__main__":
    main()