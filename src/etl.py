# src/etl.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


# ---------- Path helpers ----------
def _project_root() -> Path:
    """
    Returns the repository root (one level above /src).
    """
    return Path(__file__).resolve().parents[1]


def _resolve(p: str | Path) -> Path:
    """
    Resolve a relative path against the repo root; return absolute paths unchanged.
    """
    p = Path(p)
    return p if p.is_absolute() else (_project_root() / p)


def _ensure_parent_dir(path: Path) -> None:
    """
    Create parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------- IO ----------
def load_raw(path: str | Path = "data/raw/synthetic_sales.csv") -> pd.DataFrame:
    """
    Load a raw CSV. Works from notebooks, scripts, or Streamlit.
    Accepts 'ds' or 'date' as the datetime column; normalizes to 'ds'.
    """
    path = _resolve(path)
    df = pd.read_csv(path)

    # Normalize timestamp column
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    elif "date" in df.columns:
        df.rename(columns={"date": "ds"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError("Expected a datetime column named 'ds' or 'date'.")

    df = df.sort_values("ds").reset_index(drop=True)
    return df


def save_interim(df: pd.DataFrame, path: str | Path = "data/interim/interim.csv") -> None:
    """
    Save an intermediate CSV and create parent dirs if needed.
    """
    path = _resolve(path)
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)


def save_processed(df: pd.DataFrame, path: str | Path = "data/processed/model_ready.csv") -> None:
    """
    Save a processed CSV and create parent dirs if needed.
    """
    path = _resolve(path)
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)


def load_processed(path: str | Path = "data/processed/model_ready.csv") -> pd.DataFrame:
    """
    Load a processed CSV; expects a 'ds' datetime column.
    """
    path = _resolve(path)
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df