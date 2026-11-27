"""Cached data access for the dashboard.

All functions are read-only and operate on local files. Nothing here makes a
network call; the raw dataset and model artifacts are bundled with the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from . import config

REQUIRED_RAW_COLUMNS = {
    config.DATE_COL,
    config.CITY_COL,
    config.WARD_COL,
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    config.TARGET_COL,
}

REQUIRED_ONSITE_COLUMNS = REQUIRED_RAW_COLUMNS | set(config.SUPPLY_COLUMNS)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _read_csv(path: Path, parse_dates: Iterable[str] = ()) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _load_raw_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = _read_csv(Path(path_str), parse_dates=(config.DATE_COL,))
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=[config.DATE_COL, config.WARD_COL]).copy()
    df[config.WARD_COL] = df[config.WARD_COL].astype(str)
    df[config.CITY_COL] = df[config.CITY_COL].astype(str)
    return df.sort_values([config.WARD_COL, config.DATE_COL]).reset_index(drop=True)


def load_raw() -> pd.DataFrame:
    """Load and validate the bundled raw ward-week dataset."""
    path = config.RAW_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python scripts/fetch_data.py`."
        )
    return _load_raw_cached(str(path), _mtime(path))


@st.cache_data(show_spinner=False)
def _load_onsite_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = _read_csv(Path(path_str), parse_dates=(config.DATE_COL,))
    missing = REQUIRED_ONSITE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Onsite dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=[config.DATE_COL, config.WARD_COL]).copy()
    df[config.WARD_COL] = df[config.WARD_COL].astype(str)
    df[config.CITY_COL] = df[config.CITY_COL].astype(str)
    for col in config.SUPPLY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    return df.sort_values([config.WARD_COL, config.DATE_COL]).reset_index(drop=True)


def load_onsite() -> pd.DataFrame:
    """Load the Phase 2 onsite dataset (Phase 1 columns + supply targets).

    The case and weather columns are byte-identical to ``dengue_weather.csv``;
    only the three deterministic supply targets were added. The bundled Phase 1
    model therefore stays valid and does not need retraining.
    """
    path = config.ONSITE_RAW
    if not path.exists():
        raise FileNotFoundError(
            f"Onsite dataset not found at {path}. Keep it bundled for offline judging."
        )
    return _load_onsite_cached(str(path), _mtime(path))


@st.cache_data(show_spinner=False)
def _load_features_cached(path_str: str, mtime: float) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    df = _read_csv(path, parse_dates=(config.DATE_COL,))
    if config.WARD_COL in df.columns:
        df[config.WARD_COL] = df[config.WARD_COL].astype(str)
    return df.sort_values([config.WARD_COL, config.DATE_COL]).reset_index(drop=True)


def load_features() -> pd.DataFrame | None:
    """Load the engineered feature table, if the ML pipeline has produced it."""
    path = config.FEATURES
    if not path.exists():
        return None
    return _load_features_cached(str(path), _mtime(path))


@st.cache_data(show_spinner=False)
def _ward_meta_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(
        path_str,
        usecols=[
            config.WARD_COL,
            config.CITY_COL,
            "population_density_sqkm",
            "water_logging_risk_index",
            "sanitation_score",
        ],
    )
    df[config.WARD_COL] = df[config.WARD_COL].astype(str)
    df[config.CITY_COL] = df[config.CITY_COL].astype(str)
    return (
        df.drop_duplicates(subset=[config.WARD_COL])
        .sort_values([config.CITY_COL, config.WARD_COL])
        .reset_index(drop=True)
    )


def load_ward_meta() -> pd.DataFrame:
    """Return one row of static vulnerability data per ward."""
    return _ward_meta_cached(str(config.RAW_DATA), _mtime(config.RAW_DATA))


def available_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(df[config.DATE_COL].dropna().unique()))


def wards_for_city(meta: pd.DataFrame, city: str) -> list[str]:
    subset = meta.loc[meta[config.CITY_COL] == city, config.WARD_COL]
    return sorted(subset.astype(str).tolist())


def ward_row(meta: pd.DataFrame, ward_id: str) -> pd.Series:
    subset = meta.loc[meta[config.WARD_COL] == ward_id]
    if subset.empty:
        raise KeyError(f"Unknown ward: {ward_id}")
    return subset.iloc[0]


def ward_history(raw: pd.DataFrame, ward_id: str) -> pd.DataFrame:
    return raw.loc[raw[config.WARD_COL] == ward_id].sort_values(config.DATE_COL)


def _percentile(series: pd.Series, value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean <= value).mean() * 100.0)


def _value_on(history: pd.DataFrame, col: str, when: pd.Timestamp) -> float | None:
    row = history.loc[history[config.DATE_COL] == when, col]
    if row.empty:
        return None
    return float(row.iloc[-1])


def lag_context(raw: pd.DataFrame, meta: pd.DataFrame, ward_id: str, as_of: pd.Timestamp) -> dict:
    """Assemble the weather / vulnerability context shown next to a prediction.

    This is display logic only. The model consumes the engineered features from
    ``data/processed/features.csv``; here we reconstruct the same recent window
    so the dashboard can explain what is driving the score.
    """
    history = ward_history(raw, ward_id)
    history = history.loc[history[config.DATE_COL] <= as_of].copy()
    if history.empty:
        raise ValueError(f"No observations for {ward_id} on or before {as_of}")

    recent = history.tail(16).copy()
    rain_lags: list[tuple[int, float | None, float | None]] = []
    for weeks in range(1, 5):
        when = as_of - pd.Timedelta(weeks=weeks)
        value = _value_on(history, "rainfall_mm", when)
        rain_lags.append((weeks, value, _percentile(history["rainfall_mm"], value)))

    last4 = history.tail(4)
    rain_4w = float(last4["rainfall_mm"].fillna(0).sum())
    humidity = _value_on(history, "relative_humidity_pct", as_of)
    temp = _value_on(history, "avg_temp_c", as_of)
    info = ward_row(meta, ward_id)

    return {
        "history": recent,
        "rain_lags": rain_lags,
        "rain_4w": rain_4w,
        "rain_4w_pct": _percentile(history["rainfall_mm"].rolling(4).sum(), rain_4w),
        "humidity": humidity,
        "humidity_pct": _percentile(history["relative_humidity_pct"], humidity),
        "temp": temp,
        "temp_pct": _percentile(history["avg_temp_c"], temp),
        "density": float(info["population_density_sqkm"]),
        "water": float(info["water_logging_risk_index"]),
        "sanitation": float(info["sanitation_score"]),
        "city": str(info[config.CITY_COL]),
    }


def format_date(value: pd.Timestamp | None, fmt: str = "%d %b %Y") -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime(fmt)
