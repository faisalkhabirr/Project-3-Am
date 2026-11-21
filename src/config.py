"""Application configuration and shared constants."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA = ROOT / "data" / "raw" / "dengue_weather.csv"
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(ROOT / "data" / "processed")))
FEATURES = PROCESSED_DIR / "features.csv"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "models")))

MODEL_PATH = MODEL_DIR / "dengue_model.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLDS_PATH = MODEL_DIR / "risk_thresholds.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"

# ---------------------------------------------------------------------------
# Product copy
# ---------------------------------------------------------------------------
APP_TITLE = "Aedes Sentinel"
APP_TAGLINE = "Ward-level dengue early warning for Bangladesh"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
DATE_COL = "date"
WARD_COL = "ward_id"
CITY_COL = "city"
TARGET_COL = "dengue_cases"
HORIZON_COL = "horizon_weeks"

CITY_ORDER = [
    "Dhaka North",
    "Dhaka South",
    "Chattogram",
    "Gazipur",
    "Narayanganj",
]

TIER_COLORS = {
    "Low": "#2F7D4A",
    "Medium": "#C98A00",
    "High": "#D96A1A",
    "Critical": "#B3261E",
}

TIER_BG = {
    "Low": "#E9F5EC",
    "Medium": "#FFF3D6",
    "High": "#FBE9DD",
    "Critical": "#F7E3E0",
}

TIER_TEXT = {
    "Low": "#1F5A34",
    "Medium": "#7A5300",
    "High": "#8A3F0A",
    "Critical": "#8C1D18",
}

TIER_ICONS = {
    "Low": "●",
    "Medium": "●",
    "High": "●",
    "Critical": "●",
}

# Readable names for model features. Unknown features fall back to a prettifier.
FEATURE_LABELS: dict[str, str] = {
    "avg_temp_c": "Average temperature",
    "max_temp_c": "Maximum temperature",
    "min_temp_c": "Minimum temperature",
    "rainfall_mm": "Weekly rainfall",
    "relative_humidity_pct": "Relative humidity",
    "population_density_sqkm": "Population density",
    "water_logging_risk_index": "Waterlogging risk",
    "sanitation_score": "Sanitation score",
    "week_sin": "Seasonality (sin)",
    "week_cos": "Seasonality (cos)",
    "month": "Month",
    "year": "Year",
    "horizon_weeks": "Forecast horizon",
    "rain_cum_2w": "Rainfall, last 2 weeks",
    "rain_cum_3w": "Rainfall, last 3 weeks",
    "rain_cum_4w": "Rainfall, last 4 weeks",
    "rain_mean_4w": "Mean weekly rainfall, last 4 weeks",
    "temp_mean_4w": "Mean temperature, last 4 weeks",
    "humidity_mean_4w": "Mean humidity, last 4 weeks",
    "temp_suitability": "Temperature suitability",
    "cases_lag_1w": "Cases 1 week ago",
    "cases_lag_2w": "Cases 2 weeks ago",
    "cases_lag_3w": "Cases 3 weeks ago",
    "cases_lag_4w": "Cases 4 weeks ago",
    "rain_lag_1w": "Rainfall 1 week ago",
    "rain_lag_2w": "Rainfall 2 weeks ago",
    "rain_lag_3w": "Rainfall 3 weeks ago",
    "rain_lag_4w": "Rainfall 4 weeks ago",
    "rain_lag_7d": "Rainfall 7 days ago",
    "rain_lag_14d": "Rainfall 14 days ago",
    "rain_lag_21d": "Rainfall 21 days ago",
    "rain_lag_28d": "Rainfall 28 days ago",
}

_RAW_LABELS = {
    "avg_temp_c": "Average temperature",
    "max_temp_c": "Maximum temperature",
    "min_temp_c": "Minimum temperature",
    "rainfall_mm": "Weekly rainfall",
    "relative_humidity_pct": "Relative humidity",
    "population_density_sqkm": "Population density",
    "water_logging_risk_index": "Waterlogging risk",
    "sanitation_score": "Sanitation score",
    "dengue_cases": "Dengue cases",
}


def pretty_feature(name: str) -> str:
    """Return a human-readable label for a model/raw feature name."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name in _RAW_LABELS:
        return _RAW_LABELS[name]

    match = re.match(r"^(rain|temp|humidity|cases)_lag_(\d+)(w|d)$", name)
    if match:
        base, value, unit = match.groups()
        base_label = {
            "rain": "Rainfall",
            "temp": "Temperature",
            "humidity": "Humidity",
            "cases": "Cases",
        }.get(base, base.title())
        unit_label = "week" if unit == "w" else "day"
        plural = "" if value == "1" else "s"
        return f"{base_label} {value} {unit_label}{plural} ago"

    match = re.match(r"^(rain|temp|humidity)_(mean|cum|sum)_(\d+)w$", name)
    if match:
        base, agg, value = match.groups()
        base_label = {"rain": "Rainfall", "temp": "Temperature", "humidity": "Humidity"}.get(
            base, base.title()
        )
        agg_label = {"mean": "mean", "cum": "total", "sum": "total"}.get(agg, agg)
        return f"{base_label}, {agg_label} last {value} weeks"

    return name.replace("_", " ").strip().capitalize()


def tier_style(tier: str) -> str:
    """Return an inline CSS rule for a tier chip."""
    color = TIER_COLORS.get(tier, "#475569")
    bg = TIER_BG.get(tier, "#F1F5F9")
    text = TIER_TEXT.get(tier, "#334155")
    return f"background:{bg};color:{text};border:1px solid {color}33;"
