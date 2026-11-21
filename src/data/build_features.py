"""Build the ward-level feature table used by the dashboard.

The pipeline is deliberately boring and auditable:

* sort by ward and week;
* compute every lag/rolling statistic **inside** each ward;
* expand the last four weekly rows into one row per (ward, as-of week, horizon);
* keep the final `h` rows of each ward even when `target_cases` is NaN so the app
  can forecast from the most recent weather.

Run:
    python -m src.data.build_features
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WARD = "ward_id"
CITY = "city"
DATE = "date"
TARGET = "dengue_cases"
HORIZON = "horizon_weeks"

ENGINEERED_COLUMNS = [
    "avg_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "rain_lag_1w",
    "rain_lag_2w",
    "rain_lag_3w",
    "rain_lag_4w",
    "rain_cum_2w",
    "rain_cum_4w",
    "humidity_lag_1w",
    "humidity_lag_2w",
    "humidity_lag_3w",
    "humidity_lag_4w",
    "humidity_mean_4w",
    "temp_lag_1w",
    "temp_lag_2w",
    "temp_lag_3w",
    "temp_lag_4w",
    "temp_mean_4w",
    "temp_suitability",
    "cases_current",
    "cases_lag_1w",
    "cases_lag_2w",
    "cases_lag_3w",
    "cases_lag_4w",
    "week_sin",
    "week_cos",
    "month",
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
]

MODEL_FEATURES = [HORIZON] + ENGINEERED_COLUMNS

BASE_COLUMNS = [
    WARD,
    CITY,
    DATE,
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "rain_lag_1w",
    "rain_lag_2w",
    "rain_lag_3w",
    "rain_lag_4w",
    "rain_cum_2w",
    "rain_cum_4w",
    "humidity_lag_1w",
    "humidity_lag_2w",
    "humidity_lag_3w",
    "humidity_lag_4w",
    "humidity_mean_4w",
    "temp_lag_1w",
    "temp_lag_2w",
    "temp_lag_3w",
    "temp_lag_4w",
    "temp_mean_4w",
    "temp_suitability",
    "cases_current",
    "cases_lag_1w",
    "cases_lag_2w",
    "cases_lag_3w",
    "cases_lag_4w",
    "week_sin",
    "week_cos",
    "month",
]

REQUIRED_RAW = {
    WARD,
    CITY,
    DATE,
    TARGET,
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
}


def build_features(
    raw: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """Return one row per (ward, as-of week, horizon)."""
    missing = REQUIRED_RAW - set(raw.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing)}")
    if not horizons:
        raise ValueError("At least one horizon is required")

    df = raw.copy()
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df = df.dropna(subset=[DATE, WARD]).sort_values([WARD, DATE]).reset_index(drop=True)
    groups = df.groupby(WARD, sort=False)

    for k in range(1, 5):
        df[f"rain_lag_{k}w"] = groups["rainfall_mm"].shift(k)
        df[f"humidity_lag_{k}w"] = groups["relative_humidity_pct"].shift(k)
        df[f"temp_lag_{k}w"] = groups["avg_temp_c"].shift(k)
        df[f"cases_lag_{k}w"] = groups[TARGET].shift(k)

    df["rain_cum_2w"] = groups["rainfall_mm"].transform(
        lambda series: series.shift(1).rolling(2).sum()
    )
    df["rain_cum_4w"] = groups["rainfall_mm"].transform(
        lambda series: series.shift(1).rolling(4).sum()
    )
    df["humidity_mean_4w"] = groups["relative_humidity_pct"].transform(
        lambda series: series.rolling(4).mean()
    )
    df["temp_mean_4w"] = groups["avg_temp_c"].transform(
        lambda series: series.rolling(4).mean()
    )
    df["temp_suitability"] = (
        (df["avg_temp_c"] >= 25) & (df["avg_temp_c"] <= 32)
    ).astype(float)
    df["cases_current"] = df[TARGET]
    week = df[DATE].dt.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    df["month"] = df[DATE].dt.month.astype(float)

    frames = []
    for horizon in horizons:
        part = df[BASE_COLUMNS].copy()
        part[HORIZON] = int(horizon)
        part["target_date"] = part[DATE] + pd.Timedelta(weeks=int(horizon))
        part["target_cases"] = groups[TARGET].shift(-int(horizon))
        frames.append(part)

    features = pd.concat(frames, ignore_index=True)
    features = features.sort_values([WARD, DATE, HORIZON]).reset_index(drop=True)
    ordered = [WARD, CITY, DATE, HORIZON, "target_date", "target_cases"] + MODEL_FEATURES
    return features[ordered]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "dengue_weather.csv",
        help="raw ward-week CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "features.csv",
        help="output feature table",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,4",
        help="comma-separated lead times (weeks)",
    )
    args = parser.parse_args(argv)

    if not args.raw.exists():
        parser.error(f"raw dataset not found: {args.raw}")
    horizons = tuple(sorted({int(value) for value in args.horizons.split(",") if value.strip()}))
    raw = pd.read_csv(args.raw, parse_dates=[DATE])
    features = build_features(raw, horizons=horizons)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    size_mb = args.out.stat().st_size / 1_000_000
    print(
        f"Wrote {len(features):,} rows to {args.out} ({size_mb:.1f} MB); "
        f"{features[WARD].nunique()} wards, horizons {list(horizons)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
