#!/usr/bin/env python3
"""Independent sanity checks for the engineered feature table.

This does not trust the training script. It re-derives lag columns from the raw
CSV, checks that targets match future case counts, and verifies weekly coverage
per ward. Run it after the ML pipeline exports features:

    python scripts/verify_lags.py

Exit code 0 means every checked invariant passed; 1 means something is wrong.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "dengue_onsite.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

WARD = "ward_id"
DATE = "date"
TARGET = "dengue_cases"
HORIZON = "horizon_weeks"


def _lag_offset(column: str) -> tuple[str, int] | None:
    """Map a column name to (raw base column, offset in weeks)."""
    match = re.match(r"^(rain|rainfall|humidity|relative_humidity|temp|temperature|cases|dengue_cases)_lag_(\d+)(w|d)?$", column)
    if not match:
        return None
    base, value, unit = match.groups()
    base_map = {
        "rain": "rainfall_mm",
        "rainfall": "rainfall_mm",
        "humidity": "relative_humidity_pct",
        "relative_humidity": "relative_humidity_pct",
        "temp": "avg_temp_c",
        "temperature": "avg_temp_c",
        "cases": "dengue_cases",
        "dengue_cases": "dengue_cases",
    }
    weeks = int(value) if unit != "d" else max(1, int(value) // 7)
    return base_map[base], weeks


def _compare(
    name: str,
    frame: pd.DataFrame,
    expected: pd.Series,
    failures: list[str],
) -> None:
    actual = pd.to_numeric(frame[name], errors="coerce")
    expected = pd.to_numeric(expected, errors="coerce")
    both = actual.notna() & expected.notna()
    if not both.any():
        return
    diff = (actual[both] - expected[both]).abs()
    bad = int((diff > 1e-6).sum())
    total = int(both.sum())
    if bad:
        worst = float(diff.max())
        failures.append(f"{name}: {bad:,}/{total:,} rows differ (max abs diff {worst:.6g})")
    else:
        print(f"  PASS {name}: {total:,} rows match the raw-series shift")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"Raw dataset missing: {args.raw}", file=sys.stderr)
        return 1
    if not args.features.exists():
        print(f"Feature table missing: {args.features}", file=sys.stderr)
        return 1

    raw = pd.read_csv(args.raw, parse_dates=[DATE]).sort_values([WARD, DATE]).reset_index(drop=True)
    features = pd.read_csv(args.features, parse_dates=[DATE])
    print(f"Raw: {len(raw):,} rows · Features: {len(features):,} rows")

    failures: list[str] = []
    required = {WARD, DATE}
    missing = required - set(features.columns)
    if missing:
        print(f"FAIL missing keys: {sorted(missing)}", file=sys.stderr)
        return 1

    # 1) Weekly coverage per ward (the lag math assumes a 7-day grid).
    print("\n[1] Weekly coverage")
    for ward, group in raw.groupby(WARD):
        gaps = group[DATE].sort_values().diff().dropna()
        bad = gaps[gaps != pd.Timedelta(days=7)]
        if not bad.empty:
            failures.append(f"{ward}: {len(bad)} non-weekly date gaps")
    if not failures:
        print(f"  PASS all {raw[WARD].nunique()} wards are on a 7-day grid")

    # 2) Lag columns.
    print("\n[2] Lag columns")
    merged = features.merge(
        raw[[WARD, DATE, "rainfall_mm", "relative_humidity_pct", "avg_temp_c", TARGET]],
        on=[WARD, DATE],
        how="left",
        suffixes=("", "_raw"),
        validate="many_to_one",
    )
    for column in features.columns:
        parsed = _lag_offset(column)
        if parsed is None:
            continue
        base, weeks = parsed
        expected = (
            raw.assign(_expected=raw.groupby(WARD)[base].shift(weeks))
            .set_index([WARD, DATE])["_expected"]
        )
        aligned = merged.set_index([WARD, DATE]).index.map(expected)
        _compare(column, merged, pd.Series(aligned, index=merged.index), failures)

    # 3) Rolling rainfall totals: accept either "last k complete weeks" or "including current".
    print("\n[3] Rolling rainfall")
    for column in features.columns:
        match = re.match(r"^rain_(?:cum|sum)_(\d+)w$", column)
        if not match:
            continue
        k = int(match.group(1))
        candidates = {
            "shift(1).rolling(k)": raw.assign(
                _expected=raw.groupby(WARD)["rainfall_mm"].transform(
                    lambda s, k=k: s.shift(1).rolling(k).sum()
                )
            ).set_index([WARD, DATE])["_expected"],
            "rolling(k)": raw.assign(
                _expected=raw.groupby(WARD)["rainfall_mm"].transform(
                    lambda s, k=k: s.rolling(k).sum()
                )
            ).set_index([WARD, DATE])["_expected"],
        }
        matched = False
        for label, expected in candidates.items():
            aligned = merged.set_index([WARD, DATE]).index.map(expected)
            actual = pd.to_numeric(merged[column], errors="coerce")
            exp = pd.Series(aligned, index=merged.index)
            both = actual.notna() & exp.notna()
            if both.any() and (actual[both] - exp[both]).abs().max() <= 1e-6:
                print(f"  PASS {column}: matches {label}")
                matched = True
                break
        if not matched:
            failures.append(f"{column}: no supported rolling definition matched")

    # 4) Targets match the future raw case count.
    print("\n[4] Targets")
    if "target_cases" in features.columns and HORIZON in features.columns:
        target_col = "target_date" if "target_date" in features.columns else None
        if target_col:
            features["_target_date"] = pd.to_datetime(features[target_col], errors="coerce")
        else:
            features["_target_date"] = features[DATE] + pd.to_timedelta(
                features[HORIZON].astype(int), unit="W"
            )
        future = raw[[WARD, DATE, TARGET]].rename(
            columns={DATE: "_target_date", TARGET: "_expected_target"}
        )
        check = features.merge(future, on=[WARD, "_target_date"], how="left")
        actual = pd.to_numeric(check["target_cases"], errors="coerce")
        expected = pd.to_numeric(check["_expected_target"], errors="coerce")
        both = actual.notna() & expected.notna()
        if both.any():
            bad = int(((actual[both] - expected[both]).abs() > 1e-6).sum())
            if bad:
                failures.append(f"target_cases: {bad:,}/{int(both.sum()):,} rows do not match")
            else:
                print(f"  PASS target_cases: {int(both.sum()):,} rows match future cases")
        if (features["_target_date"] <= features[DATE]).any():
            failures.append("target_date is not strictly after date for every row")
        else:
            print("  PASS target_date is strictly after date")

    # 5) Basic numeric health.
    print("\n[5] Numeric health")
    infested = []
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            if np.isinf(pd.to_numeric(features[column], errors="coerce")).any():
                infested.append(column)
    if infested:
        failures.append(f"infinite values in: {infested}")
    else:
        print("  PASS no infinite values in numeric columns")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} issue(s):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASS — feature table is consistent with the raw ward-week series.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
