#!/usr/bin/env python3
"""Prove the supply conversion matches the Phase 2 onsite dataset row-for-row.

This is the audit artifact behind the allocation engine. It does not trust the
training pipeline: it reads the bundled CSV, recomputes demand with the
production ``cases_to_supplies`` functions, and compares every row.

Run:

    python scripts/verify_supply_formulas.py

Exit code 0 means the formulas reproduce all supply columns exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.operations.supplies import (  # noqa: E402
    BEDS_PER_CASE,
    KITS_PER_CASE,
    SALINE_PER_CASE,
    cases_to_supplies,
    supplies_for_cases,
)

KEYS = [config.WARD_COL, config.DATE_COL]


def _fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"  FAIL {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.ONSITE_RAW)
    parser.add_argument("--raw", type=Path, default=config.RAW_DATA)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Onsite dataset missing: {args.data}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.data, parse_dates=[config.DATE_COL])
    print(f"Onsite dataset: {len(df):,} rows · {df[config.WARD_COL].nunique()} wards")
    print(
        f"Rates: saline {SALINE_PER_CASE:.6f}/case · "
        f"kits {KITS_PER_CASE:.6f}/case · beds {BEDS_PER_CASE:.3f}/case"
    )

    failures: list[str] = []

    # 1) Structural coverage.
    print("\n[1] Coverage and keys")
    missing = set(config.SUPPLY_COLUMNS) - set(df.columns)
    if missing:
        _fail(f"missing supply columns: {sorted(missing)}", failures)
        print("\nFAIL — cannot continue without the supply targets.")
        return 1
    duplicates = int(df.duplicated(subset=KEYS).sum())
    if duplicates:
        _fail(f"{duplicates:,} duplicate (ward, date) rows", failures)
    else:
        print(f"  PASS unique (ward, date) keys: {len(df):,}")
    if df[KEYS].isna().any().any():
        _fail("null ward or date values", failures)
    gaps = 0
    for _, group in df.groupby(config.WARD_COL):
        diff = group[config.DATE_COL].sort_values().diff().dropna()
        gaps += int((diff != pd.Timedelta(days=7)).sum())
    if gaps:
        _fail(f"{gaps:,} non-weekly date gaps", failures)
    else:
        print(f"  PASS all {df[config.WARD_COL].nunique()} wards are on a 7-day grid")

    # 2) Exact row-level match.
    print("\n[2] Exact supply reproduction")
    predicted = supplies_for_cases(df[config.TARGET_COL])
    for col in config.SUPPLY_COLUMNS:
        actual = pd.to_numeric(df[col], errors="coerce")
        bad = actual != predicted[col]
        if bad.any():
            examples = (
                df.loc[bad, [config.TARGET_COL, col]]
                .assign(predicted=predicted.loc[bad, col])
                .drop_duplicates()
                .head(5)
            )
            _fail(
                f"{col}: {int(bad.sum()):,}/{len(df):,} rows differ; "
                f"examples:\n{examples.to_string(index=False)}",
                failures,
            )
        else:
            print(f"  PASS {col}: {len(df):,}/{len(df):,} rows exact")

    # 3) Scalar API must agree with the vectorized path for every observed count.
    print("\n[3] Scalar / vectorized agreement")
    observed = sorted(pd.to_numeric(df[config.TARGET_COL], errors="coerce").dropna().unique())
    mismatches = []
    for cases in observed:
        scalar = cases_to_supplies(cases)
        for col in config.SUPPLY_COLUMNS:
            if scalar[col] != int(predicted.loc[df[config.TARGET_COL] == cases, col].iloc[0]):
                mismatches.append((cases, col, scalar[col]))
    if mismatches:
        _fail(f"{len(mismatches)} scalar/vectorized disagreements: {mismatches[:5]}", failures)
    else:
        print(f"  PASS {len(observed)} distinct case counts agree across both APIs")

    # 4) Domain invariants.
    print("\n[4] Domain invariants")
    zero = df[config.TARGET_COL] == 0
    if (df.loc[zero, config.SUPPLY_COLUMNS] != 0).any().any():
        _fail("zero-case rows have non-zero supply", failures)
    else:
        print("  PASS zero cases imply zero supply")
    if (df[config.SUPPLY_COLUMNS] < 0).any().any():
        _fail("negative supply values", failures)
    else:
        print("  PASS no negative supply values")
    ordered = (
        (df["saline_bags_needed"] >= df["diagnostic_kits_needed"])
        & (df["diagnostic_kits_needed"] >= df["hospital_beds_required"])
    ).all()
    if not ordered:
        _fail("expected saline >= kits >= beds for every row", failures)
    else:
        print("  PASS saline >= kits >= beds for every row")

    # 5) Phase 1 continuity: prove the new file did not change the model inputs.
    print("\n[5] Phase 1 continuity")
    if args.raw.exists():
        raw = pd.read_csv(args.raw, parse_dates=[config.DATE_COL])
        check = df.merge(
            raw,
            on=KEYS,
            how="outer",
            suffixes=("", "_phase1"),
            indicator=True,
        )
        changed = int((check["_merge"] != "both").sum())
        if changed:
            _fail(f"{changed:,} ward-weeks do not join to the Phase 1 file", failures)
        else:
            for col in [config.CITY_COL, config.TARGET_COL, "rainfall_mm", "avg_temp_c"]:
                if not check[col].equals(check[f"{col}_phase1"]):
                    _fail(f"{col} differs from Phase 1", failures)
            if not failures:
                print(
                    "  PASS cases and weather are identical to Phase 1 "
                    "(supply targets excluded from model inputs)"
                )
    else:
        print(f"  SKIP Phase 1 file not found at {args.raw}")

    print("\n" + "=" * 64)
    if failures:
        print(f"FAIL — {len(failures)} issue(s):")
        for item in failures:
            print(f"  - {item.splitlines()[0]}")
        return 1
    print("PASS — supply formulas reproduce the onsite dataset exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
