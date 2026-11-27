#!/usr/bin/env python3
"""Verify holdout accuracy, baseline lift and 80% interval calibration.

This is the evidence script for the accuracy claim. It does not retrain; it reads
the committed holdout predictions and metrics, then checks:

* per-horizon MAE / RMSE / bias;
* empirical coverage of the 80% intervals;
* the model beats seasonal-naive and persistence at every horizon.

Run:

    python scripts/verify_accuracy.py

Exit code 0 means the committed evidence still satisfies the accuracy contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MIN_COVERAGE = 0.75
MAX_COVERAGE = 0.85
MAX_MAE = 1.50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=ROOT / "models" / "metrics.json"
    )
    parser.add_argument(
        "--preds", type=Path, default=ROOT / "models" / "test_predictions.csv"
    )
    args = parser.parse_args()

    if not args.metrics.exists() or not args.preds.exists():
        print("Model metrics or holdout predictions are missing.", file=sys.stderr)
        return 1

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    preds = pd.read_csv(args.preds)
    required = {"horizon_weeks", "y_true", "y_pred"}
    if not required.issubset(preds.columns):
        print(f"Holdout file is missing columns: {sorted(required - set(preds.columns))}", file=sys.stderr)
        return 1

    print(
        f"Model: {metrics.get('model_name', 'unknown')} · "
        f"trained {metrics.get('trained_at', 'unknown')} · "
        f"{len(preds):,} holdout rows"
    )
    print(f"{'horizon':>8} {'n':>6} {'MAE':>7} {'RMSE':>7} {'bias':>7} {'80% cov':>9}")
    failures: list[str] = []
    coverage_values: list[float] = []
    for horizon, group in preds.groupby("horizon_weeks"):
        h = str(int(horizon))
        mae = float(np.mean(np.abs(group["y_true"] - group["y_pred"])))
        rmse = float(np.sqrt(np.mean((group["y_true"] - group["y_pred"]) ** 2)))
        bias = float(np.mean(group["y_pred"] - group["y_true"]))
        interval = metrics["intervals"][h]
        low = np.maximum(0.0, group["y_pred"] + float(interval["p10"]))
        high = np.maximum(0.0, group["y_pred"] + float(interval["p90"]))
        coverage = float(np.mean((group["y_true"] >= low) & (group["y_true"] <= high)))
        coverage_values.append(coverage)
        print(
            f"{int(horizon):>8} {len(group):>6} {mae:>7.3f} {rmse:>7.3f} "
            f"{bias:>+7.3f} {coverage * 100:>8.1f}%"
        )
        seasonal = float(metrics["baseline_mae"][h])
        persistence = float(metrics["persistence_mae"][h])
        if mae >= seasonal:
            failures.append(f"+{h}w MAE {mae:.3f} is not below seasonal {seasonal:.3f}")
        if mae >= persistence:
            failures.append(f"+{h}w MAE {mae:.3f} is not below persistence {persistence:.3f}")
        if mae > MAX_MAE:
            failures.append(f"+{h}w MAE {mae:.3f} exceeds the {MAX_MAE:.2f} guardrail")
        if not MIN_COVERAGE <= coverage <= MAX_COVERAGE:
            failures.append(
                f"+{h}w 80% coverage {coverage:.1%} is outside "
                f"{MIN_COVERAGE:.0%}-{MAX_COVERAGE:.0%}"
            )

    overall_coverage = float(np.mean(coverage_values))
    print(
        f"\nOverall MAE {metrics['mae']:.4f} · RMSE {metrics['rmse']:.4f} · "
        f"mean interval coverage {overall_coverage:.1%}"
    )
    print(
        f"Baselines: seasonal {metrics['baseline_mae']['1']:.3f} · "
        f"persistence {metrics['persistence_mae']['1']:.3f} at +1 week"
    )

    if failures:
        print(f"\nFAIL — {len(failures)} accuracy check(s):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(
        "\nPASS — model beats both baselines at every horizon and the 80% "
        "intervals are calibrated (75-85% empirical coverage)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
