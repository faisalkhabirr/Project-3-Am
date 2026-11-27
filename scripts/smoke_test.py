#!/usr/bin/env python3
"""End-to-end smoke test for the bundled model and feature table.

Run after the ML handoff:

    python scripts/smoke_test.py

It loads the serialized artifact, predicts one ward/horizon, builds the driver
breakdown, and batch-predicts a city heatmap. Exit code 0 means the app's
inference path is healthy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, inference
from src import data_access as data


def main() -> int:
    print("Artifacts:")
    status = inference.artifact_status()
    for name, path in {
        "model": config.MODEL_PATH,
        "feature columns": config.FEATURE_COLUMNS_PATH,
        "thresholds": config.THRESHOLDS_PATH,
        "metrics": config.METRICS_PATH,
        "feature table": config.FEATURES,
        "test predictions": config.TEST_PREDICTIONS_PATH,
    }.items():
        print(f"  {'OK ' if path.exists() else 'MISSING'} {name}: {path}")
    if not status["ready"]:
        print("\nCannot run the smoke test until the artifacts exist.", file=sys.stderr)
        return 1

    data.load_raw()
    meta = data.load_ward_meta()
    features = data.load_features()
    bundle = inference.load_bundle()
    if features is None:
        print("Feature table is missing.", file=sys.stderr)
        return 1

    ward = str(features[config.WARD_COL].iloc[0])
    dates = sorted(pd.to_datetime(features.loc[features[config.WARD_COL] == ward, config.DATE_COL]))
    as_of = pd.Timestamp(dates[-5] if len(dates) > 5 else dates[-1])
    horizon = 3

    print(f"\nPredicting {ward} as of {as_of.date()} at +{horizon} weeks")
    pred = inference.predict(ward, as_of, horizon, features=features, bundle=bundle)
    print(
        f"  cases={pred.cases:.2f} tier={pred.tier} "
        f"interval=({pred.low:.2f}, {pred.high:.2f}) index={pred.risk_index:.1f}"
    )
    if not (pred.low <= pred.cases <= pred.high or abs(pred.high - pred.low) < 1e-9):
        print("  WARN: point prediction is outside its interval", file=sys.stderr)

    drivers = inference.explain(ward, as_of, horizon, features=features, bundle=bundle)
    print(f"  drivers: {len(drivers)} features")
    if not drivers.empty:
        print(drivers[["label", "value", "delta"]].head(3).to_string(index=False))

    wards = data.wards_for_city(meta, str(meta.loc[meta[config.WARD_COL] == ward, config.CITY_COL].iloc[0]))
    batch = inference.predict_batch(wards, as_of, [1, 2, 3, 4], features=features, bundle=bundle)
    print(f"  batch rows: {len(batch)}, non-null predictions: {int(batch['cases'].notna().sum())}")

    test = inference.load_test_predictions()
    if test is not None:
        print(f"  holdout rows: {len(test):,}")
        mae = float((test["y_true"] - test["y_pred"]).abs().mean())
        print(f"  holdout MAE (all horizons): {mae:.3f}")

    print("\nPASS — inference path is healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
