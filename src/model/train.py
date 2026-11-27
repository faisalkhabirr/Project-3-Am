"""Train the dengue risk model and export the dashboard artifacts.

Usage:
    python -m src.data.build_features
    python -m src.model.train

The split is chronological and applied to the **target date**:

    train      target_date <= 2022-12-31
    validation target_date in 2023
    test       target_date >= 2024-01-01

Validation residuals become the 80% prediction intervals; the final model is
refit on train + validation before scoring the untouched test window.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data.build_features import (
    DATE,
    HORIZON,
    MODEL_FEATURES,
    ROOT,
    TARGET,
    WARD,
)

DEFAULT_PARAMS = dict(
    loss="poisson",
    learning_rate=0.05,
    max_iter=350,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42,
)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def chronological_split(
    features: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {WARD, DATE, HORIZON, "target_date", "target_cases"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")

    known = features.loc[features["target_cases"].notna()].copy()
    target_date = pd.to_datetime(known["target_date"])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = known.loc[target_date <= train_end_ts].copy()
    val = known.loc[(target_date > train_end_ts) & (target_date <= val_end_ts)].copy()
    test = known.loc[target_date > val_end_ts].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One of the chronological splits is empty; check the date bounds.")
    return train, val, test


def ward_thresholds(train: pd.DataFrame) -> dict:
    thresholds: dict[str, dict[str, float]] = {}
    for ward, group in train.groupby(WARD):
        y = pd.to_numeric(group["target_cases"], errors="coerce").dropna()
        low = float(y.quantile(0.50))
        medium = float(y.quantile(0.75))
        high = float(y.quantile(0.90))
        if medium <= low:
            medium = low + 1.0
        if high <= medium:
            high = medium + 1.0
        thresholds[str(ward)] = {"low_max": low, "medium_max": medium, "high_max": high}
    thresholds["__default__"] = {"low_max": 0.0, "medium_max": 3.0, "high_max": 8.0}
    return {"mode": "per_ward", "thresholds": thresholds}


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    def summary(frame: pd.DataFrame) -> dict:
        target = pd.to_datetime(frame["target_date"])
        return {
            "start": str(target.min().date()),
            "end": str(target.max().date()),
            "weeks": int(target.nunique()),
        }

    return {
        "train": summary(train),
        "validation": summary(val),
        "test": summary(test),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "features.csv",
        help="engineered feature table",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "dengue_onsite.csv",
        help="raw CSV for the seasonal-naive baseline (onsite dataset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models",
        help="artifact directory",
    )
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.features.exists():
        parser.error(
            f"feature table not found: {args.features}. "
            "Run `python -m src.data.build_features` first."
        )
    features = pd.read_csv(args.features, parse_dates=[DATE, "target_date"])
    train, val, test = chronological_split(features, args.train_end, args.val_end)
    print(f"rows: train={len(train):,} val={len(val):,} test={len(test):,}")

    params = dict(DEFAULT_PARAMS, random_state=args.seed)
    validation_model = HistGradientBoostingRegressor(**params)
    validation_model.fit(train[MODEL_FEATURES], train["target_cases"])
    val_pred = np.clip(validation_model.predict(val[MODEL_FEATURES]), 0, None)

    intervals: dict[str, dict[str, float]] = {}
    for horizon, group in val.assign(pred=val_pred).groupby(HORIZON):
        residual = group["target_cases"] - group["pred"]
        intervals[str(int(horizon))] = {
            "p10": float(residual.quantile(0.10)),
            "p90": float(residual.quantile(0.90)),
        }
        print(
            f"  val horizon +{int(horizon)}w: "
            f"MAE={mae(group['target_cases'], group['pred']):.3f}"
        )

    trainval = pd.concat([train, val], ignore_index=True)
    model = HistGradientBoostingRegressor(**params)
    model.fit(trainval[MODEL_FEATURES], trainval["target_cases"])
    test = test.copy()
    test["y_true"] = test["target_cases"]
    test["y_pred"] = np.clip(model.predict(test[MODEL_FEATURES]), 0, None)

    # Baselines
    test["persistence_pred"] = test["cases_current"].fillna(0.0)
    if args.raw.exists():
        raw = pd.read_csv(args.raw, parse_dates=[DATE])
        base = test[[WARD, "target_date"]].copy()
        base["base_date"] = base["target_date"] - pd.Timedelta(weeks=52)
        lookup = raw[[WARD, DATE, TARGET]].rename(
            columns={DATE: "base_date", TARGET: "seasonal_pred"}
        )
        seasonal = base.merge(lookup, on=[WARD, "base_date"], how="left")[
            "seasonal_pred"
        ].fillna(0.0)
        test["seasonal_pred"] = seasonal.to_numpy()
    else:
        test["seasonal_pred"] = 0.0

    metrics = {
        "model_name": "HistGradientBoostingRegressor (Poisson)",
        "trained_at": date.today().isoformat(),
        "features": MODEL_FEATURES,
        "params": params,
        "mae": mae(test["y_true"], test["y_pred"]),
        "rmse": rmse(test["y_true"], test["y_pred"]),
        "baseline_mae": {},
        "persistence_mae": {},
        "intervals": intervals,
        "split": split_summary(train, val, test),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
    }
    for horizon, group in test.groupby(HORIZON):
        metrics["baseline_mae"][str(int(horizon))] = mae(
            group["y_true"], group["seasonal_pred"]
        )
        metrics["persistence_mae"][str(int(horizon))] = mae(
            group["y_true"], group["persistence_pred"]
        )

    thresholds = ward_thresholds(train)

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "dengue_model.joblib")
    (args.out / "feature_columns.json").write_text(
        json.dumps(MODEL_FEATURES, indent=2), encoding="utf-8"
    )
    (args.out / "risk_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    test[
        [DATE, WARD, "target_date", HORIZON, "y_true", "y_pred"]
    ].to_csv(args.out / "test_predictions.csv", index=False)

    print(
        f"test MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
        f"(seasonal {mae(test['y_true'], test['seasonal_pred']):.3f}, "
        f"persistence {mae(test['y_true'], test['persistence_pred']):.3f})"
    )
    print(f"artifacts written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
