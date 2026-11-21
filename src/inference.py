"""Model loading, inference, risk tiers and local explanations.

The dashboard talks to the ML layer only through this module. The contract is
intentionally small:

    models/dengue_model.joblib      one fitted sklearn estimator / Pipeline
    models/feature_columns.json     ordered model input columns
    models/risk_thresholds.json     case-count thresholds for Low..Critical
    models/metrics.json             split, metrics, residual intervals
    data/processed/features.csv     ward/date/horizon rows + engineered features

Every function is local and CPU-only; there are no external API calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from . import config
from .data_access import format_date, load_features


class ModelNotReady(RuntimeError):
    """Raised when the serialized model or its features are unavailable."""


@dataclass(frozen=True)
class Prediction:
    ward_id: str
    as_of: pd.Timestamp
    target_date: pd.Timestamp
    horizon: int
    cases: float
    low: float
    high: float
    tier: str
    risk_index: float


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def _load_bundle_cached(
    model_path: str,
    model_mtime: float,
    columns_path: str,
    columns_mtime: float,
    thresholds_path: str,
    thresholds_mtime: float,
    metrics_path: str,
    metrics_mtime: float,
) -> dict:
    obj = joblib.load(model_path)

    # Support both a bare estimator and a dict bundle exported by the trainer.
    embedded_columns = None
    embedded_thresholds = None
    embedded_metrics = None
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        embedded_columns = obj.get("feature_columns")
        embedded_thresholds = obj.get("risk_thresholds")
        embedded_metrics = obj.get("metrics")
    else:
        model = obj

    columns = embedded_columns if embedded_columns is not None else _read_json(Path(columns_path))
    if columns is None:
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            columns = [str(c) for c in names]

    thresholds = (
        embedded_thresholds if embedded_thresholds is not None else _read_json(Path(thresholds_path))
    )
    metrics = embedded_metrics if embedded_metrics is not None else _read_json(Path(metrics_path))

    return {
        "model": model,
        "feature_columns": list(columns or []),
        "thresholds": thresholds or {},
        "metrics": metrics or {},
    }


def load_bundle() -> dict:
    """Load the serialized model and its metadata, or raise ``ModelNotReady``."""
    missing = [
        str(path)
        for path in (config.MODEL_PATH, config.FEATURE_COLUMNS_PATH)
        if not path.exists()
    ]
    if missing:
        raise ModelNotReady(
            "Model artifacts are not available yet. Waiting for: " + ", ".join(missing)
        )

    bundle = _load_bundle_cached(
        str(config.MODEL_PATH),
        _mtime(config.MODEL_PATH),
        str(config.FEATURE_COLUMNS_PATH),
        _mtime(config.FEATURE_COLUMNS_PATH),
        str(config.THRESHOLDS_PATH),
        _mtime(config.THRESHOLDS_PATH),
        str(config.METRICS_PATH),
        _mtime(config.METRICS_PATH),
    )
    if not bundle["feature_columns"]:
        raise ModelNotReady(
            "feature_columns.json is missing or empty. Ask the trainer to export "
            "the ordered list of model inputs."
        )
    return bundle


def artifact_status() -> dict:
    """Return a small status summary for the header / empty states."""
    checks = {
        "Model": config.MODEL_PATH,
        "Feature columns": config.FEATURE_COLUMNS_PATH,
        "Thresholds": config.THRESHOLDS_PATH,
        "Metrics": config.METRICS_PATH,
        "Feature table": config.FEATURES,
    }
    missing = [name for name, path in checks.items() if not path.exists()]
    return {
        "ready": not missing,
        "missing": missing,
        "model_path": str(config.MODEL_PATH),
        "features_path": str(config.FEATURES),
        "metrics": _read_json(config.METRICS_PATH) or {},
    }


def _features_or_raise() -> pd.DataFrame:
    features = load_features()
    if features is None:
        raise ModelNotReady(
            f"Engineered feature table not found at {config.FEATURES}. "
            "Ask the ML pipeline to export data/processed/features.csv."
        )
    required = {config.WARD_COL, config.DATE_COL}
    missing = required - set(features.columns)
    if missing:
        raise ModelNotReady(f"Feature table is missing columns: {sorted(missing)}")
    return features


def _horizon_column(features: pd.DataFrame) -> str | None:
    if config.HORIZON_COL in features.columns:
        return config.HORIZON_COL
    if "horizon" in features.columns:
        return "horizon"
    return None


def _normalise_thresholds(raw: Any) -> dict[str, dict[str, float]]:
    """Accept several reasonable JSON layouts and return ward -> thresholds."""
    if not isinstance(raw, dict):
        return {}

    if isinstance(raw.get("thresholds"), dict):
        raw = raw["thresholds"]

    def clean(value: dict) -> dict[str, float] | None:
        if not isinstance(value, dict):
            return None
        low = value.get("low_max", value.get("Low", value.get("low")))
        medium = value.get("medium_max", value.get("Medium", value.get("medium")))
        high = value.get("high_max", value.get("High", value.get("high")))
        if low is None and medium is None and high is None:
            return None
        low = float(low if low is not None else 0.0)
        medium = float(medium if medium is not None else max(low + 1.0, 1.0))
        high = float(high if high is not None else max(medium + 1.0, 2.0))
        medium = max(medium, low)
        high = max(high, medium)
        return {"low_max": low, "medium_max": medium, "high_max": high}

    # Global layout: {"low_max": ..., "medium_max": ..., "high_max": ...}
    global_clean = clean(raw)
    if global_clean is not None:
        return {"__default__": global_clean}

    # Per-ward layout.
    out: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        cleaned = clean(value) if isinstance(value, dict) else None
        if cleaned is not None:
            out[str(key)] = cleaned
    if "default" in out and "__default__" not in out:
        out["__default__"] = out.pop("default")
    return out


def _fallback_thresholds(ward_id: str, features: pd.DataFrame | None) -> dict[str, float]:
    """Derive sensible thresholds from the ward's training distribution."""
    if features is not None and config.TARGET_COL in features.columns:
        subset = features.loc[features[config.WARD_COL] == ward_id, config.TARGET_COL]
        subset = pd.to_numeric(subset, errors="coerce").dropna()
        if len(subset) >= 20:
            low = float(subset.quantile(0.50))
            medium = float(subset.quantile(0.75))
            high = float(subset.quantile(0.90))
            if high <= 0:
                high = max(3.0, medium + 1.0)
            if medium <= low:
                medium = low + 1.0
            if high <= medium:
                high = medium + 1.0
            return {"low_max": low, "medium_max": medium, "high_max": high}
    return {"low_max": 0.0, "medium_max": 3.0, "high_max": 8.0}


def thresholds_for(
    ward_id: str, raw: Any, features: pd.DataFrame | None = None
) -> dict[str, float]:
    normalised = _normalise_thresholds(raw)
    value = normalised.get(ward_id) or normalised.get("__default__")
    if value is None:
        return _fallback_thresholds(ward_id, features)
    return value


def tier_for(cases: float, thresholds: dict[str, float]) -> str:
    if cases <= thresholds["low_max"]:
        return "Low"
    if cases <= thresholds["medium_max"]:
        return "Medium"
    if cases <= thresholds["high_max"]:
        return "High"
    return "Critical"


def risk_index(cases: float, thresholds: dict[str, float]) -> float:
    """Map predicted cases to a 0-100 index using the ward's tier thresholds."""
    low = max(thresholds["low_max"], 0.0)
    medium = max(thresholds["medium_max"], low + 1e-6)
    high = max(thresholds["high_max"], medium + 1e-6)
    if cases <= 0:
        return 0.0
    if cases <= low:
        return 25.0 * cases / low if low > 0 else 0.0
    if cases <= medium:
        return 25.0 + 25.0 * (cases - low) / (medium - low)
    if cases <= high:
        return 50.0 + 25.0 * (cases - medium) / (high - medium)
    return min(100.0, 75.0 + 25.0 * (cases - high) / max(high, 1.0))


def _interval(
    cases: float, horizon: int, metrics: dict
) -> tuple[float, float]:
    """Return an empirical prediction interval, falling back to RMSE."""
    raw = (
        metrics.get("intervals")
        or metrics.get("residual_quantiles")
        or metrics.get("prediction_intervals")
    )
    if isinstance(raw, dict):
        entry = raw.get(str(horizon), raw.get(horizon))
        if isinstance(entry, dict):
            low_res = entry.get("p10", entry.get("low", entry.get("q10")))
            high_res = entry.get("p90", entry.get("high", entry.get("q90")))
            if low_res is not None and high_res is not None:
                return max(0.0, cases + float(low_res)), max(0.0, cases + float(high_res))

    rmse = metrics.get("rmse") or metrics.get("test_rmse")
    if isinstance(rmse, dict):
        rmse = rmse.get(str(horizon))
    try:
        rmse_value = float(rmse)
    except (TypeError, ValueError):
        rmse_value = 0.0
    if rmse_value > 0:
        return max(0.0, cases - 1.28 * rmse_value), cases + 1.28 * rmse_value

    spread = max(1.0, cases * 0.25)
    return max(0.0, cases - spread), cases + spread


def _lookup_row(
    features: pd.DataFrame,
    ward_id: str,
    as_of: pd.Timestamp,
    horizon: int,
) -> pd.Series:
    as_of = pd.Timestamp(as_of).normalize()
    mask = (features[config.WARD_COL] == ward_id) & (
        features[config.DATE_COL] == as_of
    )
    horizon_col = _horizon_column(features)
    if horizon_col is not None:
        mask &= features[horizon_col] == horizon
    subset = features.loc[mask]
    if subset.empty:
        raise LookupError(
            f"No feature row for {ward_id} on {as_of.date()} at horizon {horizon}. "
            "Check that the feature table covers this date and horizon."
        )
    return subset.iloc[-1]


def _predict_cases(model: Any, x: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict(x)).reshape(-1)
    return np.clip(values.astype(float), 0.0, None)


def predict(
    ward_id: str,
    as_of: pd.Timestamp,
    horizon: int,
    *,
    features: pd.DataFrame | None = None,
    bundle: dict | None = None,
) -> Prediction:
    bundle = bundle or load_bundle()
    features = features if features is not None else _features_or_raise()
    as_of = pd.Timestamp(as_of).normalize()

    row = _lookup_row(features, ward_id, as_of, horizon)
    columns = bundle["feature_columns"]
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ModelNotReady(f"Feature table is missing model inputs: {missing}")

    x = row[columns].to_frame().T
    cases = float(_predict_cases(bundle["model"], x)[0])
    thresholds = thresholds_for(ward_id, bundle["thresholds"], features)
    tier = tier_for(cases, thresholds)
    low, high = _interval(cases, horizon, bundle["metrics"])

    return Prediction(
        ward_id=ward_id,
        as_of=as_of,
        target_date=as_of + pd.Timedelta(weeks=int(horizon)),
        horizon=int(horizon),
        cases=cases,
        low=low,
        high=high,
        tier=tier,
        risk_index=risk_index(cases, thresholds),
    )


def predict_batch(
    ward_ids: list[str],
    as_of: pd.Timestamp,
    horizons: list[int],
    *,
    features: pd.DataFrame | None = None,
    bundle: dict | None = None,
) -> pd.DataFrame:
    """Vectorized prediction for the ward ranking and heatmap."""
    bundle = bundle or load_bundle()
    features = features if features is not None else _features_or_raise()
    as_of = pd.Timestamp(as_of).normalize()
    columns = bundle["feature_columns"]
    horizon_col = _horizon_column(features)

    keys = pd.DataFrame(
        [
            {config.WARD_COL: ward, config.DATE_COL: as_of, "_horizon": int(h)}
            for ward in ward_ids
            for h in horizons
        ]
    )
    if horizon_col is not None:
        keys[horizon_col] = keys["_horizon"]

    mask = features[config.WARD_COL].isin(ward_ids) & (
        features[config.DATE_COL] == as_of
    )
    if horizon_col is not None:
        mask &= features[horizon_col].isin(horizons)
    pool = features.loc[mask].drop_duplicates(
        subset=[config.WARD_COL, config.DATE_COL]
        + ([horizon_col] if horizon_col is not None else []),
        keep="last",
    )

    join_cols = [config.WARD_COL, config.DATE_COL]
    if horizon_col is not None:
        join_cols.append(horizon_col)
    merged = keys.merge(pool, on=join_cols, how="left")

    valid = merged[columns].notna().all(axis=1) if columns else pd.Series(False, index=merged.index)
    result = pd.DataFrame(
        {
            config.WARD_COL: merged[config.WARD_COL].to_numpy(),
            "horizon": merged["_horizon"].astype(int).to_numpy(),
        }
    )
    result["target_date"] = as_of + pd.to_timedelta(result["horizon"], unit="W")
    result["cases"] = np.nan
    result["tier"] = "—"
    result["risk_index"] = np.nan

    if valid.any():
        preds = _predict_cases(bundle["model"], merged.loc[valid, columns])
        result.loc[valid, "cases"] = preds
        for idx in result.index[valid]:
            ward = result.at[idx, config.WARD_COL]
            cases = float(result.at[idx, "cases"])
            thresholds = thresholds_for(ward, bundle["thresholds"], features)
            result.at[idx, "tier"] = tier_for(cases, thresholds)
            result.at[idx, "risk_index"] = risk_index(cases, thresholds)
    return result


def _final_estimator(model: Any) -> Any:
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def global_importance(bundle: dict) -> pd.Series:
    """Return global feature importance when the estimator exposes it."""
    columns = bundle["feature_columns"]
    estimator = _final_estimator(bundle["model"])
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).reshape(-1))
    if values is None or len(values) != len(columns):
        return pd.Series(1.0, index=columns, dtype=float)
    return pd.Series(np.asarray(values, dtype=float), index=columns)


def explain(
    ward_id: str,
    as_of: pd.Timestamp,
    horizon: int,
    *,
    features: pd.DataFrame | None = None,
    bundle: dict | None = None,
    top_k: int = 8,
) -> pd.DataFrame:
    """Local, model-agnostic driver breakdown via single-feature occlusion.

    Each feature is reset to its typical value (the training median) and the
    prediction is recomputed. ``delta`` is how many cases that driver adds
    relative to typical conditions. This works for any sklearn estimator and
    needs no SHAP dependency.
    """
    bundle = bundle or load_bundle()
    features = features if features is not None else _features_or_raise()
    columns = bundle["feature_columns"]
    row = _lookup_row(features, ward_id, as_of, horizon)
    base_x = row[columns].to_frame().T
    base_pred = float(_predict_cases(bundle["model"], base_x)[0])

    candidates = [
        col
        for col in columns
        if col not in {config.HORIZON_COL, "horizon"}
        and pd.api.types.is_numeric_dtype(features[col])
    ]
    medians = {
        col: float(pd.to_numeric(features[col], errors="coerce").median())
        for col in candidates
        if pd.to_numeric(features[col], errors="coerce").notna().any()
    }

    rows = []
    for col, median in medians.items():
        if pd.isna(row[col]) or not np.isfinite(float(row[col])):
            continue
        perturbed = base_x.copy()
        perturbed[col] = median
        pred = float(_predict_cases(bundle["model"], perturbed)[0])
        rows.append(
            {
                "feature": col,
                "label": config.pretty_feature(col),
                "value": float(row[col]),
                "reference": median,
                "delta": base_pred - pred,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "label", "value", "reference", "delta"])

    result = pd.DataFrame(rows)
    result = result.sort_values("delta", key=lambda s: s.abs(), ascending=False).head(top_k)
    return result.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_test_predictions_cached(path_str: str, mtime: float) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if config.DATE_COL in df.columns:
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")
    return df


def load_test_predictions() -> pd.DataFrame | None:
    return _load_test_predictions_cached(
        str(config.TEST_PREDICTIONS_PATH), _mtime(config.TEST_PREDICTIONS_PATH)
    )


def split_summary(metrics: dict) -> pd.DataFrame:
    """Best-effort rendering of the chronological split from metrics.json."""
    split = metrics.get("split") if isinstance(metrics, dict) else None
    if isinstance(split, dict):
        rows = []
        for name, value in split.items():
            if isinstance(value, dict):
                rows.append(
                    {
                        "Split": str(name).title(),
                        "Start": value.get("start", "—"),
                        "End": value.get("end", "—"),
                        "Weeks": value.get("weeks", "—"),
                    }
                )
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {"Split": "Train", "Start": "2016-01-04", "End": "2022-12-31", "Weeks": 365},
            {"Split": "Validation", "Start": "2023-01-02", "End": "2023-12-25", "Weeks": 52},
            {"Split": "Test", "Start": "2024-01-01", "End": "2025-12-29", "Weeks": 105},
        ]
    )


def model_summary(bundle: dict) -> dict:
    metrics = bundle.get("metrics") or {}
    estimator = _final_estimator(bundle["model"])
    return {
        "name": metrics.get("model_name") or type(estimator).__name__,
        "trained_at": metrics.get("trained_at", "—"),
        "n_features": len(bundle.get("feature_columns") or []),
        "metrics": metrics,
    }
