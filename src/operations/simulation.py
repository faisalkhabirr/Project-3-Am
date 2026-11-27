"""Model-driven "what-if" vector-control simulator.

This is a counterfactual, not a percentage toggle. For every simulated week we:

1. take the real feature row for that ward/week,
2. rewrite the case-history features with the arm's simulated history,
3. apply the intervention's feature-level effect (larvicide -> breeding-habitat
   features; fogging -> transmission pressure / case history),
4. re-run the **serialized model**,
5. feed the prediction back into next week's case lags.

The two arms therefore diverge through the same model the dashboard uses. The
efficacy curves below are conservative planning assumptions informed by
vector-control practice (fogging acts quickly with a short residual; larvicide
builds over 2-3 weeks and lasts longer). They are deliberately exposed as
constants so the assumptions can be challenged in Q&A.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .supplies import cases_to_supplies

# Fractional reduction in the relevant feature block at full crew strength.
# Week 1..4; the last value is reused for longer simulations. Conservative
# planning assumptions: fogging acts quickly but has a short residual, while
# larvicide builds over 2-3 weeks and lasts longer.
LARVICIDE_CURVE = {1: 0.10, 2: 0.20, 3: 0.28, 4: 0.32}
FOGGING_CURVE = {1: 0.15, 2: 0.22, 3: 0.15, 4: 0.08}

BREEDING_FEATURES = (
    "rain_lag_1w",
    "rain_lag_2w",
    "rain_lag_3w",
    "rain_lag_4w",
    "rain_cum_2w",
    "rain_cum_4w",
)
CASE_FEATURES = (
    "cases_current",
    "cases_lag_1w",
    "cases_lag_2w",
    "cases_lag_3w",
    "cases_lag_4w",
)


@dataclass(frozen=True)
class Deployment:
    ward_id: str
    crew_type: str  # 'larvicide' | 'fogging'
    crews: int
    start_week: int = 1

    def __post_init__(self) -> None:
        if self.crew_type not in {"larvicide", "fogging"}:
            raise ValueError("crew_type must be 'larvicide' or 'fogging'")
        if int(self.crews) < 0:
            raise ValueError("crews must be non-negative")
        if int(self.start_week) < 1:
            raise ValueError("start_week must be >= 1")


@dataclass
class SimulationResult:
    per_ward_week: pd.DataFrame
    roi_ranking: pd.DataFrame


def _curve(crew_type: str, week: int) -> float:
    curve = LARVICIDE_CURVE if crew_type == "larvicide" else FOGGING_CURVE
    return float(curve.get(min(max(1, int(week)), max(curve)), 0.0))


def _crew_factor(crews: int) -> float:
    """Diminishing returns: zero crews do nothing, one delivers 75%, two 100%."""
    if int(crews) <= 0:
        return 0.0
    return min(1.0, 0.5 + 0.25 * int(crews))


def _effective(crew_type: str, crews: int, week: int) -> float:
    return min(1.0, _curve(crew_type, week) * _crew_factor(crews))


def _horizon_column(features: pd.DataFrame) -> str | None:
    for candidate in ("horizon_weeks", "horizon"):
        if candidate in features.columns:
            return candidate
    return None


def _row_for_week(
    features: pd.DataFrame, ward_id: str, when: pd.Timestamp, fallback_horizon: int, as_of: pd.Timestamp
) -> pd.Series | None:
    mask = (features["ward_id"].astype(str) == str(ward_id)) & (
        pd.to_datetime(features["date"]) == pd.Timestamp(when)
    )
    horizon_col = _horizon_column(features)
    if horizon_col is not None:
        subset = features.loc[mask & (features[horizon_col] == 1)]
        if subset.empty:
            subset = features.loc[
                (features["ward_id"].astype(str) == str(ward_id))
                & (pd.to_datetime(features["date"]) == pd.Timestamp(as_of))
                & (features[horizon_col] == fallback_horizon)
            ]
    else:
        subset = features.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[-1].copy()


def _num(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _set_history(row: pd.Series, history: Iterable[float]) -> None:
    """Write the arm's simulated case history into the model's case features."""
    values = list(history)
    if "cases_current" in row.index and values:
        row["cases_current"] = _num(values[-1])
    for lag in range(1, 5):
        col = f"cases_lag_{lag}w"
        if col in row.index:
            row[col] = _num(values[-lag]) if len(values) >= lag else 0.0


def _apply_effect(
    row: pd.Series,
    crew_type: str,
    effect: float,
    history: list[float],
) -> None:
    keep = 1.0 - float(effect)
    if crew_type == "larvicide":
        for col in BREEDING_FEATURES:
            if col in row.index and pd.notna(row[col]):
                row[col] = float(row[col]) * keep
        if "water_logging_risk_index" in row.index and pd.notna(row["water_logging_risk_index"]):
            row["water_logging_risk_index"] = float(row["water_logging_risk_index"]) * (
                1.0 - 0.5 * float(effect)
            )
    else:
        # Fogging kills adult vectors. The model has no explicit adult-mosquito
        # index, so we scale the case-history features (infectious reservoir)
        # and the waterlogging/vector-pressure proxy. This is a documented
        # structural assumption, not a claim that fogging changes rainfall.
        _set_history(row, [value * keep for value in history])
        if "water_logging_risk_index" in row.index and pd.notna(row["water_logging_risk_index"]):
            row["water_logging_risk_index"] = float(row["water_logging_risk_index"]) * keep


def _predict_row(bundle: dict, row: pd.Series) -> float:
    columns = bundle["feature_columns"]
    missing = [col for col in columns if col not in row.index]
    if missing:
        raise ValueError(f"feature row is missing model inputs: {missing}")
    frame = row[columns].to_frame().T.astype(float)
    values = np.asarray(bundle["model"].predict(frame)).reshape(-1)
    return float(np.clip(values.astype(float), 0.0, None)[0])


def _forecast_arm(
    *,
    features: pd.DataFrame,
    bundle: dict,
    ward_id: str,
    as_of: pd.Timestamp,
    weeks: int,
    deployments: list[Deployment],
    intervention: bool,
    predict_fn: Callable[[dict, pd.Series], float],
) -> list[dict]:
    first = _row_for_week(features, ward_id, as_of, 1, as_of)
    if first is None:
        raise LookupError(f"no feature row for {ward_id} at {as_of.date()}")
    history = [
        _num(first.get("cases_current", 0.0)),
        _num(first.get("cases_lag_1w", 0.0)),
        _num(first.get("cases_lag_2w", 0.0)),
        _num(first.get("cases_lag_3w", 0.0)),
        _num(first.get("cases_lag_4w", 0.0)),
    ]
    rows: list[dict] = []
    for week in range(1, int(weeks) + 1):
        when = pd.Timestamp(as_of) + pd.Timedelta(weeks=week - 1)
        row = _row_for_week(features, ward_id, when, week, as_of)
        if row is None:
            break
        _set_history(row, history)
        if intervention:
            for dep in deployments:
                if int(dep.start_week) <= week:
                    effect = _effective(dep.crew_type, dep.crews, week - dep.start_week + 1)
                    if effect > 0:
                        _apply_effect(row, dep.crew_type, effect, history)
        cases = predict_fn(bundle, row)
        supplies = cases_to_supplies(cases)
        rows.append(
            {
                "ward_id": ward_id,
                "week": week,
                "target_date": when + pd.Timedelta(weeks=1),
                "cases": cases,
                "saline": supplies["saline_bags_needed"],
                "kits": supplies["diagnostic_kits_needed"],
                "beds": supplies["hospital_beds_required"],
            }
        )
        history.append(cases)
        history = history[-5:]
    return rows


def simulate(
    deployments: list[Deployment],
    weeks: int = 4,
    *,
    as_of: pd.Timestamp | str | None = None,
    features: pd.DataFrame | None = None,
    bundle: dict | None = None,
    predict_fn: Callable[[dict, pd.Series], float] | None = None,
) -> SimulationResult:
    """Compare baseline vs intervention for the deployed wards.

    ``as_of`` is the week the decision is made. The simulator uses the bundled
    feature table's future weather (a historical hindcast) so the counterfactual
    is reproducible offline. Pass explicit ``features``/``bundle`` in tests.
    """
    if not 1 <= int(weeks) <= 12:
        raise ValueError("weeks must be between 1 and 12")
    if features is None or bundle is None:
        from .. import data_access, inference

        features = features if features is not None else data_access.load_features()
        bundle = bundle if bundle is not None else inference.load_bundle()
    if features is None or bundle is None:
        raise RuntimeError("feature table and model bundle are required")
    if as_of is None:
        as_of = pd.to_datetime(features["date"]).max()
    as_of = pd.Timestamp(as_of).normalize()

    by_ward: dict[str, list[Deployment]] = {}
    for dep in deployments:
        by_ward.setdefault(str(dep.ward_id), []).append(dep)

    predict = predict_fn or _predict_row
    rows: list[dict] = []
    for ward_id, ward_deps in by_ward.items():
        base = _forecast_arm(
            features=features,
            bundle=bundle,
            ward_id=ward_id,
            as_of=as_of,
            weeks=int(weeks),
            deployments=[],
            intervention=False,
            predict_fn=predict,
        )
        inter = _forecast_arm(
            features=features,
            bundle=bundle,
            ward_id=ward_id,
            as_of=as_of,
            weeks=int(weeks),
            deployments=ward_deps,
            intervention=True,
            predict_fn=predict,
        )
        # If the intervention arm used a custom predictor, rebind it for both
        # arms. The default path is already model-driven; this hook is for tests.
        base_map = {row["week"]: row for row in base}
        for row in inter:
            base_row = base_map.get(row["week"])
            if base_row is None:
                continue
            rows.append(
                {
                    "ward_id": ward_id,
                    "week": row["week"],
                    "target_date": row["target_date"],
                    "baseline_cases": base_row["cases"],
                    "intervention_cases": row["cases"],
                    "cases_prevented": max(0.0, base_row["cases"] - row["cases"]),
                    "baseline_beds": base_row["beds"],
                    "intervention_beds": row["beds"],
                    "baseline_saline": base_row["saline"],
                    "intervention_saline": row["saline"],
                }
            )

    per_ward_week = pd.DataFrame(rows)
    if per_ward_week.empty:
        return SimulationResult(
            per_ward_week=per_ward_week,
            roi_ranking=pd.DataFrame(
                columns=["ward_id", "crews", "cases_prevented", "cases_per_crew"]
            ),
        )

    totals = (
        per_ward_week.groupby("ward_id", as_index=False)
        .agg(
            cases_prevented=("cases_prevented", "sum"),
            baseline_cases=("baseline_cases", "sum"),
            intervention_cases=("intervention_cases", "sum"),
        )
        .assign(
            crews=lambda frame: frame["ward_id"].map(
                {ward: sum(d.crews for d in deps) for ward, deps in by_ward.items()}
            )
        )
    )
    totals["cases_per_crew"] = np.where(
        totals["crews"] > 0, totals["cases_prevented"] / totals["crews"], 0.0
    )
    totals["reduction_pct"] = np.where(
        totals["baseline_cases"] > 0,
        totals["cases_prevented"] / totals["baseline_cases"],
        0.0,
    )
    roi = totals.sort_values(
        ["cases_per_crew", "cases_prevented"], ascending=False
    ).reset_index(drop=True)
    return SimulationResult(per_ward_week=per_ward_week, roi_ranking=roi)
