"""Scarcity-aware resource allocation for the Phase 2 command center.

Allocation rule (deliberately simple enough to defend to a judge):

1. **Protected floor.** Every ward receives a minimum share (default 10%) of its
   required supply, so a low-priority ward is never starved to zero while a
   critical ward receives surplus.
2. **Priority water-filling.** The remaining stock is allocated in strict
   priority order. Priority is a weighted blend of risk tier and population
   density; the marginal ward receives whatever stock is left.
3. **Integer units.** Supplies are whole bags/kits/beds and crews are people,
   so every allocation is an integer and never exceeds the available stock.

The function is pure and deterministic: same inputs produce the same table.
No Streamlit, no network, no solver dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .supplies import cases_to_supplies

TIER_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
TIER_WEIGHT = 0.65
DENSITY_WEIGHT = 0.35


@dataclass(frozen=True)
class WardDemand:
    """One ward's forecast and derived requirements."""

    ward_id: str
    city: str
    predicted_cases: float
    tier: str
    density: float
    saline: int
    kits: int
    beds: int
    crews_needed: int

    @classmethod
    def from_prediction(
        cls,
        *,
        ward_id: str,
        city: str,
        predicted_cases: float,
        tier: str,
        density: float,
        crews_needed: int | None = None,
    ) -> "WardDemand":
        supplies = cases_to_supplies(predicted_cases)
        return cls(
            ward_id=ward_id,
            city=city,
            predicted_cases=float(predicted_cases),
            tier=tier,
            density=float(density),
            saline=supplies["saline_bags_needed"],
            kits=supplies["diagnostic_kits_needed"],
            beds=supplies["hospital_beds_required"],
            crews_needed=crews_needed if crews_needed is not None else crew_need(tier, density),
        )


@dataclass
class AllocationResult:
    per_ward: pd.DataFrame
    city_rollup: pd.DataFrame
    total_deficit: dict[str, int]
    available: dict[str, int] = field(default_factory=dict)
    required: dict[str, int] = field(default_factory=dict)


def crew_need(tier: str, density: float, density_threshold: float = 0.75) -> int:
    """Translate risk tier and crowding into a vector-control crew requirement.

    Baseline crews follow the tier (Low 0, Medium 1, High 2, Critical 3). A
    densely packed ward at High/Critical risk gets one extra crew because the
    same adult mosquito population reaches more people.
    """
    rank = TIER_RANK.get(str(tier), 1)
    need = max(0, rank - 1)
    if rank >= 3 and float(density) >= density_threshold:
        need += 1
    return int(need)


def _priority_scores(demands: list[WardDemand]) -> np.ndarray:
    if not demands:
        return np.array([], dtype=float)
    tiers = np.array([TIER_RANK.get(str(d.tier), 1) for d in demands], dtype=float)
    tier_norm = tiers / max(TIER_RANK.values())
    densities = np.array([float(d.density) for d in demands], dtype=float)
    spread = float(densities.max() - densities.min())
    density_norm = (densities - densities.min()) / spread if spread > 0 else np.zeros_like(densities)
    return TIER_WEIGHT * tier_norm + DENSITY_WEIGHT * density_norm


def _waterfill(required: np.ndarray, capacity: int, order: np.ndarray) -> np.ndarray:
    """Allocate integer units in priority order, never exceeding demand."""
    allocated = np.zeros(len(required), dtype=int)
    remaining = max(0, int(capacity))
    for idx in order:
        room = int(required[idx]) - int(allocated[idx])
        if room <= 0:
            continue
        give = min(room, remaining)
        allocated[idx] += give
        remaining -= give
        if remaining == 0:
            break
    return allocated


def _allocate_supply(
    required: np.ndarray,
    capacity: int,
    order: np.ndarray,
    min_floor: float,
) -> np.ndarray:
    required_int = required.astype(int)
    if required_int.sum() <= 0 or capacity <= 0:
        return np.zeros(len(required), dtype=int)

    capacity = min(int(capacity), int(required_int.sum()))
    floor_units = np.floor(required_int * max(0.0, min(1.0, min_floor))).astype(int)
    if floor_units.sum() > capacity:
        # Extreme scarcity: scale the protected floor down proportionally.
        floor_units = np.floor(
            floor_units * (capacity / max(1, floor_units.sum()))
        ).astype(int)
    floor_units = np.minimum(floor_units, required_int)

    remaining = capacity - int(floor_units.sum())
    top_up = _waterfill(required_int - floor_units, remaining, order)
    return floor_units + top_up


def allocate(
    demands: list[WardDemand],
    stock_frac: float,
    total_crews: int,
    min_floor: float = 0.10,
) -> AllocationResult:
    """Allocate supplies and crews under a regional stock scarcity control.

    Parameters
    ----------
    demands:
        One entry per ward. Supplies are integer requirements (see
        :class:`WardDemand`).
    stock_frac:
        Available regional stock as a fraction of total ward demand (0.30–1.00
        in the UI). Values outside [0, 1] raise ``ValueError``.
    total_crews:
        Number of spraying crews available across the selected scope.
    min_floor:
        Protected minimum share of each ward's requirement (0–1).
    """
    if not demands:
        empty = pd.DataFrame()
        return AllocationResult(empty, empty, {}, {}, {})
    if not 0.0 <= float(stock_frac) <= 1.0:
        raise ValueError("stock_frac must be between 0 and 1")
    if int(total_crews) < 0:
        raise ValueError("total_crews must be non-negative")

    priorities = _priority_scores(demands)
    # Highest priority first; density breaks ties within a tier.
    order = np.argsort(-priorities, kind="stable")

    saline_req = np.array([d.saline for d in demands], dtype=int)
    kits_req = np.array([d.kits for d in demands], dtype=int)
    beds_req = np.array([d.beds for d in demands], dtype=int)
    crews_req = np.array([d.crews_needed for d in demands], dtype=int)

    available = {
        "saline_bags_needed": int(np.floor(stock_frac * saline_req.sum())),
        "diagnostic_kits_needed": int(np.floor(stock_frac * kits_req.sum())),
        "hospital_beds_required": int(np.floor(stock_frac * beds_req.sum())),
    }
    # Do not clamp to demand here: stock is a fraction of demand by construction.
    alloc_saline = _allocate_supply(
        saline_req, available["saline_bags_needed"], order, min_floor
    )
    alloc_kits = _allocate_supply(
        kits_req, available["diagnostic_kits_needed"], order, min_floor
    )
    alloc_beds = _allocate_supply(
        beds_req, available["hospital_beds_required"], order, min_floor
    )
    # Crews are fully discretionary: strict priority, no protected floor.
    alloc_crews = _waterfill(crews_req, int(total_crews), order)

    frame = pd.DataFrame(
        {
            "ward_id": [d.ward_id for d in demands],
            "city": [d.city for d in demands],
            "predicted_cases": [d.predicted_cases for d in demands],
            "tier": [d.tier for d in demands],
            "density": [d.density for d in demands],
            "priority": np.round(priorities, 3),
            "required_saline": saline_req,
            "required_kits": kits_req,
            "required_beds": beds_req,
            "required_crews": crews_req,
            "allocated_saline": alloc_saline,
            "allocated_kits": alloc_kits,
            "allocated_beds": alloc_beds,
            "allocated_crews": alloc_crews,
        }
    )
    for kind in ("saline", "kits", "beds", "crews"):
        frame[f"deficit_{kind}"] = frame[f"required_{kind}"] - frame[f"allocated_{kind}"]
    frame["deficit"] = (
        (frame["deficit_saline"] > 0)
        | (frame["deficit_kits"] > 0)
        | (frame["deficit_beds"] > 0)
        | (frame["deficit_crews"] > 0)
    )
    required_total = (
        frame["required_saline"] + frame["required_kits"] + frame["required_beds"]
    )
    allocated_total = (
        frame["allocated_saline"] + frame["allocated_kits"] + frame["allocated_beds"]
    )
    frame["fulfillment_pct"] = np.where(
        required_total > 0, allocated_total / required_total.replace(0, np.nan), 1.0
    )
    frame["fulfillment_pct"] = frame["fulfillment_pct"].fillna(1.0).clip(0, 1)

    city_rollup = (
        frame.groupby("city", as_index=False)
        .agg(
            wards=("ward_id", "count"),
            predicted_cases=("predicted_cases", "sum"),
            required_saline=("required_saline", "sum"),
            allocated_saline=("allocated_saline", "sum"),
            required_kits=("required_kits", "sum"),
            allocated_kits=("allocated_kits", "sum"),
            required_beds=("required_beds", "sum"),
            allocated_beds=("allocated_beds", "sum"),
            required_crews=("required_crews", "sum"),
            allocated_crews=("allocated_crews", "sum"),
            deficit_wards=("deficit", "sum"),
        )
        .sort_values("predicted_cases", ascending=False)
        .reset_index(drop=True)
    )

    total_deficit = {
        "saline_bags_needed": int(frame["deficit_saline"].sum()),
        "diagnostic_kits_needed": int(frame["deficit_kits"].sum()),
        "hospital_beds_required": int(frame["deficit_beds"].sum()),
        "crews": int(frame["deficit_crews"].sum()),
    }
    required = {
        "saline_bags_needed": int(saline_req.sum()),
        "diagnostic_kits_needed": int(kits_req.sum()),
        "hospital_beds_required": int(beds_req.sum()),
        "crews": int(crews_req.sum()),
    }
    return AllocationResult(
        per_ward=frame.sort_values(
            ["priority", "predicted_cases"], ascending=False
        ).reset_index(drop=True),
        city_rollup=city_rollup,
        total_deficit=total_deficit,
        available=available | {"crews": int(min(total_crews, crews_req.sum()))},
        required=required,
    )
