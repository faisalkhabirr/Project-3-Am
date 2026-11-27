"""Intra-city hospital overflow routing.

When a ward's projected bed demand exceeds its local capacity, patients are
routed to lower-burden wards **inside the same city pool**. Transfers never
cross city boundaries: Dhaka North, Dhaka South, Chattogram, Gazipur and
Narayanganj are independent referral systems in this prototype.

Bed capacity is not present in the dataset. ``derive_bed_capacity`` builds a
documented planning baseline from each ward's historical 75th-percentile weekly
bed demand (minimum 2 beds). The UI exposes a multiplier so officers and judges
can stress-test the routing live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .supplies import cases_to_supplies


@dataclass
class RoutingResult:
    transfers: list[tuple[str, str, int]]
    net_deficits: dict[str, int]
    city_utilization: dict[str, float]
    escalated_cities: list[str]


def derive_bed_capacity(
    onsite: pd.DataFrame,
    *,
    as_of: pd.Timestamp | str | None = None,
    quantile: float = 0.75,
    headroom: float = 1.00,
    floor: int = 2,
) -> dict[str, int]:
    """Derive a per-ward weekly bed capacity from historical demand.

    The rule is intentionally transparent: a ward's baseline is its
    75th-percentile week, so peak weeks genuinely test the overflow routing.
    This is a planning baseline, not a claim about literal ward bed counts, and
    the UI lets the user scale it.
    """
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    frame = onsite
    if as_of is not None:
        frame = frame.loc[pd.to_datetime(frame["date"]) <= pd.Timestamp(as_of)]
    if frame.empty:
        return {}

    grouped = frame.groupby("ward_id")["hospital_beds_required"]
    p90 = grouped.quantile(float(quantile))
    capacity = np.ceil(p90.to_numpy() * float(headroom)).astype(int)
    capacity = np.maximum(capacity, int(floor))
    return dict(zip(p90.index.astype(str), capacity.tolist(), strict=True))


def route_overflow(
    bed_demand: dict[str, int],
    bed_capacity: dict[str, int],
    ward_city: dict[str, str],
) -> RoutingResult:
    """Route ward bed deficits to lower-burden wards within the same city.

    Parameters
    ----------
    bed_demand:
        Projected weekly beds required per ward (``hospital_beds_required``).
    bed_capacity:
        Local weekly bed capacity per ward.
    ward_city:
        Ward -> city mapping. Every demand ward must have a city.
    """
    demand = {str(k): max(0, int(v)) for k, v in bed_demand.items()}
    capacity = {str(k): max(0, int(v)) for k, v in bed_capacity.items()}
    city_of = {str(k): str(v) for k, v in ward_city.items()}

    missing = set(demand) - set(city_of)
    if missing:
        raise ValueError(f"missing city mapping for wards: {sorted(missing)}")

    unmet = {
        ward: max(0, demand[ward] - capacity.get(ward, 0)) for ward in demand
    }
    transfers: list[tuple[str, str, int]] = []
    escalated: list[str] = []

    for city in sorted(set(city_of[ward] for ward in demand)):
        wards = [ward for ward in demand if city_of[ward] == city]
        donors: list[list] = []
        for ward in wards:
            headroom = max(0, capacity.get(ward, 0) - demand[ward])
            if headroom > 0:
                util = demand[ward] / capacity[ward] if capacity.get(ward, 0) > 0 else 1.0
                donors.append([util, ward, headroom])
        # Lowest-burden donors first; deterministic tie-break by ward id.
        donors.sort(key=lambda item: (item[0], item[1]))

        receivers = sorted(
            (ward for ward in wards if unmet[ward] > 0),
            key=lambda ward: (-unmet[ward], ward),
        )
        for receiver in receivers:
            for donor in donors:
                if unmet[receiver] <= 0:
                    break
                if donor[2] <= 0:
                    continue
                moved = min(unmet[receiver], donor[2])
                if moved <= 0:
                    continue
                transfers.append((donor[1], receiver, int(moved)))
                unmet[receiver] -= moved
                donor[2] -= moved

        if any(unmet[ward] > 0 for ward in wards):
            escalated.append(city)

    utilization: dict[str, float] = {}
    for city in sorted(set(city_of[ward] for ward in demand)):
        wards = [ward for ward in demand if city_of[ward] == city]
        total_demand = sum(demand[ward] for ward in wards)
        total_capacity = sum(capacity.get(ward, 0) for ward in wards)
        utilization[city] = (
            total_demand / total_capacity if total_capacity > 0 else (1.0 if total_demand else 0.0)
        )

    return RoutingResult(
        transfers=transfers,
        net_deficits={ward: int(unmet[ward]) for ward in demand},
        city_utilization=utilization,
        escalated_cities=sorted(escalated),
    )


def capacity_scaled(base: dict[str, int], multiplier: float) -> dict[str, int]:
    """Apply the UI's live capacity multiplier to the planning baseline."""
    if multiplier < 0:
        raise ValueError("capacity multiplier must be non-negative")
    return {
        str(ward): max(0, int(math.ceil(int(value) * float(multiplier))))
        for ward, value in base.items()
    }


def demand_from_cases(cases: dict[str, float]) -> dict[str, int]:
    """Convenience: forecast cases -> projected weekly beds per ward."""
    return {
        str(ward): cases_to_supplies(float(value))["hospital_beds_required"]
        for ward, value in cases.items()
    }
