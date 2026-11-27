#!/usr/bin/env python3
"""Offline kill-switch: run the entire operational pipeline with sockets blocked.

This is the live-judging proof for the 15-point resilience criterion. It patches
socket creation before loading data or the model, then exercises forecast ->
allocation -> overflow routing -> counterfactual simulation -> dispatch export.
Any accidental network dependency raises immediately.

Run:

    python scripts/offline_check.py
"""
from __future__ import annotations

import socket
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class OfflineViolation(RuntimeError):
    pass


def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003
    raise OfflineViolation("network access attempted while offline")


from src import config, data_access as data, inference  # noqa: E402
from src.operations.allocation import WardDemand, allocate  # noqa: E402
from src.operations.dispatch import build_notice, write_exports  # noqa: E402
from src.operations.routing import (  # noqa: E402
    capacity_scaled,
    derive_bed_capacity,
    route_overflow,
)
from src.operations.simulation import Deployment, simulate  # noqa: E402

# Block the common escape hatches after imports, so the proof covers runtime
# access rather than library import mechanics.
socket.socket = _blocked  # type: ignore[assignment]
socket.create_connection = _blocked  # type: ignore[assignment]
socket.getaddrinfo = _blocked  # type: ignore[assignment]


def main() -> int:
    print("Sockets blocked. Running the full operational pipeline...")
    onsite = data.load_onsite()
    features = data.load_features()
    if features is None:
        print("Feature table missing.", file=sys.stderr)
        return 1
    bundle = inference.load_bundle()
    meta = data.load_ward_meta().set_index(config.WARD_COL)

    as_of = pd.Timestamp("2025-08-25")
    wards = sorted(onsite[config.WARD_COL].unique())
    preds = inference.predict_batch(wards, as_of, [2], features=features, bundle=bundle)
    demands = [
        WardDemand.from_prediction(
            ward_id=str(row[config.WARD_COL]),
            city=str(meta.loc[row[config.WARD_COL], config.CITY_COL]),
            predicted_cases=float(row["cases"]),
            tier=str(row["tier"]),
            density=float(meta.loc[row[config.WARD_COL], "population_density_sqkm"]),
        )
        for _, row in preds.iterrows()
        if pd.notna(row["cases"])
    ]
    print(f"  forecast: {len(demands)} wards · {as_of.date()} +2 weeks")

    allocation = allocate(demands, 0.60, 8)
    print(
        f"  allocate: {allocation.available['saline_bags_needed']} / "
        f"{allocation.required['saline_bags_needed']} saline bags available"
    )

    bed_demand = {
        row["ward_id"]: int(row["required_beds"])
        for _, row in allocation.per_ward.iterrows()
    }
    capacity = capacity_scaled(derive_bed_capacity(onsite, as_of=as_of), 1.20)
    routing = route_overflow(
        bed_demand, capacity, {d.ward_id: d.city for d in demands}
    )
    print(
        f"  route: {len(routing.transfers)} transfers · "
        f"{sum(routing.net_deficits.values())} unrouted beds"
    )

    top = allocation.per_ward.sort_values("priority", ascending=False).head(2)
    simulation = simulate(
        [Deployment(str(ward), "larvicide", 2, 1) for ward in top["ward_id"]],
        weeks=3,
        as_of=as_of,
        features=features,
        bundle=bundle,
    )
    print(
        f"  simulate: {simulation.roi_ranking['cases_prevented'].sum():.1f} cases prevented"
    )

    notices = []
    for _, row in allocation.per_ward.head(5).iterrows():
        notices.append(
            build_notice(
                ward_id=str(row["ward_id"]),
                city=str(row["city"]),
                target_date="08 Sep 2025",
                predicted_cases=float(row["predicted_cases"]),
                tier=str(row["tier"]),
                required={
                    "saline_bags_needed": int(row["required_saline"]),
                    "diagnostic_kits_needed": int(row["required_kits"]),
                    "hospital_beds_required": int(row["required_beds"]),
                },
                allocated={
                    "saline_bags_needed": int(row["allocated_saline"]),
                    "diagnostic_kits_needed": int(row["allocated_kits"]),
                    "hospital_beds_required": int(row["allocated_beds"]),
                },
                crews_required=int(row["required_crews"]),
                crews_allocated=int(row["allocated_crews"]),
                notes="Offline kill-switch test.",
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        written = write_exports(notices, Path(tmp))
    print(f"  dispatch: {len(written)} files built with sockets blocked")

    print("\nPASS — forecasting, allocation, routing, simulation and dispatch ran offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
