#!/usr/bin/env python3
"""Generate offline dispatch samples for the Phase 2 submission.

Run:

    python scripts/export_dispatch_samples.py

Writes SMS, WhatsApp and printable HTML briefs to ``exports/`` plus a scenario
manifest so judges can see the exact scarcity assumptions behind each notice.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data_access as data, inference  # noqa: E402
from src.operations.allocation import WardDemand, allocate  # noqa: E402
from src.operations.dispatch import build_notice, write_exports  # noqa: E402
from src.operations.routing import (  # noqa: E402
    capacity_scaled,
    derive_bed_capacity,
    route_overflow,
)


def default_as_of(dates: list[pd.Timestamp]) -> pd.Timestamp:
    if not dates:
        return pd.Timestamp.today().normalize()
    latest = dates[-1].year
    august = [d for d in dates if d.year == latest and d.month == 8]
    return august[-1] if august else dates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--stock", type=float, default=0.60)
    parser.add_argument("--crews", type=int, default=8)
    parser.add_argument("--capacity-mult", type=float, default=1.20)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "exports")
    args = parser.parse_args()

    onsite = data.load_onsite()
    features = data.load_features()
    if features is None:
        print("Feature table missing; run the ML pipeline first.", file=sys.stderr)
        return 1
    bundle = inference.load_bundle()
    meta = data.load_ward_meta().set_index(config.WARD_COL)

    dates = data.available_dates(onsite)
    as_of = pd.Timestamp(args.as_of) if args.as_of else default_as_of(dates)
    wards = sorted(onsite[config.WARD_COL].unique())
    preds = inference.predict_batch(
        wards, as_of, [args.horizon], features=features, bundle=bundle
    )
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
    allocation = allocate(demands, args.stock, args.crews)
    ward_city = {d.ward_id: d.city for d in demands}
    bed_demand = {
        row["ward_id"]: int(row["required_beds"])
        for _, row in allocation.per_ward.iterrows()
    }
    base_capacity = derive_bed_capacity(onsite, as_of=as_of)
    bed_capacity = capacity_scaled(base_capacity, args.capacity_mult)
    routing = route_overflow(bed_demand, bed_capacity, ward_city)

    overflow_by_receiver: dict[str, dict[str, int]] = defaultdict(dict)
    for donor, receiver, moved in routing.transfers:
        overflow_by_receiver[receiver][donor] = moved

    target_date = (as_of + pd.Timedelta(weeks=args.horizon)).strftime("%d %b %Y")
    candidates = allocation.per_ward.loc[allocation.per_ward["deficit"]].copy()
    candidates = candidates.sort_values(
        ["priority", "predicted_cases"], ascending=False
    ).head(args.limit)
    notices = []
    for _, row in candidates.iterrows():
        ward = str(row["ward_id"])
        notices.append(
            build_notice(
                ward_id=ward,
                city=str(row["city"]),
                target_date=target_date,
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
                overflow=overflow_by_receiver.get(ward, {}),
                notes=(
                    f"Scarcity scenario: {args.stock:.0%} regional stock, "
                    f"{args.crews} crews, {args.capacity_mult:.2f}x bed capacity. "
                    "Verify against the ward register before dispatch."
                ),
            )
        )

    written = write_exports(notices, args.out)
    manifest = {
        "generated_locally": True,
        "as_of": str(as_of.date()),
        "horizon_weeks": args.horizon,
        "target_date": target_date,
        "stock_fraction": args.stock,
        "total_crews": args.crews,
        "capacity_multiplier": args.capacity_mult,
        "wards": len(demands),
        "required": allocation.required,
        "available": allocation.available,
        "total_deficit": allocation.total_deficit,
        "bed_capacity_total": int(sum(bed_capacity.values())),
        "bed_demand_total": int(sum(bed_demand.values())),
        "transfers": [
            {"from": donor, "to": receiver, "patients": moved}
            for donor, receiver, moved in routing.transfers
        ],
        "escalated_cities": routing.escalated_cities,
        "exports": [str(path.relative_to(ROOT)) for path in written],
    }
    (args.out / "scenario.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(notices)} dispatch packets ({len(written)} files) to {args.out}")
    print(
        f"Scenario: {as_of.date()} +{args.horizon}w · stock {args.stock:.0%} · "
        f"{args.crews} crews · bed demand {sum(bed_demand.values())}/"
        f"{sum(bed_capacity.values())} · transfers {len(routing.transfers)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
