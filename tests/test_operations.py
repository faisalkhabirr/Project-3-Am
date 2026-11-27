"""Invariant tests for the operational engines."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import config
from src.operations.allocation import WardDemand, allocate
from src.operations.dispatch import build_notice, write_exports
from src.operations.routing import (
    capacity_scaled,
    derive_bed_capacity,
    route_overflow,
)
from src.operations.simulation import Deployment, _crew_factor, simulate


def _demand(ward: str, city: str, cases: float, tier: str, density: float, crews: int) -> WardDemand:
    d = WardDemand.from_prediction(
        ward_id=ward,
        city=city,
        predicted_cases=cases,
        tier=tier,
        density=density,
        crews_needed=crews,
    )
    return d


class AllocationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.demands = [
            _demand("A1", "City A", 20, "Critical", 70000, 4),
            _demand("A2", "City A", 8, "High", 40000, 2),
            _demand("B1", "City B", 3, "Medium", 20000, 1),
            _demand("B2", "City B", 1, "Low", 18000, 0),
        ]

    def test_conservation_and_caps(self) -> None:
        result = allocate(self.demands, 0.6, total_crews=3)
        for kind in ("saline", "kits", "beds"):
            self.assertLessEqual(
                int(result.per_ward[f"allocated_{kind}"].sum()),
                result.available[
                    {
                        "saline": "saline_bags_needed",
                        "kits": "diagnostic_kits_needed",
                        "beds": "hospital_beds_required",
                    }[kind]
                ],
            )
            self.assertTrue(
                (
                    result.per_ward[f"allocated_{kind}"]
                    <= result.per_ward[f"required_{kind}"]
                ).all()
            )
        self.assertLessEqual(int(result.per_ward["allocated_crews"].sum()), 3)

    def test_monotonicity(self) -> None:
        previous = allocate(self.demands, 0.30, total_crews=2).per_ward.set_index("ward_id")
        for stock in (0.4, 0.5, 0.6, 0.8, 1.0):
            current = allocate(self.demands, stock, total_crews=2).per_ward.set_index("ward_id")
            for col in ("allocated_saline", "allocated_kits", "allocated_beds"):
                self.assertTrue((current[col] >= previous[col]).all(), msg=f"{col} at {stock}")
            previous = current

    def test_priority_gets_covered_first(self) -> None:
        result = allocate(self.demands, 0.30, total_crews=2).per_ward.set_index("ward_id")
        self.assertGreater(
            result.loc["A1", "allocated_saline"], result.loc["B2", "allocated_saline"]
        )
        self.assertTrue(result.loc["A1", "fulfillment_pct"] >= result.loc["B2", "fulfillment_pct"])

    def test_zero_stock_and_validation(self) -> None:
        result = allocate(self.demands, 0.0, total_crews=0)
        self.assertEqual(int(result.per_ward["allocated_saline"].sum()), 0)
        self.assertTrue(result.per_ward["deficit"].all())
        with self.assertRaises(ValueError):
            allocate(self.demands, 1.2, total_crews=0)
        with self.assertRaises(ValueError):
            allocate(self.demands, 0.5, total_crews=-1)


class RoutingTest(unittest.TestCase):
    def test_routes_only_inside_city_and_respects_headroom(self) -> None:
        demand = {"A1": 10, "A2": 2, "A3": 3, "B1": 3, "B2": 1}
        capacity = {"A1": 2, "A2": 8, "A3": 3, "B1": 1, "B2": 1}
        city = {"A1": "A", "A2": "A", "A3": "A", "B1": "B", "B2": "B"}
        result = route_overflow(demand, capacity, city)

        for donor, receiver, _ in result.transfers:
            self.assertEqual(city[donor], city[receiver])
        self.assertEqual(result.net_deficits["A1"], 2)
        self.assertEqual(result.net_deficits["B1"], 2)
        self.assertIn("B", result.escalated_cities)
        self.assertEqual(sum(moved for _, _, moved in result.transfers), 6)
        self.assertEqual(result.city_utilization["A"], 15 / 13)

    def test_capacity_scaling(self) -> None:
        self.assertEqual(capacity_scaled({"W1": 2}, 1.2), {"W1": 3})
        self.assertEqual(capacity_scaled({"W1": 2}, 1.0), {"W1": 2})

    def test_derive_capacity(self) -> None:
        frame = pd.DataFrame(
            {
                "ward_id": ["W1"] * 10,
                "date": pd.date_range("2025-01-01", periods=10, freq="7D"),
                "hospital_beds_required": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            }
        )
        caps = derive_bed_capacity(frame, as_of="2025-02-01")
        self.assertGreaterEqual(caps["W1"], 2)


class SimulationTest(unittest.TestCase):
    def test_crew_factor(self) -> None:
        self.assertEqual(_crew_factor(0), 0.0)
        self.assertGreater(_crew_factor(1), 0.0)
        self.assertEqual(_crew_factor(2), 1.0)
        self.assertEqual(_crew_factor(20), 1.0)

    @unittest.skipUnless(
        config.FEATURES.exists() and config.MODEL_PATH.exists(),
        "model artifacts not available",
    )
    def test_zero_crews_is_identity_and_effect_is_positive(self) -> None:
        from src import data_access, inference

        features = data_access.load_features()
        bundle = inference.load_bundle()
        as_of = pd.Timestamp("2025-08-25")
        ward = "CCC_Ward_01"
        zero = simulate(
            [Deployment(ward, "larvicide", 0, 1)],
            weeks=3,
            as_of=as_of,
            features=features,
            bundle=bundle,
        )
        self.assertAlmostEqual(
            float(
                (zero.per_ward_week["baseline_cases"] - zero.per_ward_week["intervention_cases"])
                .abs()
                .max()
            ),
            0.0,
            places=6,
        )
        active = simulate(
            [Deployment(ward, "larvicide", 2, 1)],
            weeks=3,
            as_of=as_of,
            features=features,
            bundle=bundle,
        )
        self.assertGreater(float(active.roi_ranking["cases_prevented"].sum()), 0.0)
        self.assertTrue(
            (active.per_ward_week["intervention_cases"] <= active.per_ward_week["baseline_cases"]).all()
        )


class DispatchTest(unittest.TestCase):
    def _notice(self):
        return build_notice(
            ward_id="DNCC_Ward_12",
            city="Dhaka North",
            target_date="2025-09-01",
            predicted_cases=14.2,
            tier="Critical",
            required={
                "saline_bags_needed": 34,
                "diagnostic_kits_needed": 17,
                "hospital_beds_required": 4,
            },
            allocated={
                "saline_bags_needed": 26,
                "diagnostic_kits_needed": 17,
                "hospital_beds_required": 4,
            },
            crews_required=3,
            crews_allocated=2,
            overflow={"DNCC_Ward_07": 2},
            notes="Deploy two larvicide teams.",
        )

    def test_sms_limit_and_content(self) -> None:
        notice = self._notice()
        self.assertLessEqual(len(notice.sms), 160)
        self.assertIn("DEF", notice.sms)
        self.assertEqual(notice.deficits["saline_bags_needed"], 8)
        self.assertIn("DNCC Ward 12", notice.whatsapp)
        self.assertIn("<html", notice.html)

    def test_exports(self) -> None:
        notice = self._notice()
        with tempfile.TemporaryDirectory() as tmp:
            written = write_exports([notice], Path(tmp))
            self.assertEqual(len(written), 3)
            self.assertTrue(all(path.exists() for path in written))


if __name__ == "__main__":
    unittest.main()
