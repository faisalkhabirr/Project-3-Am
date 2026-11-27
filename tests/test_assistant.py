"""Tests for the deterministic, application-aware assistant."""
from __future__ import annotations

import unittest

import pandas as pd

from src.operations.allocation import WardDemand, allocate
from src.operations.assistant import (
    ROLE_COMMISSIONER,
    ROLE_HOSPITAL,
    AssistantContext,
    interval_coverage,
    normalise_role,
)
from src.operations.assistant import (
    answer as assistant_answer,
)
from src.operations.routing import route_overflow


def _context(role: str = ROLE_COMMISSIONER) -> AssistantContext:
    demands = [
        WardDemand.from_prediction(
            ward_id="DNCC_Ward_12",
            city="Dhaka North",
            predicted_cases=10,
            tier="High",
            density=70000,
            crews_needed=3,
        ),
        WardDemand.from_prediction(
            ward_id="DNCC_Ward_05",
            city="Dhaka North",
            predicted_cases=5,
            tier="Critical",
            density=60000,
            crews_needed=2,
        ),
        WardDemand.from_prediction(
            ward_id="CCC_Ward_01",
            city="Chattogram",
            predicted_cases=8,
            tier="Critical",
            density=50000,
            crews_needed=3,
        ),
    ]
    allocation = allocate(demands, 0.60, total_crews=2)
    bed_demand = {
        row["ward_id"]: int(row["required_beds"])
        for _, row in allocation.per_ward.iterrows()
    }
    bed_capacity = {ward: 1 for ward in bed_demand}
    routing = route_overflow(
        bed_demand,
        bed_capacity,
        {demand.ward_id: demand.city for demand in demands},
    )
    metrics = {
        "mae": 1.0543,
        "rmse": 2.0968,
        "baseline_mae": {"1": 1.6933},
        "persistence_mae": {"1": 1.4094},
        "intervals": {"1": {"p10": -1.4, "p90": 2.8}},
    }
    return AssistantContext(
        role=role,
        city="Dhaka North",
        ward_id="DNCC_Ward_12",
        as_of=pd.Timestamp("2025-08-25"),
        horizon=2,
        stock_frac=0.60,
        total_crews=2,
        capacity_mult=1.2,
        conservative=False,
        allocation=allocation,
        routing=routing,
        demands=demands,
        metrics=metrics,
        bed_capacity=bed_capacity,
        bed_demand=bed_demand,
        interval_coverage=0.815,
    )


class AssistantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _context()

    def test_role_normalisation(self) -> None:
        self.assertEqual(normalise_role("Hospital manager"), ROLE_HOSPITAL)
        self.assertEqual(normalise_role("commissioner"), ROLE_COMMISSIONER)
        self.assertEqual(normalise_role("unknown"), ROLE_COMMISSIONER)

    def test_ward_status_uses_live_allocation(self) -> None:
        result = assistant_answer("Why is my ward short?", self.context)
        self.assertEqual(result.intent, "ward")
        self.assertIn("DNCC Ward 12", result.text)
        self.assertIn("Crews", result.text)
        self.assertTrue(result.sources)

    def test_supply_answer_contains_verified_formula(self) -> None:
        result = assistant_answer("How are saline bags calculated?", self.context)
        self.assertEqual(result.intent, "supply")
        self.assertIn("31/13", result.text)
        self.assertIn("31,320", result.text)

    def test_accuracy_answer_contains_calibration(self) -> None:
        result = assistant_answer("How accurate is the forecast?", self.context)
        self.assertEqual(result.intent, "accuracy")
        self.assertIn("1.054", result.text)
        self.assertIn("81.5%", result.text)

    def test_offline_answer(self) -> None:
        result = assistant_answer("Does this work without internet?", self.context)
        self.assertEqual(result.intent, "offline")
        self.assertIn("offline", result.text.lower())
        self.assertIn("0", result.text)

    def test_assistant_is_not_an_llm(self) -> None:
        result = assistant_answer("Are you an LLM?", self.context)
        self.assertEqual(result.intent, "assistant")
        self.assertIn("not an LLM", result.text)
        self.assertIn("no external API", result.text)

    def test_biggest_gap_question(self) -> None:
        result = assistant_answer("What is my biggest gap?", self.context)
        self.assertEqual(result.intent, "ward")
        self.assertIn("Saline", result.text)

    def test_stock_what_if_is_read_only_and_changes_numbers(self) -> None:
        result = assistant_answer("What if stock drops to 30%?", self.context)
        self.assertEqual(result.intent, "stock_what_if")
        self.assertIn("30%", result.text)
        self.assertIn("Regional saline", result.text)
        self.assertIn("does not change the live Command Center", result.text)
        # The live context must remain unchanged.
        self.assertAlmostEqual(self.context.stock_frac, 0.60)

    def test_overflow_answer_contains_city_bed_position(self) -> None:
        result = assistant_answer("Where do overflow patients go?", self.context)
        self.assertEqual(result.intent, "routing")
        self.assertIn("City demand", result.text)
        self.assertIn("Unrouted", result.text)

    def test_crew_question(self) -> None:
        result = assistant_answer("Should I send crews to Ward 12?", self.context)
        self.assertEqual(result.intent, "crews")
        self.assertIn("Required", result.text)
        self.assertIn("Simulator", result.text)

    def test_role_changes_wording_not_numbers(self) -> None:
        commissioner = assistant_answer("What should I do first?", self.context)
        hospital = assistant_answer(
            "What should I do first?", _context(ROLE_HOSPITAL)
        )
        self.assertIn("Commissioner operations view", commissioner.text)
        self.assertIn("Hospital surge view", hospital.text)
        self.assertNotEqual(commissioner.text, hospital.text)

    def test_unknown_question_falls_back_safely(self) -> None:
        result = assistant_answer("Who won the football match?", self.context)
        self.assertEqual(result.intent, "unknown")
        self.assertIn("grounded answer", result.text)
        self.assertTrue(result.suggestions)

    def test_natural_language_check_passes_with_matching_outputs(self) -> None:
        available = int(self.context.allocation.available["saline_bags_needed"])
        required = int(self.context.allocation.required["saline_bags_needed"])
        result = assistant_answer(
            f"At 60% stock with 2 crews, regional saline should be {available} of {required}.",
            self.context,
        )
        self.assertEqual(result.intent, "verify")
        self.assertIn("Regional saline available", result.text)
        self.assertIn("Overall: PASS", result.text)

    def test_natural_language_check_fails_on_wrong_expected_value(self) -> None:
        result = assistant_answer(
            "At 60% stock with 2 crews, regional saline should be 1 of 2.",
            self.context,
        )
        self.assertEqual(result.intent, "verify")
        self.assertIn("Overall: FAIL", result.text)

    def test_natural_language_ward_check(self) -> None:
        row = self.context.allocation.per_ward.set_index("ward_id").loc["DNCC_Ward_12"]
        result = assistant_answer(
            f"Ward 12 should have {int(row['allocated_crews'])} crews.",
            self.context,
        )
        self.assertEqual(result.intent, "verify")
        self.assertIn("Ward crews allocated", result.text)
        self.assertIn("Overall: PASS", result.text)

    def test_natural_language_ward_pair_check(self) -> None:
        row = self.context.allocation.per_ward.set_index("ward_id").loc["DNCC_Ward_12"]
        result = assistant_answer(
            f"Ward 12 should have {int(row['allocated_saline'])} of "
            f"{int(row['required_saline'])} saline and "
            f"{int(row['allocated_crews'])} of {int(row['required_crews'])} crews.",
            self.context,
        )
        self.assertIn("Ward saline allocated", result.text)
        self.assertIn("Ward crews allocated", result.text)
        self.assertIn("Overall: PASS", result.text)

    def test_natural_language_invariant_and_accuracy_checks(self) -> None:
        invariant = assistant_answer(
            "No transfer should cross a city boundary.", self.context
        )
        self.assertIn("Cross-city transfers", invariant.text)
        self.assertIn("Overall: PASS", invariant.text)
        accuracy = assistant_answer("The model MAE should be below 1.5.", self.context)
        self.assertIn("Holdout MAE", accuracy.text)
        self.assertIn("Overall: PASS", accuracy.text)

    def test_natural_language_supply_formula_check(self) -> None:
        result = assistant_answer("Saline for 10 cases should be 24.", self.context)
        self.assertIn("Saline for 10 cases", result.text)
        self.assertIn("Overall: PASS", result.text)

    def test_scenario_without_claim_returns_actual_output(self) -> None:
        result = assistant_answer("Check at 30% stock with 2 crews.", self.context)
        self.assertEqual(result.intent, "verify")
        self.assertNotIn("Overall:", result.text)
        self.assertIn("actual output", result.text)
        self.assertIn("Regional saline", result.text)

    def test_deterministic(self) -> None:
        first = assistant_answer("Why is my ward short?", self.context)
        second = assistant_answer("Why is my ward short?", self.context)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.intent, second.intent)

    def test_interval_coverage_helper(self) -> None:
        preds = pd.DataFrame(
            {
                "horizon_weeks": [1, 1, 1],
                "y_true": [1.0, 2.0, 10.0],
                "y_pred": [1.0, 1.0, 1.0],
            }
        )
        metrics = {"intervals": {"1": {"p10": -1.0, "p90": 2.0}}}
        coverage = interval_coverage(metrics, preds)
        self.assertAlmostEqual(float(coverage), 2 / 3)


if __name__ == "__main__":
    unittest.main()
