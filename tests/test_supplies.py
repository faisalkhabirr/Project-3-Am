"""Unit tests for the deterministic supply conversion."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import config
from src.operations.supplies import cases_to_supplies, supplies_for_cases


class CasesToSuppliesTest(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(
            cases_to_supplies(0),
            {
                "saline_bags_needed": 0,
                "diagnostic_kits_needed": 0,
                "hospital_beds_required": 0,
            },
        )

    def test_known_rows(self) -> None:
        self.assertEqual(
            cases_to_supplies(1),
            {
                "saline_bags_needed": 2,
                "diagnostic_kits_needed": 1,
                "hospital_beds_required": 0,
            },
        )
        self.assertEqual(
            cases_to_supplies(3),
            {
                "saline_bags_needed": 7,
                "diagnostic_kits_needed": 3,
                "hospital_beds_required": 1,
            },
        )
        self.assertEqual(
            cases_to_supplies(5),
            {
                "saline_bags_needed": 12,
                "diagnostic_kits_needed": 6,
                "hospital_beds_required": 1,
            },
        )
        self.assertEqual(
            cases_to_supplies(41),
            {
                "saline_bags_needed": 98,
                "diagnostic_kits_needed": 48,
                "hospital_beds_required": 12,
            },
        )

    def test_rejects_negative_and_non_finite(self) -> None:
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                cases_to_supplies(value)

    def test_vectorized_matches_scalar(self) -> None:
        cases = pd.Series([0, 1, 2, 3, 5, 9, 15, 22, 41], dtype=float)
        frame = supplies_for_cases(cases)
        for idx, value in cases.items():
            expected = cases_to_supplies(value)
            for col, supply in expected.items():
                self.assertEqual(frame.loc[idx, col], supply)

    def test_monotone_and_ordered(self) -> None:
        frame = supplies_for_cases(np.arange(0, 200, 0.25))
        self.assertTrue((frame.diff().fillna(0) >= 0).all().all())
        self.assertTrue((frame["saline_bags_needed"] >= frame["diagnostic_kits_needed"]).all())
        self.assertTrue(
            (frame["diagnostic_kits_needed"] >= frame["hospital_beds_required"]).all()
        )

    @unittest.skipUnless(config.ONSITE_RAW.exists(), "onsite dataset not bundled")
    def test_reproduces_every_onsite_row(self) -> None:
        df = pd.read_csv(config.ONSITE_RAW)
        predicted = supplies_for_cases(df[config.TARGET_COL])
        for col in config.SUPPLY_COLUMNS:
            self.assertTrue(
                (predicted[col].to_numpy() == df[col].to_numpy()).all(),
                msg=f"{col} does not match the onsite dataset",
            )


if __name__ == "__main__":
    unittest.main()
