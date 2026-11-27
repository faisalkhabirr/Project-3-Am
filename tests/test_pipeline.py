"""Feature-pipeline tests: onsite source, no supply leakage, correct lags."""
from __future__ import annotations

import unittest

import pandas as pd

from src import config
from src.data.build_features import MODEL_FEATURES, build_features
from src.operations.supplies import SUPPLY_COLUMNS


class FeaturePipelineTest(unittest.TestCase):
    def test_training_source_is_the_onsite_file(self) -> None:
        self.assertEqual(config.TRAIN_RAW, config.ONSITE_RAW)

    def test_supply_targets_are_not_model_inputs(self) -> None:
        self.assertTrue(set(SUPPLY_COLUMNS).isdisjoint(MODEL_FEATURES))

    def test_builder_ignores_supply_columns_and_keeps_lags_causal(self) -> None:
        dates = pd.date_range("2025-01-06", periods=12, freq="7D")
        frame = pd.DataFrame(
            {
                "date": dates,
                "city": ["City A"] * 12,
                "ward_id": ["W1"] * 12,
                "population_density_sqkm": [50000.0] * 12,
                "water_logging_risk_index": [0.5] * 12,
                "sanitation_score": [0.6] * 12,
                "avg_temp_c": [28.0] * 12,
                "rainfall_mm": [0.0] * 12,
                "relative_humidity_pct": [70.0] * 12,
                "dengue_cases": list(range(12)),
                "saline_bags_needed": [999] * 12,
                "diagnostic_kits_needed": [999] * 12,
                "hospital_beds_required": [999] * 12,
            }
        )
        features = build_features(frame, horizons=(1, 2))
        self.assertTrue(set(SUPPLY_COLUMNS).isdisjoint(features.columns))
        self.assertEqual(len(features), 12 * 2)

        row = features.loc[
            (features["date"] == dates[5]) & (features["horizon_weeks"] == 1)
        ].iloc[0]
        self.assertEqual(float(row["cases_current"]), 5.0)
        self.assertEqual(float(row["cases_lag_1w"]), 4.0)
        self.assertEqual(float(row["target_cases"]), 6.0)
        self.assertEqual(pd.Timestamp(row["target_date"]), dates[6])


if __name__ == "__main__":
    unittest.main()
