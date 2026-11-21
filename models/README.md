# Model artifacts

The Streamlit app loads these files at runtime. They are produced by the training pipeline
(`src/model/train.py`) and must be committed so a fresh clone runs offline.

| File | Required | Contract |
|---|---|---|
| `dengue_model.joblib` | yes | One fitted scikit-learn estimator or `Pipeline` exposing `.predict(X)`. `X` is a DataFrame with the columns in `feature_columns.json`, in order. The output is **weekly dengue cases on the original scale** (not log). |
| `feature_columns.json` | yes | Ordered JSON list of model input columns, e.g. `["rain_lag_1w", "rain_lag_2w", ...]`. |
| `risk_thresholds.json` | recommended | Per-ward case thresholds. Schema below. Missing wards fall back to `__default__`, then to the training distribution. |
| `metrics.json` | recommended | Split dates, MAE/RMSE, baselines, residual intervals. Schema below. |
| `test_predictions.csv` | recommended | Chronological holdout predictions used by the Validation tab. Columns: `date`, `ward_id`, `horizon_weeks`, `target_date`, `y_true`, `y_pred`. |

## Target definition

`date` in the feature table is the **as-of week**. For horizon `h`:

```
target_date = date + h weeks
target_cases = dengue_cases at target_date for the same ward_id
```

Use `groupby("ward_id").shift(-h)`. Keep the last `h` weeks of each ward even though
`target_cases` is `NaN`; the app needs those feature rows to forecast from the latest data.

## `risk_thresholds.json`

```json
{
  "mode": "per_ward",
  "thresholds": {
    "DNCC_Ward_12": {"low_max": 0, "medium_max": 6, "high_max": 15},
    "...": {"low_max": 0, "medium_max": 5, "high_max": 10},
    "__default__": {"low_max": 0, "medium_max": 3, "high_max": 8}
  }
}
```

Tier rule: `cases <= low_max` → Low; `<= medium_max` → Medium; `<= high_max` → High;
otherwise Critical. The app applies a 0.5-case floor so a near-zero forecast stays Low.
A global object `{"low_max": ..., "medium_max": ..., "high_max": ...}` is also accepted.

## `metrics.json`

```json
{
  "model_name": "HistGradientBoostingRegressor (Poisson)",
  "trained_at": "2026-11-21",
  "mae": 1.05,
  "rmse": 2.10,
  "baseline_mae": {"1": 1.69, "2": 1.69, "3": 1.69, "4": 1.69},
  "persistence_mae": {"1": 1.41, "2": 1.66, "3": 1.87, "4": 2.06},
  "intervals": {
    "1": {"p10": -1.9, "p90": 2.2},
    "2": {"p10": -2.1, "p90": 2.4}
  },
  "split": {
    "train": {"start": "2016-01-04", "end": "2022-12-26", "weeks": 365},
    "validation": {"start": "2023-01-02", "end": "2023-12-25", "weeks": 52},
    "test": {"start": "2024-01-01", "end": "2025-12-22", "weeks": 105}
  }
}
```

`intervals` contains **residual** quantiles per horizon (added to the point prediction).
If omitted, the app uses `rmse` for an approximate interval.

## Verify before committing

```bash
python scripts/verify_lags.py   # re-derives lags and targets from the raw CSV
python scripts/smoke_test.py    # loads the artifacts and runs one end-to-end prediction
```
