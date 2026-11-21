# Processed feature table

`features.csv` is the contract between the ML pipeline and the dashboard. One row per
**(ward, as-of week, horizon)**.

| Column | Meaning |
|---|---|
| `ward_id` | Ward key, matching `data/raw/dengue_weather.csv`. |
| `date` | As-of week (the last week of weather the model is allowed to see). |
| `horizon_weeks` | Lead time, `1`–`4`. |
| `target_date` | `date + horizon_weeks` weeks. |
| `target_cases` | Dengue cases at `target_date` for the same ward. `NaN` for the last `h` weeks (kept for live inference). |
| model features | Everything listed in `models/feature_columns.json`: rainfall/humidity/temperature lags, rolling rainfall, seasonality, case history and ward vulnerability. |

## Rules

- Sort by `ward_id, date` before computing `shift`/`rolling`; never let a rolling window
  cross a ward boundary.
- Shift rolling features so a row never sees its own future.
- Keep the last `h` weeks of each ward even when `target_cases` is `NaN`.
- Do not include the as-of week's target or any future weather.

## Checks

```bash
python scripts/verify_lags.py
```

The script re-derives every `*_lag_*` column and the target from the raw CSV, checks the
7-day grid, and fails on leakage.
