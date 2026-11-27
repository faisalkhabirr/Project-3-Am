# Synthetic ensemble experiment — **not the shipped model**

This folder preserves the R&D notebook that
**Muhammad Anamur Rashid (`loser-2-o`)** explored during the build. It is kept for transparency and
as a starting point for a future real-data ensemble. **The live dashboard does not load any file
from this folder.**

## What it is

| Item | Detail |
|---|---|
| Notebook | `Team_3_AM_synthetic_benchmark.ipynb` (outputs cleared) |
| Artifact | Not committed — regenerate locally from the notebook if needed |
| Model | 5-model stack: LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees + Ridge meta-learner |
| Features | 1w–4w weather lags, 2w/4w rolling means and max, rainfall × vulnerability interactions |
| Split | 80/20 chronological on the generated data (380 train / 100 validation rows) |
| Reported metrics | R² 0.97 · MAE 8.38 · RMSE 12.94 (synthetic holdout) |
| Serialized bundle | `base_models`, `meta_learner`, `feature_cols`, `residuals` |

## Why it is not shipped

1. **Synthetic data, not the official dataset.** The notebook calls
   `generate_hackathon_benchmark_data(num_wards=10, weeks_per_ward=52)` and never reads
   `data/raw/dengue_weather.csv`. The synthetic cases range roughly 12–415 per ward-week; the real
   dataset is 0–41. The reported metrics are therefore not comparable to the live model
   (real-data MAE ≈ 1.05, RMSE ≈ 2.10).
2. **Same-week target, not a 2–4 week forecast.** The target is `dengue_cases` on the same row.
   There is no `horizon_weeks` and no future target (`shift(-h)`); the app's contract requires
   predictions for the selected lead time.
3. **Global tiers.** The notebook uses `<20 Low / <50 Medium / <100 High / else Critical`. On the
   real dataset (max 41 cases/ward-week), High and Critical would never occur. The live app uses
   ward-specific thresholds derived from each ward's training distribution.
4. **Incompatible feature schema.** The bundle's `feature_cols` use synthetic names such as
   `avg_temp_C` and `rainfall_mm_lag_1w`, and include synthetic interaction columns. They do not
   match `models/feature_columns.json` or the columns in `data/processed/features.csv`.
5. **Extra heavy dependencies.** Loading the bundle requires LightGBM, XGBoost and CatBoost, which
   are not in `requirements.txt`. Adding them would materially increase install time and memory for
   no proven real-data benefit.
6. **Stacking leakage.** The meta-learner is fit on in-sample base-model predictions rather than
   out-of-fold predictions, so the synthetic validation result is optimistic even within the
   synthetic benchmark.
7. **Uncalibrated intervals.** The 95% bootstrap uses training residuals; its own examples do not
   cover several validation actuals. The live app uses validation residual quantiles for an
   explicitly labeled 80% interval.

## If you want to run it

The notebook was written for Google Colab:

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy matplotlib seaborn shap joblib
```

Then open the notebook and run all cells. It regenerates the synthetic data locally; it
does not touch the official dataset or the app. The serialized `.joblib` is intentionally
**not committed**: it is unused by the app and joblib deserialization is a code-execution
surface. Only regenerate it from this trusted notebook.

## What a real-data version would need

- Ingest `data/raw/dengue_onsite.csv` (60 wards, 2016–2025).
- Build one row per `(ward_id, date, horizon_weeks 1–4)` with future `target_cases`.
- Use the exact feature list in `models/feature_columns.json` (or publish a new contract and update
  the app).
- Fit the meta-learner on out-of-fold base-model predictions.
- Use per-ward thresholds and validation-residual intervals.
- Export `features.csv`, `dengue_model.joblib`, `feature_columns.json`,
  `risk_thresholds.json`, `metrics.json`, `test_predictions.csv`.
- Prove the install stays under the reproducibility budget on Python 3.11/3.12.

Until then, the shipped engine remains the real-data Poisson HistGradientBoosting model documented
in the root `README.md` and `ARCHITECTURE.md`.
