# Aedes Sentinel

**Ward-level dengue early warning for Bangladesh — a 2-to-4 week window to act before outbreaks peak.**

Dengue outbreaks in Dhaka, Chattogram and nearby cities follow the monsoon, but not immediately.
Standing water needs 1–2 weeks to produce adult *Aedes aegypti* mosquitoes, followed by viral
incubation and symptom onset. Aedes Sentinel turns that biological lag into a public-health tool:
it ingests weekly rainfall, temperature and humidity, computes ward-level multi-week lags, and
serves a live risk tier and driver breakdown for every ward.

> Built for the **Hackathon Phase 1: 24-Hour Remote Qualifier — AI for Urban Dengue Outbreak**.

---

## Quickstart (under 3 minutes)

**Requirements:** Python 3.11 or 3.12 (tested on both), no API keys, no external services.

```bash
git clone https://github.com/faisalkhabirr/Project-3-Am.git
cd Project-3-Am

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Open **http://localhost:8501**.

The dataset and serialized model are committed to the repo, so the app runs fully offline.
There is no build step, no database, and no network call at runtime.

<details>
<summary>Using uv instead of pip (optional)</summary>

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py
```
</details>

---

## What the dashboard shows

| Area | What you get |
|---|---|
| **Risk forecast** | Ward-specific **Low / Medium / High / Critical** tier, predicted weekly cases, and an 80% prediction interval for a chosen lead time (1–4 weeks). |
| **Driver breakdown** | Local, model-agnostic contribution bars: which weather lags and vulnerability features push the forecast up or down. |
| **Weather context** | Rainfall 1–4 weeks ago, 4-week rainfall totals, humidity and temperature percentiles, plus the historical rainfall-vs-cases curve with the 2–4 week lag window highlighted. |
| **Ward vulnerability** | Population density, waterlogging risk and sanitation score — the reason identical rainfall produces different risk in adjacent wards. |
| **City view** | Ranked ward table and a ward-level risk chart for the selected city/week. |
| **Ward × week heatmap** | Predicted weekly cases for every ward in the city across the next four weeks. |
| **Validation** | Chronological holdout MAE/RMSE, comparison with seasonal-naive and persistence baselines, actual-vs-predicted curve, and error by horizon. |
| **Method & cost** | Lag definitions, split logic, model rationale, and the concrete cost/API controls. |

---

## Dataset

The official Phase 1 dataset is bundled at `data/raw/dengue_weather.csv`:

- **31,320 rows** — 60 wards × 522 weeks
- **Period:** 2016-01-04 → 2025-12-29, weekly
- **Cities:** Dhaka North (20 wards), Dhaka South (20), Chattogram (10), Gazipur (5), Narayanganj (5)
- **Weather:** `avg_temp_c`, `max_temp_c`, `min_temp_c`, `rainfall_mm`, `relative_humidity_pct`
- **Vulnerability:** `population_density_sqkm`, `water_logging_risk_index`, `sanitation_score`
- **Target:** `dengue_cases` (weekly, per ward)

Data source: [Hackathon Phase 1 dataset](https://docs.google.com/spreadsheets/d/1QFI0ACJRvJQkiQT3LfGHklM3g4PgaJWUt8xSVU07NIE/edit?usp=sharing).

To refresh the bundled file (one-time, not needed to run the app):

```bash
python scripts/fetch_data.py --force
```

---

## How the pipeline works

```text
Raw ward-week weather + cases
        │
        ▼
Group by ward_id → 1–4 week rainfall lags → rolling rainfall totals
        │                              humidity / temperature lags
        │                              seasonal terms + ward vulnerability
        ▼
One row per (ward, as-of week, horizon 1–4)
        │
        ▼
Chronological split: train ≤2022 · validate 2023 · test 2024–2025
        │
        ▼
Serialized regressor (joblib) + feature schema + risk thresholds
        │
        ▼
Streamlit app → local inference → tier + interval + drivers
```

The app never invents a prediction: it looks up the engineered feature row for the selected
`(ward, as-of week, horizon)` and calls the serialized estimator. Drivers are computed locally
with single-feature occlusion — no SHAP install, no external explanation API.

Verify the lag math independently (re-derives every lag from the raw CSV and checks targets):

```bash
python scripts/verify_lags.py
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full data-flow diagram, leakage controls and
model-selection rationale.

---

## Model and validation

<!-- MODEL_SECTION_START -->
The prediction engine consumes the engineered feature table and returns weekly cases for the
selected ward and lead time. It is trained only on weeks up to 2022, checked on 2023, and
evaluated on the untouched 2024–2025 holdout.

| Metric (2024–2025 holdout) | Value |
|---|---|
| MAE | _populated from `models/metrics.json`_ |
| RMSE | _populated from `models/metrics.json`_ |
| Seasonal-naive MAE | _populated from `models/metrics.json`_ |
| Persistence MAE | _populated from `models/metrics.json`_ |

The exact model name and metrics are rendered live in the **Validation** tab and read from
`models/metrics.json`. Training and feature code live in `src/data/` and `src/model/`.
<!-- MODEL_SECTION_END -->

---

## Repository layout

```text
app.py                      Streamlit entry point
src/
  config.py                 paths, labels, tier colours
  data_access.py            cached data loaders + ward context
  inference.py              model loading, prediction, tiers, drivers
  charts.py                 Plotly figures
  data/                     feature pipeline (ML owner)
  model/                    training + artifact export (ML owner)
data/
  raw/dengue_weather.csv    bundled official dataset
  processed/features.csv    engineered ward/week/horizon features
models/
  dengue_model.joblib       serialized estimator
  feature_columns.json      ordered model inputs
  risk_thresholds.json      per-ward tier thresholds
  metrics.json              split, metrics, prediction intervals
  test_predictions.csv      chronological holdout predictions
scripts/
  fetch_data.py             one-time data refresh
  verify_lags.py            independent lag/leakage sanity checks
ARCHITECTURE.md             system design + validation + cost controls
README.md
requirements.txt
```

---

## Compute and API cost

Aedes Sentinel is designed for a public-health office laptop, not a GPU cluster:

- **0 external API calls at runtime.** Data and model are bundled.
- **One-time data fetch**, cached in the repo (`scripts/fetch_data.py`).
- **`joblib` model loaded once** per process via `st.cache_resource`.
- **Cached data frames** via `st.cache_data`, keyed on file mtime.
- **One vectorized prediction** for the city ranking/heatmap, not a per-ward API loop.
- **No LLM in the product loop** — no token spend during a prediction.
- Typical warm inference: **under 300 ms** per prediction on CPU.

The detailed control table is in [ARCHITECTURE.md](ARCHITECTURE.md#6-compute--api-cost-controls)
and in the app's **Method & cost** tab.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `streamlit: command not found` | Activate the venv first: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`). |
| Port 8501 already in use | `streamlit run app.py --server.port 8502`. |
| "Model handoff pending" banner | The serialized artifacts are missing; restore `models/` and `data/processed/features.csv` from the repo, or run the training pipeline. |
| App starts but no data | Confirm `data/raw/dengue_weather.csv` exists; re-run `python scripts/fetch_data.py`. |
| Slow first page load | The first run imports scikit-learn and loads the model; subsequent interactions are cached and fast. |

---

## Team and contributions

| Person | Role |
|---|---|
| **Faisal Khabir** | Dashboard/UI, inference integration, validation views, architecture, demo, repo delivery |
| **ML teammate** | Data processing, multi-week lag pipeline, model training, artifact export |

Git history is intentionally incremental across the build window. Feature, model and UI work
land in separate commits from each contributor.

---

## AI assistants and external APIs

We disclose every AI assistant and external API used in this project, as required by the
judging criteria.

| Tool / API | Used for | Runtime dependency? |
|---|---|---|
| **OpenCode (DeepSeek V4.1 Flash)** | Coding assistant: Streamlit dashboard, data-access/inference modules, documentation, integration review | No |
| **Google Sheets `gviz` CSV endpoint** | One-time download of the official hackathon dataset | No — CSV is bundled |
| **scikit-learn / pandas / numpy / joblib** | Training and local inference libraries | Local libraries, no API |
| **Streamlit / Plotly** | Dashboard and charts | Local libraries, no API |

No LLM API is called when the app runs. No paid inference endpoint, map service, weather API or
database is used at runtime. If any teammate used an additional assistant during model
development, it is listed here before submission.

---

## Limitations

- Historical ward-week data, not a live weather feed. A production version would add a cached
  daily weather adapter and a scheduled retrain.
- Ward polygons/coordinates are not in the dataset, so risk is shown as a ward × week matrix
  rather than a geographic map.
- Weekly resolution matches the surveillance data; daily lags would need finer input.
- Risk tiers are decision support, not a clinical diagnosis.

---

## License

MIT — see [LICENSE](LICENSE).
