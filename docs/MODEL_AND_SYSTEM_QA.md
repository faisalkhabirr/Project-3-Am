<div class="cover">
  <div class="cover-top">
    <div class="cover-mark">3AM</div>
    <div class="cover-kicker">Team 3AM · Phase 2 Onsite Final</div>
  </div>
  <h1 class="cover-title">Aedes Sentinel</h1>
  <div class="cover-sub">Model, Validation &amp; Operational System Design</div>
  <div class="cover-rule"></div>
  <p class="cover-lede">
    A judge-facing reference for the dengue early-warning model, the chronological
    train/validation/test split, the five Phase 2 operational capabilities, and the
    complete local-first system design.
  </p>
  <div class="cover-grid">
    <div><span>Coverage</span><strong>60 wards · 5 cities</strong></div>
    <div><span>Period</span><strong>2016–2025 · weekly</strong></div>
    <div><span>Runtime APIs</span><strong>0</strong></div>
    <div><span>Holdout MAE</span><strong>1.054</strong></div>
  </div>
  <div class="cover-foot">
    <div>
      <span class="cover-label">Live app</span>
      <div class="cover-value">project3am.streamlit.app</div>
    </div>
    <div>
      <span class="cover-label">Code snapshot</span>
      <div class="cover-value">ce0fbb6a65f3</div>
      <div class="cover-subvalue">version-onsite-final</div>
    </div>
    <div>
      <span class="cover-label">Generated</span>
      <div class="cover-value">27 November 2026</div>
    </div>
  </div>
</div>

<div class="pagebreak"></div>

<div class="toc-title">Contents</div>

[TOC]

<div class="pagebreak"></div>

# 1. Executive summary

> **30-second pitch:** “Phase 1 gave us a validated dengue forecaster — MAE 1.054 on an untouched chronological holdout. Phase 2 turns that forecast into operations: constrained supply allocation, intra-city hospital overflow routing, a model-driven crew simulator, offline field dispatch, and a grounded assistant. Everything runs locally with zero runtime APIs.”

## At a glance

| Item | Value |
|---|---|
| Problem | Dengue outbreaks in Bangladesh; vector control and hospital readiness lag the case curve. |
| Forecast target | Weekly dengue cases per ward, 1–4 weeks ahead. |
| Coverage | 60 wards · 5 cities · 522 weeks · 2016-01-04 → 2025-12-29. |
| Model | `HistGradientBoostingRegressor(loss="poisson")`, 32 features. |
| Holdout | MAE **1.054** · RMSE **2.097** · 80% interval coverage **81.5%**. |
| Baselines | Seasonal-naive **1.693** · persistence **1.748** (all horizons). |
| Operational layer | Allocation · overflow routing · crew simulator · dispatch · offline proof. |
| Bonus layer | Read-only, application-aware assistant. |
| Runtime | Local CPU · bundled data/model · **0 runtime APIs**. |

## What this document answers

1. How the model works.
2. How the train / validation / test split was constructed.
3. What the “five new features” actually are, and how they connect.
4. The complete system design and data flow.
5. The numbers and short answers to use with visitor judges.

# 2. How the model works

**One-sentence version:** a Poisson gradient-boosted model learns the relationship between recent weather, case history, ward vulnerability and forecast horizon, then predicts weekly dengue cases per ward 1–4 weeks ahead.

## Input and output contract

| Element | Definition |
|---|---|
| Target | `target_cases` — reported cases in a ward at `as-of date + horizon_weeks`. |
| Unit | One row = one ward × one as-of week × one horizon. |
| Horizons | +1, +2, +3, +4 weeks. One model serves all four via `horizon_weeks`. |
| Model | `HistGradientBoostingRegressor(loss="poisson")`. |
| Output | Expected weekly cases, risk tier, 0–100 risk index and an 80% interval. |
| Excluded inputs | `saline_bags_needed`, `diagnostic_kits_needed`, `hospital_beds_required` — deterministic targets, not model features. |

## The 32 features

| Group | Features | Why it matters |
|---|---|---|
| Horizon | `horizon_weeks` | One model serves four lead times. |
| Current weather | `avg_temp_c`, `rainfall_mm`, `relative_humidity_pct` | Current environmental conditions. |
| Rainfall lags | `rain_lag_1w` … `rain_lag_4w` | The breeding clock: rain 1–4 weeks before cases. |
| Rainfall accumulation | `rain_cum_2w`, `rain_cum_4w` | Sustained standing water, not a single downpour. |
| Humidity lags | `humidity_lag_1w` … `humidity_lag_4w`, `humidity_mean_4w` | Vector survival and activity. |
| Temperature lags | `temp_lag_1w` … `temp_lag_4w`, `temp_mean_4w`, `temp_suitability` | Virus incubation and vector activity. |
| Case history | `cases_current`, `cases_lag_1w` … `cases_lag_4w` | Outbreak momentum and autocorrelation. |
| Seasonality | `week_sin`, `week_cos`, `month` | Monsoon cycle without hard-coded calendar rules. |
| Vulnerability | `population_density_sqkm`, `water_logging_risk_index`, `sanitation_score` | Why identical rainfall produces different risk in adjacent wards. |

## Why this model

- **Poisson loss** matches non-negative, overdispersed, zero-inflated counts.
- **Gradient boosting** captures non-linear interactions such as rain × humidity × density.
- **HistGradientBoosting** is CPU-only, handles missing values natively, produces a small artifact and is fast for local inference.
- **One model for four horizons** shares statistical strength and keeps deployment simple.
- **No deep sequence model** — 60 wards × 522 weeks is small, and the deployment risk is higher for no proven real-data gain.

Shipped parameters: `learning_rate=0.05`, `max_iter=350`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=False`, `random_state=42`.

## Prediction path

```text
as-of ward / week / horizon
  -> lookup engineered feature row
  -> model.predict()
  -> clip at zero
  -> ward-specific thresholds -> tier + risk index
  -> validation residual quantiles -> 80% interval
  -> single-feature occlusion -> local driver breakdown
```

Risk tiers are **ward-specific**, derived from the training distribution: median (`Low`), p75 (`Medium`), p90 (`High`), above p90 (`Critical`). A half-case floor keeps a near-zero forecast in `Low`.

The 80% interval is empirical: validation residual p10/p90 per horizon, added to the point prediction and clipped at zero. It is not a second model.

## Holdout performance

| Metric | Value |
|---|---:|
| MAE | **1.054** cases per ward-week |
| RMSE | **2.097** |
| Seasonal-naive MAE | **1.693** |
| Persistence MAE | **1.748** all horizons |
| 80% interval coverage | **81.5%** |

| Horizon | MAE | RMSE | Bias | 80% coverage |
|---|---:|---:|---:|---:|
| +1 week | 0.981 | 1.949 | +0.194 | 81.9% |
| +2 weeks | 0.996 | 2.011 | +0.249 | 81.5% |
| +3 weeks | 1.029 | 1.995 | +0.278 | 80.7% |
| +4 weeks | 1.211 | 2.401 | +0.384 | 81.8% |

Interpretation: about **1 case error per ward-week** against a mean of 2.64 cases; the model is slightly conservative; the 80% interval is almost perfectly calibrated.

# 3. Train / validation / test split

**Chronological by target date — never random.**

Each feature row has `target_date = date + horizon_weeks`. The split is applied to `target_date`, not the as-of date. That prevents any row from predicting into a later window.

| Split | Target-date window | Rows | Weeks | Purpose |
|---|---|---:|---:|---|
| **Train** | 2016-01-11 → 2022-12-26 | **87,000** | 364 | Fit the model and ward thresholds. |
| **Validation** | 2023-01-02 → 2023-12-25 | **12,480** | 52 | Model comparison and residual intervals. |
| **Test** | 2024-01-01 → 2025-12-29 | **25,200** | 105 | Untouched holdout, scored once. |

Total known-target rows: **124,680**.

## Why train is 87,000, not 87,360

60 wards × 4 horizons × 364 weeks = 87,360. The first three target weeks only have 1, 2 and 3 horizons available, so **360 early-boundary rows** do not exist. That gives exactly **87,000** — a useful detail if a judge checks the arithmetic.

## Leakage controls

1. Every lag and rolling statistic is computed **within each ward** after sorting by date.
2. Lags are shifted before use; no row sees its own future.
3. Rolling rainfall is shifted to use only completed prior weeks.
4. The split is by **target date**; rows crossing a boundary are removed from training.
5. The final model is refit on **train + validation** (99,480 rows) before scoring test.
6. Test is scored **once**; it is not used for tuning, thresholds or intervals.
7. HGB handles missing values natively; no scaler or imputer is fit on test.
8. `scripts/verify_lags.py` independently re-derives the lag columns and checks the split.
9. Supply targets are excluded from `MODEL_FEATURES`; `tests/test_pipeline.py` guards this.

## Model-selection flow

```text
train ──► validation model ──► validation MAE + residual quantiles
                                     │
                                     ├─ 80% intervals per horizon
                                     └─ model comparison
train + validation ──► final model ──► test MAE / RMSE / baselines
```

Ward thresholds come from the **training** distribution only. Intervals come from **validation** residuals only. The test window is used exactly once.

## Baselines

- **Seasonal naive:** cases in the same week last year (`target_date − 52 weeks`).
- **Persistence:** most recent observed cases (`cases_current`).

# 4. The “five new features” — correct interpretation

> **Important clarification for judges:** Phase 2 did **not** add five new ML input features. The model still uses the same **32 features**. The onsite file added **three deterministic supply targets**, and they are excluded from the model inputs. What the brief calls “five” are the five **operational capabilities**.

## The five Phase 2 capabilities

| # | Capability | What it does | Inputs | Outputs | Connects to |
|---|---|---|---|---|---|
| 1 | **Constrained allocation** | Forecast cases → saline, kits, beds, crews; 30–100% stock; priority by tier × density; live deficits. | Forecast, stock %, crews, density/tier. | Allocated vs required, deficit flags, bed demand. | Feeds routing and dispatch. |
| 2 | **Overflow routing** | Ward bed demand vs local capacity; intra-city transfers to lower-burden wards; utilization and escalation. | Required beds, capacity baseline, capacity multiplier. | Transfers, net deficits, city utilization, escalation. | Feeds dispatch and assistant. |
| 3 | **Vector-control simulator** | Rewrites breeding/transmission features; re-runs the model; feeds predictions into case lags; 2–4 week impact and ROI. | Selected wards, crew type, crews, start week, features/model. | Cases/beds/saline per week, cases prevented per crew. | Feeds dispatch and ROI. |
| 4 | **Multi-channel dispatch** | ≤160-char SMS, WhatsApp, printable HTML brief; allocated vs required, deficits, crews, overflow. | Allocation row, routing transfers, notes. | SMS, WhatsApp, HTML/PDF packet, `/exports`. | Consumes allocation and routing. |
| 5 | **Offline resilience** | Socket-blocked full pipeline; no APIs; deterministic templates. | All modules and bundled assets. | Proven offline execution. | Wraps all four. |

**Bonus layer:** the **Assistant** reads allocation, routing, metrics and supply math and answers in natural language. It is read-only and does not feed back into the model.

## How the five connect

```text
                         FORECAST
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
     ALLOCATION        SIMULATION         DASHBOARD
          │                 │
          │ bed demand      │ cases/beds/saline saved
          ▼                 │
      ROUTING               │
          │                 │
          └────────┬────────┘
                   ▼
                DISPATCH ──► /exports
                   │
                   ▼
                ASSISTANT (read-only)

     OFFLINE PROOF wraps forecast, allocation, routing,
     simulation, dispatch and assistant.
```

- **Allocation** answers: how much saline, kits, beds and how many crews.
- **Routing** answers: where overflow patients go.
- **Simulation** answers: what happens if we deploy crews.
- **Dispatch** answers: what the field officer receives.
- **Offline** proves all of it works without internet.
- **Assistant** answers questions about the live state.

# 5. Complete system design

## Layer diagram

```text
┌──────────────────────────────────────────────────────────────┐
│ DATA LAYER                                                   │
│ data/raw/dengue_onsite.csv  (31,320 ward-weeks)              │
│   -> build_features.py: lags, rolling, seasonality,          │
│      vulnerability, horizon                                  │
│   -> data/processed/features.csv  (125,280 rows)             │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ MODEL LAYER                                                  │
│ chronological split by target_date                           │
│   -> train HGB Poisson -> model.joblib                       │
│   -> validation residuals -> 80% intervals                   │
│   -> training distribution -> ward thresholds                │
│   -> metrics.json / test_predictions.csv                     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ INFERENCE LAYER                                              │
│ inference.py: predict / predict_batch / tier / interval /    │
│ driver explanation                                           │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ OPERATIONS LAYER                                             │
│ supplies.py -> allocation.py -> routing.py                   │
│                     │              │                         │
│                     ├──► dispatch.py ◄── simulation.py       │
│                     │              │                         │
│                     └──► assistant.py ◄──────────────────────┘
│ exports/                                                     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ INTERFACE LAYER                                              │
│ Streamlit: Home · Purpose · Dashboard · Command · Assistant ·│
│ About                                                        │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ VERIFICATION LAYER                                           │
│ unit tests · supply audit · lag audit · accuracy/calibration  │
│ offline socket-block proof · headless UI smoke test           │
└──────────────────────────────────────────────────────────────┘
```

## Layer responsibilities

| Layer | Files | Responsibility |
|---|---|---|
| **Data** | `data/raw/dengue_onsite.csv`, `src/data/build_features.py` | Ingest, validate, engineer lags and rolling features, export the feature table. |
| **Model** | `src/model/train.py` | Chronological split, train HGB Poisson, intervals, thresholds, metrics, artifact. |
| **Inference** | `src/inference.py` | Load model, predict one/batch, tier, risk index, interval, drivers. |
| **Operations** | `src/operations/*` | Supplies, allocation, routing, simulation, dispatch, assistant. |
| **Interface** | `app.py`, `ui/pages/*` | Home, Purpose, Dashboard, Command Center, Assistant, About. |
| **Verification** | `scripts/*`, `tests/*` | Supply audit, lag audit, accuracy/calibration, offline proof, UI smoke, unit tests. |

## Request flows

### Dashboard

```text
city / ward / as-of / horizon
  -> feature row lookup
  -> model.predict
  -> clip at zero
  -> ward thresholds -> tier + risk index
  -> validation residuals -> 80% interval
  -> occlusion -> driver breakdown
```

### Command Center — Allocate

```text
predict_batch(60 wards)
  -> cases_to_supplies
  -> WardDemand (tier, density, crews needed)
  -> allocate(stock %, total crews)
  -> per-ward ledger + city roll-up + deficit alerts
```

### Command Center — Overflow

```text
required beds
  -> derive_bed_capacity (p75 history, minimum 2 beds)
  -> capacity multiplier
  -> route_overflow (strictly intra-city)
  -> transfers + net deficits + city utilization + escalation
```

### Command Center — Simulate

```text
selected wards + crew type + crews + start week
  -> modify breeding-habitat or transmission features
  -> recursive model.predict with case-lag feedback
  -> cases / beds / saline per week
  -> ROI = cases prevented per crew
```

### Command Center — Dispatch

```text
allocation row + routing transfers
  -> SMS / WhatsApp / printable HTML
  -> /exports
```

### Assistant

```text
question + live context
  -> intent + entity extraction
  -> allocation / routing / metrics lookup
  -> bounded read-only what-if
  -> deterministic answer + sources
```

## Artifacts

| Artifact | Purpose |
|---|---|
| `data/raw/dengue_onsite.csv` | 31,320 ward-weeks, 15 columns, no missing values. |
| `data/processed/features.csv` | 125,280 rows = 60 wards × 522 weeks × 4 horizons. |
| `models/dengue_model.joblib` | Serialized HGB Poisson model. |
| `models/feature_columns.json` | Ordered 32 model inputs. |
| `models/risk_thresholds.json` | Per-ward Low / Medium / High / Critical thresholds. |
| `models/metrics.json` | MAE, RMSE, baselines, residual intervals, split. |
| `models/test_predictions.csv` | 25,200 holdout predictions. |
| `exports/` | Sample SMS, WhatsApp and printable HTML briefs + scenario manifest. |

# 6. Key numbers to memorise

| Fact | Value |
|---|---|
| Dataset | 31,320 ward-weeks · 60 wards · 5 cities · 522 weeks |
| Feature table | 125,280 rows = 60 × 522 × 4 |
| Train | 87,000 rows · 364 target weeks · through 2022-12-26 |
| Validation | 12,480 rows · 52 weeks · 2023 |
| Test | 25,200 rows · 105 weeks · 2024–2025 |
| Model | Poisson HGB · 32 features · one model for 4 horizons |
| Holdout MAE | **1.054** |
| Holdout RMSE | **2.097** |
| Seasonal-naive MAE | **1.693** |
| Persistence MAE | **1.748** all horizons |
| 80% interval coverage | **81.5%** |
| Supply match | **31,320 / 31,320 rows exact** |
| Saline | `round(31/13 × cases)` ≈ 2.385 bags/case |
| Kits | `round_half_down(7/6 × cases)` ≈ 1.167 kits/case |
| Beds | `round_half_down(0.3 × cases)` = 30% admission |
| Allocation demo | 60% stock, 8 crews → **749 / 1,249** saline; **31 / 60** supply gaps; **58 / 60** crew gaps |
| Routing demo | 1.20× capacity → **7** intra-city transfers, **0** unrouted |
| Simulator demo | 2 larvicide crews × 3 Dhaka North wards × 4 weeks → **~12.1** cases prevented |
| Runtime | **0** external API calls |

> **If a judge asks only one question:** “Where do your supply numbers come from?” Answer: we reverse-engineered the generator and match **31,320 of 31,320** rows exactly. `scripts/verify_supply_formulas.py` proves it.

# 7. Judge Q&A — short answers

### “How does the model work?”

> “It is a Poisson gradient-boosted model. For each ward/week/horizon it combines 1–4 week weather lags, case history, seasonality and vulnerability, and predicts the expected weekly case count. One model serves all four horizons.”

### “Why Poisson loss?”

> “Cases are non-negative counts, overdispersed and zero-inflated. Poisson loss matches the data-generating process better than squared error.”

### “How did you split train / validation / test?”

> “Chronologically by target date: train through 2022, validation 2023, untouched test 2024–25. No shuffling, no random split, no target leakage.”

### “Why 87,000 training rows?”

> “60 wards × 4 horizons × 364 weeks = 87,360, minus 360 early-boundary rows where the first target weeks only have 1–3 horizons available.”

### “What are the five new features?”

> “Phase 2 did not add five new model inputs. The model still uses 32 features. The five new operational capabilities are allocation, overflow routing, simulation, dispatch and offline resilience. The onsite file added three supply targets, which are excluded from the model.”

### “How do the five connect?”

> “The forecast feeds allocation; allocation’s bed demand feeds routing; allocation and routing feed dispatch; the simulator re-runs the model with modified features and produces ROI; offline wraps all of them; the assistant reads the live state.”

### “Why should we trust the forecast?”

> “Holdout MAE 1.054, RMSE 2.097, better than seasonal-naive 1.693 and persistence 1.748, with 81.5% empirical coverage for our 80% intervals. The supply formulas match all 31,320 rows exactly, and the pipeline runs with sockets blocked.”

### “Why not 100% accurate?”

> “Dengue is a noisy, partially observed process. A 100% holdout score would suggest leakage or overfitting. We publish calibrated uncertainty and beat both baselines instead.”

### “How does the simulator differ from a hardcoded percentage?”

> “It rewrites the model’s actual feature vector, re-runs the serialized model, and feeds each prediction into next week’s case lags. Zero crews reproduce the baseline exactly — that is an assertion in the test suite.”

### “Where does bed capacity come from?”

> “It is not in the dataset. We use a transparent planning baseline: each ward’s historical 75th-percentile weekly bed demand, minimum two beds. A live multiplier lets judges stress-test the router.”

### “Does the assistant use an LLM?”

> “No. It is a deterministic intent router over the live allocation, routing and metrics. It has no external API, no model download and no hallucinated numbers; unknown questions get a safe fallback.”

### “Does it work offline?”

> “Yes. `scripts/offline_check.py` blocks sockets and runs forecast, allocation, routing, assistant, simulation and dispatch to completion. The local app has zero runtime APIs.”

# 8. Honest limitations

- Bed capacity and simulator efficacy curves are **planning assumptions**, not measured causal effects.
- The simulator is a **historical hindcast**: it uses the bundled future weather from the feature table, not a live forecast ensemble.
- Weekly resolution and no ward geometry; cross-city transfers are deliberately out of scope.
- The prototype needs authentication, audit logging and a retraining schedule before production use.

# 9. Verification in five commands {.keep-with-previous}

```bash
python -m unittest discover              # 36 tests
python scripts/verify_supply_formulas.py # 31,320 / 31,320 exact
python scripts/verify_accuracy.py        # MAE 1.054, 80% coverage 81.5%
python scripts/offline_check.py          # socket-blocked full pipeline
python scripts/ui_smoke_test.py          # Home -> Command -> Assistant
```

<div class="closing">
  <div class="closing-mark">3AM</div>
  <div>
    <strong>Aedes Sentinel</strong> · Team 3AM · Phase 2 onsite final<br>
    Forecast to field action in under a minute, on a laptop, with the Wi-Fi off.
  </div>
</div>
