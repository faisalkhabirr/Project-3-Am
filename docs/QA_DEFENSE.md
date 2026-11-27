# Phase 2 Q&A defense sheet

Rehearse these out loud. The 30-point Q&A is the largest single criterion, and most
questions will probe the assumptions behind allocation, routing and simulation.

## Numbers to memorise

| Fact | Value |
|---|---|
| Dataset | 31,320 ward-weeks · 60 wards · 5 cities · 522 weeks |
| Supply match | **31,320 / 31,320 rows exact** for all three targets |
| Saline | `round(31/13 × cases)` ≈ 2.385 bags/case |
| Kits | `round_half_down(7/6 × cases)` ≈ 1.167 kits/case |
| Beds | `round_half_down(0.3 × cases)` = 30% admission |
| Model holdout | MAE **1.05**, RMSE **2.10**; seasonal-naive 1.69; persistence 1.75 |
| Allocation demo | 60% stock, 8 crews → 749/1,249 saline; supply gaps in 31/60 wards; crew gaps in 58/60 |
| Routing demo | 1.20× capacity → 7 intra-city transfers, 0 unrouted beds |
| Simulator demo | 2 larvicide crews × 3 Dhaka North wards × 4 weeks → ~12.1 cases prevented |
| Offline proof | sockets blocked, full pipeline completes, 15 dispatch files written |

## The questions

### 1. "Derive your supply conversion. Why should we believe it?"
The onsite file adds three targets to the identical Phase 1 case series. We
reverse-engineered each target and verified it row-for-row:
`saline = round(31/13 × cases)`, `kits = round_half_down(7/6 × cases)`,
`beds = round_half_down(0.3 × cases)`. `scripts/verify_supply_formulas.py`
recomputes all three on all 31,320 rows and exits non-zero on any mismatch. It
also proves cases and weather are byte-identical to Phase 1.

### 2. "Why not retrain the model on the new columns?"
Because they are deterministic functions of `dengue_cases`. There is no new
predictive information, and retraining would discard the untouched 2024–25
holdout result and the calibrated intervals. The validated Phase 1 forecaster
stays frozen and feeds the operational layer.

### 3. "How does allocation prioritise when stock is scarce?"
Priority = `0.65 × tier_rank + 0.35 × density_norm`, where density is normalised
across the 60 wards. Every ward gets a protected 10% floor; the remaining stock
is water-filled in strict priority order; crews have no floor and go to the
highest-priority wards first. Integer units only.

### 4. "Prove it behaves correctly."
Two invariant tests: **monotonicity** (raising stock never reduces any ward's
allocation) and **conservation** (allocated ≤ available for saline, kits and
beds). A cap test proves no ward receives more than its requirement. The UI
reports **Supply %** and **Crew %** separately — a ward with full supplies but
no crew is still flagged as short. The full suite is `python -m unittest discover`.

### 5. "Isn't the 10% floor unfair to high-risk wards?"
It is a deliberate policy trade-off: a floor prevents a low-risk ward from being
starved to zero while a critical ward receives surplus. The remaining 90% is
strictly priority-ordered, so high-risk, dense wards are covered first. The floor
is a parameter (`min_floor`), not a hardcode, and can be set to zero.

### 6. "At 30% stock, what happens?"
At the demo week, 30% stock leaves **374 of 1,249 saline bags** for 60 wards; supply
gaps rise from 31 to **49 of 60 wards**, while 58 wards are short on crews even at
60%. The highest tier/density wards keep the largest share, and the UI raises a live
critical-deficit banner naming the worst wards. At 100% stock, supply gaps fall to
zero but crew scarcity remains visible. Nothing is silently rounded up.

### 7. "Where does bed capacity come from? It's not in the dataset."
Correct — it is not. We use a transparent planning baseline: each ward's
historical **75th-percentile weekly bed demand**, minimum 2 beds. A 0.6–2.0×
multiplier lets judges stress-test the router live. We label this an assumption,
not a claim about literal ward beds.

### 8. "Show that routing is really intra-city."
`route_overflow()` groups wards by city, sorts donors by lowest utilization, and
matches deficits only inside that pool. Tests assert: no transfer crosses a city
boundary, no donor exceeds headroom, and transfers exactly resolve the routed
deficit. If a city cannot absorb its own demand, it is added to
`escalated_cities` and the UI shows a regional escalation alert.

### 9. "How is the simulator different from a hardcoded 20% reduction?"
It rewrites the model's actual feature vector — larvicide scales breeding-habitat
features (`rain_lag_*`, `rain_cum_*`, effective waterlogging risk); fogging
scales case-history and the vector-pressure proxy — then re-runs the serialized
Phase 1 model. Each prediction is fed back into next week's `cases_lag_*`, so the
outbreak momentum genuinely diverges. Zero crews reproduces the baseline exactly;
that is an assertion in the test suite.

### 10. "Where did the efficacy curves come from?"
They are conservative planning assumptions informed by vector-control practice:
fogging acts quickly with a short residual; larvicide builds over 2–3 weeks and
lasts longer. We expose them as constants, not hidden magic, and we do not claim
they are causal estimates. Prospective field evaluation is the next step.

### 11. "Why does fogging touch the waterlogging index?"
The model has no explicit adult-mosquito index. We use the case-history and the
waterlogging/vector-pressure proxy to represent reduced adult vector pressure.
This is a documented structural assumption; the alternative — silently changing
weather — would be less defensible.

### 12. "Prove it works offline."
`scripts/offline_check.py` patches `socket.socket`, `socket.create_connection`
and `socket.getaddrinfo` to raise, then runs forecast → allocation → routing →
simulation → dispatch. It completes and writes 15 dispatch files. The app itself
has no runtime API call; the nav says "Local · 0 APIs".

### 13. "What stops leakage in the model?"
Lags and rolling statistics are computed within ward after sorting by date and
shifted before use; splits are by **target date**, not as-of date; rows crossing a
split boundary are removed; scalers/imputers are fit on training only; the test
window is used once. `scripts/verify_lags.py` independently re-derives the lags.

### 14. "What is hardcoded vs learned?"
Learned: the case forecast (Poisson gradient boosting on 32 features). Deterministic
from the dataset: supply conversion. Explicit policy: priority weights, protected
floor and crew derivation. Explicit assumption: bed capacity baseline and efficacy
curves. We separate these categories on purpose so judges can challenge each one.

### 15. "What happens if a city runs out of beds?"
The router resolves what it can inside the pool and adds the city to
`escalated_cities`; the UI shows a red regional-escalation alert and the
unrouted deficits by ward. The system does not invent cross-city transfers
because those are outside the modelled referral boundary.

### 16. "What if the model is wrong?"
The forecast is decision support, not diagnosis. The UI shows the 80% interval and
the holdout metrics; the allocation can be run on the upper interval for
conservative planning. A production path adds prospective validation and a
retraining schedule.

### 17. "What is the operational value beyond the rubric?"
A commissioner gets one packet: forecast, allocated vs required, deficit flag,
crew assignment and overflow route — in SMS, WhatsApp and print form — generated
in seconds on a laptop. The protected floor, capacity slider and scenario
manifest make the trade-offs explicit rather than hidden in a spreadsheet.

### 18. "What would you do with more time?"
Prospective ward-level validation, entomological calibration of the efficacy
curves, a live weather adapter, authentication/audit logging, and a proper
capacity survey to replace the p75 baseline.

## Honest limitations to state before judges find them

- Bed capacity and efficacy curves are planning assumptions, not measured data.
- The simulator is a hindcast: it uses historical future weather from the feature
  table, not a live forecast ensemble.
- Weekly resolution and no ward geometry; cross-city transfers are deliberately
  out of scope.
- The current deployment is a prototype; production needs auth, audit and a
  retraining schedule.
