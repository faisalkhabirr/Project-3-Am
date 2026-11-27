# Phase 2 onsite walkthrough — 7:00 uncut recording plan

**Goal:** full marks on the 10-point video criterion. One continuous screen recording,
voiceover, no slides, no jump cuts, no edits. Target **6:40** so there is margin under 7:00.

The rubric explicitly checks: all five modules, live walkthrough, and a commit hash that
matches the repo. The app footer prints the current short git hash; leave it visible.

## Before recording

1. Start the app from the repo root with the final artifacts:
   ```bash
   source .venv/bin/activate
   streamlit run app.py
   ```
2. Open http://localhost:8501 in Chrome/Edge, clean profile, bookmarks hidden, 100% zoom, F11.
3. Confirm the footer shows the same hash as `git rev-parse --short=12 HEAD`.
4. Warm the app once: Home → Dashboard → Command, change a slider, click **Reset demo defaults**
   on the Command page, then return to the defaults below.
5. Demo defaults:
   - Dashboard: **Dhaka North**, ward **DNCC Ward 12**, weather through **25 Aug 2025**, **+3 weeks**.
   - Command: **Dhaka North**, **25 Aug 2025**, **+2 weeks**, stock **60%**, crews **8**, capacity **1.20×**.
6. Have a terminal window ready with the repo root for the offline proof and `git log`.
7. Disable notifications; use a headset mic; record the browser window only. Windows Game Bar
   (`Win+Alt+R`) or OBS, no editing.

## Script (target 6:40)

### 0:00–0:30 — Hook and claim (Home)
> "Dengue outbreaks in Bangladesh peak weeks after the monsoon. A forecast is only useful if a
> ward commissioner can turn it into saline, beds, crews and a dispatch notice. This is Aedes
> Sentinel — Team 3AM's Phase 2 command system. Everything you are about to see runs locally, with
> zero API calls. I will prove that by cutting the network at the end."

### 0:30–1:15 — Forecast (Dashboard)
> "The Phase 1 model is unchanged and still validated: Poisson gradient boosting, MAE 1.05 on the
> untouched 2024–25 holdout, beating seasonal-naive and persistence. Here is DNCC Ward 12 at a
> three-week horizon: about twelve cases, High tier, an 80% interval, and the drivers behind it —
> four-week humidity, seasonality, and rainfall in the two-to-four-week lag window. That is the
> warning. Now the operations."

### 1:15–2:15 — Constrained allocation (Command → Allocate)
> "I switch to the Command Center. Sixty wards, Dhaka North selected, 60% of regional stock, eight
> crews. The engine converts predicted cases to saline, kits and beds using the exact dataset
> conversions — 2.385 bags per case, 1.167 kits, 30% admission. It protects a ten-percent floor
> for every ward, then water-fills by risk tier times population density. At 60% stock, 31 of
> 60 wards have a supply gap and 58 are short on crews; 749 of 1,249 saline bags are available.
> The red badges are live. Watch what happens at 30%."
>
> **Drag stock to 30%.** Pause for the recompute.
>
> "More wards fall into deficit, and the highest-risk, densest wards keep their allocation first.
> The city roll-up and the ward ledger update in under a second. No solver, no API, deterministic
> math."

### 2:15–3:05 — Overflow routing (Command → Overflow)
> "Beds are the next constraint. The dataset has no bed capacity, so we use a transparent planning
> baseline: each ward's historical 75th-percentile weekly bed demand, minimum two beds. A
> capacity multiplier lets a judge stress-test it. At 1.2×, the router makes seven intra-city
> transfers across the five pools — two of them into Dhaka North wards — strictly inside each
> city; it never crosses a city boundary. The city table shows demand, capacity, headroom and
> utilization. If I drop the multiplier, every city escalates — that is the regional exhaustion
> flag."

### 3:05–4:35 — Counterfactual simulator (Command → Simulate)
> "Now the what-if. I select three wards, two larvicide crews each, start week one, four-week
> horizon, and run. This is not a hardcoded percentage: the simulator rewrites breeding-habitat
> features, re-runs the serialized Phase 1 model, and feeds each prediction back into next week's
> case lags. The baseline dotted line is what happens without crews; the green line is the
> counterfactual. Here the top ward prevents about 5.7 cases over four weeks — roughly 2.8 cases
> per crew — and the three-ward package prevents about 12.1 cases. The ROI table ranks every
> ward. Zero crews reproduces the baseline exactly, and that is a unit test."

### 4:35–5:25 — Dispatch (Command → Dispatch)
> "One click turns the allocation into field language. The SMS is under 160 characters: ward,
> cases, saline, kits, beds, crews, deficit and overflow. The WhatsApp message is formatted for a
> phone. The printable brief is a self-contained HTML page with the same numbers, ready for the
> commissioner's printer. All three come from one validated set of values, with no LLM and no
> network call. The packet also exports to the /exports folder for the repo."

### 5:25–6:15 — Offline kill switch (terminal)
> "Here is the resilience proof. I am going to block the sockets, then run the entire pipeline:
> forecast, allocation, routing, simulation and dispatch."
>
> **Run `python scripts/offline_check.py`.** Let the PASS line land.
>
> "Sockets blocked. Sixty wards forecast, 749 of 1,249 saline bags allocated, seven transfers,
> cases prevented, fifteen dispatch files — all offline. If any module reached for the network,
> this script would fail."

### 6:15–6:45 — Close and commit hash
> "The repo is the same Phase 1 repository, with continuous per-module commits and a
> version-onsite-final tag. The footer prints the build hash; let me show it matches."
>
> **Run `git log --oneline -5` and `git rev-parse --short=12 HEAD`.**
>
> "Aedes Sentinel. Forecast to field action in under a minute, on a laptop, with the Wi-Fi off.
> Team 3AM."

## After recording

- Duration **≤ 7:00**; audio clear; no edits.
- Upload unlisted YouTube or Drive with link sharing.
- Put the link in `README.md` under **Demo video**.
- Keep the raw file as a backup.
- If a click goes wrong, do not cut: narrate the correction and continue.
