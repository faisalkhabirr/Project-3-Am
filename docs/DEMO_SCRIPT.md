# 3-minute uncut demo — recording plan

**Goal:** 20/20. One continuous screen recording, ≤3:00, live app on localhost, voiceover,
no slides, no jump cuts. Target 2:45 so there is margin.

## Before recording

1. Start the app with the final artifacts:
   ```bash
   cd "Project 3Am"
   source .venv/bin/activate
   streamlit run app.py
   ```
2. Open http://localhost:8501 in Chrome/Edge. Use a clean profile, hide the bookmarks bar,
   zoom 100%, and go full screen (F11).
3. Warm the dashboard once: click **Open dashboard**, change a city/ward, return to the demo
   defaults, then click **Home** in the nav. The recording starts on the Home page.
4. Demo defaults:
   - Landing page: Home
   - Dashboard: City **Dhaka North** · Ward **DNCC Ward 12** · weather through **25 Aug 2025** ·
     horizon **+3 weeks**
5. Silence notifications (Focus Assist / Do Not Disturb), close Slack/WhatsApp, use a headset
   mic, and check the input level.
6. Recorder: **Windows Game Bar** (`Win+Alt+R` to start/stop; enable the microphone in
   Settings → Gaming → Captures) or OBS Studio (Window Capture + Mic). Record the browser
   window only. Do **not** add slides, titles, or edits.

## Script (target 2:45)

### 0:00–0:10 — Hook (Home page)
> "Dengue outbreaks in Dhaka and Chattogram spike weeks after the monsoon. By the time hospital
> admissions climb, vector-control teams are already late. Aedes Sentinel turns that biological
> lag into a two-to-four week early warning."

### 0:10–0:22 — Landing page and scope
> "This is Team 3AM's Aedes Sentinel: sixty wards across five cities, with a one-to-four week
> warning window. The whole thing runs locally. Let's open the dashboard."
>
> **Click “Open dashboard”.**

### 0:22–0:45 — Inputs and headline prediction
> "I'll use Dhaka North, DNCC Ward 12, weather observed through 25 August 2025, and a three-week
> horizon. The model predicts about twelve cases for the week of 15 September — a High risk tier,
> an eighty-percent interval of roughly eleven to fifteen, and a risk index of sixty-seven out
> of one hundred."

### 0:45–1:10 — Driver breakdown
> "The driver breakdown shows why. Seasonality and humidity over the last four weeks are pushing
> the forecast up, alongside rainfall in the two-to-four week lag window. These bars are computed
> locally by resetting each driver to its typical value and re-running the model. The tier and
> interval come from the model and this ward's own thresholds, not a static rule."

### 1:10–1:35 — Weather lag and vulnerability
> "This chart is the biological clock: rainfall today feeds cases two to four weeks later. The
> shaded band is the lag window the model uses. The vulnerability cards explain why identical
> rainfall hits this ward harder: seventy-two thousand people per square kilometre, a waterlogging
> risk of 0.92, and a sanitation score of 0.78."

### 1:35–1:55 — City targeting view
> "The city view ranks every ward for the same week. Ward 12 is highest, followed by Wards 19 and
> 09. The ward-by-week heatmap shows the next four weeks: the whole city stays elevated, but the
> top wards are consistently darker. This is the targeting view for spraying and larvicide teams."

### 1:55–2:25 — Validation and close
> "The Validation tab is the honest check. At the one-week horizon, MAE is 0.98 and RMSE is 1.95;
> across all horizons it is 1.05 and 2.10, beating the seasonal-naive MAE of 1.69 and persistence
> MAE of 1.75. Error grows with the horizon, which is exactly what an early-warning tool should
> show. Everything runs on a laptop CPU with zero external API calls. The README has the
> one-command launch, and ARCHITECTURE.md documents the split and cost controls. Aedes Sentinel
> gives public-health teams the weeks they need to act before the peak."

### 2:25–2:45 — Buffer / close on the dashboard
> "Team 3AM. Predicting dengue risk before it peaks."

## After recording

- Check duration is **≤ 3:00** and the audio is clear.
- Do not trim or edit. If a take is bad, re-record the whole thing.
- Upload as **unlisted YouTube** or Google Drive with link sharing.
- Put the link in `README.md` under a `## Demo video` heading.
- Keep the raw file as a backup.

## If a click goes wrong

Do not cut. Stay calm, narrate the correction ("let me switch back to Dhaka North"), and
continue. An honest uncut take scores better than a polished jump-cut.
