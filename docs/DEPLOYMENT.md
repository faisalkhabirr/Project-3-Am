# Deploying the Phase 2 build

The rubric requires a working live URL **and** a fully local offline path. The
local path is the primary judging path; the URL is for asynchronous review.

## 1. Push the Phase 1 repository

The build extends the existing repo; do not create a new one.

```bash
git push origin main
git tag -a version-onsite-final -m "Phase 2 onsite final"
git push origin version-onsite-final
```

## 2. Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → repository `faisalkhabirr/Project-3-Am` → branch `main`.
3. Main file path: `app.py`.
4. Advanced settings → Python version **3.12**.
5. Secrets: **none**. The app must not need any key.
6. Deploy. The first build installs `requirements.txt`; later boots are cached.
7. Open the URL and check:
   - Home renders;
   - Dashboard predicts;
   - Command Center sliders, routing, simulator and dispatch work;
   - the footer build hash matches `git rev-parse --short=12 HEAD`.

## 3. Put the URL in the repo

Add the live URL to the top of `README.md` (and optionally the About page), then
commit:

```bash
git add README.md
git commit -m "docs: add live Phase 2 URL"
git push origin main
```

Streamlit Cloud redeploys automatically. The commit hash changes, so the footer
will update too.

## 4. Offline fallback (primary demo)

If the venue Wi-Fi fails, run locally:

```bash
source .venv/bin/activate
streamlit run app.py
```

Then run the kill-switch proof in a second terminal:

```bash
python scripts/offline_check.py
```

The app, model, dataset, allocation, routing, simulator and dispatch templates
are all bundled. No secret, API key or network access is required.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Streamlit Cloud build fails on pandas/numpy | Confirm Python 3.12 in app settings; versions are pinned in `requirements.txt`. |
| Footer says `local` | The deployment omitted `.git`; set a `COMMIT_HASH` secret/env var to the release hash, or accept the local marker. |
| Command Center says model missing | Ensure `models/` and `data/processed/features.csv` were committed; they are part of the repo. |
| Simulator slow on first click | It re-runs the local model per ward-week; subsequent clicks are similar (no API latency). |
