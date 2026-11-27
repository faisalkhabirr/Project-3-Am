"""Dashboard page — the existing analytical UI, moved behind the router.

All widget keys, tab labels, data loading and prediction logic are unchanged.
Only the page chrome and card styling moved to ``ui/theme.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, config, inference
from src import data_access as data


def ordinal(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    n = int(round(float(value)))
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} percentile"


def mini_card(col, label: str, value: str, sub: str) -> None:
    col.markdown(
        f"<div class='mini-card'><div class='mini-label'>{label}</div>"
        f"<div class='mini-value'>{value}</div><div class='mini-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def water_label(value: float) -> str:
    if value >= 0.75:
        return "Very high · drainage bottleneck"
    if value >= 0.55:
        return "High · standing-water risk"
    if value >= 0.35:
        return "Moderate"
    return "Low"


def sanitation_label(value: float) -> str:
    if value < 0.40:
        return "Weak · waste collection gap"
    if value < 0.70:
        return "Moderate"
    return "Strong"


def metric_value(metrics: dict, key: str, horizon: int | None = None):
    value = metrics.get(key) if isinstance(metrics, dict) else None
    if isinstance(value, dict):
        value = value.get(str(horizon), value.get(horizon))
    return value


def default_as_of(dates: list[pd.Timestamp]) -> pd.Timestamp:
    """Prefer the last August week in the latest year for a meaningful demo."""
    if not dates:
        return pd.Timestamp.today().normalize()
    latest_year = dates[-1].year
    august = [d for d in dates if d.year == latest_year and d.month == 8]
    return august[-1] if august else dates[-1]


def load_model_and_features():
    bundle = None
    features = None
    error = None
    try:
        bundle = inference.load_bundle()
        features = data.load_features()
        if features is None:
            raise inference.ModelNotReady(
                f"Feature table not found at {config.FEATURES}."
            )
    except Exception as exc:
        error = str(exc)
    return bundle, features, error


def render_dashboard() -> None:
    """Render the full ward-level dashboard."""
    # ---------------------------------------------------------------------------
    # Data + model
    # ---------------------------------------------------------------------------
    try:
        raw = data.load_raw()
        meta = data.load_ward_meta()
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    dates = data.available_dates(raw)
    if not dates:
        st.error("The dataset has no dated rows.")
        st.stop()

    bundle, features, model_error = load_model_and_features()
    model_ready = bundle is not None and features is not None

    cities = [c for c in config.CITY_ORDER if c in set(raw[config.CITY_COL])]
    if not cities:
        cities = sorted(raw[config.CITY_COL].unique())

    default_city = "Dhaka North" if "Dhaka North" in cities else cities[0]
    default_date = default_as_of(dates)

    # ---------------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------------
    status_class = "live" if model_ready else "pending"
    status_text = "Model live" if model_ready else "Model pending"
    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-kicker">Dengue early warning · Bangladesh</div>
      <h1>{config.APP_TITLE}</h1>
      <p>{config.APP_TAGLINE}. Turn rainfall, temperature and humidity into a 2–4 week
      warning for every ward — before hospitals fill up.</p>
      <div class="hero-badges">
        <span class="hero-badge {status_class}">● {status_text}</span>
        <span class="hero-badge">Local inference · 0 API calls</span>
        <span class="hero-badge">{raw[config.WARD_COL].nunique()} wards · {len(cities)} cities</span>
        <span class="hero-badge">Weekly · 2016–2025</span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # Controls
    # ---------------------------------------------------------------------------
    with st.container(key="controls_card"):
        c1, c2, c3, c4 = st.columns([1.0, 1.35, 1.25, 1.0], gap="medium")
        city = c1.selectbox(
            "City",
            cities,
            index=cities.index(default_city),
            key="city_select",
        )
        wards = data.wards_for_city(meta, city)
        preferred_ward = "DNCC_Ward_12" if "DNCC_Ward_12" in wards else wards[0]
        ward = c2.selectbox(
            "Ward",
            wards,
            index=wards.index(preferred_ward),
            format_func=lambda w: str(w).replace("_", " "),
            key=f"ward_select_{city}",
        )
        dates_desc = list(reversed(dates))
        default_index = dates_desc.index(default_date) if default_date in dates_desc else 0
        as_of = c3.selectbox(
            "Weather observed through",
            dates_desc,
            index=default_index,
            format_func=data.format_date,
            key="as_of_select",
        )
        horizon = c4.select_slider(
            "Forecast horizon",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda h: f"+{h} week" + ("s" if int(h) > 1 else ""),
            key="horizon_select",
        )
        st.caption(
            "Example default: the last peak-season week. Change the ward, week or horizon "
            "to explore the forecast."
        )

    as_of = pd.Timestamp(as_of).normalize()
    horizon = int(horizon)
    target_date = as_of + pd.Timedelta(weeks=horizon)
    st.markdown(
        f"<div class='target-line'>Forecast target: <b>week of {data.format_date(target_date)}</b> "
        f"· weather observed through <b>{data.format_date(as_of)}</b> · horizon <b>+{horizon} weeks</b></div>",
        unsafe_allow_html=True,
    )

    if not model_ready and model_error:
        st.markdown(
            f"<div class='pending'><b>Model handoff pending.</b> The data views are live; "
            f"predictions will appear automatically once the ML artifacts land. "
            f"<br><span style='opacity:.75'>{model_error}</span></div>",
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------------
    pred = None
    drivers = None
    forecast_df = None
    if model_ready:
        try:
            pred = inference.predict(ward, as_of, horizon, features=features, bundle=bundle)
        except Exception as exc:
            st.error(f"Prediction failed for {ward} on {data.format_date(as_of)}: {exc}")

        if pred is not None:
            try:
                drivers = inference.explain(
                    ward, as_of, horizon, features=features, bundle=bundle
                )
            except Exception:
                drivers = None

            rows = []
            for h in range(1, 5):
                try:
                    item = inference.predict(ward, as_of, h, features=features, bundle=bundle)
                    rows.append(
                        {
                            "horizon": h,
                            "target_date": item.target_date,
                            "cases": item.cases,
                            "low": item.low,
                            "high": item.high,
                            "tier": item.tier,
                        }
                    )
                except Exception:
                    continue
            forecast_df = pd.DataFrame(rows)

    try:
        context = data.lag_context(raw, meta, ward, as_of)
    except Exception as exc:
        context = None
        st.warning(f"Could not build the driver context: {exc}")

    # ---------------------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------------------
    risk_tab, validation_tab, method_tab = st.tabs(
        ["Risk dashboard", "Validation", "Method & cost"]
    )

    with risk_tab:
        left, right = st.columns([1.0, 1.7], gap="large")

        with left:
            with st.container(key="risk_card"):
                st.markdown("<div class='card-label'>Risk forecast</div>", unsafe_allow_html=True)
                if pred is not None:
                    st.plotly_chart(
                        charts.risk_gauge(pred.risk_index, pred.tier),
                        width="stretch",
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"gauge_{ward}_{as_of.date()}_{horizon}",
                    )
                    st.markdown(
                        charts.tier_bar(pred.tier, f"+{pred.horizon} weeks"),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='risk-cases'>{pred.cases:.1f} "
                        f"<span>predicted cases</span></div>"
                        f"<div class='risk-sub'>80% interval: {pred.low:.1f}–{pred.high:.1f} cases</div>"
                        f"<div class='risk-sub'>Target: week of {data.format_date(pred.target_date)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='pending'>Waiting for the serialized model and engineered "
                        "features. This card will show the live risk tier, predicted cases and "
                        "the 80% interval.</div>",
                        unsafe_allow_html=True,
                    )

        with right:
            with st.container(key="forecast_card"):
                if forecast_df is not None and not forecast_df.empty:
                    st.plotly_chart(
                        charts.forecast_figure(forecast_df, as_of),
                        width="stretch",
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"forecast_{ward}_{as_of.date()}",
                    )
                else:
                    if context is not None:
                        st.plotly_chart(
                            charts.rainfall_history_figure(context["history"], as_of),
                            width="stretch",
                            theme=None,
                            config={"displayModeBar": False},
                            key=f"history_fallback_{ward}_{as_of.date()}",
                        )
                    st.caption(
                        "Historical rainfall and cases are live. The 4-week forecast will appear "
                        "when the model artifacts are in place."
                    )

        st.markdown("<div class='section-title'>Why this score?</div>", unsafe_allow_html=True)
        dleft, dright = st.columns([1.1, 1.0], gap="large")
        with dleft:
            with st.container(key="driver_card"):
                if drivers is not None and not drivers.empty:
                    st.plotly_chart(
                        charts.driver_figure(drivers),
                        width="stretch",
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"drivers_{ward}_{as_of.date()}_{horizon}",
                    )
                    st.caption(
                        "Contribution = change in predicted cases when a driver is reset to its "
                        "typical value (single-feature occlusion). Red raises risk; green lowers it."
                    )
                else:
                    st.markdown(
                        "<div class='pending'>Driver contributions will appear with the model. "
                        "They are computed locally from the same serialized estimator.</div>",
                        unsafe_allow_html=True,
                    )
        with dright:
            with st.container(key="rainfall_card"):
                if context is not None:
                    st.plotly_chart(
                        charts.rainfall_history_figure(context["history"], as_of),
                        width="stretch",
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"history_{ward}_{as_of.date()}",
                    )
                    st.caption(
                        "The shaded window is the 2–4 week lag between rainfall and the case "
                        "curve — the biological clock the model is built around."
                    )
                else:
                    st.info("Weather history is unavailable for this ward.")

        if context is not None:
            m1, m2, m3, m4 = st.columns(4, gap="medium")
            mini_card(
                m1,
                "Rainfall · last 4 weeks",
                f"{context['rain_4w']:.0f} mm",
                f"{ordinal(context['rain_4w_pct'])} for this ward",
            )
            mini_card(
                m2,
                "Relative humidity",
                "—" if context["humidity"] is None else f"{context['humidity']:.0f}%",
                f"{ordinal(context['humidity_pct'])} for this ward",
            )
            mini_card(
                m3,
                "Average temperature",
                "—" if context["temp"] is None else f"{context['temp']:.1f} °C",
                f"{ordinal(context['temp_pct'])} for this ward",
            )
            mini_card(
                m4,
                "Waterlogging risk",
                f"{context['water']:.2f}",
                water_label(context["water"]),
            )

            st.markdown("<div class='section-title'>Ward vulnerability</div>", unsafe_allow_html=True)
            v1, v2, v3 = st.columns(3, gap="medium")
            mini_card(
                v1,
                "Population density",
                f"{context['density']:,.0f} /km²",
                "Crowding amplifies transmission",
            )
            mini_card(
                v2,
                "Waterlogging risk index",
                f"{context['water']:.2f}",
                water_label(context["water"]),
            )
            mini_card(
                v3,
                "Sanitation score",
                f"{context['sanitation']:.2f}",
                sanitation_label(context["sanitation"]),
            )
            st.caption(
                "Identical rainfall hits these wards differently. The model uses vulnerability "
                "features alongside the weather lags, so the ranking is hyper-local."
            )

        st.markdown("<div class='section-title'>City view</div>", unsafe_allow_html=True)
        if model_ready:
            try:
                batch = inference.predict_batch(
                    wards, as_of, [horizon], features=features, bundle=bundle
                )
                if not batch.empty and batch["cases"].notna().any():
                    st.plotly_chart(
                        charts.ward_ranking_figure(batch, top_n=12),
                        width="stretch",
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"ranking_{city}_{as_of.date()}_{horizon}",
                    )
                    table = batch.merge(meta, on=config.WARD_COL, how="left")
                    table = table.sort_values("cases", ascending=False)
                    table = table[
                        [
                            config.WARD_COL,
                            "cases",
                            "tier",
                            "risk_index",
                            "water_logging_risk_index",
                            "sanitation_score",
                            "population_density_sqkm",
                        ]
                    ].rename(
                        columns={
                            config.WARD_COL: "Ward",
                            "cases": "Predicted cases",
                            "tier": "Tier",
                            "risk_index": "Risk index",
                            "water_logging_risk_index": "Waterlogging",
                            "sanitation_score": "Sanitation",
                            "population_density_sqkm": "Density /km²",
                        }
                    )
                    table["Ward"] = table["Ward"].str.replace("_", " ", regex=False)
                    st.dataframe(
                        table,
                        hide_index=True,
                        width="stretch",
                        height=min(430, 60 + 32 * len(table)),
                        column_config={
                            "Predicted cases": st.column_config.NumberColumn(format="%.1f"),
                            "Risk index": st.column_config.ProgressColumn(
                                min_value=0, max_value=100, format="%.0f"
                            ),
                            "Waterlogging": st.column_config.NumberColumn(format="%.2f"),
                            "Sanitation": st.column_config.NumberColumn(format="%.2f"),
                            "Density /km²": st.column_config.NumberColumn(format="%d"),
                        },
                    )
                    st.caption(
                        "Risk index is relative to each ward's own thresholds; predicted cases "
                        "are comparable across wards."
                    )
                else:
                    st.info("No batch predictions available for this city/week.")
            except Exception as exc:
                st.warning(f"Could not build the city ranking: {exc}")
        else:
            st.info("The city-wide ranking will appear when the model is available.")

        st.markdown("<div class='section-title'>Ward × week heatmap</div>", unsafe_allow_html=True)
        if model_ready:
            try:
                batch_all = inference.predict_batch(
                    wards, as_of, [1, 2, 3, 4], features=features, bundle=bundle
                )
                matrix = batch_all.pivot_table(
                    index=config.WARD_COL,
                    columns="target_date",
                    values="cases",
                    aggfunc="last",
                )
                if not matrix.empty:
                    order = matrix.sum(axis=1).sort_values(ascending=False).index
                    matrix = matrix.loc[order]
                    st.plotly_chart(
                        charts.heatmap_figure(matrix),
                        width="stretch",
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"heatmap_{city}_{as_of.date()}",
                    )
                    st.caption(
                        "Predicted weekly cases by ward and forecast week. The color scale is absolute "
                        "(cases), so a light week stays light even when it is the highest in its city."
                    )
                else:
                    st.info("No heatmap data available for this city/week.")
            except Exception as exc:
                st.warning(f"Could not build the heatmap: {exc}")
        else:
            st.info("The ward × week heatmap will appear when the model is available.")

    with validation_tab:
        st.markdown("### Chronological validation")
        st.markdown(
            "The model is evaluated only on weeks *after* its training window. "
            "No shuffling, no future rows, and no target leakage across the split."
        )
        test = inference.load_test_predictions()
        metrics = bundle["metrics"] if bundle else {}

        if test is None:
            st.markdown(
                "<div class='pending'>Validation artifacts are not available yet. "
                "The trainer will export <code>models/test_predictions.csv</code> and "
                "<code>models/metrics.json</code>.</div>",
                unsafe_allow_html=True,
            )
        elif "y_true" not in test.columns or "y_pred" not in test.columns:
            st.error("test_predictions.csv must contain `y_true` and `y_pred` columns.")
        else:
            test = test.copy()
            available_horizons = (
                sorted(test[config.HORIZON_COL].dropna().astype(int).unique())
                if config.HORIZON_COL in test.columns
                else [1]
            )
            selected = st.selectbox(
                "Validation horizon",
                available_horizons,
                format_func=lambda h: f"+{int(h)} week" + ("s" if int(h) > 1 else ""),
                key="validation_horizon",
            )
            subset = (
                test.loc[test[config.HORIZON_COL] == selected]
                if config.HORIZON_COL in test.columns
                else test
            )
            mae = float((subset["y_true"] - subset["y_pred"]).abs().mean())
            rmse = float(np.sqrt(((subset["y_true"] - subset["y_pred"]) ** 2).mean()))
            baseline = metric_value(metrics, "baseline_mae", selected) or metric_value(
                metrics, "seasonal_naive_mae", selected
            )
            persistence = metric_value(metrics, "persistence_mae", selected)

            k1, k2, k3, k4 = st.columns(4, gap="medium")
            k1.metric("MAE", f"{mae:.2f}", help="Cases per ward-week, chronological holdout")
            k2.metric("RMSE", f"{rmse:.2f}", help="Cases per ward-week, chronological holdout")
            if baseline is not None:
                delta = (mae - float(baseline)) / max(float(baseline), 1e-9) * 100
                k3.metric(
                    "Seasonal-naive MAE",
                    f"{float(baseline):.2f}",
                    f"{delta:+.1f}%",
                    delta_color="inverse",
                    help="Lower is better. Negative delta means the model beats the baseline.",
                )
            else:
                k3.metric("Seasonal-naive MAE", "—", help="Baseline not exported yet")
            if persistence is not None:
                delta = (mae - float(persistence)) / max(float(persistence), 1e-9) * 100
                k4.metric(
                    "Persistence MAE",
                    f"{float(persistence):.2f}",
                    f"{delta:+.1f}%",
                    delta_color="inverse",
                )
            else:
                k4.metric("Persistence MAE", "—", help="Baseline not exported yet")

            overall_mae = metrics.get("mae") if isinstance(metrics, dict) else None
            overall_rmse = metrics.get("rmse") if isinstance(metrics, dict) else None
            if overall_mae is None:
                overall_mae = float((test["y_true"] - test["y_pred"]).abs().mean())
                overall_rmse = float(
                    np.sqrt(((test["y_true"] - test["y_pred"]) ** 2).mean())
                )
            st.caption(
                f"Overall across horizons 1–4: MAE {float(overall_mae):.2f} · "
                f"RMSE {float(overall_rmse):.2f}. The cards above show the selected "
                f"+{int(selected)} week horizon."
            )

            st.plotly_chart(
                charts.validation_figure(test, horizon=selected),
                width="stretch",
                theme=None,
                config={"displayModeBar": False},
                key=f"validation_actual_{selected}",
            )
            st.plotly_chart(
                charts.validation_error_figure(test),
                width="stretch",
                theme=None,
                config={"displayModeBar": False},
                key="validation_error",
            )

            st.markdown("### Split")
            st.dataframe(
                inference.split_summary(metrics),
                hide_index=True,
                width="stretch",
            )
            model_name = metric_value(metrics, "model_name") or "—"
            trained_at = metric_value(metrics, "trained_at") or "—"
            st.caption(
                f"Model: {model_name} · trained {trained_at} · "
                f"{len(subset):,} holdout ward-week rows at this horizon."
            )

    with method_tab:
        st.markdown("### How it works")
        st.markdown(
            "<div class='flow'>"
            "<span class='flow-step'>Ward-week weather</span><span class='flow-arrow'>→</span>"
            "<span class='flow-step'>Grouped 1–4 week lags</span><span class='flow-arrow'>→</span>"
            "<span class='flow-step'>Chronological training</span><span class='flow-arrow'>→</span>"
            "<span class='flow-step'>Serialized model</span><span class='flow-arrow'>→</span>"
            "<span class='flow-step'>Local inference</span><span class='flow-arrow'>→</span>"
            "<span class='flow-step'>Ward risk dashboard</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
    **The biological clock.** Heavy rain does not create cases the next morning. Water sits in
    drains and containers, *Aedes aegypti* eggs mature into adults, then viral incubation and
    symptom onset follow. The model therefore sees rainfall **1–4 weeks** before the target week,
    not same-week weather.

    **Hyper-local vulnerability.** Each ward carries a population density, waterlogging risk and
    sanitation score. These static features let identical rainfall produce different risk in
    adjacent wards.
    """
        )

        st.markdown("### Multi-week lag features")
        lag_table = pd.DataFrame(
            [
                ("rain_lag_1w", "Rainfall 7 days before the as-of week", "Early breeding signal"),
                ("rain_lag_2w", "Rainfall 14 days before the as-of week", "Larva → adult window"),
                ("rain_lag_3w", "Rainfall 21 days before the as-of week", "Peak transmission window"),
                ("rain_lag_4w", "Rainfall 28 days before the as-of week", "Long-tail incubation"),
                ("rain_cum_4w", "Total rainfall over the last 4 weeks", "Sustained standing water"),
                ("humidity_lag_*", "Relative humidity at 1–4 week lags", "Vector survival"),
                ("temp_lag_*", "Average temperature at 1–4 week lags", "Extrinsic incubation"),
                ("water_logging_risk_index", "Ward drainage bottleneck", "Breeding-site density"),
                ("sanitation_score", "Ward waste collection", "Container availability"),
                ("population_density_sqkm", "Ward crowding", "Human-vector contact"),
            ],
            columns=["Feature", "Definition", "Why it matters"],
        )
        st.dataframe(lag_table, hide_index=True, width="stretch")
        st.caption(
            "Every lag and rolling statistic is computed within `ward_id`; no ward borrows "
            "another ward's weather history."
        )

        st.markdown("### Chronological split")
        st.dataframe(
            inference.split_summary(bundle["metrics"] if bundle else {}),
            hide_index=True,
            width="stretch",
        )

        st.markdown("### Model")
        model_name = metric_value(bundle["metrics"], "model_name") if bundle else None
        st.markdown(
            f"""
    The prediction engine is a **{model_name or 'tree-based regressor'}** trained on ward-week
    rows. It predicts weekly dengue cases for a selected ward and lead time, then maps the
    prediction onto ward-specific **Low / Medium / High / Critical** thresholds.

    Why this family of model: it captures non-linear weather interactions, handles the many
    zero-case weeks, needs no GPU, and serializes to a small joblib artifact for fast local
    inference.

    **Risk tiers.** Thresholds are derived from each ward's historical case distribution
    (median / p75 / p90), so a high-burden ward is not automatically flagged Critical every week.
    The app reads the thresholds from `models/risk_thresholds.json` and applies them live.
    """
        )

        st.markdown("### Compute & API cost controls")
        cost = pd.DataFrame(
            [
                ("Runtime external API calls", "0 — dataset and model are bundled with the repo"),
                ("Data acquisition", "One-time `scripts/fetch_data.py`; cached CSV is committed"),
                ("Model loading", "`joblib` once per process via `st.cache_resource`"),
                ("Data loading", "`st.cache_data` keyed on file mtime; no per-widget reloads"),
                ("City ranking / heatmap", "One vectorized `.predict()` for all wards × horizons"),
                ("LLM usage", "None in the product; AI assistants used only during development"),
                ("Hardware", "Laptop CPU; no GPU, no paid inference endpoint"),
                ("Typical latency", "Under 300 ms per prediction after warm-up"),
            ],
            columns=["Control", "Implementation"],
        )
        st.dataframe(cost, hide_index=True, width="stretch")

        st.markdown("### Limitations")
        st.markdown(
            """
    - The prototype uses the historical ward-week record, not a live weather feed. A production
      deployment would add a cached daily weather adapter and a scheduled retrain.
    - Ward boundaries and coordinates are not in the dataset, so the heatmap is a ward × week
      matrix rather than a geographic map.
    - Weekly resolution matches the surveillance data; daily lags would need finer-grained input.
    - Risk tiers are decision support, not a clinical diagnosis.
    """
        )
