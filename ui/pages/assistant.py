"""Messenger-style, read-only field assistant for the Phase 2 command system."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src import config, inference
from src import data_access as data
from src.operations.allocation import WardDemand, allocate
from src.operations.assistant import (
    ROLE_COMMISSIONER,
    ROLE_HOSPITAL,
    AssistantContext,
    interval_coverage,
    normalise_role,
    suggested_questions,
)
from src.operations.assistant import (
    answer as assistant_answer,
)
from src.operations.routing import capacity_scaled, derive_bed_capacity, route_overflow
from ui.components import go_to

ROLE_OPTIONS = {
    "Commissioner operations": ROLE_COMMISSIONER,
    "Hospital surge": ROLE_HOSPITAL,
}
USER_AVATARS = {ROLE_COMMISSIONER: "🧑‍⚕️", ROLE_HOSPITAL: "🏥"}


def _default_as_of(dates: list[pd.Timestamp]) -> pd.Timestamp:
    if not dates:
        return pd.Timestamp.today().normalize()
    latest = dates[-1].year
    august = [d for d in dates if d.year == latest and d.month == 8]
    return august[-1] if august else dates[-1]


def _context_chips(
    role: str,
    city: str,
    ward: str,
    as_of: pd.Timestamp,
    horizon: int,
    stock: int,
    crews: int,
    capacity: float,
    conservative: bool,
) -> None:
    chips = [
        ("View", "Commissioner" if normalise_role(role) == ROLE_COMMISSIONER else "Hospital"),
        ("City", city),
        ("Ward", str(ward).replace("_", " ")),
        ("Week", as_of.strftime("%d %b %Y")),
        ("Horizon", f"+{horizon}w"),
        ("Stock", f"{stock}%"),
        ("Crews", str(crews)),
        ("Beds", f"{capacity:.1f}×"),
    ]
    if conservative:
        chips.append(("Planning", "80% upper"))
    body = "".join(
        f"<span class='assistant-chip'>{html.escape(label)} <strong>{html.escape(value)}</strong></span>"
        for label, value in chips
    )
    st.markdown(f"<div class='assistant-context'>{body}</div>", unsafe_allow_html=True)


def _append_exchange(messages: list[dict], question: str, context: AssistantContext) -> None:
    messages.append(
        {
            "role": "user",
            "content": question,
            "role_view": normalise_role(context.role),
        }
    )
    result = assistant_answer(question, context)
    messages.append(
        {
            "role": "assistant",
            "content": result.text,
            "sources": result.sources,
            "suggestions": result.suggestions,
            "intent": result.intent,
        }
    )


def _render_messages(messages: list[dict]) -> None:
    if not messages:
        with st.chat_message("assistant", avatar="🦟"):
            st.markdown(
                "**Hi — I’m the Aedes Sentinel field assistant.**\n\n"
                "I answer from the current scenario: your ward, stock, crews, overflow routes "
                "and the model’s holdout metrics. I am read-only, deterministic and fully "
                "offline — no external API."
            )
        return
    for message in messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🦟"):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Grounded in: " + " · ".join(message["sources"]))
        else:
            avatar = USER_AVATARS.get(message.get("role_view"), "🧑‍⚕️")
            with st.chat_message("user", avatar=avatar):
                st.markdown(message["content"])


def render_assistant() -> None:
    try:
        onsite = data.load_onsite()
        meta = data.load_ward_meta()
        features = data.load_features()
        bundle = inference.load_bundle()
    except Exception as exc:  # noqa: BLE001 - shown to the user
        st.error(f"The assistant needs the bundled data and model: {exc}")
        st.stop()

    if features is None:
        st.error("Feature table missing. Run the ML pipeline first.")
        st.stop()

    dates = data.available_dates(onsite)
    if not dates:
        st.error("The onsite dataset has no dated rows.")
        st.stop()

    cities = [c for c in config.CITY_ORDER if c in set(meta[config.CITY_COL])]
    default_city = "Dhaka North" if "Dhaka North" in cities else cities[0]
    default_date = _default_as_of(dates)

    st.markdown(
        """
    <div class="page-head">
      <div class="page-kicker">Phase 2 · field assistant</div>
      <h1 class="page-title">Ask Aedes Sentinel</h1>
      <p class="page-sub">A read-only, application-aware assistant. It answers from the live
      forecast, allocation, routing and holdout metrics — no external API, no invented numbers.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Scenario context · shared with Command Center", expanded=False):
        r1 = st.columns([1.25, 1.15, 1.15, 1.25], gap="medium")
        role_label = r1[0].selectbox(
            "View as", list(ROLE_OPTIONS), index=0, key="assistant_role"
        )
        role = ROLE_OPTIONS[role_label]
        city = r1[1].selectbox(
            "City", cities, index=cities.index(default_city), key="cmd_city"
        )
        wards_city = data.wards_for_city(meta, city)
        preferred = "DNCC_Ward_12" if "DNCC_Ward_12" in wards_city else wards_city[0]
        ward = r1[2].selectbox(
            "Ward",
            wards_city,
            index=wards_city.index(preferred),
            format_func=lambda w: str(w).replace("_", " "),
            key=f"assistant_ward_{city}",
        )
        dates_desc = list(reversed(dates))
        date_index = dates_desc.index(default_date) if default_date in dates_desc else 0
        as_of = r1[3].selectbox(
            "Observed through",
            dates_desc,
            index=date_index,
            format_func=data.format_date,
            key="cmd_as_of",
        )
        r2 = st.columns([1.0, 1.0, 1.0, 1.0], gap="medium")
        horizon = r2[0].select_slider(
            "Horizon",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda h: f"+{h}w",
            key="cmd_horizon",
        )
        stock = r2[1].slider(
            "Stock",
            min_value=30,
            max_value=100,
            value=60,
            step=5,
            format="%d%%",
            key="cmd_stock",
        )
        crews = r2[2].slider(
            "Crews",
            min_value=0,
            max_value=40,
            value=8,
            step=1,
            key="cmd_crews",
        )
        capacity = r2[3].slider(
            "Bed capacity",
            min_value=0.60,
            max_value=2.00,
            value=1.20,
            step=0.10,
            format="%.2f×",
            key="cmd_capacity",
        )
        conservative = st.toggle(
            "Plan against the 80% upper interval",
            value=False,
            key="cmd_conservative",
        )
        st.caption("These controls are the same scenario used by the Command Center.")

    as_of = pd.Timestamp(as_of).normalize()
    horizon = int(horizon)
    all_wards = sorted(onsite[config.WARD_COL].unique())
    preds = inference.predict_batch(
        all_wards, as_of, [horizon], features=features, bundle=bundle
    )
    meta_idx = meta.set_index(config.WARD_COL)
    demands: list[WardDemand] = []
    for _, row in preds.iterrows():
        ward_id = str(row[config.WARD_COL])
        if pd.isna(row["cases"]) or ward_id not in meta_idx.index:
            continue
        info = meta_idx.loc[ward_id]
        planning_cases = (
            float(row["high"])
            if conservative and pd.notna(row.get("high"))
            else float(row["cases"])
        )
        demands.append(
            WardDemand.from_prediction(
                ward_id=ward_id,
                city=str(info[config.CITY_COL]),
                predicted_cases=planning_cases,
                tier=str(row["tier"]),
                density=float(info["population_density_sqkm"]),
            )
        )
    if not demands:
        st.error("No ward forecasts were available for this week.")
        st.stop()

    allocation = allocate(demands, stock / 100.0, int(crews))
    bed_capacity = capacity_scaled(derive_bed_capacity(onsite, as_of=as_of), capacity)
    bed_demand = {
        str(row["ward_id"]): int(row["required_beds"])
        for _, row in allocation.per_ward.iterrows()
    }
    routing = route_overflow(
        bed_demand, bed_capacity, {demand.ward_id: demand.city for demand in demands}
    )
    metrics = bundle.get("metrics") or {}
    coverage = interval_coverage(metrics, inference.load_test_predictions())

    if ward not in set(allocation.per_ward["ward_id"]):
        ward = str(allocation.per_ward.loc[allocation.per_ward["city"] == city, "ward_id"].iloc[0])

    context = AssistantContext(
        role=role,
        city=city,
        ward_id=str(ward),
        as_of=as_of,
        horizon=horizon,
        stock_frac=stock / 100.0,
        total_crews=int(crews),
        capacity_mult=float(capacity),
        conservative=bool(conservative),
        allocation=allocation,
        routing=routing,
        demands=demands,
        metrics=metrics,
        bed_capacity=bed_capacity,
        bed_demand=bed_demand,
        interval_coverage=coverage,
    )

    _context_chips(
        role, city, str(ward), as_of, horizon, int(stock), int(crews), float(capacity), bool(conservative)
    )

    left, right = st.columns([1.0, 1.0], gap="small")
    if left.button("Open Command Center", key="assistant_open_command", width="stretch"):
        go_to("command")
    if right.button("Clear chat", key="assistant_clear", width="stretch"):
        st.session_state["assistant_messages"] = []
        st.rerun()

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []
    messages: list[dict] = st.session_state["assistant_messages"]

    with st.container(key="assistant_chat", height=520, border=False):
        _render_messages(messages)

    suggestions = suggested_questions(context)
    st.markdown("**Try asking**")
    suggestion_cols = st.columns(len(suggestions), gap="small")
    for index, (column, question) in enumerate(zip(suggestion_cols, suggestions, strict=True)):
        if column.button(question, key=f"assistant_suggest_{index}", width="stretch"):
            _append_exchange(messages, question, context)
            st.rerun()

    with st.container(key="assistant_input"):
        prompt = st.chat_input("Ask about your ward, supplies, overflow, crews, or accuracy…")
    if prompt:
        _append_exchange(messages, prompt, context)
        st.rerun()

    st.caption(
        "Read-only: answers are computed from the scenario above. Nothing is dispatched or "
        "changed from chat. No external API is called."
    )
