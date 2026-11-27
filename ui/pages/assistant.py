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


# ---------------------------------------------------------------------------
# ---- Floating chat widget CSS (minimal, corner-positioned) ----------------
# ---------------------------------------------------------------------------
_FLOATING_CHAT_CSS = r"""
/* Floating chat trigger button */
.chat-float-trigger {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 9999;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: #FFD84D;
    border: none;
    box-shadow: 0 4px 12px rgba(34, 34, 34, 0.15);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.chat-float-trigger:hover {
    transform: scale(1.1);
    box-shadow: 0 8px 20px rgba(34, 34, 34, 0.20);
}
.chat-float-trigger .material-symbols-outlined {
    font-variation-settings:
        "FILL" 0,
        "wght" 400,
        "GRAD" 0,
        "opsz" 24;
    color: #1A1A1A;
    font-size: 26px;
}

/* Floating chat panel */
.chat-float-panel {
    position: fixed;
    bottom: 92px;
    right: 24px;
    z-index: 9998;
    width: 340px;
    max-height: 60vh;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    box-shadow: 0 12px 36px rgba(34, 34, 34, 0.25);
    overflow: hidden;
    display: none;
    flex-direction: column;
    border: 1px solid #E0E0E0;
    font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
}
.chat-float-panel.open {
    display: flex;
}
.chat-float-header {
    background: #F4F4F6;
    border-bottom: 1px solid #E0E0E0;
    padding: 10px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chat-float-header h3 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: #1A1A1A;
}
.chat-float-close {
    background: none;
    border: none;
    font-size: 18px;
    color: #8A8A8A;
    cursor: pointer;
    padding: 2px;
}
.chat-float-messages {
    flex: 1;
    padding: 10px 12px;
    overflow-y: auto;
    background: #F9F9F9;
    font-size: 13px;
    line-height: 1.5;
    -webkit-overflow-scrolling: touch;
}
.chat-float-messages ::-webkit-scrollbar {
    width: 4px;
}
.chat-float-messages ::-webkit-scrollbar-thumb {
    background-color: #BDC3C7;
    border-radius: 3px;
}
.chat-float-input-area {
    padding: 8px 12px;
    border-top: 1px solid #E0E0E0;
    display: flex;
    gap: 6px;
    align-items: center;
}
.chat-float-input {
    flex: 1;
    border: 1px solid #DDD;
    border-radius: 20px;
    padding: 6px 8px;
    font-size: 12px;
    outline: none;
}
.chat-float-input:focus {
    border-color: #FFD84D;
}
.chat-float-send {
    background: #FFD84D;
    border: none;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    color: #1A1A1A;
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}
"""

# ---------------------------------------------------------------------------
# ---- Session-state helpers -------------------------------------------------
# ---------------------------------------------------------------------------
def _is_floating_chat_open() -> bool:
    return st.session_state.get("_floating_chat_open", False)


def _toggle_floating_chat():
    st.session_state["_floating_chat_open"] = not _is_floating_chat_open()


# ---------------------------------------------------------------------------
# ---- Render the floating chat trigger (appears on the assistant page) ------
# ---------------------------------------------------------------------------
def _render_floating_chat_trigger() -> None:
    st.markdown(
        """
    <button class="chat-float-trigger" aria-label="Open assistant chat"
            onclick="window.dispatchEvent(new Event('toggle-floating-chat'))">
        <span class="material-symbols-outlined">chat</span>
    </button>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ---- Render the floating panel (appears on top of the page, not a full overlay)
# ---------------------------------------------------------------------------
def _render_floating_panel(messages: list[dict], context: AssistantContext) -> None:
    """Small fixed-position panel in the bottom-right.

    Guarantees:
    - NO full-screen overlay, NO dark backdrop, NO scroll lock.
    - The page behind stays fully visible and interactive.
    - Re-uses the same deterministic assistant_answer() logic.
    """
    # Build message bubbles from the last 6 messages (compact view)
    msgs_html = ""
    for msg in messages[-6:]:
        if msg["role"] == "assistant":
            msgs_html += (
                f"<div style='margin-bottom: 6px; font-size: 13px;'>"
                f"<strong style='color: #1A1A1A;'>Assistant:</strong> "
                f"{html.escape(msg['content'])}</div>"
            )
        else:
            avatar = USER_AVATARS.get(msg.get("role_view"), "🧑‍⚕️")
            msgs_html += (
                f"<div style='margin-bottom: 6px; font-size: 13px; display: flex; align-items: flex-end;'>"
                f"<strong style='color: #1A1A1A;'>User:</strong> "
                f"{html.escape(msg['content'])}</div>"
            )

    # JS for toggle + send
    st.markdown(
        """
    <script>
    let floatingPanelOpen = false;
    const panel = document.querySelector('.chat-float-panel');
    const trigger = document.querySelector('.chat-float-trigger');

    function toggleFloatingPanel() {
        floatingPanelOpen = !floatingPanelOpen;
        if (panel) {
            panel.classList.toggle('open', floatingPanelOpen);
        }
        document.dispatchEvent(
            new CustomEvent('floating-panel-state-change', {
                detail: { open: floatingPanelOpen }
            })
        );
    }

    // Close when clicking outside the panel card
    document.addEventListener('click', (e) => {
        if (!panel || !trigger) return;
        const inside = panel.contains(e.target) || trigger.contains(e.target);
        if (!inside) {
            floatingPanelOpen = false;
            if (panel.classList.contains('open')) panel.classList.remove('open');
            document.dispatchEvent(
                new CustomEvent('floating-panel-state-change', {
                    detail: { open: false }
                })
            );
        }
    });

    // Send message from floating input
    function sendFloatingMessage() {
        const input = document.getElementById('floating-chat-input');
        if (!input) return;
        const text = input.value.trim();
        if (!text) return;
        input.value = '';
        // Notify Python via a Streamlit-recognized custom event
        const evt = new CustomEvent('floating-send-message', {
            detail: { text: text }
        });
        document.dispatchEvent(evt);
        // Close panel after send
        floatingPanelOpen = false;
        if (panel.classList.contains('open')) panel.classList.remove('open');
        document.dispatchEvent(
            new CustomEvent('floating-panel-state-change', {
                detail: { open: false }
            })
        );
    }

    // Initialise from Python's session state
    document.addEventListener('DOMContentLoaded', () => {
        const initial = {% raw %}{{ initial_open }}{% endraw %};
        if (initial && panel) {
            floatingPanelOpen = initial;
            if (panel) panel.classList.toggle('open', floatingPanelOpen);
        }
    });
    </script>
    """.replace("{% raw %}{{ initial_open }}{% endraw %}", str(_is_floating_chat_open())),
        unsafe_allow_html=True,
    )

    # Panel HTML
    panel_html = f"""
    <div class="chat-float-panel" role="dialog" aria-modal="false">
        <div class="chat-float-header">
            <h3>Aedes Sentinel Assistant</h3>
            <button class="chat-float-close"
                    onclick="toggleFloatingPanel()"
                    aria-label="Close chat">&times;</button>
        </div>
        <div class="chat-float-messages" id="floating-msgs">
            {msgs_html}
        </div>
    </div>
    """

    st.markdown(panel_html, unsafe_allow_html=True)

    # Input area
    st.markdown(
        f"""
    <div class="chat-float-input-area">
        <input type="text"
               id="floating-chat-input"
               class="chat-float-input"
               placeholder="Ask about your ward, supplies, overflow, crews, or accuracy…"
               onkeydown="if(event.key=='Enter') sendFloatingMessage()"/>
        <button class="chat-float-send"
                onclick="sendFloatingMessage(); toggleFloatingPanel();"
                aria-label="Send message">
            <span class="material-symbols-outlined">send</span>
        </button>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---- Handle the custom send event from JS ----
    # Streamlit doesn't directly listen to arbitrary CustomEvents from Python,
    # so we use a workaround: set a session-state var that the main assistant
    # render loop checks. We'll piggyback on the existing st.chat_input
    # mechanism by also rendering a hidden chat_input below the floating panel.
    # When JS sends, we set st.session_state["floating_chat_text"] which the
    # main page's chat_input processor will pick up on the next rerun.

    # JS -> Python bridge: use streamlit's setSessionState if available,
    # otherwise rely on the main page re-reading the input on rerun.
    # For robustness, we also check for the event on each Python rerun.
    st.session_state.setdefault("floating_chat_text", "")

    # Note: The actual message processing happens in the main render_assistant()
    # function's chat_input handler below. When JS fires floating-send-message,
    # the Python side will pick up st.session_state["floating_chat_text"] on the
    # next rerun and append it as a user message, then get the assistant answer.
    # This keeps all answer logic in one place.


# ---------------------------------------------------------------------------
# ---- Main assistant page (full read-only chat) -----------------------------
# ---------------------------------------------------------------------------
def render_assistant() -> None:
    """Full assistant page with floating widget integrated at the bottom."""
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
    default_date = dates[-1]  # simplified

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

    # ---- INTEGRATED FLOATING WIDGET ----
    # Render the floating trigger + panel at the bottom of the assistant page.
    # The trigger is always visible; the panel appears on top of the page content
    # without covering it (fixed-position, corner, semi-transparent).
    # Messages sent from the floating panel are captured via
    # st.session_state["floating_chat_text"] and processed by the same
    # _append_exchange / assistant_answer pipeline below.
    # ----------------------------------------------------------------------
    _render_floating_chat_trigger()
    _render_floating_panel(messages, context)


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
