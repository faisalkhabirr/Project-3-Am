"""Read-only, application-aware field assistant.

The assistant is deliberately **not** a free-form LLM. It is a deterministic
intent router over the live Command Center state:

* ward/city status comes from ``AllocationResult``;
* overflow answers come from ``RoutingResult``;
* supply answers use the verified conversion;
* accuracy answers come from the committed holdout metrics;
* stock what-if answers recompute allocation without mutating the app.

No network, no LLM, no embeddings service. Same question + same context gives
the same answer. Unknown questions get a safe fallback instead of an invention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from .allocation import AllocationResult, WardDemand, allocate
from .checks import verify_explanation
from .routing import RoutingResult
from .supplies import cases_to_supplies

ROLE_COMMISSIONER = "commissioner"
ROLE_HOSPITAL = "hospital_manager"
ROLE_LABELS = {
    ROLE_COMMISSIONER: "Commissioner operations",
    ROLE_HOSPITAL: "Hospital surge",
}
_ROLE_ALIASES = {
    "commissioner": ROLE_COMMISSIONER,
    "commissioner operations": ROLE_COMMISSIONER,
    "operations": ROLE_COMMISSIONER,
    "hospital": ROLE_HOSPITAL,
    "hospital manager": ROLE_HOSPITAL,
    "hospital surge": ROLE_HOSPITAL,
    "hospital_manager": ROLE_HOSPITAL,
}


@dataclass
class AssistantContext:
    role: str
    city: str
    ward_id: str
    as_of: pd.Timestamp
    horizon: int
    stock_frac: float
    total_crews: int
    capacity_mult: float
    conservative: bool
    allocation: AllocationResult
    routing: RoutingResult
    demands: list[WardDemand]
    metrics: dict
    bed_capacity: dict[str, int]
    bed_demand: dict[str, int]
    interval_coverage: float | None = None


@dataclass
class AssistantAnswer:
    text: str
    intent: str
    suggestions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def normalise_role(value: str) -> str:
    return _ROLE_ALIASES.get(str(value).strip().lower(), ROLE_COMMISSIONER)


def interval_coverage(metrics: dict, preds: pd.DataFrame | None) -> float | None:
    """Empirical coverage of the nominal 80% interval on the holdout."""
    if preds is None or preds.empty or "horizon_weeks" not in preds.columns:
        return None
    covered = []
    for horizon, group in preds.groupby("horizon_weeks"):
        key = str(int(horizon))
        interval = (metrics.get("intervals") or {}).get(key)
        if not interval:
            continue
        low = np.maximum(0.0, group["y_pred"] + float(interval["p10"]))
        high = np.maximum(0.0, group["y_pred"] + float(interval["p90"]))
        covered.append((group["y_true"] >= low) & (group["y_true"] <= high))
    if not covered:
        return None
    return float(np.concatenate([np.asarray(mask) for mask in covered]).mean())


def suggested_questions(context: AssistantContext) -> list[str]:
    ward = str(context.ward_id).replace("_", " ")
    available = int(context.allocation.available["saline_bags_needed"])
    required = int(context.allocation.required["saline_bags_needed"])
    stock = context.stock_frac * 100
    if normalise_role(context.role) == ROLE_HOSPITAL:
        return [
            f"Where do overflow patients from {ward} go?",
            f"At {stock:.0f}% stock, regional saline should be {available:,} of {required:,}.",
            "How accurate is the forecast?",
            "Does this work offline?",
        ]
    return [
        f"Why is {ward} short?",
        f"At {stock:.0f}% stock, regional saline should be {available:,} of {required:,}.",
        "Where do overflow patients go?",
        "How accurate is the forecast?",
    ]


def answer(question: str, context: AssistantContext) -> AssistantAnswer:
    """Answer a question from the current operational context."""
    role = normalise_role(context.role)
    q = _normalise(question)
    if not q:
        return _help(context, role, "help")
    verify_phrases = (
        "check ",
        "verify ",
        "make sure",
        "confirm ",
        "assert ",
        "test whether",
        "test that",
    )
    assertion_phrases = re.compile(
        r"\b(?:should|must)\s+(?:be|have|get|receive|equal|show|cross|remain|stay)\b"
    )
    if any(phrase in q for phrase in verify_phrases) or assertion_phrases.search(q):
        return _verify(context, question)

    city = _mentioned_city(q) or context.city
    ward_id = _resolve_ward(q, context, city)
    row = _ward_row(context, ward_id)
    if row is not None:
        city = str(row["city"])

    intent = _intent(q)
    if intent == "help":
        return _help(context, role, "help")
    if intent == "offline":
        return _offline(role)
    if intent == "assistant":
        return _assistant_meta(role)
    if intent == "accuracy":
        return _accuracy(context, role)
    if intent == "dispatch":
        return _dispatch(context, role, row, ward_id)
    if intent == "dataset":
        return _dataset(role)
    if intent == "stock_what_if":
        return _stock_what_if(context, role, _extract_pct(q), ward_id, city)
    if intent == "routing":
        return _routing(context, role, ward_id, city)
    if intent == "crews":
        return _crews(context, role, row, ward_id)
    if intent == "supply":
        return _supply(context, role, row, ward_id)
    if intent == "recommend":
        return _recommend(context, role, row, ward_id)
    if intent == "city":
        return _city(context, role, city)
    if intent == "model":
        return _model(role)
    if intent == "ward" and row is not None:
        return _ward(context, role, row, ward_id)
    return _unknown(context, role)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _ward_row(context: AssistantContext, ward_id: str) -> pd.Series | None:
    frame = context.allocation.per_ward
    subset = frame.loc[frame["ward_id"] == ward_id]
    return subset.iloc[0] if not subset.empty else None


def _city_rows(context: AssistantContext, city: str) -> pd.DataFrame:
    frame = context.allocation.per_ward
    return frame.loc[frame["city"] == city]


def _mentioned_city(q: str) -> str | None:
    for city in config.CITY_ORDER:
        if city.lower() in q:
            return city
    return None


def _resolve_ward(q: str, context: AssistantContext, city: str) -> str:
    wards = set(context.allocation.per_ward["ward_id"].astype(str))
    if any(phrase in q for phrase in ("my ward", "this ward", "selected ward")):
        if str(context.ward_id) in wards and context.city == city:
            return str(context.ward_id)
        city_wards = sorted(
            context.allocation.per_ward.loc[
                context.allocation.per_ward["city"] == city, "ward_id"
            ].astype(str)
        )
        if city_wards:
            return city_wards[0]

    match = re.search(r"\b(dncc|dscc|ccc|gcc|ncc)[\s_-]*ward[\s_-]*(\d{1,2})\b", q)
    if match:
        prefix, number = match.group(1).upper(), int(match.group(2))
        candidate = f"{prefix}_Ward_{number:02d}"
        if candidate in wards:
            return candidate

    match = re.search(r"\bward[\s_-]*(\d{1,2})\b", q)
    if match:
        number = int(match.group(1))
        city_wards = sorted(
            context.allocation.per_ward.loc[
                context.allocation.per_ward["city"] == city, "ward_id"
            ].astype(str)
        )
        for ward in city_wards:
            if ward.endswith(f"_{number:02d}"):
                return ward
        for ward in sorted(wards):
            if ward.endswith(f"_{number:02d}"):
                return ward

    if str(context.ward_id) in wards and context.city == city:
        return str(context.ward_id)
    city_wards = sorted(
        context.allocation.per_ward.loc[
            context.allocation.per_ward["city"] == city, "ward_id"
        ].astype(str)
    )
    return city_wards[0] if city_wards else str(context.ward_id)


def _extract_pct(q: str) -> int | None:
    match = re.search(r"(\d{2,3})\s*(?:%|percent)", q)
    if match:
        value = int(match.group(1))
        return min(100, max(30, value))
    return None


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _role_header(role: str) -> str:
    return (
        "**Hospital surge view**"
        if normalise_role(role) == ROLE_HOSPITAL
        else "**Commissioner operations view**"
    )


def _role_action(role: str, row: pd.Series) -> str:
    saline_gap = int(row["deficit_saline"])
    kit_gap = int(row["deficit_kits"])
    bed_gap = int(row["deficit_beds"])
    crew_gap = int(row["deficit_crews"])
    if normalise_role(role) == ROLE_HOSPITAL:
        if bed_gap > 0:
            return (
                f"Escalate **{bed_gap} bed(s)** through the Overflow tab; "
                "the router keeps transfers inside the city pool."
            )
        if saline_gap > 0 or kit_gap > 0:
            return (
                f"Request **{saline_gap} saline bag(s)** and **{kit_gap} kit(s)**; "
                "bed supply is currently covered."
            )
        return "Bed and commodity supply are covered; monitor the next forecast."
    if crew_gap > 0:
        return (
            f"Assign crews here first — **{crew_gap} crew(s) short**. "
            "Use the Simulator tab to compare cases prevented per crew."
        )
    if saline_gap > 0 or kit_gap > 0:
        return (
            f"Raise a supply request for **{saline_gap} saline bag(s)** and "
            f"**{kit_gap} kit(s)**; then send the ward brief from Dispatch."
        )
    return "This ward is fully covered at the current stock and crew levels."


def _intent(q: str) -> str:
    if re.search(r"\b(hi|hello|hey|help|what can you do|commands|capabilities)\b", q):
        return "help"
    if any(word in q for word in ("offline", "internet", "network", "wifi", "disconnect", "api")):
        return "offline"
    if any(word in q for word in ("llm", "chatbot", "assistant", "hallucin")):
        return "assistant"
    if "who are you" in q:
        return "assistant"
    if any(word in q for word in ("accurate", "accuracy", "mae", "rmse", "reliable", "coverage", "validation", "trust")):
        return "accuracy"
    if any(word in q for word in ("sms", "whatsapp", "brief", "print", "dispatch", "notice")):
        return "dispatch"
    if any(word in q for word in ("dataset", "data set", "rows", "cities", "weeks")):
        return "dataset"
    if ("stock" in q or "scarcity" in q) and (
        "what if" in q or "if " in q or "%" in q or "percent" in q or "drops" in q
    ):
        return "stock_what_if"
    if any(word in q for word in ("overflow", "transfer", "route", "routing", "capacity", "headroom", "utilization", "patients go")):
        return "routing"
    if any(word in q for word in ("crew", "crews", "spray", "spraying", "larvicide", "fogging", "team", "teams", "roi")):
        return "crews"
    if any(word in q for word in ("saline", "kit", "kits", "supply", "supplies", "formula", "calculate", "calculated", "how many", "need", "required", "beds")):
        return "supply"
    if any(word in q for word in ("what should", "recommend", "priority", "first", "do next", "action")):
        return "recommend"
    if any(word in q for word in ("city", "overview", "summary", "how is")):
        return "city"
    if any(word in q for word in ("ward", "short", "deficit", "tier", "risk", "why", "status", "gap", "biggest", "worst")):
        return "ward"
    if any(word in q for word in ("model", "poisson", "leakage", "train", "feature", "machine learning", "how does it work")):
        return "model"
    return "unknown"


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


def _help(context: AssistantContext, role: str, intent: str) -> AssistantAnswer:
    text = (
        f"{_role_header(role)} · I answer from the live scenario, not from an LLM.\n\n"
        "You can ask me about:\n"
        "- **My ward** — risk, supply gaps, crew gaps, priority\n"
        "- **Overflow** — where patients are routed and city bed pressure\n"
        "- **Supplies** — saline, kits, beds and the verified formulas\n"
        "- **Crews** — required vs assigned and who is next in priority\n"
        "- **What-if** — “What if stock drops to 30%?” (read-only)\n"
        "- **Verify** — describe a test in words and get expected vs actual\n"
        "- **Trust** — MAE, calibration, leakage, offline proof\n\n"
        "Try one of the suggested questions below."
    )
    return AssistantAnswer(text, intent, suggested_questions(context), ["assistant.py", "live Command context"])


def _offline(role: str) -> AssistantAnswer:
    text = (
        f"{_role_header(role)} · **Yes, the local app works offline.**\n\n"
        "- Data, model, feature table and dispatch templates are bundled.\n"
        "- `scripts/offline_check.py` blocks sockets and runs forecast → allocation → "
        "routing → simulation → dispatch to completion.\n"
        "- Runtime API calls: **0**.\n\n"
        "The deployed Streamlit URL needs internet, but `streamlit run app.py` on localhost "
        "does not."
    )
    return AssistantAnswer(text, "offline", [], ["offline_check.py", ".streamlit/config.toml"])


def _verify(context: AssistantContext, question: str) -> AssistantAnswer:
    report = verify_explanation(question, context)
    return AssistantAnswer(
        text=report.markdown,
        intent="verify",
        suggestions=report.suggestions,
        sources=report.sources,
    )


def _assistant_meta(role: str) -> AssistantAnswer:
    text = (
        f"{_role_header(role)} · **This is a deterministic assistant, not an LLM.**\n\n"
        "- It routes intents and reads the live allocation, routing and metrics.\n"
        "- It recomputes bounded what-ifs without changing the scenario.\n"
        "- It has no external API, no model download and no hallucinated numbers.\n"
        "- Unknown questions fall back to suggested topics."
    )
    return AssistantAnswer(text, "assistant", [], ["assistant.py"])


def _accuracy(context: AssistantContext, role: str) -> AssistantAnswer:
    metrics = context.metrics or {}
    coverage = context.interval_coverage
    lines = [
        f"{_role_header(role)} · **Forecast quality on the untouched 2024–25 holdout**",
        "",
        f"- MAE **{float(metrics.get('mae', 0)):.3f}** cases/ward-week",
        f"- RMSE **{float(metrics.get('rmse', 0)):.3f}**",
        f"- Seasonal-naive MAE **{float(metrics.get('baseline_mae', {}).get('1', 0)):.3f}**",
        f"- Persistence MAE **{float(metrics.get('persistence_mae', {}).get('1', 0)):.3f}** at +1 week",
    ]
    if coverage is not None:
        lines.append(f"- 80% interval coverage **{coverage * 100:.1f}%**")
    lines.extend(
        [
            "",
            "This is a probabilistic forecast, not a claim of 100% accuracy. "
            "The supply math is exact; the case forecast is calibrated decision support.",
        ]
    )
    if normalise_role(role) == ROLE_HOSPITAL:
        lines.append("For surge planning, use the **upper 80% interval** in the Command Center.")
    else:
        lines.append("For stock planning, the conservative upper-interval mode is available in Command.")
    return AssistantAnswer("\n".join(lines), "accuracy", suggested_questions(context), ["metrics.json", "verify_accuracy.py"])


def _dispatch(context: AssistantContext, role: str, row: pd.Series | None, ward_id: str) -> AssistantAnswer:
    if row is None:
        return _unknown(context, role)
    text = (
        f"{_role_header(role)} · **Dispatch for {str(ward_id).replace('_', ' ')}**\n\n"
        f"- Predicted cases: **{float(row['predicted_cases']):.1f}**\n"
        f"- Saline: **{int(row['allocated_saline'])} / {int(row['required_saline'])}**\n"
        f"- Kits: **{int(row['allocated_kits'])} / {int(row['required_kits'])}**\n"
        f"- Beds: **{int(row['allocated_beds'])} / {int(row['required_beds'])}**\n"
        f"- Crews: **{int(row['allocated_crews'])} / {int(row['required_crews'])}**\n\n"
        "Open the **Dispatch** tab for the ≤160-character SMS, WhatsApp message and printable brief."
    )
    return AssistantAnswer(text, "dispatch", suggested_questions(context), ["dispatch.py", "allocation.per_ward"])


def _dataset(role: str) -> AssistantAnswer:
    text = (
        f"{_role_header(role)} · **Dataset**\n\n"
        "- **31,320 ward-weeks** · **60 wards** · **5 cities** · **522 weeks**\n"
        "- Period: **2016-01-04 → 2025-12-29**, weekly\n"
        "- Onsite file adds saline, kits and beds; all three match **31,320 / 31,320** rows exactly.\n"
        "- Training uses the onsite file; the supply columns are excluded from model inputs."
    )
    return AssistantAnswer(text, "dataset", [], ["verify_supply_formulas.py", "build_features.py"])


def _stock_what_if(
    context: AssistantContext, role: str, pct: int | None, ward_id: str, city: str
) -> AssistantAnswer:
    target = pct if pct is not None else 30
    hypothetical = allocate(context.demands, target / 100.0, context.total_crews)
    frame = hypothetical.per_ward
    gap_wards = int(
        (
            frame[["deficit_saline", "deficit_kits", "deficit_beds"]].sum(axis=1) > 0
        ).sum()
    )
    current_gap_wards = int(
        (
            context.allocation.per_ward[
                ["deficit_saline", "deficit_kits", "deficit_beds"]
            ].sum(axis=1)
            > 0
        ).sum()
    )
    row = frame.loc[frame["ward_id"] == ward_id]
    current = context.allocation.per_ward.loc[
        context.allocation.per_ward["ward_id"] == ward_id
    ]
    city_rows = frame.loc[frame["city"] == city]
    lines = [
        f"{_role_header(role)} · **Read-only what-if: stock at {target}%**",
        "",
        f"- Regional saline: **{hypothetical.available['saline_bags_needed']:,} / "
        f"{hypothetical.required['saline_bags_needed']:,}**",
        f"- Regional supply gaps: **{gap_wards} / {len(frame)} wards** "
        f"(currently {current_gap_wards})",
    ]
    if not row.empty and not current.empty:
        r = row.iloc[0]
        c = current.iloc[0]
        lines.append(
            f"- {str(ward_id).replace('_', ' ')} saline: **{int(r['allocated_saline'])} / "
            f"{int(r['required_saline'])}** (currently {int(c['allocated_saline'])})"
        )
        lines.append(
            f"- {str(ward_id).replace('_', ' ')} crews: **{int(r['allocated_crews'])} / "
            f"{int(r['required_crews'])}**"
        )
    lines.append(
        f"- {city}: **{int((city_rows[['deficit_saline', 'deficit_kits', 'deficit_beds']].sum(axis=1) > 0).sum())} / "
        f"{len(city_rows)}** wards with a supply gap"
    )
    lines.extend(
        [
            "",
            "This does not change the live Command Center; it is a scenario calculation.",
        ]
    )
    if normalise_role(role) == ROLE_HOSPITAL:
        lines.append("Protect bed and saline capacity first; defer elective bed use if the gap widens.")
    else:
        lines.append("Expect to ration saline first; keep crews on the highest-risk, densest wards.")
    return AssistantAnswer("\n".join(lines), "stock_what_if", suggested_questions(context), ["allocation.py", "recomputed scenario"])


def _routing(context: AssistantContext, role: str, ward_id: str, city: str) -> AssistantAnswer:
    city_rows = _city_rows(context, city)
    city_demand = sum(int(context.bed_demand.get(w, 0)) for w in city_rows["ward_id"])
    city_capacity = sum(int(context.bed_capacity.get(w, 0)) for w in city_rows["ward_id"])
    incoming = [
        (donor, receiver, moved)
        for donor, receiver, moved in context.routing.transfers
        if receiver == ward_id
    ]
    outgoing = [
        (donor, receiver, moved)
        for donor, receiver, moved in context.routing.transfers
        if donor == ward_id
    ]
    lines = [
        f"{_role_header(role)} · **Bed position for {str(ward_id).replace('_', ' ')}**",
        "",
        f"- City demand: **{city_demand} / {city_capacity}** beds "
        f"({city_demand / city_capacity * 100:.0f}% utilization)" if city_capacity else "- City capacity unavailable",
        f"- Unrouted deficit in this ward: **{int(context.routing.net_deficits.get(ward_id, 0))}**",
    ]
    if incoming:
        lines.append("- Incoming overflow: " + ", ".join(f"**{moved}** from {donor.replace('_', ' ')}" for donor, _, moved in incoming))
    else:
        lines.append("- Incoming overflow: none")
    if outgoing:
        lines.append("- Outgoing transfers: " + ", ".join(f"**{moved}** to {receiver.replace('_', ' ')}" for _, receiver, moved in outgoing))
    if context.routing.escalated_cities:
        lines.append("- Regional escalation: " + ", ".join(context.routing.escalated_cities))
    if normalise_role(role) == ROLE_HOSPITAL:
        lines.append("\nCoordinate the transfer list with the receiving ward before admitting.")
    else:
        lines.append("\nThe router keeps every transfer inside the same city pool.")
    return AssistantAnswer("\n".join(lines), "routing", suggested_questions(context), ["routing.py", "routing.transfers"])


def _crews(context: AssistantContext, role: str, row: pd.Series | None, ward_id: str) -> AssistantAnswer:
    if row is None:
        return _unknown(context, role)
    city_rows = _city_rows(context, str(row["city"])).sort_values("priority", ascending=False)
    top = city_rows.head(3)
    lines = [
        f"{_role_header(role)} · **Crew position for {str(ward_id).replace('_', ' ')}**",
        "",
        f"- Required: **{int(row['required_crews'])}** · assigned: **{int(row['allocated_crews'])}** · gap: **{int(row['deficit_crews'])}**",
        f"- Priority score: **{float(row['priority']):.2f}**",
        "- City top priorities: " + ", ".join(
            f"{str(item.ward_id).replace('_', ' ')} ({int(item.required_crews)} needed)"
            for item in top.itertuples()
        ),
        "",
        "Crews are allocated by risk tier × density; the Simulator tab ranks cases prevented per crew.",
    ]
    if normalise_role(role) == ROLE_HOSPITAL:
        lines.append("Crews reduce the case load feeding beds; coordinate timing with the vector-control lead.")
    else:
        lines.append("Send crews to the highest-priority wards first; a second crew has diminishing returns.")
    return AssistantAnswer("\n".join(lines), "crews", suggested_questions(context), ["allocation.py", "simulation.py"])


def _supply(context: AssistantContext, role: str, row: pd.Series | None, ward_id: str) -> AssistantAnswer:
    if row is None:
        return _unknown(context, role)
    need = cases_to_supplies(float(row["predicted_cases"]))
    text = (
        f"{_role_header(role)} · **Supply requirement for {str(ward_id).replace('_', ' ')}**\n\n"
        f"- Predicted cases: **{float(row['predicted_cases']):.2f}**\n"
        f"- Saline: `round(31/13 × c)` → **{need['saline_bags_needed']}** "
        f"(allocated {int(row['allocated_saline'])}, gap {int(row['deficit_saline'])})\n"
        f"- Kits: `round_half_down(7/6 × c)` → **{need['diagnostic_kits_needed']}** "
        f"(allocated {int(row['allocated_kits'])}, gap {int(row['deficit_kits'])})\n"
        f"- Beds: `round_half_down(0.3 × c)` → **{need['hospital_beds_required']}** "
        f"(allocated {int(row['allocated_beds'])}, gap {int(row['deficit_beds'])})\n\n"
        "These formulas reproduce all 31,320 onsite rows exactly. "
        f"Current stock scenario: **{_pct(context.stock_frac)}**."
    )
    return AssistantAnswer(text, "supply", suggested_questions(context), ["supplies.py", "verify_supply_formulas.py"])


def _ward(context: AssistantContext, role: str, row: pd.Series, ward_id: str) -> AssistantAnswer:
    text = (
        f"{_role_header(role)} · **{str(ward_id).replace('_', ' ')} · {row['tier']}**\n\n"
        f"- Predicted cases: **{float(row['predicted_cases']):.1f}**\n"
        f"- Saline: **{int(row['allocated_saline'])} / {int(row['required_saline'])}**\n"
        f"- Kits: **{int(row['allocated_kits'])} / {int(row['required_kits'])}**\n"
        f"- Beds: **{int(row['allocated_beds'])} / {int(row['required_beds'])}**\n"
        f"- Crews: **{int(row['allocated_crews'])} / {int(row['required_crews'])}**\n"
        f"- Priority score: **{float(row['priority']):.2f}**\n\n"
        f"{_role_action(role, row)}"
    )
    return AssistantAnswer(text, "ward", suggested_questions(context), ["allocation.per_ward", "routing.py"])


def _recommend(context: AssistantContext, role: str, row: pd.Series | None, ward_id: str) -> AssistantAnswer:
    if row is None:
        return _unknown(context, role)
    text = (
        f"{_role_header(role)} · **Next action for {str(ward_id).replace('_', ' ')}**\n\n"
        f"{_role_action(role, row)}\n\n"
        "Then open the corresponding tab to execute it: "
        + ("**Overflow** for transfers." if normalise_role(role) == ROLE_HOSPITAL else "**Dispatch** for the ward brief.")
    )
    return AssistantAnswer(text, "recommend", suggested_questions(context), ["allocation.per_ward", "dispatch.py"])


def _city(context: AssistantContext, role: str, city: str) -> AssistantAnswer:
    rows = _city_rows(context, city)
    if rows.empty:
        return _unknown(context, role)
    supply_gap_wards = int(
        (rows[["deficit_saline", "deficit_kits", "deficit_beds"]].sum(axis=1) > 0).sum()
    )
    top = rows.sort_values("priority", ascending=False).head(3)
    text = (
        f"{_role_header(role)} · **{city} overview**\n\n"
        f"- Forecast cases: **{rows['predicted_cases'].sum():.1f}** across {len(rows)} wards\n"
        f"- Supply gaps: **{supply_gap_wards} / {len(rows)}** wards\n"
        f"- Crews: **{int(rows['allocated_crews'].sum())} / {int(rows['required_crews'].sum())}**\n"
        f"- Top priority: " + ", ".join(
            f"{str(item.ward_id).replace('_', ' ')} ({item.tier})" for item in top.itertuples()
        )
    )
    if normalise_role(role) == ROLE_HOSPITAL:
        text += "\n\nCheck the Overflow tab for transfers and city bed utilization."
    else:
        text += "\n\nStart with the top-priority wards; open Dispatch for the first brief."
    return AssistantAnswer(text, "city", suggested_questions(context), ["allocation.city_rollup", "routing.py"])


def _model(role: str) -> AssistantAnswer:
    text = (
        f"{_role_header(role)} · **Model**\n\n"
        "- Poisson HistGradientBoosting on 32 features.\n"
        "- Chronological split: train through 2022, validation 2023, untouched test 2024–25.\n"
        "- Supply targets are **excluded** from the inputs, so there is no target leakage.\n"
        "- Holdout MAE 1.054, RMSE 2.097; 80% interval coverage 81.5%."
    )
    return AssistantAnswer(text, "model", [], ["inference.py", "metrics.json", "ARCHITECTURE.md"])


def _unknown(context: AssistantContext, role: str) -> AssistantAnswer:
    text = (
        f"{_role_header(role)} · I don’t have a grounded answer for that question. "
        "I can answer ward status, supply formulas, overflow, crews, stock what-ifs, "
        "model accuracy and offline operation. Pick one of these:"
    )
    return AssistantAnswer(text, "unknown", suggested_questions(context), ["assistant.py"])
