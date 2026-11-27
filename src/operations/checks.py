"""Natural-language verification over the live operational scenario.

The user describes a test in words, for example:

    "At 60% stock with 8 crews, regional saline should be 749 of 1,249,
     and 31 of 60 wards should have a supply gap."

The parser extracts the scenario overrides and the assertions, re-runs the
relevant local engine, and returns expected vs actual PASS/FAIL. It is read-only:
the live Command scenario is never mutated.

No LLM, no shell, no arbitrary code execution. Unparsed claims are reported as
clarification instead of being invented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .. import config
from .allocation import allocate
from .supplies import cases_to_supplies

_NUM_RE = r"[\d,]+(?:\.\d+)?"
_CITY_WORDS = {city.lower(): city for city in config.CITY_ORDER}


@dataclass
class CheckAssertion:
    label: str
    metric: str
    expected: float | int | str
    op: str
    actual: float | int | str | None = None
    passed: bool = False
    cases: int | None = None


@dataclass
class CheckReport:
    markdown: str
    passed: bool | None
    parsed: dict
    assertions: list[CheckAssertion] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def _num(value: str) -> float | int:
    number = float(str(value).replace(",", "").rstrip("."))
    return int(number) if number.is_integer() else number


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _match_city(q: str) -> str | None:
    for needle, city in _CITY_WORDS.items():
        if needle in q:
            return city
    return None


def _match_ward(q: str, context: Any, city: str) -> str:
    wards = set(context.allocation.per_ward["ward_id"].astype(str))
    if "my ward" in q or "this ward" in q:
        if context.city == city and str(context.ward_id) in wards:
            return str(context.ward_id)
        city_wards = sorted(
            context.allocation.per_ward.loc[
                context.allocation.per_ward["city"] == city, "ward_id"
            ].astype(str)
        )
        return city_wards[0] if city_wards else str(context.ward_id)

    match = re.search(r"\b(dncc|dscc|ccc|gcc|ncc)[\s_-]*ward[\s_-]*(\d{1,2})\b", q)
    if match:
        candidate = f"{match.group(1).upper()}_Ward_{int(match.group(2)):02d}"
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

    if context.city == city and str(context.ward_id) in wards:
        return str(context.ward_id)
    city_wards = sorted(
        context.allocation.per_ward.loc[
            context.allocation.per_ward["city"] == city, "ward_id"
        ].astype(str)
    )
    return city_wards[0] if city_wards else str(context.ward_id)


def _scenario(q: str, context: Any) -> tuple[dict, Any, str, str]:
    stock = float(context.stock_frac * 100)
    if "full stock" in q:
        stock = 100.0
    elif "half stock" in q:
        stock = 50.0
    for pattern in (
        r"(?:at|with)\s+(\d{2,3})\s*(?:%|percent)\s+stock",
        r"stock\s*(?:at|=|:|of)?\s*(\d{2,3})\s*(?:%|percent)",
        r"(\d{2,3})\s*(?:%|percent)\s+stock",
    ):
        match = re.search(pattern, q)
        if match:
            stock = (float(match.group(1)))
            break
    stock = min(100.0, max(0.0, stock))

    crews = int(context.total_crews)
    if "no crews" in q or "zero crews" in q:
        crews = 0
    for pattern in (
        r"(?:with|at|using)\s+(\d+)\s+crews?\b",
        r"(\d+)\s+crews?\s+(?:available|total|deployed)",
        r"crews?\s*(?:=|:|of)\s*(\d+)\b",
    ):
        match = re.search(pattern, q)
        if match:
            crews = max(0, int(match.group(1)))
            break

    city = _match_city(q) or context.city
    ward_id = _match_ward(q, context, city)
    row = context.allocation.per_ward.loc[
        context.allocation.per_ward["ward_id"] == ward_id
    ]
    if not row.empty:
        city = str(row.iloc[0]["city"])
    allocation = allocate(context.demands, stock / 100.0, crews)
    parsed = {
        "stock": stock,
        "crews": crews,
        "city": city,
        "ward": ward_id,
    }
    return parsed, allocation, ward_id, city


def _row(allocation: Any, ward_id: str) -> pd.Series | None:
    frame = allocation.per_ward
    subset = frame.loc[frame["ward_id"] == ward_id]
    return subset.iloc[0] if not subset.empty else None


def _city_rows(allocation: Any, city: str) -> pd.DataFrame:
    return allocation.per_ward.loc[allocation.per_ward["city"] == city]


def _ward_city(allocation: Any) -> dict[str, str]:
    return dict(
        zip(
            allocation.per_ward["ward_id"].astype(str),
            allocation.per_ward["city"].astype(str),
            strict=True,
        )
    )


def _metric(
    metric: str,
    allocation: Any,
    context: Any,
    ward_id: str,
    city: str,
    cases: int | None = None,
) -> float | int | str | None:
    frame = allocation.per_ward
    city_frame = _city_rows(allocation, city)
    row = _row(allocation, ward_id)
    if metric == "regional_saline_available":
        return int(allocation.available["saline_bags_needed"])
    if metric == "regional_saline_required":
        return int(allocation.required["saline_bags_needed"])
    if metric == "regional_wards":
        return int(len(frame))
    if metric == "city_wards":
        return int(len(city_frame))
    if metric == "regional_supply_gap_wards":
        return int(
            (frame[["deficit_saline", "deficit_kits", "deficit_beds"]].sum(axis=1) > 0).sum()
        )
    if metric == "regional_crew_gap_wards":
        return int((frame["deficit_crews"] > 0).sum())
    if metric == "city_supply_gap_wards":
        return int(
            (
                city_frame[["deficit_saline", "deficit_kits", "deficit_beds"]].sum(axis=1)
                > 0
            ).sum()
        )
    if metric == "city_crew_gap_wards":
        return int((city_frame["deficit_crews"] > 0).sum())
    if metric == "city_saline_allocated":
        return int(city_frame["allocated_saline"].sum())
    if metric == "city_saline_required":
        return int(city_frame["required_saline"].sum())
    if metric == "city_bed_demand":
        return int(sum(context.bed_demand.get(w, 0) for w in city_frame["ward_id"]))
    if metric == "city_bed_capacity":
        return int(sum(context.bed_capacity.get(w, 0) for w in city_frame["ward_id"]))
    if metric == "city_utilization":
        demand = _metric("city_bed_demand", allocation, context, ward_id, city)
        capacity = _metric("city_bed_capacity", allocation, context, ward_id, city)
        return (float(demand) / float(capacity)) if capacity else 0.0
    if metric == "transfers_count":
        return int(len(context.routing.transfers))
    if metric == "unrouted_beds":
        return int(sum(context.routing.net_deficits.values()))
    if metric == "cross_city_transfers":
        ward_city = _ward_city(allocation)
        return int(
            sum(
                1
                for donor, receiver, _ in context.routing.transfers
                if ward_city.get(donor) != ward_city.get(receiver)
            )
        )
    if metric == "mae":
        return float((context.metrics or {}).get("mae", 0.0))
    if metric == "rmse":
        return float((context.metrics or {}).get("rmse", 0.0))
    if metric == "interval_coverage":
        return None if context.interval_coverage is None else float(context.interval_coverage)
    if metric == "saline_for_cases" and cases is not None:
        return int(cases_to_supplies(float(cases))["saline_bags_needed"])
    if row is None:
        return None
    if metric == "ward_cases":
        return float(row["predicted_cases"])
    if metric == "ward_tier":
        return str(row["tier"])
    if metric == "ward_priority":
        return float(row["priority"])
    if metric == "ward_saline_allocated":
        return int(row["allocated_saline"])
    if metric == "ward_saline_required":
        return int(row["required_saline"])
    if metric == "ward_kits_allocated":
        return int(row["allocated_kits"])
    if metric == "ward_kits_required":
        return int(row["required_kits"])
    if metric == "ward_beds_allocated":
        return int(row["allocated_beds"])
    if metric == "ward_beds_required":
        return int(row["required_beds"])
    if metric == "ward_crews_allocated":
        return int(row["allocated_crews"])
    if metric == "ward_crews_required":
        return int(row["required_crews"])
    return None


def _add(
    assertions: list[CheckAssertion],
    *,
    label: str,
    metric: str,
    expected: float | int | str,
    op: str = "==",
    cases: int | None = None,
) -> None:
    for existing in assertions:
        if (
            existing.metric == metric
            and existing.op == op
            and existing.expected == expected
            and existing.cases == cases
        ):
            return
    assertions.append(
        CheckAssertion(
            label=label,
            metric=metric,
            expected=expected,
            op=op,
            cases=cases,
        )
    )


def _parse_assertions(q: str, context: Any, city: str, ward_id: str) -> list[CheckAssertion]:
    assertions: list[CheckAssertion] = []
    city_lower = city.lower()
    city_mentioned = _match_city(q)

    # Regional saline pair.
    match = re.search(
        rf"regional saline[^0-9]*({_NUM_RE})\s*(?:of|out of|/)\s*({_NUM_RE})",
        q,
    )
    if match:
        _add(assertions, label="Regional saline available", metric="regional_saline_available", expected=_num(match.group(1)))
        _add(assertions, label="Regional saline required", metric="regional_saline_required", expected=_num(match.group(2)))

    # City saline pair.
    for needle, name in _CITY_WORDS.items():
        match = re.search(
            rf"{re.escape(needle)} saline[^0-9]*({_NUM_RE})\s*(?:of|out of|/)\s*({_NUM_RE})",
            q,
        )
        if match:
            _add(assertions, label=f"{name} saline allocated", metric="city_saline_allocated", expected=_num(match.group(1)))
            _add(assertions, label=f"{name} saline required", metric="city_saline_required", expected=_num(match.group(2)))

    # Ward saline / kits / beds / crews pairs.
    if re.search(r"\bward\b", q):
        for word, metric in (
            ("saline", "ward_saline"),
            ("kits", "ward_kits"),
            ("kit", "ward_kits"),
            ("beds", "ward_beds"),
            ("bed", "ward_beds"),
            ("crews", "ward_crews"),
            ("crew", "ward_crews"),
        ):
            match = re.search(
                rf"({_NUM_RE})\s*(?:of|/)\s*({_NUM_RE})\s*{word}",
                q,
            )
            if match:
                _add(
                    assertions,
                    label=f"Ward {word} allocated",
                    metric=f"{metric}_allocated",
                    expected=_num(match.group(1)),
                )
                _add(
                    assertions,
                    label=f"Ward {word} required",
                    metric=f"{metric}_required",
                    expected=_num(match.group(2)),
                )

        for word, metric, label in (
            ("saline", "ward_saline_allocated", "Ward saline allocated"),
            ("kits", "ward_kits_allocated", "Ward kits allocated"),
            ("kit", "ward_kits_allocated", "Ward kits allocated"),
            ("beds", "ward_beds_allocated", "Ward beds allocated"),
            ("bed", "ward_beds_allocated", "Ward beds allocated"),
            ("crews", "ward_crews_allocated", "Ward crews allocated"),
            ("crew", "ward_crews_allocated", "Ward crews allocated"),
        ):
            match = re.search(
                rf"ward[^.]*?(?:get|have|receive|allocated|needs?)[^.]*?({_NUM_RE})\s*{word}",
                q,
            )
            if match:
                required_metric = metric.replace("_allocated", "_required")
                if not any(item.metric == required_metric for item in assertions):
                    _add(assertions, label=label, metric=metric, expected=_num(match.group(1)))

        match = re.search(r"ward[^.]*?cases?[^0-9]*({_NUM_RE})", q)
        if match:
            _add(assertions, label="Ward predicted cases", metric="ward_cases", expected=_num(match.group(1)))
        match = re.search(r"ward[^.]*?(?:tier|risk)\s*(?:should be|is|:)?\s*(low|medium|high|critical)", q)
        if match:
            _add(assertions, label="Ward tier", metric="ward_tier", expected=match.group(1).capitalize())

    # Supply-gap / crew-gap ward pairs.
    match = re.search(
        rf"({_NUM_RE})\s*(?:of|out of|/)\s*({_NUM_RE})\s+wards?.*?supply gap",
        q,
    )
    if match:
        scope = "city" if (city_mentioned or f"in {city_lower}" in q) else "regional"
        prefix = "city" if scope == "city" else "regional"
        _add(assertions, label=f"{scope.title()} supply-gap wards", metric=f"{prefix}_supply_gap_wards", expected=_num(match.group(1)))
        _add(assertions, label=f"{scope.title()} wards", metric=f"{prefix}_wards", expected=_num(match.group(2)))
    elif re.search(r"no ward.*?supply gap", q):
        scope = "city" if (city_mentioned or f"in {city_lower}" in q) else "regional"
        prefix = "city" if scope == "city" else "regional"
        _add(assertions, label=f"{scope.title()} supply-gap wards", metric=f"{prefix}_supply_gap_wards", expected=0)

    match = re.search(
        rf"({_NUM_RE})\s*(?:of|out of|/)\s*({_NUM_RE})\s+wards?.*?crew.*?(?:gap|short)",
        q,
    )
    if match:
        scope = "city" if (city_mentioned or f"in {city_lower}" in q) else "regional"
        prefix = "city" if scope == "city" else "regional"
        _add(assertions, label=f"{scope.title()} crew-gap wards", metric=f"{prefix}_crew_gap_wards", expected=_num(match.group(1)))
        _add(assertions, label=f"{scope.title()} wards", metric=f"{prefix}_wards", expected=_num(match.group(2)))

    # Invariants.
    if re.search(r"no (?:patient )?transfer.*?(?:cross|outside).*?city", q):
        _add(assertions, label="Cross-city transfers", metric="cross_city_transfers", expected=0)
    if re.search(r"no ward.*?supply gap", q):
        scope = "city" if (city_mentioned or f"in {city_lower}" in q) else "regional"
        prefix = "city" if scope == "city" else "regional"
        _add(assertions, label=f"{scope.title()} supply-gap wards", metric=f"{prefix}_supply_gap_wards", expected=0)

    # Accuracy / calibration.
    match = re.search(r"(?:mae|mean absolute error).*?(below|under|less than|<=|at most)\s*([\d.]+)", q)
    if match:
        _add(assertions, label="Holdout MAE", metric="mae", expected=_num(match.group(2)), op="<=")
    else:
        match = re.search(r"(?:mae|mean absolute error).*?(?:should be|is|=)\s*([\d.]+)", q)
        if match:
            _add(assertions, label="Holdout MAE", metric="mae", expected=_num(match.group(1)))
    match = re.search(r"(?:coverage|interval coverage).*?(at least|above|>=)?\s*([\d.]+)\s*%", q)
    if match:
        op = ">=" if match.group(1) else "=="
        _add(assertions, label="80% interval coverage", metric="interval_coverage", expected=_num(match.group(2)) / 100.0, op=op)

    # Supply formula by case count.
    match = re.search(rf"saline for\s+({_NUM_RE})\s+cases?.*?(?:should be|is|=|needs?)\s*({_NUM_RE})", q)
    if match:
        _add(
            assertions,
            label=f"Saline for {_fmt(_num(match.group(1)))} cases",
            metric="saline_for_cases",
            expected=_num(match.group(2)),
            cases=int(_num(match.group(1))),
        )

    # Generic named metrics.
    generic = {
        "regional saline available": "regional_saline_available",
        "regional saline required": "regional_saline_required",
        "regional supply gap wards": "regional_supply_gap_wards",
        "regional crew gap wards": "regional_crew_gap_wards",
        "supply gap wards": "regional_supply_gap_wards",
        "crew gap wards": "regional_crew_gap_wards",
        "transfers": "transfers_count",
        "unrouted beds": "unrouted_beds",
        "city bed demand": "city_bed_demand",
        "city bed capacity": "city_bed_capacity",
    }
    for phrase, metric in generic.items():
        match = re.search(rf"{re.escape(phrase)}.*?(?:should be|is|=|:)\s*({_NUM_RE})", q)
        if match:
            _add(assertions, label=phrase.title(), metric=metric, expected=_num(match.group(1)))

    return assertions


def _compare(actual: float | int | str | None, expected: float | int | str, op: str) -> bool:
    if actual is None:
        return False
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual).lower() == str(expected).lower() if op == "==" else False
    if op == "==":
        return abs(float(actual) - float(expected)) <= 1e-6
    if op == "!=":
        return abs(float(actual) - float(expected)) > 1e-6
    if op == "<":
        return float(actual) < float(expected)
    if op == "<=":
        return float(actual) <= float(expected) + 1e-9
    if op == ">":
        return float(actual) > float(expected)
    if op == ">=":
        return float(actual) >= float(expected) - 1e-9
    return False


def _suggestions(context: Any, city: str, ward_id: str) -> list[str]:
    return [
        "At 60% stock with 8 crews, regional saline should be 749 of 1,249.",
        "31 of 60 wards should have a supply gap at 60% stock.",
        "No transfer should cross a city boundary.",
        f"Ward {str(ward_id).replace('_', ' ')} should have crews allocated.",
        "The model MAE should be below 1.5.",
    ]


def verify_explanation(text: str, context: Any) -> CheckReport:
    """Parse a natural-language test and evaluate it against the engines."""
    q = _normalise(text)
    parsed, allocation, ward_id, city = _scenario(q, context)
    assertions = _parse_assertions(q, context, city, ward_id)

    for assertion in assertions:
        assertion.actual = _metric(
            assertion.metric,
            allocation,
            context,
            ward_id,
            city,
            cases=assertion.cases,
        )
        assertion.passed = _compare(assertion.actual, assertion.expected, assertion.op)

    scenario_line = (
        f"**Parsed scenario:** {parsed['stock']:.0f}% stock · "
        f"{parsed['crews']} crews · {city} · {str(ward_id).replace('_', ' ')}"
    )
    if not assertions:
        region = allocation
        gap_wards = _metric("regional_supply_gap_wards", region, context, ward_id, city)
        crew_wards = _metric("regional_crew_gap_wards", region, context, ward_id, city)
        row = _row(allocation, ward_id)
        lines = [
            scenario_line,
            "",
            "I could not find a checkable claim. Here is the actual output for that scenario:",
            f"- Regional saline: **{_fmt(region.available['saline_bags_needed'])} / "
            f"{_fmt(region.required['saline_bags_needed'])}**",
            f"- Regional supply-gap wards: **{_fmt(gap_wards)} / {len(region.per_ward)}**",
            f"- Regional crew-gap wards: **{_fmt(crew_wards)} / {len(region.per_ward)}**",
        ]
        if row is not None:
            lines.append(
                f"- {str(ward_id).replace('_', ' ')}: **{int(row['allocated_saline'])} / "
                f"{int(row['required_saline'])}** saline · "
                f"**{int(row['allocated_crews'])} / {int(row['required_crews'])}** crews"
            )
        lines.extend(
            [
                "",
                "Add an expected value — for example: *“regional saline should be "
                f"{_fmt(region.available['saline_bags_needed'])} of "
                f"{_fmt(region.required['saline_bags_needed'])}”* — to turn it into a test.",
            ]
        )
        return CheckReport(
            markdown="\n".join(lines),
            passed=None,
            parsed=parsed,
            suggestions=_suggestions(context, city, ward_id),
            sources=["checks.py", "allocation.py"],
        )

    rows = [
        "| Check | Expected | Actual | Result |",
        "|---|---:|---:|---|",
    ]
    for assertion in assertions:
        actual = "n/a" if assertion.actual is None else _fmt(assertion.actual)
        if assertion.metric == "interval_coverage" and assertion.actual is not None:
            actual = f"{float(assertion.actual) * 100:.1f}%"
        expected = _fmt(assertion.expected)
        if assertion.metric == "interval_coverage":
            expected = f"{float(assertion.expected) * 100:.1f}%"
        rows.append(
            f"| {assertion.label} | {expected} | {actual} | "
            f"{'PASS' if assertion.passed else 'FAIL'} |"
        )
    overall = all(a.passed for a in assertions)
    lines = [
        scenario_line,
        "",
        "\n".join(rows),
        "",
        f"**Overall: {'PASS' if overall else 'FAIL'}**",
        "",
        "Read-only: this check re-ran the engines but did not change the live Command Center.",
    ]
    if not overall:
        lines.append("If a value differs, check the scenario controls or the assertion wording.")
    return CheckReport(
        markdown="\n".join(lines),
        passed=overall,
        parsed=parsed,
        assertions=assertions,
        suggestions=_suggestions(context, city, ward_id),
        sources=["checks.py", "allocation.py", "routing.py", "metrics.json"],
    )
