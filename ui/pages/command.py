"""Phase 2 Command Center — allocate, route, simulate and dispatch in one flow."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import charts, config, data_access as data, inference
from src.operations.allocation import WardDemand, allocate
from src.operations.dispatch import build_notice, write_exports
from src.operations.routing import (
    capacity_scaled,
    derive_bed_capacity,
    route_overflow,
)
from src.operations.simulation import Deployment, simulate

SUPPLY_LABELS = {
    "saline_bags_needed": "Saline",
    "diagnostic_kits_needed": "Kits",
    "hospital_beds_required": "Beds",
}


def _default_as_of(dates: list[pd.Timestamp]) -> pd.Timestamp:
    if not dates:
        return pd.Timestamp.today().normalize()
    latest = dates[-1].year
    august = [d for d in dates if d.year == latest and d.month == 8]
    return august[-1] if august else dates[-1]


def _fulfilment(allocated: float, required: float) -> float:
    return 100.0 * allocated / required if required else 100.0


def _kpi_row(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items), gap="small")
    for column, (label, value, sub) in zip(columns, items, strict=True):
        column.markdown(
            f"<div class='mini-card'><div class='mini-label'>{label}</div>"
            f"<div class='mini-value'>{value}</div><div class='mini-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )


def _allocation_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame[
        [
            "ward_id",
            "tier",
            "predicted_cases",
            "required_saline",
            "allocated_saline",
            "deficit_saline",
            "required_kits",
            "allocated_kits",
            "deficit_kits",
            "required_beds",
            "allocated_beds",
            "deficit_beds",
            "required_crews",
            "allocated_crews",
            "deficit",
            "fulfillment_supply_pct",
            "fulfillment_crew_pct",
        ]
    ].copy()
    display["ward_id"] = display["ward_id"].astype(str).str.replace("_", " ")
    for col in ("fulfillment_supply_pct", "fulfillment_crew_pct"):
        display[col] = display[col] * 100
    display = display.rename(
        columns={
            "ward_id": "Ward",
            "tier": "Tier",
            "predicted_cases": "Cases",
            "required_saline": "Saline need",
            "allocated_saline": "Saline got",
            "deficit_saline": "Saline gap",
            "required_kits": "Kits need",
            "allocated_kits": "Kits got",
            "deficit_kits": "Kits gap",
            "required_beds": "Beds need",
            "allocated_beds": "Beds got",
            "deficit_beds": "Beds gap",
            "required_crews": "Crews need",
            "allocated_crews": "Crews got",
            "deficit": "Deficit",
            "fulfillment_supply_pct": "Supply %",
            "fulfillment_crew_pct": "Crew %",
        }
    )
    return display


def _city_rollup_table(rollup: pd.DataFrame) -> pd.DataFrame:
    display = rollup.copy()
    display["saline"] = display.apply(
        lambda row: f"{int(row['allocated_saline'])} / {int(row['required_saline'])}", axis=1
    )
    display["beds"] = display.apply(
        lambda row: f"{int(row['allocated_beds'])} / {int(row['required_beds'])}", axis=1
    )
    display["crews"] = display.apply(
        lambda row: f"{int(row['allocated_crews'])} / {int(row['required_crews'])}", axis=1
    )
    display["supply_%"] = display["supply_fulfillment_pct"] * 100
    display["crew_%"] = display["crew_fulfillment_pct"] * 100
    return display[
        [
            "city",
            "wards",
            "predicted_cases",
            "saline",
            "beds",
            "crews",
            "supply_%",
            "crew_%",
            "deficit_wards",
        ]
    ].rename(
        columns={
            "city": "City",
            "wards": "Wards",
            "predicted_cases": "Cases",
            "saline": "Saline got / need",
            "beds": "Beds got / need",
            "crews": "Crews got / need",
            "supply_%": "Supply %",
            "crew_%": "Crew %",
            "deficit_wards": "Short wards",
        }
    )


def render_command() -> None:
    try:
        onsite = data.load_onsite()
        meta = data.load_ward_meta()
        features = data.load_features()
        bundle = inference.load_bundle()
    except Exception as exc:  # noqa: BLE001 - shown to the user
        st.error(f"Command Center needs the bundled data and model: {exc}")
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
      <div class="page-kicker">Phase 2 · operational command</div>
      <h1 class="page-title">Command Center</h1>
      <p class="page-sub">Forecast → constrained stock → intra-city overflow → counterfactual
      crews → field dispatch. Every step is deterministic and runs on this laptop.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ controls
    with st.container(key="controls_card"):
        c1, c2, c3 = st.columns([1.05, 1.35, 1.0], gap="medium")
        city = c1.selectbox("City", cities, index=cities.index(default_city), key="cmd_city")
        dates_desc = list(reversed(dates))
        date_index = dates_desc.index(default_date) if default_date in dates_desc else 0
        as_of = c2.selectbox(
            "Forecast observed through",
            dates_desc,
            index=date_index,
            format_func=data.format_date,
            key="cmd_as_of",
        )
        horizon = c3.select_slider(
            "Forecast horizon",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda h: f"+{h} week" + ("s" if int(h) > 1 else ""),
            key="cmd_horizon",
        )
        c4, c5, c6 = st.columns([1.25, 1.25, 1.25], gap="medium")
        stock_pct = c4.slider(
            "Available regional stock",
            min_value=30,
            max_value=100,
            value=60,
            step=5,
            format="%d%%",
            key="cmd_stock",
            help="Shared across all 60 wards. The engine protects a floor and then "
            "water-fills by risk × density.",
        )
        total_crews = c5.slider(
            "Total spraying crews",
            min_value=0,
            max_value=40,
            value=8,
            step=1,
            key="cmd_crews",
        )
        capacity_mult = c6.slider(
            "Bed capacity multiplier",
            min_value=0.60,
            max_value=2.00,
            value=1.20,
            step=0.10,
            key="cmd_capacity",
            help="Stress-test the overflow router against the historical p75 baseline.",
        )
        st.caption(
            "Local execution · 0 API calls · the same inputs always produce the same numbers."
        )
        conservative = st.toggle(
            "Plan against the 80% upper interval",
            value=False,
            key="cmd_conservative",
            help="Size supplies, beds and routing against each ward's upper 80% forecast "
            "interval instead of the point forecast. Useful for conservative stock planning.",
        )
        if conservative:
            st.caption(
                "Conservative mode: operational demand is sized against the upper 80% interval."
            )
        if st.button("Reset demo defaults", key="cmd_reset"):
            for key in list(st.session_state):
                if key.startswith(("cmd_", "sim_", "dispatch_ward_")):
                    del st.session_state[key]
            st.rerun()

    as_of = pd.Timestamp(as_of).normalize()
    horizon = int(horizon)

    # ------------------------------------------------------------------ forecast
    all_wards = sorted(onsite[config.WARD_COL].unique())
    preds = inference.predict_batch(
        all_wards, as_of, [horizon], features=features, bundle=bundle
    )
    meta_idx = meta.set_index(config.WARD_COL)
    demands: list[WardDemand] = []
    for _, row in preds.iterrows():
        ward = str(row[config.WARD_COL])
        if pd.isna(row["cases"]) or ward not in meta_idx.index:
            continue
        info = meta_idx.loc[ward]
        planning_cases = (
            float(row["high"])
            if conservative and pd.notna(row.get("high"))
            else float(row["cases"])
        )
        demands.append(
            WardDemand.from_prediction(
                ward_id=ward,
                city=str(info[config.CITY_COL]),
                predicted_cases=planning_cases,
                tier=str(row["tier"]),
                density=float(info["population_density_sqkm"]),
            )
        )
    if not demands:
        st.error("No ward forecasts were available for this week.")
        st.stop()

    allocation = allocate(demands, stock_pct / 100.0, int(total_crews))
    city_frame = allocation.per_ward.loc[
        allocation.per_ward["city"] == city
    ].sort_values(["priority", "predicted_cases"], ascending=False)
    target_date = as_of + pd.Timedelta(weeks=horizon)

    # ------------------------------------------------------------------ routing
    base_capacity = derive_bed_capacity(onsite, as_of=as_of)
    bed_capacity = capacity_scaled(base_capacity, capacity_mult)
    bed_demand = {
        str(row["ward_id"]): int(row["required_beds"])
        for _, row in allocation.per_ward.iterrows()
    }
    ward_city = {d.ward_id: d.city for d in demands}
    routing = route_overflow(bed_demand, bed_capacity, ward_city)
    overflow_by_receiver: dict[str, dict[str, int]] = {}
    for donor, receiver, moved in routing.transfers:
        overflow_by_receiver.setdefault(receiver, {})[donor] = moved
    city_transfers = [
        (donor, receiver, moved)
        for donor, receiver, moved in routing.transfers
        if ward_city.get(receiver) == city
    ]

    city_required_saline = int(city_frame["required_saline"].sum())
    city_allocated_saline = int(city_frame["allocated_saline"].sum())
    city_supply_gap_wards = int(
        (city_frame[["deficit_saline", "deficit_kits", "deficit_beds"]].sum(axis=1) > 0).sum()
    )
    city_crew_gap_wards = int((city_frame["deficit_crews"] > 0).sum())
    city_capacity = sum(bed_capacity.get(ward, 0) for ward in city_frame["ward_id"])
    city_bed_demand = sum(bed_demand.get(ward, 0) for ward in city_frame["ward_id"])
    city_unrouted = sum(
        routing.net_deficits.get(ward, 0) for ward in city_frame["ward_id"]
    )
    regional_required_saline = int(allocation.required["saline_bags_needed"])
    regional_available_saline = int(allocation.available["saline_bags_needed"])
    regional_supply_gap_wards = int(
        (
            allocation.per_ward[["deficit_saline", "deficit_kits", "deficit_beds"]].sum(axis=1)
            > 0
        ).sum()
    )
    regional_crew_gap_wards = int((allocation.per_ward["deficit_crews"] > 0).sum())
    regional_wards = len(allocation.per_ward)

    _kpi_row(
        [
            (
                "Forecast week",
                target_date.strftime("%d %b"),
                f"+{horizon} weeks · {city}" + (" · 80% upper" if conservative else ""),
            ),
            (
                "Regional saline",
                f"{regional_available_saline:,} / {regional_required_saline:,}",
                f"{city}: {city_allocated_saline:,}/{city_required_saline:,} "
                f"({_fulfilment(city_allocated_saline, city_required_saline):.0f}%)",
            ),
            (
                "Regional supply gaps",
                f"{regional_supply_gap_wards} / {regional_wards} wards",
                f"{city}: {city_supply_gap_wards} / {len(city_frame)} · "
                f"crews short in {regional_crew_gap_wards}/{regional_wards}",
            ),
            (
                "City bed pressure",
                f"{city_bed_demand} / {city_capacity}",
                f"{city_unrouted} unrouted · {len(city_transfers)} transfers in",
            ),
        ]
    )

    allocate_tab, route_tab, simulate_tab, dispatch_tab = st.tabs(
        ["1 · Allocate", "2 · Overflow", "3 · Simulate", "4 · Dispatch"]
    )

    # ---------------------------------------------------------------- allocate
    with allocate_tab:
        left, right = st.columns([1.55, 1.0], gap="large")
        with left:
            st.plotly_chart(
                charts.allocation_figure(city_frame),
                width="stretch",
                config={"displayModeBar": False},
            )
        with right:
            st.markdown("#### City roll-up · all 60 wards")
            st.dataframe(
                _city_rollup_table(allocation.city_rollup),
                hide_index=True,
                width="stretch",
                height=300,
                column_config={
                    "Cases": st.column_config.NumberColumn("Cases", format="%.1f"),
                    "Supply %": st.column_config.ProgressColumn(
                        "Supply %", min_value=0, max_value=100, format="%.0f%%"
                    ),
                    "Crew %": st.column_config.ProgressColumn(
                        "Crew %", min_value=0, max_value=100, format="%.0f%%"
                    ),
                },
            )
            st.caption(
                "Supply % is the bottleneck across saline, kits and beds. "
                "Crew % is assigned vs required spraying teams. A ward is short if either "
                "is below 100%."
            )

        city_supply_gap = city_frame[
            ["deficit_saline", "deficit_kits", "deficit_beds"]
        ].sum(axis=1)
        if city_supply_gap_wards > 0:
            if (city_supply_gap > 0).any():
                worst = city_frame.assign(_gap=city_supply_gap).sort_values(
                    "_gap", ascending=False
                ).head(3)
                names = ", ".join(
                    f"{ward.replace('_', ' ')} ({int(gap)} units)"
                    for ward, gap in zip(worst["ward_id"], worst["_gap"], strict=True)
                )
                supply_text = f"Largest supply gaps: {names}."
            else:
                supply_text = ""
            st.error(
                f"Critical supply deficit in {city}: {city_supply_gap_wards} of "
                f"{len(city_frame)} wards cannot be 100% fulfilled at this stock level. "
                f"{supply_text} {city_crew_gap_wards} wards are short on crews."
            )
        elif city_crew_gap_wards > 0:
            st.warning(
                f"Supplies are covered in {city}, but {city_crew_gap_wards} of "
                f"{len(city_frame)} wards are short on spraying crews."
            )
        else:
            st.success("All ward demand is covered at this stock level.")

        st.markdown("#### Ward allocation ledger")
        st.dataframe(
            _allocation_table(city_frame),
            hide_index=True,
            width="stretch",
            height=430,
            column_config={
                "Supply %": st.column_config.ProgressColumn(
                    "Supply %", min_value=0, max_value=100, format="%.0f%%"
                ),
                "Crew %": st.column_config.ProgressColumn(
                    "Crew %", min_value=0, max_value=100, format="%.0f%%"
                ),
                "Deficit": st.column_config.CheckboxColumn("Deficit"),
                "Cases": st.column_config.NumberColumn("Cases", format="%.1f"),
            },
        )
        st.caption(
            "Rule: a 10% protected floor for every ward, then strict priority water-filling "
            "by risk tier × population density. Crews are fully discretionary and go to the "
            "highest-priority wards first."
        )

    # ---------------------------------------------------------------- routing
    with route_tab:
        st.markdown("#### Intra-city overflow routing")
        st.caption(
            "Bed capacity baseline = each ward's historical 75th-percentile weekly bed demand "
            f"(minimum 2). Current capacity multiplier: {capacity_mult:.2f}×."
        )
        left, right = st.columns([1.0, 1.25], gap="large")
        with left:
            st.plotly_chart(
                charts.city_utilization_figure(routing.city_utilization),
                width="stretch",
                config={"displayModeBar": False},
            )
            if routing.escalated_cities:
                st.error(
                    "Regional escalation: " + ", ".join(routing.escalated_cities)
                    + " cannot absorb their own projected bed demand inside the city pool."
                )
            else:
                st.success("Every city can absorb its projected demand inside its own pool.")
        with right:
            st.markdown("#### City bed position")
            rows = []
            for name in cities:
                wards = [w for w in bed_demand if ward_city.get(w) == name]
                demand = sum(bed_demand[w] for w in wards)
                capacity = sum(bed_capacity.get(w, 0) for w in wards)
                rows.append(
                    {
                        "City": name,
                        "Demand": demand,
                        "Capacity": capacity,
                        "Headroom": capacity - demand,
                        "Utilization": 100 * demand / capacity if capacity else 0.0,
                        "Unrouted": sum(routing.net_deficits.get(w, 0) for w in wards),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                height=250,
                column_config={
                    "Utilization": st.column_config.ProgressColumn(
                        "Utilization", min_value=0, max_value=150, format="%.0f%%"
                    )
                },
            )
            if city_transfers:
                st.markdown(f"#### Transfers into {city}")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "From": donor.replace("_", " "),
                                "To": receiver.replace("_", " "),
                                "Patients": moved,
                            }
                            for donor, receiver, moved in city_transfers
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("No patient transfer is needed in this city at the current capacity.")

        st.markdown("#### Unrouted deficits after routing")
        deficit_rows = [
            {
                "Ward": ward.replace("_", " "),
                "City": ward_city.get(ward, ""),
                "Projected beds": bed_demand[ward],
                "Capacity": bed_capacity.get(ward, 0),
                "Unrouted": deficit,
            }
            for ward, deficit in routing.net_deficits.items()
            if deficit > 0
        ]
        if deficit_rows:
            st.dataframe(
                pd.DataFrame(deficit_rows).sort_values("Unrouted", ascending=False),
                hide_index=True,
                width="stretch",
                height=260,
            )
        else:
            st.success("All bed deficits were routed within city pools.")

    # -------------------------------------------------------------- simulation
    with simulate_tab:
        st.markdown("#### What-if vector control")
        st.caption(
            "The simulator rewrites breeding-habitat features (larvicide) or transmission "
            "pressure (fogging), re-runs the serialized Phase 1 model, and feeds predictions "
            "back into next week's case lags. No hardcoded percentage reduction."
        )
        wards_city = data.wards_for_city(meta, city)
        default_wards = [
            ward
            for ward in city_frame["ward_id"].head(3).tolist()
            if ward in wards_city
        ]
        s1, s2, s3, s4 = st.columns([1.7, 1.0, 1.0, 1.0], gap="medium")
        selected_wards = s1.multiselect(
            "Wards to treat",
            wards_city,
            default=default_wards,
            key=f"sim_wards_{city}",
        )
        crew_type = s2.radio(
            "Crew type",
            ["larvicide", "fogging"],
            horizontal=True,
            key="sim_type",
        )
        crews_per_ward = s3.slider("Crews per ward", 1, 4, 2, key="sim_crews")
        start_week = s4.slider("Start week", 1, 4, 1, key="sim_start")
        weeks = st.slider("Simulation horizon", 2, 4, 4, key="sim_weeks")
        run = st.button("Run counterfactual", type="primary", key="sim_run")
        signature = (
            tuple(selected_wards),
            crew_type,
            crews_per_ward,
            start_week,
            weeks,
            str(as_of),
            city,
        )
        if run:
            if not selected_wards:
                st.warning("Pick at least one ward to treat.")
            else:
                with st.spinner("Re-running the model on modified features..."):
                    result = simulate(
                        [
                            Deployment(ward, crew_type, crews_per_ward, start_week)
                            for ward in selected_wards
                        ],
                        weeks=weeks,
                        as_of=as_of,
                        features=features,
                        bundle=bundle,
                    )
                st.session_state["sim_result"] = result
                st.session_state["sim_signature"] = signature

        if st.session_state.get("sim_signature") == signature and "sim_result" in st.session_state:
            result = st.session_state["sim_result"]
            frame = result.per_ward_week
            prevented = float(frame["cases_prevented"].sum())
            beds_saved = float((frame["baseline_beds"] - frame["intervention_beds"]).sum())
            saline_saved = float(
                (frame["baseline_saline"] - frame["intervention_saline"]).sum()
            )
            _kpi_row(
                [
                    ("Cases prevented", f"{prevented:.1f}", "4-week cumulative"),
                    ("Beds freed", f"{beds_saved:.0f}", "ward-weeks"),
                    ("Saline avoided", f"{saline_saved:.0f}", "bags"),
                ]
            )
            st.plotly_chart(
                charts.simulation_figure(frame, selected_wards[0]),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.markdown("#### Intervention ROI")
            roi = result.roi_ranking.copy()
            roi["reduction_pct"] = roi["reduction_pct"] * 100
            st.dataframe(
                roi.rename(
                    columns={
                        "ward_id": "Ward",
                        "crews": "Crews",
                        "cases_prevented": "Cases prevented",
                        "cases_per_crew": "Cases / crew",
                        "baseline_cases": "Baseline cases",
                        "intervention_cases": "With crews",
                        "reduction_pct": "Reduction",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Cases / crew": st.column_config.NumberColumn(
                        "Cases / crew", format="%.2f"
                    ),
                    "Reduction": st.column_config.ProgressColumn(
                        "Reduction", min_value=0, max_value=100, format="%.0f%%"
                    ),
                },
            )
        else:
            st.info("Choose wards and crews, then run the counterfactual.")

    # ---------------------------------------------------------------- dispatch
    with dispatch_tab:
        st.markdown("#### Field dispatch")
        st.caption(
            "SMS, WhatsApp and a printable commissioner brief generated from the same "
            "numbers. Templates only — no LLM, no network."
        )
        wards_city = data.wards_for_city(meta, city)
        dispatch_ward = st.selectbox(
            "Ward to brief",
            wards_city,
            index=0,
            format_func=lambda w: str(w).replace("_", " "),
            key=f"dispatch_ward_{city}",
        )
        row = allocation.per_ward.loc[
            allocation.per_ward["ward_id"] == dispatch_ward
        ].iloc[0]
        notice = build_notice(
            ward_id=dispatch_ward,
            city=str(row["city"]),
            target_date=target_date.strftime("%d %b %Y"),
            predicted_cases=float(row["predicted_cases"]),
            tier=str(row["tier"]),
            required={
                "saline_bags_needed": int(row["required_saline"]),
                "diagnostic_kits_needed": int(row["required_kits"]),
                "hospital_beds_required": int(row["required_beds"]),
            },
            allocated={
                "saline_bags_needed": int(row["allocated_saline"]),
                "diagnostic_kits_needed": int(row["allocated_kits"]),
                "hospital_beds_required": int(row["allocated_beds"]),
            },
            crews_required=int(row["required_crews"]),
            crews_allocated=int(row["allocated_crews"]),
            overflow=overflow_by_receiver.get(dispatch_ward, {}),
            notes=(
                f"{stock_pct}% regional stock · {total_crews} crews · "
                f"{capacity_mult:.2f}× bed capacity."
            ),
        )
        left, right = st.columns([1.0, 1.15], gap="large")
        with left:
            st.markdown("**SMS alert**")
            st.code(notice.sms, language=None)
            st.download_button(
                "Download SMS",
                notice.sms,
                file_name=f"{dispatch_ward}_sms.txt",
                mime="text/plain",
                key="dl_sms",
            )
            st.markdown("**WhatsApp**")
            st.code(notice.whatsapp, language=None)
            st.download_button(
                "Download WhatsApp",
                notice.whatsapp,
                file_name=f"{dispatch_ward}_whatsapp.txt",
                mime="text/plain",
                key="dl_wa",
            )
            st.download_button(
                "Download printable brief",
                notice.html,
                file_name=f"{dispatch_ward}_brief.html",
                mime="text/html",
                key="dl_html",
            )
        with right:
            components.html(notice.html, height=620, scrolling=True)

        if st.button("Export all city packets to /exports", key="export_all"):
            city_notices = []
            for _, ward_row in city_frame.iterrows():
                ward = str(ward_row["ward_id"])
                city_notices.append(
                    build_notice(
                        ward_id=ward,
                        city=str(ward_row["city"]),
                        target_date=target_date.strftime("%d %b %Y"),
                        predicted_cases=float(ward_row["predicted_cases"]),
                        tier=str(ward_row["tier"]),
                        required={
                            "saline_bags_needed": int(ward_row["required_saline"]),
                            "diagnostic_kits_needed": int(ward_row["required_kits"]),
                            "hospital_beds_required": int(ward_row["required_beds"]),
                        },
                        allocated={
                            "saline_bags_needed": int(ward_row["allocated_saline"]),
                            "diagnostic_kits_needed": int(ward_row["allocated_kits"]),
                            "hospital_beds_required": int(ward_row["allocated_beds"]),
                        },
                        crews_required=int(ward_row["required_crews"]),
                        crews_allocated=int(ward_row["allocated_crews"]),
                        overflow=overflow_by_receiver.get(ward, {}),
                        notes=f"{stock_pct}% regional stock · local template.",
                    )
                )
            written = write_exports(city_notices, Path("exports"))
            st.success(f"Wrote {len(written)} files for {len(city_notices)} wards to /exports.")
