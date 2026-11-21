"""Purpose / problem statement page."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown(
        """
<div class="page-head anim">
  <div class="page-kicker">Purpose</div>
  <h1 class="page-title">A warning that arrives <br>before the ward fills up.</h1>
  <p class="page-sub">
    Dengue outbreaks in Dhaka, Chattogram and nearby cities follow the monsoon — but
    the response usually starts after hospital admissions climb. By then, transmission
    chains are already weeks ahead. Aedes Sentinel exists to move the decision earlier,
    from <b>“how many cases today?”</b> to <b>“which ward is at risk in the next 2–4 weeks?”</b>
  </p>
</div>

<div class="stat-strip anim delay-1">
  <div class="stat-card">
    <div class="stat-label">Annual hospitalizations</div>
    <div class="stat-number">150k+</div>
    <div class="stat-foot">Dengue-linked admissions in Bangladesh in recent years</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Biological lag</div>
    <div class="stat-number">2–4<span class="stat-unit">weeks</span></div>
    <div class="stat-foot">From rainfall to adult mosquitoes, incubation and symptoms</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Ward-level coverage</div>
    <div class="stat-number">60</div>
    <div class="stat-foot">Weekly weather and surveillance records, 2016–2025</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="purpose-grid anim delay-2">
  <div class="purpose-card">
    <div class="purpose-icon">01</div>
    <h3>The breeding clock</h3>
    <p>
      Heavy rain does not create cases the next morning. Water sits in drains and
      containers; <i>Aedes aegypti</i> eggs take 7–14 days to become adults; viral
      incubation and symptom onset add more delay. That is why the model looks at
      rainfall <b>1–4 weeks before</b> the target week, not same-day weather.
    </p>
  </div>
  <div class="purpose-card">
    <div class="purpose-icon">02</div>
    <h3>Same rain, different risk</h3>
    <p>
      Identical rainfall hits wards differently. A ward with blocked drainage, high
      crowding and weak waste collection becomes a hotspot far faster than an adjacent
      ward. Population density, waterlogging risk and sanitation are part of every
      prediction, not an afterthought.
    </p>
  </div>
  <div class="purpose-card">
    <div class="purpose-icon">03</div>
    <h3>The gap we close</h3>
    <p>
      Larvicide and spraying often happen after admissions spike. A forward-looking
      risk tier gives city corporation teams a defensible way to prioritise wards before
      the peak — and gives hospitals time to prepare beds, saline and testing kits.
    </p>
  </div>
  <div class="purpose-card">
    <div class="purpose-icon">04</div>
    <h3>Who it helps</h3>
    <p>
      Public-health officers, city corporation vector-control teams and hospital
      preparedness units. The dashboard is decision support, not a clinical diagnosis:
      it shows where the next 2–4 weeks are likely to be hardest, and which weather
      drivers are moving the forecast.
    </p>
  </div>
</div>

<div class="home-section anim delay-3">
  <div class="section-kicker">What the dashboard answers</div>
  <div class="flow-row">
    <span class="flow-chip">Which ward is at risk?</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip">In which week?</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip">What is driving it?</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip">How confident is the model?</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
