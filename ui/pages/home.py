"""Landing page for Aedes Sentinel."""
from __future__ import annotations

import streamlit as st

from ui.components import go_to


def render() -> None:
    st.markdown(
        """
<section class="home-hero">
  <div>
    <div class="home-kicker">Team 3AM · Dengue early warning</div>
    <h1 class="home-title">Predicting dengue risk <span class="accent">before it peaks.</span></h1>
    <p class="home-sub">
      A ward-level early-warning dashboard that turns 1–4 weeks of weather lag
      into a 2–4 week window for vector control, hospital readiness and
      targeted larvicide — then turns the forecast into saline, beds, crews and
      field dispatch in the Phase 2 command center.
    </p>
  </div>
  <div class="home-hero-right">
    <div class="preview-card">
      <div class="preview-top">
        <span class="preview-kicker">Example forecast</span>
        <span class="preview-chip">High</span>
      </div>
      <div class="preview-ring">
        <div class="preview-ring-inner">
          <div><b>67</b><span>/100 risk</span></div>
        </div>
      </div>
      <div class="preview-row"><span>DNCC Ward 12 · Dhaka North</span><strong>12.3 cases</strong></div>
      <div class="preview-row"><span>80% interval</span><strong>10.8–14.6</strong></div>
      <div class="preview-row"><span>Target week</span><strong>15 Sep 2025</strong></div>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

    with st.container(key="home_cta"):
        c1, c2, c3, _ = st.columns([1.15, 1.15, 1.35, 2.0], gap="small")
        if c1.button(
            "Open dashboard",
            key="open_dashboard",
            type="primary",
            width="stretch",
        ):
            go_to("dashboard")
        if c2.button(
            "Phase 2 command center",
            key="open_command",
            type="primary",
            width="stretch",
        ):
            go_to("command")
        if c3.button(
            "Why we built this",
            key="why_built",
            type="secondary",
            width="stretch",
        ):
            go_to("purpose")

    st.markdown(
        """
<div class="stat-grid anim delay-1">
  <div class="stat-card">
    <div class="stat-label">Wards covered</div>
    <div class="stat-number count-60"></div>
    <div class="stat-foot">Across five city corporations</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Cities</div>
    <div class="stat-number count-5"></div>
    <div class="stat-foot">Dhaka North &amp; South, Chattogram, Gazipur, Narayanganj</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Early warning</div>
    <div class="stat-number">1–4<span class="stat-unit">weeks</span></div>
    <div class="stat-foot">Ahead of the reported case peak</div>
  </div>
</div>

<div class="home-section anim delay-2">
  <div class="section-kicker">How it works</div>
  <div class="flow-row">
    <span class="flow-chip">Weekly weather</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip">Multi-week lags</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip">Poisson model</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip">Risk tier + 80% interval</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
