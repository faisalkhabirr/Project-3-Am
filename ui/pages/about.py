"""About Team 3AM."""
from __future__ import annotations

import streamlit as st

TEAM = (
    {
        "initials": "FK",
        "name": "Faisal Khabir",
        "handle": "faisalkhabirr",
        "affiliation": "Student, East Delta University, Chattogram",
    },
    {
        "initials": "MR",
        "name": "Muhammad Anamur Rashid",
        "handle": "loser-2-o",
        "affiliation": "Student, East Delta University, Chattogram",
    },
    {
        "initials": "TJ",
        "name": "Tanvir Jonayed",
        "handle": "tanvir-jonayed",
        "affiliation": "Student, East Delta University, Chattogram",
    },
)


def render() -> None:
    st.markdown(
        """
<div class="page-head anim">
  <div class="page-kicker">About us</div>
  <h1 class="page-title">Built by Team 3AM.</h1>
  <p class="page-sub">
    We are three students at East Delta University, Chattogram, building a local-first
    dengue early-warning system for Bangladesh. The shipped dashboard runs entirely on a
    laptop, with no external API calls and no patient data.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    cards = "".join(
        f"""
<div class="member-card anim delay-{index + 1}">
  <div class="avatar">{member['initials']}</div>
  <div class="member-name">{member['name']}</div>
  <div class="member-handle">@{member['handle']}</div>
  <div class="member-affiliation">{member['affiliation']}</div>
  <div class="member-links">
    <a href="https://github.com/{member['handle']}" target="_blank" rel="noopener">GitHub ↗</a>
  </div>
</div>
"""
        for index, member in enumerate(TEAM)
    )
    st.markdown(f'<div class="team-grid">{cards}</div>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="project-card anim delay-3">
  <h3>Project info</h3>
  <p>
    Aedes Sentinel is a Poisson gradient-boosting model trained on weekly ward-level
    weather and dengue surveillance. The app maps each prediction to a ward-specific
    risk tier, an 80% interval and a local driver breakdown. The Phase 2 command
    center turns that forecast into constrained supplies, intra-city overflow
    routing, counterfactual crew simulation and offline field dispatch. A read-only
    assistant answers questions from the live scenario without an external API.
  </p>
  <div class="project-grid">
    <div class="project-item"><b>Poisson HistGBR</b><span>Weekly cases, 1–4 week horizons</span></div>
    <div class="project-item"><b>60 wards · 5 cities</b><span>2016–2025, weekly resolution</span></div>
    <div class="project-item"><b>Phase 2 operations</b><span>Allocate · route · simulate · dispatch</span></div>
    <div class="project-item"><b>Field assistant</b><span>Read-only · grounded · offline</span></div>
  </div>
  <div class="chip-row">
    <span class="chip">Python</span>
    <span class="chip">Streamlit</span>
    <span class="chip">scikit-learn</span>
    <span class="chip">Plotly</span>
    <span class="chip">pandas</span>
  </div>
  <p style="margin-top:16px; margin-bottom:0;">
    <a href="https://github.com/faisalkhabirr/Project-3-Am" target="_blank" rel="noopener">
      View the repository on GitHub →
    </a>
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
