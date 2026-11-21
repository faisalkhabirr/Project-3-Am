"""Reusable shell components: navbar, footer, navigation helpers."""
from __future__ import annotations

import streamlit as st

ALLOWED_PAGES = ("home", "purpose", "dashboard", "about")

NAV_ITEMS = (
    ("Home", "home"),
    ("Purpose", "purpose"),
    ("Dashboard", "dashboard"),
    ("About Us", "about"),
)

GITHUB_URL = "https://github.com/faisalkhabirr/Project-3-Am"


def go_to(page: str) -> None:
    """Switch page, keep the URL shareable, and rerun."""
    if page not in ALLOWED_PAGES:
        return
    st.session_state["page"] = page
    st.query_params["page"] = page
    st.rerun()


def render_navbar(current_page: str) -> None:
    """Pill navigation, sticky at the top of the app sheet."""
    with st.container(key="topnav"):
        columns = st.columns(
            [1.55, 0.82, 0.88, 1.0, 0.9, 1.35],
            gap="small",
            vertical_alignment="center",
        )
        columns[0].markdown(
            '<a class="brand" href="?page=home"><span class="brand-mark">3</span>AM</a>',
            unsafe_allow_html=True,
        )
        for column, (label, page) in zip(columns[1:5], NAV_ITEMS):
            active = current_page == page
            if column.button(
                label,
                key=f"nav_{page}",
                type="primary" if active else "secondary",
                width="content",
            ):
                go_to(page)
        columns[5].markdown(
            '<div class="nav-note"><span class="status-dot"></span>Local · 0 APIs</div>',
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        f"""
<div class="site-footer">
  <div>
    <div class="footer-brand"><span class="brand-mark">3</span>AM · Aedes Sentinel</div>
    <div class="footer-tag">Ward-level dengue early warning for Bangladesh.</div>
  </div>
  <div class="footer-meta">
    Not a clinical diagnostic tool.<br>
    © 2026 Team 3AM · <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
