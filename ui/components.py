"""Reusable shell components: navbar, footer, navigation helpers."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import streamlit as st

ALLOWED_PAGES = ("home", "purpose", "dashboard", "command", "about")

NAV_ITEMS = (
    ("Home", "home"),
    ("Purpose", "purpose"),
    ("Dashboard", "dashboard"),
    ("Command", "command"),
    ("About Us", "about"),
)

GITHUB_URL = "https://github.com/faisalkhabirr/Project-3-Am"
ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def build_hash() -> str:
    """Short git hash for the on-screen build stamp (works offline)."""
    override = os.getenv("COMMIT_HASH")
    if override:
        return override[:12]
    try:
        secret = st.secrets.get("COMMIT_HASH")  # type: ignore[union-attr]
    except Exception:
        secret = None
    if secret:
        return str(secret)[:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "local"


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
            [1.45, 0.78, 0.86, 0.94, 1.02, 0.9, 1.2],
            gap="small",
            vertical_alignment="center",
        )
        columns[0].markdown(
            '<a class="brand" href="?page=home"><span class="brand-mark">3</span>AM</a>',
            unsafe_allow_html=True,
        )
        for column, (label, page) in zip(columns[1:6], NAV_ITEMS):
            active = current_page == page
            if column.button(
                label,
                key=f"nav_{page}",
                type="primary" if active else "secondary",
                width="content",
            ):
                go_to(page)
        columns[6].markdown(
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
    Build <span style="font-family:monospace;color:rgba(255,255,255,0.75)">{build_hash()}</span><br>
    Not a clinical diagnostic tool.<br>
    © 2026 Team 3AM · <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
