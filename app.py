"""Aedes Sentinel — thin page router for Team 3AM.

Run locally with:
    streamlit run app.py

The analytical dashboard lives in ``ui/pages/dashboard.py``. This module only
sets the page config, applies the shared theme, renders the navigation and
dispatches to Home / Purpose / Dashboard / About.
"""
from __future__ import annotations

import streamlit as st

from ui import theme
from ui.components import render_footer, render_navbar

ALLOWED_PAGES = ("home", "purpose", "dashboard", "command", "about")

st.set_page_config(
    page_title="Aedes Sentinel · Team 3AM",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme.apply()

# Page state: session_state is the source of truth; ?page=... keeps links shareable.
query_page = st.query_params.get("page")
if "page" not in st.session_state:
    st.session_state["page"] = query_page if query_page in ALLOWED_PAGES else "home"
elif query_page in ALLOWED_PAGES and query_page != st.session_state["page"]:
    st.session_state["page"] = query_page

current_page = st.session_state["page"]
render_navbar(current_page)

if current_page == "home":
    from ui.pages.home import render as render_home

    render_home()
elif current_page == "purpose":
    from ui.pages.purpose import render as render_purpose

    render_purpose()
elif current_page == "about":
    from ui.pages.about import render as render_about

    render_about()
elif current_page == "command":
    from ui.pages.command import render_command

    render_command()
else:
    # Lazy import keeps the landing page from loading pandas/sklearn/model artifacts.
    from ui.pages.dashboard import render_dashboard

    render_dashboard()

render_footer()
