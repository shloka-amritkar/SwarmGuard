"""SwarmGuard Streamlit app: router, stylesheet and navigation.

Run with:  .venv\\Scripts\\streamlit run app.py

st.navigation builds the top navigation from the page list below. Each
st.Page points to a file in views/ (not pages/, because Streamlit would
otherwise auto-build a second, conflicting menu from a folder named pages/).

Every number the app shows is a simulation result (rule R4).
"""

from __future__ import annotations

import logging

import streamlit as st

from ui.components import navbar
from ui.theme import inject_css

PAGES: dict[str, list[tuple[str, str, str]]] = {
    "Operations": [
        ("views/home.py", "Home", ":material/home:"),
        ("views/dashboard.py", "Dashboard", ":material/dashboard:"),
        ("views/simulator.py", "Swarm Simulator", ":material/flight:"),
        ("views/attack_lab.py", "Attack Lab", ":material/science:"),
    ],
    "Analysis": [
        ("views/detection.py", "Detection Analysis", ":material/troubleshoot:"),
        ("views/experiments.py", "Experiments", ":material/query_stats:"),
        ("views/recovery.py", "Recovery", ":material/my_location:"),
    ],
    "Forensics": [
        ("views/incidents.py", "Incidents", ":material/report:"),
        ("views/evidence.py", "Evidence Centre", ":material/verified_user:"),
    ],
    "Reference": [
        ("views/methodology.py", "Methodology", ":material/menu_book:"),
    ],
}


def build_pages() -> dict[str, list[st.Page]]:
    """st.Page objects grouped by section; Home is the default page."""
    return {
        section: [st.Page(path, title=title, icon=icon, default=(path == "views/home.py"))
                  for path, title, icon in items]
        for section, items in PAGES.items()
    }


def main() -> None:
    """Configure the page, add the stylesheet, show the navigation and run the chosen view."""
    st.set_page_config(page_title="SwarmGuard", page_icon=":material/radar:", layout="wide",
                       initial_sidebar_state="collapsed")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    inject_css()
    nav = st.navigation(build_pages(), position="top")
    navbar(nav.title)
    nav.run()


main()
