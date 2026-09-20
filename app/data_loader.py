"""
Graph loading for the dashboard — tries the real cached Koramangala data
first (from Day 3's osm_loader.py), falls back to a synthetic graph if the
cache isn't present (e.g. running in a fresh environment with no internet
access yet). This is what lets the dashboard demo work identically whether
or not the OSM fetch has been run on this machine.
"""

import streamlit as st

from src.network.osm_loader import load_osm_graph, DATA_DIR
from src.network.synthetic import make_synthetic_graph
from src.network.traffic_simulator import TrafficSimulator


@st.cache_resource(show_spinner="Loading road network...")
def load_graph_and_source():
    """Returns (G, source_label). Cached for the whole Streamlit session so
    we don't re-fetch or re-generate on every widget interaction."""
    cache_path = DATA_DIR / "koramangala_bengaluru_india.graphml"
    if cache_path.exists():
        G = load_osm_graph()
        return G, "Koramangala, Bengaluru (real OpenStreetMap data)"

    G = make_synthetic_graph(n_nodes=300, seed=7)
    return G, "Synthetic demo network (real Koramangala data not cached on this machine — " \
               "run `python src/network/osm_loader.py` once to fetch it)"


def get_simulator(_G, seed: int = 7) -> TrafficSimulator:
    """Not cached with @st.cache_resource on purpose — the simulator holds
    mutable state (congestion values change via inject_incident/step) that
    must persist in st.session_state across reruns, not be silently reset."""
    return TrafficSimulator(_G, seed=seed)
