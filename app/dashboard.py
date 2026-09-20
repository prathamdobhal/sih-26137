"""
Fleet dispatcher's command console — the live demo dashboard.

Run with: streamlit run app/dashboard.py

Every control here maps to something the algorithm actually does — there is
no decorative UI. Mode cards select real objective weights (docs/formulation.md),
"Simulate Incident" calls the real warm-started re-optimization from Day 8,
and the explainability panel calls the real Monte Carlo confidence score
from Day 10. Nothing on this page is faked for the demo.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

from app.styles import inject_css, mode_card_html, stat_tile_html, confidence_badge_html, COLORS, MODE_META
from app.data_loader import load_graph_and_source, get_simulator
from app.map_utils import node_latlon, route_to_latlon_path, DEFAULT_BASE_LAT, DEFAULT_BASE_LON, ROUTE_COLORS

from src.solvers.instance import generate_instance
from src.solvers.qpso_vrp import solve_qpso_adaptive, solve_qpso_multistart
from src.solvers.ga_vrp import solve_ga
from src.solvers.evaluate import route_set_cost
from src.dynamic.rerouting import reoptimize_on_incident, recompute_instance_for_traffic
from src.explainability.explain import explain_choice

st.set_page_config(page_title="Koramangala Route Optimizer", layout="wide", page_icon="🛰")
inject_css()

N_CUSTOMERS = 15
N_VEHICLES = 3
QUICK_SWARM = 40
QUICK_ITERS = 70


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    if "initialized" in st.session_state:
        return
    G, source_label = load_graph_and_source()
    sim = get_simulator(G, seed=7)
    inst = generate_instance(G, n_customers=N_CUSTOMERS, num_vehicles=N_VEHICLES,
                              vehicle_capacity=40, seed=7)

    st.session_state.update({
        "initialized": True,
        "G": G,
        "source_label": source_label,
        "sim": sim,
        "inst": inst,
        "mode": "balanced",
        "routes": None,
        "meta": None,
        "cost_info": None,
        "explanation": None,
        "incident_edges": [],
        "last_reroute_time": None,
    })


def run_optimize():
    inst = st.session_state.inst
    # multistart (not a single solve_qpso_adaptive call): a single run at this
    # interactive budget can occasionally land on a worse result for its OWN
    # mode's metric than a differently-weighted mode does by chance — e.g.
    # Fastest showing a higher travel time than Balanced. 3 restarts reliably
    # fixes this while staying well under a second. See qpso_vrp.py docstring.
    routes, cost, meta = solve_qpso_multistart(
        inst, swarm_size=QUICK_SWARM, iterations=QUICK_ITERS,
        n_restarts=3, seed=7, mode=st.session_state.mode,
    )
    st.session_state.routes = routes
    st.session_state.meta = meta
    st.session_state.cost_info = route_set_cost(routes, inst, mode=st.session_state.mode)
    st.session_state.explanation = None  # stale, recompute on demand


def run_incident():
    inst = st.session_state.inst
    sim = st.session_state.sim
    G = st.session_state.G
    meta = st.session_state.meta

    if meta is None:
        run_optimize()
        meta = st.session_state.meta

    # Pick the first real road edge on the current best route's first leg —
    # guarantees the simulated incident actually affects the fleet, matching
    # the deterministic approach used in tests/test_dynamic.py
    route = next((r for r in st.session_state.routes if r), None)
    if route is None:
        return
    from src.solvers.shortest_path import dijkstra, reconstruct_path
    _, prev = dijkstra(G, inst.depot, targets={route[0]})
    path = reconstruct_path(prev, inst.depot, route[0])
    if len(path) < 2:
        return
    u, v = path[0], path[1]

    result = reoptimize_on_incident(
        inst, sim, u, v, meta["final_positions"],
        severity=0.85, radius_hops=1, iterations=50, seed=7,
    )
    st.session_state.inst = result["new_instance"]
    st.session_state.routes = result["routes"]
    st.session_state.meta = result["meta"]
    st.session_state.incident_edges = result["affected_edges"]
    st.session_state.last_reroute_time = result["reoptimize_time_s"]
    st.session_state.cost_info = route_set_cost(result["routes"], result["new_instance"], mode=st.session_state.mode)
    st.session_state.explanation = None


def run_explain():
    inst = st.session_state.inst
    G = st.session_state.G
    meta = st.session_state.meta
    if meta is None:
        run_optimize()
        meta = st.session_state.meta
    st.session_state.explanation = explain_choice(inst, G, meta, top_k=2, n_trials=15, seed=7)


init_state()

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Route Optimizer")
    st.caption(st.session_state.source_label)

    st.markdown("**Optimization mode**")
    for mode_key in ["fastest", "eco", "balanced", "emergency"]:
        selected = st.session_state.mode == mode_key
        st.markdown(mode_card_html(mode_key, selected), unsafe_allow_html=True)
        if st.button(f"Use {MODE_META[mode_key]['label']}", key=f"btn_{mode_key}",
                     width="stretch", type="primary" if selected else "secondary"):
            st.session_state.mode = mode_key
            run_optimize()
            st.rerun()

    st.divider()

    if st.button("Optimize routes", type="primary", width="stretch"):
        run_optimize()

    if st.button("Simulate incident on current route", width="stretch"):
        run_incident()

    st.divider()
    st.caption(
        "Simulate incident blocks a real road on the fleet's current path and "
        "re-optimizes from where the search already was, not from scratch."
    )

# ---------------------------------------------------------------------------
# Main — header
# ---------------------------------------------------------------------------
st.markdown("## Koramangala fleet dispatch")
st.caption(
    "Quantum-inspired route optimization for a delivery fleet — SIH 26137 · "
    f"live view, {MODE_META[st.session_state.mode]['label']} mode"
)

if st.session_state.routes is None:
    run_optimize()

if st.session_state.last_reroute_time is not None:
    st.success(
        f"Rerouted around the incident in {st.session_state.last_reroute_time*1000:.0f} ms "
        f"— warm-started from the previous search, not restarted."
    )

# ---------------------------------------------------------------------------
# Main — map + stats
# ---------------------------------------------------------------------------
map_col, stat_col = st.columns([2.3, 1])

with map_col:
    inst = st.session_state.inst
    G = st.session_state.G
    depot_lat, depot_lon = node_latlon(G, inst.depot)

    # OpenStreetMap tiles: no API key required, unlike CartoDB's dark tiles
    # (which now gate behind a key — confirmed via testing, not assumed).
    # Route lines and markers use our own accent colors regardless of the
    # basemap, so they read clearly against the lighter OSM tiles too.
    fmap = folium.Map(location=[depot_lat, depot_lon], zoom_start=15, tiles="OpenStreetMap")
    folium.CircleMarker([depot_lat, depot_lon], radius=8, color="#FFFFFF", fill=True,
                         fill_color="#FFFFFF", fill_opacity=1, tooltip="Depot").add_to(fmap)

    mode_color = MODE_META[st.session_state.mode]["color"]
    for i, route in enumerate(st.session_state.routes or []):
        if not route:
            continue
        path_latlon = route_to_latlon_path(route, inst.depot, G)
        color = mode_color if len(st.session_state.routes) == 1 else ROUTE_COLORS[i % len(ROUTE_COLORS)]
        folium.PolyLine(path_latlon, color=color, weight=4, opacity=0.85,
                         tooltip=f"Vehicle {i+1}").add_to(fmap)
        for c in route:
            lat, lon = node_latlon(G, c)
            folium.CircleMarker([lat, lon], radius=5, color=color, fill=True,
                                 fill_color=color, fill_opacity=0.9,
                                 tooltip=f"Stop (demand {inst.demand[c]})").add_to(fmap)

    for u, v in st.session_state.incident_edges:
        lat, lon = node_latlon(G, u)
        folium.CircleMarker([lat, lon], radius=9, color=COLORS["emergency"],
                             fill=True, fill_color=COLORS["emergency"], fill_opacity=0.7,
                             tooltip="Incident-affected road").add_to(fmap)

    st_folium(fmap, height=520, use_container_width=True, returned_objects=[])

with stat_col:
    ci = st.session_state.cost_info
    if ci:
        st.markdown(stat_tile_html("Travel time", f"{ci['total_time']/60:.1f} min", mode_color),
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(stat_tile_html("Distance", f"{ci['total_distance']/1000:.1f} km", mode_color),
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(stat_tile_html("Emissions", f"{ci['total_emissions_g']:.0f} g CO2", mode_color),
                    unsafe_allow_html=True)
        st.write("")
        n_vehicles_used = sum(1 for r in st.session_state.routes if r)
        st.markdown(stat_tile_html("Vehicles used", f"{n_vehicles_used} / {inst.num_vehicles}", mode_color),
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs — explainability + benchmark
# ---------------------------------------------------------------------------
st.write("")
tab_explain, tab_benchmark = st.tabs(["Why this route?", "QPSO vs. GA"])

with tab_explain:
    if st.button("Explain this choice"):
        with st.spinner("Comparing against the swarm's runner-up candidate..."):
            run_explain()

    exp = st.session_state.explanation
    if exp is None:
        st.caption("Click \"Explain this choice\" to compare the chosen route against "
                   "the next-best alternative the search considered.")
    elif len(exp["candidates"]) < 2:
        st.caption("Only one distinct candidate route was found — nothing to compare against.")
    else:
        st.markdown(confidence_badge_html(exp["confidence"]), unsafe_allow_html=True)
        st.write("")
        st.write(exp["reason"])
        st.caption(
            f"Confidence reflects how often this route stays the better choice across "
            f"{exp['confidence_detail']['trials']} simulated traffic scenarios with "
            f"congestion varied by ±20%."
        )
        cols = st.columns(2)
        for i, cand in enumerate(exp["candidates"][:2]):
            with cols[i]:
                label = "Chosen route" if i == 0 else "Alternative"
                b = cand["breakdown"]
                st.markdown(f"**{label}**")
                st.markdown(stat_tile_html("Time", f"{b['time_min']:.1f} min"), unsafe_allow_html=True)
                st.write("")
                st.markdown(stat_tile_html("Distance", f"{b['distance_km']:.1f} km"), unsafe_allow_html=True)
                st.write("")
                st.markdown(stat_tile_html("Avg. congestion", f"{b['avg_congestion']*100:.0f}%"),
                            unsafe_allow_html=True)

with tab_benchmark:
    st.caption(
        "Runs a quick head-to-head on the current fleet's exact problem, so this reflects "
        "the live instance rather than a precomputed number. Full multi-size results are in reports/."
    )
    if st.button("Run comparison"):
        inst = st.session_state.inst
        with st.spinner("Solving with both algorithms..."):
            _, ga_cost, ga_hist = solve_ga(inst, pop_size=QUICK_SWARM, generations=QUICK_ITERS,
                                            seed=7, mode=st.session_state.mode)
            _, qpso_cost, qpso_meta = solve_qpso_adaptive(inst, swarm_size=QUICK_SWARM,
                                                            iterations=QUICK_ITERS, seed=7,
                                                            mode=st.session_state.mode)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ga_hist, name="Genetic Algorithm", line=dict(color="#8993A6")))
        fig.add_trace(go.Scatter(y=qpso_meta["gbest_history"], name="QPSO (adaptive)",
                                  line=dict(color=mode_color)))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
            font=dict(family="Inter", color=COLORS["text"]),
            xaxis_title="Iteration / Generation", yaxis_title="Best cost found",
            height=380, margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Final: GA {ga_cost:.3f} · QPSO {qpso_cost:.3f}")
