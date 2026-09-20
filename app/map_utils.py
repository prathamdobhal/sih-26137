"""
Map utilities for the dashboard — kept separate from app.py (which is
Streamlit-specific and can't be unit-tested directly) so the actual geometry
logic can be verified in isolation.

Handles two graph types uniformly:
  - OSM graphs (osm_loader.py): node x/y are real lon/lat
  - Synthetic graphs (synthetic.py): node x/y are meters on a flat local grid
Both get converted to (lat, lon) so the map renders correctly either way —
this is what lets the dashboard work in a from-scratch demo (synthetic
fallback) and with the real Koramangala data with no code changes.
"""

import math
import networkx as nx

from src.solvers.shortest_path import dijkstra, reconstruct_path

# Koramangala 4th Block — matches src/network/osm_loader.py's DEFAULT_POINT,
# used as the reference origin when georeferencing a synthetic (meters-based) graph.
DEFAULT_BASE_LAT = 12.9352
DEFAULT_BASE_LON = 77.6245

ROUTE_COLORS = ["#25726A", "#254688", "#71398B", "#A25034", "#E73DAC", "#C73F3F"]


def node_latlon(G: nx.DiGraph, node, base_lat: float = DEFAULT_BASE_LAT,
                 base_lon: float = DEFAULT_BASE_LON) -> tuple:
    """Returns (lat, lon) for any node, whether the graph is real OSM data
    (x=lon, y=lat already) or a synthetic flat-meter grid (converted via a
    standard meters-to-degrees approximation around the base point)."""
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]

    if -180 <= x <= 180 and -90 <= y <= 90:
        return y, x  # already real lon/lat

    lat = base_lat + (y / 111_320)
    lon = base_lon + (x / (111_320 * math.cos(math.radians(base_lat))))
    return lat, lon


def route_to_latlon_path(route: list, depot, G: nx.DiGraph) -> list:
    """Reconstructs the REAL road path (via Dijkstra) for one vehicle's full
    route (depot -> stops -> depot) and returns it as a list of (lat, lon)
    points — so the drawn line hugs actual roads, not straight lines between
    stops."""
    if not route:
        return []
    leg_nodes = [depot] + route + [depot]
    full_path = []
    for a, b in zip(leg_nodes[:-1], leg_nodes[1:]):
        _, prev = dijkstra(G, a, targets={b})
        path = reconstruct_path(prev, a, b)
        if full_path and path:
            path = path[1:]  # avoid duplicating the shared junction node
        full_path.extend(path)
    return [node_latlon(G, n) for n in full_path]


def edge_latlon_pair(u, v, G: nx.DiGraph) -> tuple:
    return node_latlon(G, u), node_latlon(G, v)
