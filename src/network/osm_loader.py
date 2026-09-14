"""
Real-world road network loader — Koramangala, Bengaluru (default), via OSMnx.

Data source: OpenStreetMap (openstreetmap.org), community-contributed map data,
fetched through the Overpass API using the `osmnx` Python library. This gives you
a REAL road topology (actual intersections, actual streets, actual one-way/two-way
structure) — but NOT real-time traffic. Congestion is simulated on top of this real
graph by `traffic_simulator.py`. Be explicit about this split when presenting:
"real road network, simulated traffic dynamics."

Run this file directly once to download and cache the graph locally — after the
first run it loads from `data/` instead of hitting the API again.
"""

from pathlib import Path
import networkx as nx
import osmnx as ox

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_PLACE = "Koramangala, Bengaluru, India"

# Koramangala doesn't have a registered administrative boundary polygon in
# Nominatim, so graph_from_place() fails. Fetching by center point + radius
# sidesteps that entirely and gives predictable graph size — this is the
# more reliable approach for a demo area regardless of whether the place
# name resolves to a polygon.
DEFAULT_POINT = (12.9352, 77.6245)  # Koramangala 4th Block/Sony Signal area
DEFAULT_DIST_M = 1500               # radius in meters -> roughly a 3km-wide demo area


def load_osm_graph(place: str = DEFAULT_PLACE, force_refresh: bool = False,
                    point: tuple = DEFAULT_POINT, dist_m: int = DEFAULT_DIST_M) -> nx.DiGraph:
    """
    Load a drivable road network centered on `point` within `dist_m` meters.
    Caches to data/<slug>.graphml so repeated runs don't re-hit the API.

    `place` is kept only for the cache filename/logging — the actual fetch uses
    point+radius (see module docstring for why).
    """
    slug = place.lower().replace(",", "").replace(" ", "_")
    cache_path = DATA_DIR / f"{slug}.graphml"

    if cache_path.exists() and not force_refresh:
        print(f"Loading cached graph from {cache_path}")
        G = ox.load_graphml(cache_path)
    else:
        print(f"Fetching area around {point} (radius {dist_m}m) from OpenStreetMap via Overpass API...")
        G = ox.graph_from_point(point, dist=dist_m, network_type="drive")
        # Add real speed limits (falls back to sensible defaults where OSM has none)
        # and computes travel times from length + speed.
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, cache_path)
        print(f"Cached to {cache_path}")

    # osmnx returns a MultiDiGraph; collapse to DiGraph (keep the shortest parallel
    # edge where duplicates exist) so it matches the interface of synthetic.py
    G_simple = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        length = data.get("length", 1.0)
        if G_simple.has_edge(u, v):
            if length >= G_simple[u][v]["length_m"]:
                continue
        speed_kph = data.get("speed_kph", 30.0)
        G_simple.add_edge(
            u, v,
            length_m=float(length),
            speed_kph=float(speed_kph),
            congestion=0.0,  # baseline congestion set by TrafficSimulator, not here
        )
    for n, data in G.nodes(data=True):
        G_simple.add_node(n, x=data.get("x", 0.0), y=data.get("y", 0.0))

    return G_simple


if __name__ == "__main__":
    G = load_osm_graph()
    print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
