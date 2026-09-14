"""
Synthetic graph generator — used for fast local testing and scalability sweeps
(50 / 200 / 1000 nodes) without hitting any external API.

This is deliberately separate from the OSM loader (osm_loader.py): the synthetic
generator is for algorithm development and the Phase 7/13 scalability benchmarks;
the OSM loader is for the real-world demo graph.
"""

import networkx as nx
import numpy as np


def make_synthetic_graph(n_nodes: int = 50, seed: int = 42,
                          avg_degree: int = 4) -> nx.DiGraph:
    """
    Build a random, connected, directed graph with realistic-ish edge attributes,
    for use as a drop-in replacement for an OSM graph during development.

    Each edge gets:
      - length_m      : straight-line-ish distance in meters (from random 2D layout)
      - speed_kph      : a random speed limit typical of urban roads
      - congestion     : baseline congestion in [0, 0.3] (traffic simulator raises this)

    Returns a directed graph so it has the same interface as an osmnx graph
    (osmnx graphs are always MultiDiGraph).
    """
    rng = np.random.default_rng(seed)

    # Random geometric graph gives a locally-connected, road-network-like topology
    # (nearby nodes connect, unlike Erdos-Renyi which connects uniformly at random)
    radius = np.sqrt(avg_degree / (np.pi * n_nodes))
    G_undirected = nx.random_geometric_graph(n_nodes, radius, seed=seed)

    # Ensure connectivity — random geometric graphs can leave isolated components
    if not nx.is_connected(G_undirected):
        components = list(nx.connected_components(G_undirected))
        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
            G_undirected.add_edge(u, v)

    G = nx.DiGraph()
    pos = nx.get_node_attributes(G_undirected, "pos")
    for node, (x, y) in pos.items():
        # Scale unit-square layout to a ~5km x 5km area for realistic distances
        G.add_node(node, x=x * 5000, y=y * 5000)

    for u, v in G_undirected.edges():
        dist = float(np.hypot(
            G.nodes[u]["x"] - G.nodes[v]["x"],
            G.nodes[u]["y"] - G.nodes[v]["y"],
        ))
        speed = float(rng.choice([30, 40, 50, 60]))  # kph, typical urban speed tiers
        base_congestion = float(rng.uniform(0.0, 0.3))
        # Add both directions — synthetic roads are two-way by default
        for a, b in [(u, v), (v, u)]:
            G.add_edge(a, b, length_m=dist, speed_kph=speed,
                       congestion=base_congestion)

    return G


def graph_summary(G: nx.DiGraph) -> dict:
    """Quick sanity-check stats — use this after generating or loading any graph."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "is_connected": nx.is_weakly_connected(G),
        "avg_out_degree": sum(dict(G.out_degree()).values()) / G.number_of_nodes(),
    }


if __name__ == "__main__":
    for n in (50, 200, 1000):
        G = make_synthetic_graph(n)
        print(n, graph_summary(G))
