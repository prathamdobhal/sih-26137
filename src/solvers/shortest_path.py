"""
Dijkstra shortest-path solver + distance-matrix builder.

This is a hand-implemented Dijkstra (not just a call to nx.shortest_path) for two
reasons: (1) the problem statement asks for exact-method benchmarking, and a real
implementation is far more defensible under judge questioning than a library call,
and (2) every other solver (GA, QPSO) needs a travel-time matrix between customer
nodes, and this is where that matrix comes from.

Edge weight = the exact travel_time formula from docs/formulation.md Section 1:
    travel_time_ij(t) = distance_ij / (speed_limit_ij * (1 - congestion_ij(t)))
"""

import heapq
import networkx as nx
import numpy as np


def edge_travel_time(data: dict) -> float:
    """Matches TrafficSimulator.travel_time() exactly — single source of truth
    for the formula, duplicated here only because Dijkstra needs a pure
    edge-dict -> float function rather than a (u, v) graph lookup."""
    length_m = data["length_m"]
    speed_kph = max(data["speed_kph"], 1.0)
    speed_mps = speed_kph * 1000 / 3600
    congestion = min(data.get("congestion", 0.0), 0.9)
    return length_m / (speed_mps * (1 - congestion))


def dijkstra(G: nx.DiGraph, source, targets: set = None):
    """
    Custom Dijkstra's algorithm. Returns (dist, prev) where:
      - dist[node] = shortest travel time from source to node
      - prev[node] = predecessor of node on the shortest path (for reconstruction)

    If `targets` is given, stops early once all targets are settled (optimization
    for building a distance matrix over a small customer set in a large graph).
    """
    dist = {source: 0.0}
    prev = {}
    visited = set()
    pq = [(0.0, source)]
    remaining_targets = set(targets) if targets else None

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if remaining_targets is not None:
            remaining_targets.discard(u)
            if not remaining_targets:
                break

        for v, data in G[u].items():
            w = edge_travel_time(data)
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(prev: dict, source, target) -> list:
    """Walk predecessor pointers from target back to source."""
    if target == source:
        return [source]
    if target not in prev:
        return []  # unreachable
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    return path[::-1]


def build_distance_matrix(G: nx.DiGraph, nodes: list) -> np.ndarray:
    """
    Builds an NxN travel-time matrix between `nodes` (typically depot + customers).
    Runs one Dijkstra per node, targeted at the rest of `nodes` — cheap even on a
    2000+ node city graph since we only need ~N Dijkstra runs, not N^2.
    """
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    matrix = np.full((n, n), np.inf)
    np.fill_diagonal(matrix, 0.0)

    for i, src in enumerate(nodes):
        targets = set(nodes) - {src}
        dist, _ = dijkstra(G, src, targets=targets)
        for node, d in dist.items():
            if node in idx:
                matrix[i, idx[node]] = d

    return matrix


def edge_emissions(data: dict, base_rate: float = 120.0) -> float:
    """
    Emissions estimate for a single edge, in grams CO2 (order-of-magnitude
    demo figure, not a sensor-calibrated value — matches docs/formulation.md
    Section 5's stated design: no real sensor data required).

    speed_factor is a simple U-shaped curve: low speed (stop-start idling)
    and very high speed (inefficiency) both cost more than a mid-range
    cruising speed — a standard qualitative shape in traffic-emissions
    literature (MOVES/COPERT-style speed-emission curves), simplified to a
    3-tier piecewise constant for the prototype.
    """
    speed_kph = data.get("speed_kph", 30.0)
    if speed_kph < 20:
        speed_factor = 1.5
    elif speed_kph <= 60:
        speed_factor = 1.0
    else:
        speed_factor = 1.3
    distance_km = data["length_m"] / 1000
    return base_rate * distance_km * speed_factor


def build_full_metrics_matrices(G: nx.DiGraph, nodes: list) -> dict:
    """
    Builds FOUR NxN matrices in one pass (reusing the same Dijkstra runs
    build_distance_matrix already does, at no extra Dijkstra cost):
      - time      : travel time in seconds (same as build_distance_matrix)
      - distance  : real road distance in meters
      - congestion: congestion EXPOSURE (congestion * time_on_edge), summed along the path
      - emissions : grams CO2 (distance/speed term only — idle-per-stop is
                    added separately in evaluate.py, since it's a per-customer-visit
                    cost, not a per-edge one)

    These back the full multi-objective F(R) = a.T + b.D + g.C + d.E from
    docs/formulation.md, and the mode weight presets (Fastest/Eco/Balanced/
    Emergency) that select between them.
    """
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    time_m = np.full((n, n), np.inf)
    dist_m = np.zeros((n, n))
    cong_m = np.zeros((n, n))
    emis_m = np.zeros((n, n))
    np.fill_diagonal(time_m, 0.0)

    for i, src in enumerate(nodes):
        targets = set(nodes) - {src}
        _, prev = dijkstra(G, src, targets=targets)
        for target in targets:
            if target not in prev and target != src:
                continue  # unreachable, leave as inf/0
            path = reconstruct_path(prev, src, target)
            if not path:
                continue
            j = idx[target]
            t_total = d_total = c_total = e_total = 0.0
            for p1, p2 in zip(path[:-1], path[1:]):
                edge = G[p1][p2]
                t = edge_travel_time(edge)
                t_total += t
                d_total += edge["length_m"]
                c_total += min(edge.get("congestion", 0.0), 0.9) * t
                e_total += edge_emissions(edge)
            time_m[i, j] = t_total
            dist_m[i, j] = d_total
            cong_m[i, j] = c_total
            emis_m[i, j] = e_total

    return {"time": time_m, "distance": dist_m, "congestion": cong_m, "emissions": emis_m}
