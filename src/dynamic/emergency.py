"""
Emergency vehicle yield behavior — Phase 11, Idea B from the original problem
statement. When an emergency vehicle is routed, other vehicles' subsequent
re-optimization should be nudged away from the roads the emergency vehicle is
using, modeling real-world yielding without needing full multi-agent
simultaneous optimization (out of scope for the prototype, noted as future
work in docs/architecture.md).

Mechanism: temporarily bump congestion on the emergency route's real
(Dijkstra-reconstructed) path. Since a later re-optimization of the normal
fleet reads live congestion off the same graph, this makes those roads look
less attractive without hard-blocking them — a soft, congestion-based
yield rather than a hard constraint.
"""

import networkx as nx

from src.solvers.instance import VRPInstance
from src.solvers.shortest_path import dijkstra, reconstruct_path


def apply_yield_penalty(G: nx.DiGraph, emergency_routes: list, inst: VRPInstance,
                         bump: float = 0.3) -> list:
    """Raises congestion by `bump` on every edge the emergency vehicle's real
    path traverses. Returns the affected edges so the caller can clear the
    penalty later (clear_yield_penalty) once the emergency has passed."""
    affected_edges = []
    for route in emergency_routes:
        if not route:
            continue
        leg_nodes = [inst.depot] + route + [inst.depot]
        for a, b in zip(leg_nodes[:-1], leg_nodes[1:]):
            _, prev = dijkstra(G, a, targets={b})
            path = reconstruct_path(prev, a, b)
            for p1, p2 in zip(path[:-1], path[1:]):
                current = G[p1][p2].get("congestion", 0.0)
                G[p1][p2]["congestion"] = float(min(current + bump, 0.9))
                affected_edges.append((p1, p2))
    return affected_edges


def clear_yield_penalty(G: nx.DiGraph, affected_edges: list, bump: float = 0.3):
    """Reverts the yield bump once the emergency vehicle has passed."""
    for p1, p2 in affected_edges:
        current = G[p1][p2]["congestion"]
        G[p1][p2]["congestion"] = float(max(current - bump, 0.0))
