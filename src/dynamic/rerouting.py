"""
Dynamic rerouting — this is the "Inject Incident" demo button's backend logic.

Flow: incident spikes congestion on the live graph (TrafficSimulator, Day 3) ->
distance matrix is recomputed via real Dijkstra on the updated graph (Day 4) ->
QPSO re-optimizes WARM-STARTED from its previous swarm position (Day 8), not
from random initialization — matching the "incremental re-optimization, not
brute-force re-solving" claim in docs/architecture.md.
"""

from dataclasses import replace
import time

from src.network.traffic_simulator import TrafficSimulator
from src.solvers.instance import VRPInstance
from src.solvers.shortest_path import build_distance_matrix
from src.solvers.qpso_vrp import solve_qpso_adaptive


def recompute_instance_for_traffic(inst: VRPInstance, sim: TrafficSimulator) -> VRPInstance:
    """
    After congestion changes on sim.G, the distance matrix (which encodes
    travel times, not raw distances) is stale. Recompute it via Dijkstra on
    the current graph state and return a NEW VRPInstance (dataclasses.replace
    keeps depot/customers/demand identical — only the matrix changes).
    """
    new_matrix = build_distance_matrix(sim.G, inst.nodes)
    return replace(inst, distance_matrix=new_matrix)


def reoptimize_on_incident(inst: VRPInstance, sim: TrafficSimulator,
                            u, v, prev_final_positions,
                            severity: float = 0.85, radius_hops: int = 1,
                            iterations: int = 60, seed: int = 42):
    """
    The full "inject incident -> reroute" pipeline. Returns everything the
    dashboard needs: the new routes, cost, affected edges to highlight on the
    map, and timing (for the live demo to show "rerouted in X ms").
    """
    affected_edges = sim.inject_incident(u, v, severity=severity, radius_hops=radius_hops)
    new_inst = recompute_instance_for_traffic(inst, sim)

    t0 = time.time()
    routes, cost, meta = solve_qpso_adaptive(
        new_inst, iterations=iterations, seed=seed, init_positions=prev_final_positions
    )
    elapsed = time.time() - t0

    return {
        "routes": routes,
        "cost": cost,
        "meta": meta,
        "new_instance": new_inst,
        "affected_edges": affected_edges,
        "reoptimize_time_s": elapsed,
    }
