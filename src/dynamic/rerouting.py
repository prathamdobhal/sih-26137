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
from src.solvers.instance import VRPInstance, compute_term_scales
from src.solvers.shortest_path import build_full_metrics_matrices
from src.solvers.qpso_vrp import solve_qpso_adaptive


def recompute_instance_for_traffic(inst: VRPInstance, sim: TrafficSimulator) -> VRPInstance:
    """
    After congestion changes on sim.G, ALL FOUR matrices (time, distance,
    congestion, emissions) are stale, not just travel time — a congested
    edge changes both how long it takes AND how much congestion exposure a
    route through it accumulates. Recompute everything together via one
    Dijkstra pass (build_full_metrics_matrices) so time/congestion/emissions
    stay mutually consistent; recomputing only the time matrix here was a
    real bug caught during Day 11 testing (mode-weighted cost was scoring
    routes against stale congestion data after an incident).
    """
    matrices = build_full_metrics_matrices(sim.G, inst.nodes)
    term_scales = compute_term_scales(matrices)
    return replace(
        inst,
        distance_matrix=matrices["time"],
        raw_distance_matrix=matrices["distance"],
        congestion_matrix=matrices["congestion"],
        emissions_matrix=matrices["emissions"],
        term_scales=term_scales,
    )


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
