"""Day 8 smoke test — run with: python -m pytest tests/test_dynamic.py -v -s"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network.synthetic import make_synthetic_graph
from src.network.traffic_simulator import TrafficSimulator
from src.solvers.instance import generate_instance
from src.solvers.qpso_vrp import solve_qpso_adaptive
from src.solvers.evaluate import route_set_cost
from src.dynamic.rerouting import reoptimize_on_incident, recompute_instance_for_traffic


def _setup(seed=42, n_nodes=200, n_customers=20):
    G = make_synthetic_graph(n_nodes=n_nodes, seed=seed)
    sim = TrafficSimulator(G, seed=seed)
    inst = generate_instance(G, n_customers=n_customers, num_vehicles=4,
                              vehicle_capacity=40, seed=seed)
    return G, sim, inst


def _pick_relevant_edge(G, inst, customer_idx=0):
    """Picks an edge guaranteed to lie on a real shortest path to a customer,
    so incident tests are deterministic rather than relying on a random edge
    happening to matter."""
    from src.solvers.shortest_path import dijkstra, reconstruct_path
    target = inst.customers[customer_idx]
    _, prev = dijkstra(G, inst.depot, targets={target})
    path = reconstruct_path(prev, inst.depot, target)
    return path[0], path[1]


def test_recompute_instance_reflects_new_congestion():
    G, sim, inst = _setup()
    u, v = _pick_relevant_edge(G, inst)
    sim.inject_incident(u, v, severity=0.9)
    new_inst = recompute_instance_for_traffic(inst, sim)
    assert not (new_inst.distance_matrix == inst.distance_matrix).all()


def test_incident_reroute_produces_feasible_routes():
    G, sim, inst = _setup()
    _, _, meta0 = solve_qpso_adaptive(inst, swarm_size=40, iterations=80, seed=1)

    u, v = _pick_relevant_edge(G, inst)
    result = reoptimize_on_incident(inst, sim, u, v, meta0["final_positions"],
                                     iterations=60, seed=1)

    cost_check = route_set_cost(result["routes"], result["new_instance"])
    assert cost_check["feasible"]
    assert len(result["affected_edges"]) >= 1
    assert result["reoptimize_time_s"] < 30  # sanity bound, not a strict perf test


def test_warmstart_faster_than_coldstart_to_reach_same_quality():
    """
    THE key claim to validate: after an incident, warm-starting from the
    previous swarm's positions should reach a good solution in fewer
    iterations than restarting from random positions on the new landscape.
    """
    G, sim, inst = _setup(seed=7, n_customers=25)

    # Establish a good initial solution (this is the "before the incident" state)
    _, _, meta0 = solve_qpso_adaptive(inst, swarm_size=50, iterations=100, seed=7)

    u, v = _pick_relevant_edge(G, inst)
    sim.inject_incident(u, v, severity=0.88, radius_hops=1)
    new_inst = recompute_instance_for_traffic(inst, sim)

    # Cold start: random positions on the NEW (post-incident) landscape
    _, cold_cost, cold_meta = solve_qpso_adaptive(new_inst, swarm_size=50, iterations=100, seed=7)

    # Warm start: same iteration budget, but starting from the pre-incident swarm
    _, warm_cost, warm_meta = solve_qpso_adaptive(
        new_inst, swarm_size=50, iterations=100, seed=7,
        init_positions=meta0["final_positions"]
    )

    def iters_to_within(history, target, pct=1.05):
        threshold = target * pct
        for i, v in enumerate(history):
            if v <= threshold:
                return i
        return len(history)

    best_reachable = min(cold_cost, warm_cost)
    cold_iters = iters_to_within(cold_meta["gbest_history"], best_reachable)
    warm_iters = iters_to_within(warm_meta["gbest_history"], best_reachable)

    print(f"\nCold-start iterations to within 5% of best: {cold_iters}")
    print(f"Warm-start iterations to within 5% of best: {warm_iters}")
    print(f"Cold-start final cost: {cold_cost:.1f}  Warm-start final cost: {warm_cost:.1f}")

    assert warm_iters <= cold_iters, (
        f"Expected warm-start to reach good quality in fewer or equal iterations, "
        f"got warm={warm_iters} vs cold={cold_iters}"
    )


if __name__ == "__main__":
    test_recompute_instance_reflects_new_congestion()
    test_incident_reroute_produces_feasible_routes()
    test_warmstart_faster_than_coldstart_to_reach_same_quality()
    print("\nAll Day 8 smoke tests passed.")
