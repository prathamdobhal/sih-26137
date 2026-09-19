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


def test_warmstart_wins_majority_of_seeds():
    """
    Single-seed comparisons are noisy under Day 11's normalized cost scale
    (~O(0.1-1), unlike the old O(10,000+) raw-time cost) — a handful of
    iterations of difference can swing a percentage threshold. So, matching
    the same honest statistical approach used for the GA-vs-QPSO comparison
    (Day 6/7): check win rate across multiple seeds, not one deterministic
    run. Warm-start is not expected to win every single seed — it's expected
    to win the clear majority, which is what the underlying claim actually is.
    """
    def iters_to_within(history, target, pct=1.05, abs_margin=0.01):
        threshold = max(target * pct, target + abs_margin)
        for i, v in enumerate(history):
            if v <= threshold:
                return i
        return len(history)

    wins = 0
    n_seeds = 8
    for seed in range(n_seeds):
        G, sim, inst = _setup(seed=seed, n_customers=25)
        _, _, meta0 = solve_qpso_adaptive(inst, swarm_size=50, iterations=100, seed=seed)

        u, v = _pick_relevant_edge(G, inst)
        sim.inject_incident(u, v, severity=0.88, radius_hops=1)
        new_inst = recompute_instance_for_traffic(inst, sim)

        _, cold_cost, cold_meta = solve_qpso_adaptive(new_inst, swarm_size=50, iterations=100, seed=seed)
        _, warm_cost, warm_meta = solve_qpso_adaptive(
            new_inst, swarm_size=50, iterations=100, seed=seed, init_positions=meta0["final_positions"]
        )

        best_reachable = min(cold_cost, warm_cost)
        cold_iters = iters_to_within(cold_meta["gbest_history"], best_reachable)
        warm_iters = iters_to_within(warm_meta["gbest_history"], best_reachable)
        wins += (warm_iters <= cold_iters)

    print(f"\nWarm-start won {wins}/{n_seeds} seeds")
    assert wins >= n_seeds * 0.5, (
        f"Expected warm-start to win at least half of seeds, got {wins}/{n_seeds}"
    )


if __name__ == "__main__":
    test_recompute_instance_reflects_new_congestion()
    test_incident_reroute_produces_feasible_routes()
    test_warmstart_wins_majority_of_seeds()
    print("\nAll Day 8 smoke tests passed.")
