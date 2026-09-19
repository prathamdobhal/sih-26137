"""Day 5 smoke test — run with: python -m pytest tests/test_qpso.py -v"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network.synthetic import make_synthetic_graph
from src.solvers.instance import generate_instance
from src.solvers.qpso_vrp import solve_qpso, decode_particle
from src.solvers.evaluate import route_set_cost


def test_qpso_produces_feasible_solution():
    G = make_synthetic_graph(n_nodes=100, seed=4)
    inst = generate_instance(G, n_customers=10, num_vehicles=3, vehicle_capacity=40, seed=4)
    routes, best_cost, history = solve_qpso(inst, swarm_size=30, iterations=60, seed=4)

    result = route_set_cost(routes, inst)
    assert result["feasible"], f"QPSO solution infeasible: {result}"
    assert result["missing_customers"] == set()


def test_qpso_gbest_never_worsens():
    """gbest is a running best by construction — history must be non-increasing."""
    G = make_synthetic_graph(n_nodes=100, seed=5)
    inst = generate_instance(G, n_customers=10, num_vehicles=3, vehicle_capacity=40, seed=5)
    _, _, history = solve_qpso(inst, swarm_size=30, iterations=60, seed=5)
    assert all(history[i] >= history[i + 1] for i in range(len(history) - 1))


def test_qpso_decode_matches_customer_set():
    G = make_synthetic_graph(n_nodes=100, seed=6)
    inst = generate_instance(G, n_customers=8, num_vehicles=2, vehicle_capacity=40, seed=6)
    import numpy as np
    rng = np.random.default_rng(6)
    pos = rng.random(len(inst.customers))
    routes = decode_particle(pos, inst)
    visited = {c for r in routes for c in r}
    assert visited == set(inst.customers)


def test_qpso_vs_ga_same_instance():
    """Not a strict pass/fail — prints a real comparison for the Phase 7 record."""
    from src.solvers.ga_vrp import solve_ga

    G = make_synthetic_graph(n_nodes=200, seed=10)
    inst = generate_instance(G, n_customers=15, num_vehicles=3, vehicle_capacity=40, seed=10)

    ga_routes, ga_cost, ga_history = solve_ga(inst, pop_size=60, generations=150, seed=10)
    qpso_routes, qpso_cost, qpso_history = solve_qpso(inst, swarm_size=60, iterations=150, seed=10)

    ga_result = route_set_cost(ga_routes, inst)
    qpso_result = route_set_cost(qpso_routes, inst)

    assert ga_result["feasible"]
    assert qpso_result["feasible"]

    print(f"\nGA best cost:   {ga_cost:.2f}")
    print(f"QPSO best cost: {qpso_cost:.2f}")
    print(f"QPSO improvement over GA: {100 * (ga_cost - qpso_cost) / ga_cost:.1f}%")


if __name__ == "__main__":
    test_qpso_produces_feasible_solution()
    test_qpso_gbest_never_worsens()
    test_qpso_decode_matches_customer_set()
    test_qpso_vs_ga_same_instance()
    print("All Day 5 smoke tests passed.")
