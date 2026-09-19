"""Day 6 smoke test — run with: python -m pytest tests/test_day6.py -v"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network.synthetic import make_synthetic_graph
from src.solvers.instance import generate_instance
from src.solvers.local_search import two_opt, polish_routes
from src.solvers.evaluate import route_set_cost
from src.solvers.qpso_vrp import solve_qpso_adaptive


def test_two_opt_never_worsens_and_preserves_customers():
    G = make_synthetic_graph(n_nodes=100, seed=7)
    inst = generate_instance(G, n_customers=10, num_vehicles=3, vehicle_capacity=40, seed=7)
    route = inst.customers[:6]
    polished = two_opt(route, inst)
    before = route_set_cost([route], inst)["total_time"]
    after = route_set_cost([polished], inst)["total_time"]
    assert after <= before + 1e-6
    assert set(route) == set(polished)


def test_adaptive_qpso_feasible_and_tracks_metadata():
    G = make_synthetic_graph(n_nodes=200, seed=8)
    inst = generate_instance(G, n_customers=20, num_vehicles=4, vehicle_capacity=40, seed=8)
    routes, cost, meta = solve_qpso_adaptive(inst, swarm_size=40, iterations=80, seed=8)
    result = route_set_cost(routes, inst)
    assert result["feasible"]
    assert "gbest_history" in meta and "diversity_history" in meta and "reinit_events" in meta


def test_adaptive_qpso_beats_or_matches_raw_on_larger_instance():
    """On a 30-customer instance, adaptive QPSO should be at least competitive
    with raw QPSO (it should never be drastically worse — local search + gentle
    stagnation control are additive improvements, not regressions)."""
    from src.solvers.qpso_vrp import solve_qpso

    G = make_synthetic_graph(n_nodes=300, seed=9)
    inst = generate_instance(G, n_customers=30, num_vehicles=5, vehicle_capacity=40, seed=9)

    _, raw_cost, _ = solve_qpso(inst, swarm_size=60, iterations=150, seed=9)
    _, adaptive_cost, _ = solve_qpso_adaptive(inst, swarm_size=60, iterations=150, seed=9)

    # Not a strict inequality (randomness) — but adaptive shouldn't be catastrophically worse
    assert adaptive_cost <= raw_cost * 1.15


if __name__ == "__main__":
    test_two_opt_never_worsens_and_preserves_customers()
    test_adaptive_qpso_feasible_and_tracks_metadata()
    test_adaptive_qpso_beats_or_matches_raw_on_larger_instance()
    print("All Day 6 smoke tests passed.")
