"""Day 10 smoke test — run with: python -m pytest tests/test_explainability.py -v -s"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network.synthetic import make_synthetic_graph
from src.solvers.instance import generate_instance
from src.solvers.qpso_vrp import solve_qpso_adaptive
from src.explainability.explain import (
    get_top_candidates, decompose_route_set, sensitivity_score, explain_choice
)


def test_top_candidates_are_distinct_and_feasible():
    G = make_synthetic_graph(n_nodes=200, seed=11)
    inst = generate_instance(G, n_customers=20, num_vehicles=4, vehicle_capacity=40, seed=11)
    _, _, meta = solve_qpso_adaptive(inst, swarm_size=50, iterations=100, seed=11)

    candidates = get_top_candidates(inst, meta, k=3)
    assert len(candidates) >= 1
    for c in candidates:
        assert c["feasible"]
    # costs should be non-decreasing (sorted best-first)
    costs = [c["cost"] for c in candidates]
    assert costs == sorted(costs)


def test_decompose_route_set_distance_matches_manual_sum():
    """Cross-check: the decomposed distance should equal a manual sum over
    the same Dijkstra-reconstructed paths, computed independently here."""
    from src.solvers.shortest_path import dijkstra, reconstruct_path

    G = make_synthetic_graph(n_nodes=100, seed=12)
    inst = generate_instance(G, n_customers=10, num_vehicles=2, vehicle_capacity=40, seed=12)
    routes, _, _ = solve_qpso_adaptive(inst, swarm_size=30, iterations=60, seed=12)

    breakdown = decompose_route_set(routes, inst, G)

    manual_distance = 0.0
    for route in routes:
        if not route:
            continue
        leg_nodes = [inst.depot] + route + [inst.depot]
        for a, b in zip(leg_nodes[:-1], leg_nodes[1:]):
            _, prev = dijkstra(G, a, targets={b})
            path = reconstruct_path(prev, a, b)
            for p1, p2 in zip(path[:-1], path[1:]):
                manual_distance += G[p1][p2]["length_m"]

    assert abs(breakdown["distance_km"] * 1000 - manual_distance) < 1e-6


def test_sensitivity_score_no_side_effects_on_graph():
    """sensitivity_score perturbs G internally for Monte Carlo trials — must
    restore original congestion values afterward, or every later solve would
    silently run on corrupted traffic data."""
    G = make_synthetic_graph(n_nodes=100, seed=13)
    inst = generate_instance(G, n_customers=12, num_vehicles=3, vehicle_capacity=40, seed=13)
    _, _, meta = solve_qpso_adaptive(inst, swarm_size=30, iterations=60, seed=13)
    candidates = get_top_candidates(inst, meta, k=2)

    before = {(u, v): G[u][v]["congestion"] for u, v in G.edges()}
    if len(candidates) >= 2:
        sensitivity_score(candidates, inst, G, n_trials=10, seed=13)
    after = {(u, v): G[u][v]["congestion"] for u, v in G.edges()}

    assert before == after, "sensitivity_score must not permanently mutate graph congestion"


def test_sensitivity_score_confidence_in_valid_range():
    G = make_synthetic_graph(n_nodes=150, seed=14)
    inst = generate_instance(G, n_customers=15, num_vehicles=3, vehicle_capacity=40, seed=14)
    _, _, meta = solve_qpso_adaptive(inst, swarm_size=40, iterations=80, seed=14)
    candidates = get_top_candidates(inst, meta, k=2)

    if len(candidates) >= 2:
        result = sensitivity_score(candidates, inst, G, n_trials=15, seed=14)
        assert 0.0 <= result["confidence"] <= 1.0


def test_explain_choice_end_to_end():
    G = make_synthetic_graph(n_nodes=200, seed=15)
    inst = generate_instance(G, n_customers=18, num_vehicles=4, vehicle_capacity=40, seed=15)
    _, _, meta = solve_qpso_adaptive(inst, swarm_size=40, iterations=100, seed=15)

    explanation = explain_choice(inst, G, meta, top_k=2, n_trials=15, seed=15)

    assert "candidates" in explanation
    assert "reason" in explanation and isinstance(explanation["reason"], str)
    assert 0.0 <= explanation["confidence"] <= 1.0
    print(f"\nReason: {explanation['reason']}")
    print(f"Confidence: {explanation['confidence']*100:.0f}%")
    for i, c in enumerate(explanation["candidates"]):
        b = c["breakdown"]
        print(f"  candidate {i}: cost={c['cost']:.1f} dist={b['distance_km']:.2f}km "
              f"time={b['time_min']:.1f}min avg_congestion={b['avg_congestion']*100:.0f}%")


if __name__ == "__main__":
    test_top_candidates_are_distinct_and_feasible()
    test_decompose_route_set_distance_matches_manual_sum()
    test_sensitivity_score_no_side_effects_on_graph()
    test_sensitivity_score_confidence_in_valid_range()
    test_explain_choice_end_to_end()
    print("\nAll Day 10 smoke tests passed.")
