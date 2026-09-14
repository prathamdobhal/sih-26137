"""Day 4 smoke test — run with: python -m pytest tests/test_solvers.py -v"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from src.network.synthetic import make_synthetic_graph
from src.solvers.shortest_path import dijkstra, reconstruct_path, edge_travel_time, build_distance_matrix
from src.solvers.instance import generate_instance, instance_summary
from src.solvers.ga_vrp import solve_ga, split_into_routes
from src.solvers.evaluate import route_set_cost


def test_dijkstra_matches_networkx():
    """Cross-validate our hand-rolled Dijkstra against networkx's implementation."""
    G = make_synthetic_graph(n_nodes=50, seed=1)
    source = list(G.nodes())[0]

    our_dist, _ = dijkstra(G, source)
    nx_dist = nx.single_source_dijkstra_path_length(
        G, source, weight=lambda u, v, d: edge_travel_time(d)
    )

    assert set(our_dist.keys()) == set(nx_dist.keys())
    for node in our_dist:
        assert abs(our_dist[node] - nx_dist[node]) < 1e-6


def test_reconstruct_path_valid():
    G = make_synthetic_graph(n_nodes=50, seed=1)
    nodes = list(G.nodes())
    source, target = nodes[0], nodes[10]
    dist, prev = dijkstra(G, source, targets={target})
    path = reconstruct_path(prev, source, target)
    assert path[0] == source
    assert path[-1] == target
    # every consecutive pair must be a real edge
    for u, v in zip(path[:-1], path[1:]):
        assert G.has_edge(u, v)


def test_distance_matrix_symmetric_diagonal_zero():
    G = make_synthetic_graph(n_nodes=50, seed=2)
    nodes = list(G.nodes())[:10]
    matrix = build_distance_matrix(G, nodes)
    assert matrix.shape == (10, 10)
    for i in range(10):
        assert matrix[i, i] == 0.0


def test_generate_instance_feasible_by_capacity():
    G = make_synthetic_graph(n_nodes=100, seed=3)
    inst = generate_instance(G, n_customers=12, num_vehicles=4, vehicle_capacity=40, seed=3)
    summary = instance_summary(inst)
    assert summary["n_customers"] == 12
    assert len(inst.distance_matrix) == 13  # depot + 12 customers


def test_ga_solves_and_visits_all_customers():
    G = make_synthetic_graph(n_nodes=100, seed=4)
    inst = generate_instance(G, n_customers=10, num_vehicles=3, vehicle_capacity=40, seed=4)
    routes, best_cost, history = solve_ga(inst, pop_size=30, generations=40, seed=4)

    result = route_set_cost(routes, inst)
    assert result["feasible"], f"GA solution infeasible: {result}"
    assert result["missing_customers"] == set()
    assert best_cost < float("inf")


def test_ga_converges_improves_over_generations():
    G = make_synthetic_graph(n_nodes=100, seed=5)
    inst = generate_instance(G, n_customers=10, num_vehicles=3, vehicle_capacity=40, seed=5)
    _, best_cost, history = solve_ga(inst, pop_size=30, generations=40, seed=5)
    # convergence history should be non-increasing (we always track the running best)
    assert all(history[i] >= history[i + 1] for i in range(len(history) - 1))
    assert history[-1] <= history[0]


if __name__ == "__main__":
    test_dijkstra_matches_networkx()
    test_reconstruct_path_valid()
    test_distance_matrix_symmetric_diagonal_zero()
    test_generate_instance_feasible_by_capacity()
    test_ga_solves_and_visits_all_customers()
    test_ga_converges_improves_over_generations()
    print("All Day 4 smoke tests passed.")
