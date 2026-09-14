"""Day 3 smoke test — run with: python -m pytest tests/test_network.py -v"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network.synthetic import make_synthetic_graph, graph_summary
from src.network.traffic_simulator import TrafficSimulator


def test_synthetic_graph_connected():
    G = make_synthetic_graph(n_nodes=50, seed=1)
    summary = graph_summary(G)
    assert summary["is_connected"]
    assert summary["nodes"] == 50


def test_traffic_simulator_baseline():
    G = make_synthetic_graph(n_nodes=50, seed=1)
    sim = TrafficSimulator(G, seed=1)
    for u, v in G.edges():
        assert 0.0 <= G[u][v]["congestion"] <= 0.9


def test_travel_time_matches_formulation():
    G = make_synthetic_graph(n_nodes=20, seed=2)
    sim = TrafficSimulator(G, seed=2)
    u, v = next(iter(G.edges()))
    tt = sim.travel_time(u, v)
    assert tt > 0


def test_incident_injection_raises_congestion():
    G = make_synthetic_graph(n_nodes=50, seed=3)
    sim = TrafficSimulator(G, seed=3)
    u, v = next(iter(G.edges()))
    before = G[u][v]["congestion"]
    affected = sim.inject_incident(u, v, severity=0.9)
    after = G[u][v]["congestion"]
    assert after == 0.9
    assert after >= before
    assert (u, v) in affected


def test_clear_incident_restores_baseline():
    G = make_synthetic_graph(n_nodes=50, seed=4)
    sim = TrafficSimulator(G, seed=4)
    u, v = next(iter(G.edges()))
    baseline = G[u][v]["_base_congestion"]
    sim.inject_incident(u, v, severity=0.9)
    sim.clear_incident([(u, v)])
    assert G[u][v]["congestion"] == baseline


def test_step_evolves_congestion():
    G = make_synthetic_graph(n_nodes=50, seed=5)
    sim = TrafficSimulator(G, seed=5)
    snapshot_before = {(u, v): G[u][v]["congestion"] for u, v in G.edges()}
    sim.step()
    snapshot_after = {(u, v): G[u][v]["congestion"] for u, v in G.edges()}
    assert snapshot_before != snapshot_after


if __name__ == "__main__":
    test_synthetic_graph_connected()
    test_traffic_simulator_baseline()
    test_travel_time_matches_formulation()
    test_incident_injection_raises_congestion()
    test_clear_incident_restores_baseline()
    test_step_evolves_congestion()
    print("All Day 3 smoke tests passed.")
