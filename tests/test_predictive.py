"""Day 9 smoke test — run with: python -m pytest tests/test_predictive.py -v -s"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dynamic.predictor import HoltForecaster
from src.dynamic.predictive_rerouting import run_ab_comparison, route_uses_edge
from src.network.synthetic import make_synthetic_graph
from src.solvers.instance import generate_instance
from src.solvers.qpso_vrp import solve_qpso_adaptive


def test_holt_forecaster_picks_up_rising_trend():
    f = HoltForecaster(alpha=0.5, beta=0.3)
    for v in [0.1, 0.15, 0.22, 0.30, 0.38, 0.47]:
        f.update(v)
    assert f.trend > 0, "forecaster should detect a positive trend from a rising series"
    assert f.forecast(steps_ahead=3) > f.level, "forecast ahead should exceed current level given positive trend"


def test_holt_forecaster_flat_series_no_false_trigger():
    f = HoltForecaster(alpha=0.5, beta=0.3)
    for _ in range(10):
        f.update(0.2)  # constant congestion, no real trend
    assert abs(f.trend) < 0.01
    assert not f.is_rising_toward(threshold=0.6, within_steps=3)


def test_holt_forecaster_triggers_before_threshold_crossed():
    """The core mechanism: is_rising_toward should fire while the CURRENT
    value is still below threshold, given a clear rising trend."""
    f = HoltForecaster(alpha=0.6, beta=0.4)
    values = [0.1, 0.18, 0.27, 0.37, 0.48]  # steadily rising, still all < 0.6
    triggered_early = False
    for v in values:
        f.update(v)
        if f.level < 0.6 and f.is_rising_toward(threshold=0.6, within_steps=3):
            triggered_early = True
    assert triggered_early, "predictor should flag the rising trend before actual congestion hits threshold"


def test_route_uses_edge_detects_real_path_membership():
    G = make_synthetic_graph(n_nodes=60, seed=3)
    inst = generate_instance(G, n_customers=8, num_vehicles=2, vehicle_capacity=40, seed=3)
    routes, _, _ = solve_qpso_adaptive(inst, swarm_size=30, iterations=50, seed=3)

    from src.solvers.shortest_path import dijkstra, reconstruct_path
    # Grab a real edge from the actual path of the first non-empty route's first leg
    route = next(r for r in routes if r)
    _, prev = dijkstra(G, inst.depot, targets={route[0]})
    path = reconstruct_path(prev, inst.depot, route[0])
    real_edge = (path[0], path[1])

    assert route_uses_edge(routes, inst, G, real_edge)
    assert not route_uses_edge(routes, inst, G, (999999, 999998))  # nonexistent edge


def test_predictive_never_worse_than_reactive():
    """The actual Phase 9 deliverable claim, checked across multiple seeds:
    predictive should reroute at or before reactive, with equal-or-lower
    congestion exposure — never strictly worse."""
    for seed in [1, 2, 3]:
        results, edge = run_ab_comparison(n_nodes=200, n_customers=12, seed=seed,
                                           ramp_steps=20, threshold=0.6,
                                           forecast_horizon=3, verbose=False)
        reactive = results["reactive"]
        predictive = results["predictive"]

        if reactive["rerouted_at"] is not None and predictive["rerouted_at"] is not None:
            assert predictive["rerouted_at"] <= reactive["rerouted_at"], (
                f"seed {seed}: predictive triggered later than reactive "
                f"({predictive['rerouted_at']} vs {reactive['rerouted_at']})"
            )
        assert predictive["exposure_steps"] <= reactive["exposure_steps"], (
            f"seed {seed}: predictive exposure ({predictive['exposure_steps']}) "
            f"exceeded reactive ({reactive['exposure_steps']})"
        )
        print(f"seed {seed}: reactive triggered@{reactive['rerouted_at']} "
              f"exposure={reactive['exposure_steps']} | "
              f"predictive triggered@{predictive['rerouted_at']} "
              f"exposure={predictive['exposure_steps']}")


if __name__ == "__main__":
    test_holt_forecaster_picks_up_rising_trend()
    test_holt_forecaster_flat_series_no_false_trigger()
    test_holt_forecaster_triggers_before_threshold_crossed()
    test_route_uses_edge_detects_real_path_membership()
    test_predictive_never_worse_than_reactive()
    print("\nAll Day 9 smoke tests passed.")
