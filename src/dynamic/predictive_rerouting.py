"""
Predictive vs. reactive rerouting — A/B comparison harness.

Simulates a congestion "ramp" (a jam gradually forming on one edge, rather
than an instant incident) and runs two independent strategies against it:

  - REACTIVE: only re-optimizes once the edge's ACTUAL congestion crosses
    the threshold (this is what Day 8's incident injection already does).
  - PREDICTIVE: re-optimizes as soon as the Holt forecaster predicts the
    edge WILL cross the threshold within a few steps — before it actually
    does.

Metric: congestion exposure = number of simulated time steps during which
the vehicle's actual route still travels through the worsening edge, before
each strategy reroutes away from it. Predictive should show measurably lower
exposure, since it acts earlier — this is checked with real path
reconstruction (Dijkstra), not assumed.
"""

from dataclasses import replace

from src.network.synthetic import make_synthetic_graph
from src.solvers.instance import generate_instance, VRPInstance
from src.solvers.shortest_path import dijkstra, reconstruct_path, build_full_metrics_matrices
from src.solvers.instance import compute_term_scales
from src.solvers.qpso_vrp import solve_qpso_adaptive
from src.dynamic.predictor import HoltForecaster


def route_uses_edge(routes: list, inst: VRPInstance, G, edge: tuple) -> bool:
    """Reconstructs the REAL road-level path (via Dijkstra) for every leg of
    every route and checks whether `edge` appears anywhere in it. This is the
    honest check — customer-to-customer doesn't mean edge-to-edge, the actual
    road path between two stops can pass through many intermediate edges."""
    u_target, v_target = edge
    for route in routes:
        if not route:
            continue
        leg_nodes = [inst.depot] + route + [inst.depot]
        for a, b in zip(leg_nodes[:-1], leg_nodes[1:]):
            _, prev = dijkstra(G, a, targets={b})
            path = reconstruct_path(prev, a, b)
            for p1, p2 in zip(path[:-1], path[1:]):
                if (p1, p2) == (u_target, v_target):
                    return True
    return False


def _find_edge_on_route(routes: list, inst: VRPInstance, G) -> tuple:
    """Picks the first real road edge on the current best route set's actual
    paths — this is the edge the ramp will be applied to, guaranteeing the
    simulated jam actually matters to the vehicle."""
    for route in routes:
        if not route:
            continue
        leg_nodes = [inst.depot] + route + [inst.depot]
        for a, b in zip(leg_nodes[:-1], leg_nodes[1:]):
            _, prev = dijkstra(G, a, targets={b})
            path = reconstruct_path(prev, a, b)
            if len(path) >= 2:
                return path[0], path[1]
    raise ValueError("No usable edge found on any route — instance may be degenerate")


def run_ab_comparison(n_nodes: int = 200, n_customers: int = 15, seed: int = 42,
                       ramp_steps: int = 20, threshold: float = 0.6,
                       forecast_horizon: int = 3, verbose: bool = True):
    G = make_synthetic_graph(n_nodes=n_nodes, seed=seed)
    inst = generate_instance(G, n_customers=n_customers, num_vehicles=3,
                              vehicle_capacity=40, seed=seed)

    best_routes, best_cost, meta0 = solve_qpso_adaptive(
        inst, swarm_size=40, iterations=100, seed=seed
    )
    target_edge = _find_edge_on_route(best_routes, inst, G)
    u, v = target_edge
    ramp_values = [0.05 + i * (0.85 / ramp_steps) for i in range(ramp_steps + 1)]

    results = {}
    for strategy in ("reactive", "predictive"):
        sim_g = make_synthetic_graph(n_nodes=n_nodes, seed=seed)  # fresh graph per strategy
        current_routes = best_routes
        current_positions = meta0["final_positions"]
        current_inst = inst
        forecaster = HoltForecaster()
        exposure_steps = 0
        rerouted_at = None

        for t, congestion_val in enumerate(ramp_values):
            sim_g[u][v]["congestion"] = congestion_val
            forecaster.update(congestion_val)

            if rerouted_at is None and route_uses_edge(current_routes, current_inst, sim_g, (u, v)):
                exposure_steps += 1

            should_reroute = False
            if rerouted_at is None:
                if strategy == "reactive" and congestion_val >= threshold:
                    should_reroute = True
                elif strategy == "predictive" and forecaster.is_rising_toward(threshold, forecast_horizon):
                    should_reroute = True

            if should_reroute:
                new_matrices = build_full_metrics_matrices(sim_g, current_inst.nodes)
                current_inst = replace(
                    current_inst,
                    distance_matrix=new_matrices["time"],
                    raw_distance_matrix=new_matrices["distance"],
                    congestion_matrix=new_matrices["congestion"],
                    emissions_matrix=new_matrices["emissions"],
                    term_scales=compute_term_scales(new_matrices),
                )
                current_routes, _, meta = solve_qpso_adaptive(
                    current_inst, iterations=60, seed=seed, init_positions=current_positions
                )
                current_positions = meta["final_positions"]
                rerouted_at = t

        results[strategy] = {
            "exposure_steps": exposure_steps,
            "rerouted_at": rerouted_at,
            "total_steps": len(ramp_values),
        }
        if verbose:
            print(f"{strategy:>10}: rerouted at step {rerouted_at} "
                  f"(exposure: {exposure_steps}/{len(ramp_values)} steps)")

    return results, target_edge


if __name__ == "__main__":
    run_ab_comparison()
