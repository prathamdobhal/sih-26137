"""
Explainable Route Decision Layer — Phase 10 (Idea 1 from the original problem
statement: don't just show the "optimal" route, show WHY it was chosen over
alternatives, plus a confidence score reflecting sensitivity to traffic
uncertainty).

Three pieces:
  1. get_top_candidates  — pull the swarm's best K distinct personal-best
     particles (not just the single gbest) as real alternative route sets.
  2. decompose_route_set — break a route set's cost into real distance,
     time, and congestion-exposure components via actual Dijkstra path
     reconstruction (same honest approach as Day 9's route_uses_edge),
     not the aggregated distance-matrix number alone.
  3. sensitivity_score   — Monte Carlo perturbation (+/-20% congestion,
     matching docs/formulation.md's stated confidence definition) to see
     how often the chosen winner would still win under uncertain traffic.
"""

from dataclasses import replace
import numpy as np
import networkx as nx

from src.solvers.instance import VRPInstance
from src.solvers.evaluate import route_set_cost
from src.solvers.shortest_path import dijkstra, reconstruct_path, build_distance_matrix
from src.solvers.qpso_vrp import decode_particle


def get_top_candidates(inst: VRPInstance, meta: dict, k: int = 2) -> list:
    """
    Decodes the K best DISTINCT personal-best particles from the swarm's
    final state into real, independently-arrived-at route sets — these are
    genuine alternatives the swarm actually considered, not synthetic
    perturbations of the winner.
    """
    pbest = meta["final_pbest"]
    pbest_fitness = meta["final_pbest_fitness"]
    order = np.argsort(pbest_fitness)

    candidates = []
    seen_route_signatures = set()
    for idx in order:
        routes = decode_particle(pbest[idx], inst)
        signature = tuple(tuple(sorted(r)) for r in routes if r)
        if signature in seen_route_signatures:
            continue  # skip near-duplicate particles, we want genuinely different alternatives
        seen_route_signatures.add(signature)
        cost_info = route_set_cost(routes, inst)
        candidates.append({"routes": routes, "cost": cost_info["cost"],
                            "feasible": cost_info["feasible"]})
        if len(candidates) >= k:
            break

    return candidates


def decompose_route_set(routes: list, inst: VRPInstance, G: nx.DiGraph) -> dict:
    """
    Real path-level breakdown: reconstructs the actual road route for every
    leg (depot -> customer -> ... -> depot) via Dijkstra, and sums genuine
    distance (meters) and time-weighted congestion exposure across every
    edge actually traversed — not the aggregated matrix number alone.
    """
    total_distance_m = 0.0
    total_time_s = 0.0
    congestion_exposure = 0.0  # sum of congestion * time_on_edge, matches C(R) in formulation.md
    num_edges_traversed = 0

    for route in routes:
        if not route:
            continue
        leg_nodes = [inst.depot] + route + [inst.depot]
        for a, b in zip(leg_nodes[:-1], leg_nodes[1:]):
            _, prev = dijkstra(G, a, targets={b})
            path = reconstruct_path(prev, a, b)
            for p1, p2 in zip(path[:-1], path[1:]):
                edge_data = G[p1][p2]
                length_m = edge_data["length_m"]
                speed_kph = max(edge_data["speed_kph"], 1.0)
                speed_mps = speed_kph * 1000 / 3600
                congestion = min(edge_data.get("congestion", 0.0), 0.9)
                edge_time = length_m / (speed_mps * (1 - congestion))

                total_distance_m += length_m
                total_time_s += edge_time
                congestion_exposure += congestion * edge_time
                num_edges_traversed += 1

    return {
        "distance_km": total_distance_m / 1000,
        "time_min": total_time_s / 60,
        "congestion_exposure": congestion_exposure,
        "avg_congestion": congestion_exposure / total_time_s if total_time_s > 0 else 0.0,
        "num_edges": num_edges_traversed,
    }


def sensitivity_score(candidates: list, inst: VRPInstance, G: nx.DiGraph,
                       n_trials: int = 20, perturb_pct: float = 0.20, seed: int = 42) -> dict:
    """
    Monte Carlo confidence check: perturb every edge's congestion by up to
    +/-perturb_pct (multiplicative), recompute the REAL distance matrix under
    each perturbation, and see how often candidate 0 (the current winner)
    stays the cheapest option. This directly implements the "confidence
    score" defined in docs/formulation.md — sensitivity of the ranking to
    +/-20% congestion error.
    """
    rng = np.random.default_rng(seed)
    winner_stays_best = 0

    original_congestion = {(u, v): G[u][v]["congestion"] for u, v in G.edges()}

    for _ in range(n_trials):
        for u, v in G.edges():
            factor = 1.0 + rng.uniform(-perturb_pct, perturb_pct)
            G[u][v]["congestion"] = float(np.clip(original_congestion[(u, v)] * factor, 0.0, 0.9))

        perturbed_matrix = build_distance_matrix(G, inst.nodes)
        perturbed_inst = replace(inst, distance_matrix=perturbed_matrix)

        costs = [route_set_cost(c["routes"], perturbed_inst)["cost"] for c in candidates]
        if int(np.argmin(costs)) == 0:
            winner_stays_best += 1

    # Restore original congestion values — this function must not have side effects on G
    for (u, v), val in original_congestion.items():
        G[u][v]["congestion"] = val

    confidence = winner_stays_best / n_trials
    return {"confidence": confidence, "trials": n_trials, "winner_stayed_best_count": winner_stays_best}


def explain_choice(inst: VRPInstance, G: nx.DiGraph, meta: dict,
                    top_k: int = 2, n_trials: int = 20, seed: int = 42) -> dict:
    """
    Top-level entry point the dashboard calls. Returns everything the
    Explainability Panel needs to render: candidates with real breakdowns,
    a confidence score, and a plain-language reason string.
    """
    candidates = get_top_candidates(inst, meta, k=top_k)
    for c in candidates:
        c["breakdown"] = decompose_route_set(c["routes"], inst, G)

    reason = "Only one distinct candidate was found in the swarm's final state."
    confidence_info = {"confidence": 1.0, "trials": 0, "winner_stayed_best_count": 0}

    if len(candidates) >= 2:
        winner, runner_up = candidates[0], candidates[1]
        wb, rb = winner["breakdown"], runner_up["breakdown"]

        dist_delta = rb["distance_km"] - wb["distance_km"]
        time_delta = rb["time_min"] - wb["time_min"]

        if dist_delta > 0.1 and time_delta < -0.1:
            reason = (f"Chosen route is {abs(time_delta):.1f} min faster despite being "
                      f"{dist_delta:.1f} km longer than the alternative — the extra distance "
                      f"avoids more congested roads.")
        elif dist_delta < -0.1 and time_delta > 0.1:
            reason = (f"Alternative route is {abs(dist_delta):.1f} km shorter but "
                      f"{time_delta:.1f} min slower due to higher congestion exposure "
                      f"({rb['avg_congestion']*100:.0f}% vs {wb['avg_congestion']*100:.0f}%) — "
                      f"the chosen route trades distance for reliability.")
        else:
            reason = (f"Chosen route wins on total cost ({winner['cost']:.0f} vs "
                      f"{runner_up['cost']:.0f}) with {wb['time_min']:.1f} min travel time "
                      f"vs the alternative's {rb['time_min']:.1f} min.")

        confidence_info = sensitivity_score(candidates, inst, G, n_trials=n_trials, seed=seed)

    return {
        "candidates": candidates,
        "reason": reason,
        "confidence": confidence_info["confidence"],
        "confidence_detail": confidence_info,
    }
