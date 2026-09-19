"""
Shared route evaluation — every solver (Dijkstra sanity check, GA baseline, and
later QPSO) scores route sets through this SAME function, so benchmarking
comparisons in Phase 7 are apples-to-apples.

Day 4 scope note: this evaluates on total travel time (the dominant T(R) term
from docs/formulation.md) plus capacity-violation penalty. The full weighted
multi-objective F(R) = α·T + β·D + γ·C + δ·E and mode presets (Fastest/Eco/
Balanced/Emergency) get wired in when QPSO's mode selector is built (Phase 11) —
introducing all four terms now, before any solver exists to compare against,
would make today's baselines harder to sanity-check by hand.
"""

from src.solvers.instance import VRPInstance

CAPACITY_PENALTY_LAMBDA = 2.0   # penalty weights re-tuned for the normalized [~0,3] cost
VISIT_PENALTY_LAMBDA = 50.0     # scale used starting Day 11 (see mode weights below);
                                # large enough that an infeasible solution never wins
IDLE_EMISSION_RATE_G = 40.0     # grams CO2 per stop (idling/start-stop overhead)

# Mode weight presets — locked in docs/formulation.md Section 3 on Day 2,
# wired into the actual cost function here on Day 11. Must each sum to 1.0.
MODE_WEIGHTS = {
    "fastest":   {"alpha": 0.60, "beta": 0.10, "gamma": 0.30, "delta": 0.00},
    "eco":       {"alpha": 0.20, "beta": 0.20, "gamma": 0.20, "delta": 0.40},
    "balanced":  {"alpha": 0.35, "beta": 0.15, "gamma": 0.25, "delta": 0.25},
    "emergency": {"alpha": 0.90, "beta": 0.05, "gamma": 0.05, "delta": 0.00},
}


def route_set_cost(routes: list, inst: VRPInstance, mode: str = "fastest") -> dict:
    """
    `routes`: list of vehicle routes, each a list of customer NODE IDS
    (not matrix indices) in visit order, e.g. [[c1, c3], [c2, c4, c5]].
    An implicit depot->first, last->depot leg is added for every non-empty route.

    `mode`: one of "fastest" / "eco" / "balanced" / "emergency" — selects the
    weight preset from MODE_WEIGHTS, applied to the normalized multi-objective
    F(R) = alpha.T + beta.D + gamma.C + delta.E (docs/formulation.md Section 3).

    Normalization: each raw term (seconds, meters, congestion-seconds, grams)
    lives on a wildly different scale, so each is divided by inst.term_scales
    (the instance's own average pairwise magnitude for that term) before
    weighting — this is a pragmatic per-instance normalization rather than a
    literal "solve a reference instance first" approach, but serves the same
    purpose: no single term numerically dominates just because of its units.

    Returns a breakdown dict so callers can inspect feasibility, the raw
    T/D/C/E components, and the final weighted cost separately.
    """
    weights = MODE_WEIGHTS[mode]
    node_to_idx = {node: i for i, node in enumerate(inst.nodes)}
    depot_idx = node_to_idx[inst.depot]
    scales = inst.term_scales or {"time": 1.0, "distance": 1.0, "congestion": 1.0, "emissions": 1.0}

    total_time = total_distance = total_congestion = total_emissions = 0.0
    capacity_violation = 0.0
    all_customers_visited = set()

    for route in routes:
        if not route:
            continue

        route_demand = sum(inst.demand[c] for c in route)
        if route_demand > inst.vehicle_capacity:
            capacity_violation += (route_demand - inst.vehicle_capacity)

        prev_idx = depot_idx
        for customer in route:
            cust_idx = node_to_idx[customer]
            total_time += inst.distance_matrix[prev_idx, cust_idx]
            if inst.raw_distance_matrix is not None:
                total_distance += inst.raw_distance_matrix[prev_idx, cust_idx]
                total_congestion += inst.congestion_matrix[prev_idx, cust_idx]
                total_emissions += inst.emissions_matrix[prev_idx, cust_idx]
            prev_idx = cust_idx
            all_customers_visited.add(customer)
        total_time += inst.distance_matrix[prev_idx, depot_idx]
        if inst.raw_distance_matrix is not None:
            total_distance += inst.raw_distance_matrix[prev_idx, depot_idx]
            total_congestion += inst.congestion_matrix[prev_idx, depot_idx]
            total_emissions += inst.emissions_matrix[prev_idx, depot_idx]

    total_emissions += IDLE_EMISSION_RATE_G * len(all_customers_visited)

    missing = set(inst.customers) - all_customers_visited
    visit_violation = len(missing)

    # Normalize each term to a comparable ~O(1) scale before weighting
    n_legs = max(sum(len(r) + 1 for r in routes if r), 1)  # +1 per route for the return-to-depot leg
    t_norm = total_time / (scales["time"] * n_legs)
    d_norm = total_distance / (scales["distance"] * n_legs)
    c_norm = total_congestion / (scales["congestion"] * n_legs)
    e_norm = total_emissions / (scales["emissions"] * n_legs)

    F = (weights["alpha"] * t_norm + weights["beta"] * d_norm
         + weights["gamma"] * c_norm + weights["delta"] * e_norm)

    penalized_cost = F + CAPACITY_PENALTY_LAMBDA * capacity_violation + VISIT_PENALTY_LAMBDA * visit_violation

    return {
        "total_time": total_time,
        "total_distance": total_distance,
        "total_congestion_exposure": total_congestion,
        "total_emissions_g": total_emissions,
        "capacity_violation": capacity_violation,
        "missing_customers": missing,
        "feasible": capacity_violation == 0 and not missing,
        "mode": mode,
        "F": F,
        "cost": penalized_cost,
    }
