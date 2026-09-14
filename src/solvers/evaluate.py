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

CAPACITY_PENALTY_LAMBDA = 5.0  # matches docs/formulation.md Section 4 default (lambda1)


def route_set_cost(routes: list, inst: VRPInstance) -> dict:
    """
    `routes`: list of vehicle routes, each a list of customer NODE IDS
    (not matrix indices) in visit order, e.g. [[c1, c3], [c2, c4, c5]].
    An implicit depot->first, last->depot leg is added for every non-empty route.

    Returns a breakdown dict so callers can inspect feasibility separately
    from raw cost (useful for the explainability layer later).
    """
    node_to_idx = {node: i for i, node in enumerate(inst.nodes)}
    depot_idx = node_to_idx[inst.depot]

    total_time = 0.0
    capacity_violation = 0.0
    all_customers_visited = set()

    for route in routes:
        if not route:
            continue

        route_demand = sum(inst.demand[c] for c in route)
        if route_demand > inst.vehicle_capacity:
            capacity_violation += (route_demand - inst.vehicle_capacity)

        # depot -> first customer
        prev_idx = depot_idx
        for customer in route:
            cust_idx = node_to_idx[customer]
            total_time += inst.distance_matrix[prev_idx, cust_idx]
            prev_idx = cust_idx
            all_customers_visited.add(customer)
        # last customer -> depot
        total_time += inst.distance_matrix[prev_idx, depot_idx]

    missing = set(inst.customers) - all_customers_visited
    # Missing/duplicate visits are also constraint violations — penalize heavily,
    # the repair operators built in Phase 5 should make this case rare in practice.
    visit_violation = len(missing) * 1000.0

    penalized_cost = total_time + CAPACITY_PENALTY_LAMBDA * capacity_violation + visit_violation

    return {
        "total_time": total_time,
        "capacity_violation": capacity_violation,
        "missing_customers": missing,
        "feasible": capacity_violation == 0 and not missing,
        "cost": penalized_cost,
    }
