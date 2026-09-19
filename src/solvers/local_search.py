"""
2-opt local search — polishes a single vehicle route by repeatedly reversing
segments that shorten the route, until no improving reversal remains.

Operates on the SAME distance matrix every solver uses (inst.distance_matrix),
so improvements here are measured in the identical units (travel time) as
everything else in the pipeline.
"""

from src.solvers.instance import VRPInstance


def _route_cost(route: list, inst: VRPInstance, node_to_idx: dict, depot_idx: int) -> float:
    if not route:
        return 0.0
    idxs = [depot_idx] + [node_to_idx[c] for c in route] + [depot_idx]
    return sum(
        inst.distance_matrix[idxs[i], idxs[i + 1]] for i in range(len(idxs) - 1)
    )


def two_opt(route: list, inst: VRPInstance, max_passes: int = 20) -> list:
    """
    Standard 2-opt: for every pair of positions (i, j) in the route, try
    reversing the segment between them; keep the reversal if it reduces cost.
    Repeats until a full pass makes no improvement, or max_passes is hit
    (safety cap — real routes converge in a handful of passes).
    """
    if len(route) < 3:
        return route  # nothing to reorder

    node_to_idx = {node: i for i, node in enumerate(inst.nodes)}
    depot_idx = node_to_idx[inst.depot]

    best_route = route[:]
    best_cost = _route_cost(best_route, inst, node_to_idx, depot_idx)

    for _ in range(max_passes):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                candidate = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                candidate_cost = _route_cost(candidate, inst, node_to_idx, depot_idx)
                if candidate_cost < best_cost - 1e-9:
                    best_route = candidate
                    best_cost = candidate_cost
                    improved = True
        if not improved:
            break

    return best_route


def polish_routes(routes: list, inst: VRPInstance) -> list:
    """Applies 2-opt independently to every vehicle's route in a route set."""
    return [two_opt(r, inst) for r in routes]
