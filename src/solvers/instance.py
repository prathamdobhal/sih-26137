"""
VRP instance generator — samples a depot + N customers with random demand from
any graph (synthetic or OSM-loaded), producing the input every solver (GA, QPSO)
consumes: a distance matrix, demands, and vehicle capacity.
"""

from dataclasses import dataclass, field
import numpy as np
import networkx as nx

from src.solvers.shortest_path import build_distance_matrix


@dataclass
class VRPInstance:
    depot: object                  # graph node id acting as depot
    customers: list                # list of graph node ids
    demand: dict                   # {node_id: demand_units}
    distance_matrix: np.ndarray    # travel times, index 0 = depot, 1..N = customers
    num_vehicles: int
    vehicle_capacity: int
    nodes: list = field(init=False)  # [depot] + customers, matches matrix indexing

    def __post_init__(self):
        self.nodes = [self.depot] + self.customers


def generate_instance(G: nx.DiGraph, n_customers: int = 15, num_vehicles: int = 3,
                       vehicle_capacity: int = 40, seed: int = 42) -> VRPInstance:
    """
    Randomly selects a depot + n_customers from G's nodes (all guaranteed
    reachable from the depot), assigns random demand, and precomputes the
    real shortest-path distance matrix between them via Dijkstra.
    """
    rng = np.random.default_rng(seed)

    # Depot + customers must all be mutually reachable, so restrict sampling to
    # the largest weakly-connected component (matters on OSM graphs, which can
    # have small disconnected slivers at the query boundary).
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    candidates = list(largest_cc)

    if len(candidates) < n_customers + 1:
        raise ValueError(
            f"Graph's largest connected component ({len(candidates)} nodes) is too "
            f"small for {n_customers} customers + depot. Reduce n_customers or use "
            f"a larger graph/radius."
        )

    chosen = rng.choice(candidates, size=n_customers + 1, replace=False)
    depot = chosen[0]
    customers = list(chosen[1:])

    demand = {c: int(rng.integers(3, 12)) for c in customers}

    nodes = [depot] + customers
    dist_matrix = build_distance_matrix(G, nodes)

    return VRPInstance(
        depot=depot,
        customers=customers,
        demand=demand,
        distance_matrix=dist_matrix,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
    )


def instance_summary(inst: VRPInstance) -> dict:
    total_demand = sum(inst.demand.values())
    total_capacity = inst.num_vehicles * inst.vehicle_capacity
    return {
        "n_customers": len(inst.customers),
        "total_demand": total_demand,
        "total_fleet_capacity": total_capacity,
        "feasible_by_capacity_alone": total_demand <= total_capacity,
    }
