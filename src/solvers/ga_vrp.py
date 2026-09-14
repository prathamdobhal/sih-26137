"""
Genetic Algorithm solver for CVRP — the one baseline built in full (per the
solo-scope decision: ACO and classical PSO are cited from literature in the
report rather than re-implemented, since their only job is to give QPSO a
credible comparison point, not to be production-grade themselves).

Encoding: a chromosome is a permutation of all customers (a "giant tour").
It's decoded into feasible vehicle routes by a greedy capacity-based split:
walk the permutation, start a new route whenever adding the next customer
would exceed vehicle capacity. This split step is intentionally the SAME
kind of decode QPSO will use (Day 2 formulation, Section 7) so the two
algorithms are compared on equal footing — the only real difference is how
each one searches over permutations.
"""

import random
from src.solvers.instance import VRPInstance
from src.solvers.evaluate import route_set_cost


def split_into_routes(permutation: list, inst: VRPInstance) -> list:
    """Greedy capacity split of a giant tour into feasible-by-construction routes."""
    routes = []
    current_route = []
    current_load = 0

    for customer in permutation:
        demand = inst.demand[customer]
        if current_load + demand > inst.vehicle_capacity:
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_load = demand
        else:
            current_route.append(customer)
            current_load += demand

    if current_route:
        routes.append(current_route)

    return routes


def _fitness(permutation: list, inst: VRPInstance) -> float:
    routes = split_into_routes(permutation, inst)
    return route_set_cost(routes, inst)["cost"]


def _order_crossover(parent1: list, parent2: list, rng: random.Random) -> list:
    """Standard OX crossover — preserves relative order, guarantees a valid permutation."""
    size = len(parent1)
    a, b = sorted(rng.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    fill_values = [c for c in parent2 if c not in child[a:b]]
    fill_idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[fill_idx]
            fill_idx += 1
    return child


def _swap_mutation(perm: list, rng: random.Random, rate: float = 0.1) -> list:
    perm = perm.copy()
    for i in range(len(perm)):
        if rng.random() < rate:
            j = rng.randrange(len(perm))
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def _tournament_select(population: list, fitnesses: list, rng: random.Random, k: int = 3):
    idxs = rng.sample(range(len(population)), k)
    best = min(idxs, key=lambda i: fitnesses[i])
    return population[best]


def solve_ga(inst: VRPInstance, pop_size: int = 60, generations: int = 150,
             elite_frac: float = 0.1, seed: int = 42, verbose: bool = False):
    """
    Runs the GA and returns (best_routes, best_cost, convergence_history) —
    the history list is what feeds the Phase 7 convergence chart.
    """
    rng = random.Random(seed)
    n_elite = max(1, int(pop_size * elite_frac))

    population = [rng.sample(inst.customers, len(inst.customers)) for _ in range(pop_size)]
    history = []

    best_perm, best_cost = None, float("inf")

    for gen in range(generations):
        fitnesses = [_fitness(p, inst) for p in population]

        gen_best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_cost:
            best_cost = fitnesses[gen_best_idx]
            best_perm = population[gen_best_idx]
        history.append(best_cost)

        if verbose and gen % 25 == 0:
            print(f"gen {gen}: best_cost={best_cost:.2f}")

        # Elitism: carry the best individuals forward unchanged
        ranked = sorted(range(pop_size), key=lambda i: fitnesses[i])
        new_population = [population[i] for i in ranked[:n_elite]]

        while len(new_population) < pop_size:
            parent1 = _tournament_select(population, fitnesses, rng)
            parent2 = _tournament_select(population, fitnesses, rng)
            child = _order_crossover(parent1, parent2, rng)
            child = _swap_mutation(child, rng)
            new_population.append(child)

        population = new_population

    best_routes = split_into_routes(best_perm, inst)
    return best_routes, best_cost, history
