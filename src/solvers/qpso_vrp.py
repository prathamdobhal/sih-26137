"""
Quantum-behaved Particle Swarm Optimization (QPSO) for CVRP — the actual
problem-statement algorithm.

Encoding: SPV (Smallest-Position-Value), per docs/formulation.md Section 7.
Each particle holds a continuous position vector x in R^n (n = number of
customers). To evaluate a particle, its values are sorted (argsort) to
produce a permutation — the "giant tour" — which is then split into
feasible routes by the SAME greedy capacity-split GA uses (ga_vrp.split_into_routes),
so the two algorithms are compared on identical decode logic; only the
search mechanism differs.

Position update (no velocity term — this is what distinguishes QPSO from
classical PSO):

    mbest_d = mean over the swarm of each particle's personal-best position, per dimension d
    p_id    = phi * pbest_id + (1 - phi) * gbest_d,   phi ~ U(0,1)
    x_id(t+1) = p_id +/- beta * |mbest_d - x_id(t)| * ln(1/u),   u ~ U(0,1)

beta (contraction-expansion coefficient) is linearly annealed from beta_max to
beta_min over the run — standard practice from the QPSO literature (Sun et al.),
balancing early exploration against late-stage exploitation.
"""

import random
import numpy as np

from src.solvers.instance import VRPInstance
from src.solvers.evaluate import route_set_cost
from src.solvers.ga_vrp import split_into_routes
from src.solvers.local_search import polish_routes


def decode_particle(position: np.ndarray, inst: VRPInstance) -> list:
    """SPV decode: sort customer indices by position value -> permutation -> routes."""
    order = np.argsort(position)
    permutation = [inst.customers[i] for i in order]
    return split_into_routes(permutation, inst)


def _fitness(position: np.ndarray, inst: VRPInstance) -> float:
    routes = decode_particle(position, inst)
    return route_set_cost(routes, inst)["cost"]


def solve_qpso(inst: VRPInstance, swarm_size: int = 40, iterations: int = 150,
               beta_max: float = 1.0, beta_min: float = 0.4,
               seed: int = 42, verbose: bool = False):
    """
    Runs QPSO and returns (best_routes, best_cost, convergence_history) — same
    return shape as solve_ga so Phase 7 benchmarking can treat them identically.
    """
    rng = np.random.default_rng(seed)
    n = len(inst.customers)

    # Positions initialized in [0, 1)^n — SPV decoding only cares about relative
    # order, so the absolute range doesn't matter, just needs to be continuous.
    positions = rng.random((swarm_size, n))
    pbest = positions.copy()
    pbest_fitness = np.array([_fitness(positions[i], inst) for i in range(swarm_size)])

    gbest_idx = int(np.argmin(pbest_fitness))
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    history = [gbest_fitness]

    for it in range(iterations):
        beta = beta_max - (beta_max - beta_min) * (it / max(1, iterations - 1))
        mbest = pbest.mean(axis=0)

        for i in range(swarm_size):
            phi = rng.random(n)
            p = phi * pbest[i] + (1 - phi) * gbest

            u = rng.random(n)
            u = np.clip(u, 1e-9, 1 - 1e-9)  # avoid log(0)/log(inf)
            sign = rng.choice([-1.0, 1.0], size=n)
            positions[i] = p + sign * beta * np.abs(mbest - positions[i]) * np.log(1.0 / u)

            fit = _fitness(positions[i], inst)
            if fit < pbest_fitness[i]:
                pbest[i] = positions[i].copy()
                pbest_fitness[i] = fit
                if fit < gbest_fitness:
                    gbest = positions[i].copy()
                    gbest_fitness = fit

        history.append(gbest_fitness)
        if verbose and it % 25 == 0:
            print(f"iter {it}: gbest_cost={gbest_fitness:.2f} beta={beta:.2f}")

    best_routes = decode_particle(gbest, inst)
    return best_routes, gbest_fitness, history


def _swarm_diversity(positions: np.ndarray) -> float:
    """Mean pairwise Euclidean distance across the swarm — collapses toward 0
    as particles converge on the same region of the search space."""
    centroid = positions.mean(axis=0)
    return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))


def solve_qpso_adaptive(inst: VRPInstance, swarm_size: int = 40, iterations: int = 150,
                         beta_max: float = 1.0, beta_min: float = 0.4,
                         local_search_every: int = 5, reinit_fraction: float = 0.10,
                         stagnation_window: int = 30, diversity_threshold: float = 0.02,
                         seed: int = 42, verbose: bool = False,
                         init_positions: np.ndarray = None):
    """
    QPSO + 2-opt local search + stagnation control (Phase 6). Same update rule
    as solve_qpso, plus:

    init_positions: optional (swarm_size, n_customers) array to WARM-START the
    swarm from a previous run's final positions instead of random init — used
    by src/dynamic/rerouting.py when traffic conditions change. Personal bests
    are re-evaluated fresh against `inst` (which may have an updated distance
    matrix), not carried over, since the fitness landscape may have shifted.

    - Every `local_search_every` iterations, 2-opt polishes the current gbest's
      decoded routes; if the polished version is better, it REPLACES gbest by
      re-encoding isn't possible (2-opt operates on route order, not the
      continuous position vector) — so instead we track a separate
      "polished_best" routes/cost pair alongside the swarm's own gbest, and
      report whichever is better at the end. This keeps the swarm's own
      search dynamics untouched by the polish step.
    - Stagnation control: if the swarm's positional diversity drops below
      `diversity_threshold`, OR gbest hasn't improved in `stagnation_window`
      iterations, the worst `reinit_fraction` of particles (by pbest fitness)
      are reinitialized to random positions — matching the chaos/mutation
      escape mechanism from Li et al. (2012), simplified to reinitialization.
    """
    rng = np.random.default_rng(seed)
    n = len(inst.customers)

    if init_positions is not None:
        positions = init_positions.copy()
        swarm_size = positions.shape[0]
    else:
        positions = rng.random((swarm_size, n))
    pbest = positions.copy()
    pbest_fitness = np.array([_fitness(positions[i], inst) for i in range(swarm_size)])

    gbest_idx = int(np.argmin(pbest_fitness))
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    polished_routes = decode_particle(gbest, inst)
    polished_cost = route_set_cost(polished_routes, inst)["cost"]

    history = [gbest_fitness]
    diversity_history = [_swarm_diversity(positions)]
    reinit_events = []
    iters_since_improvement = 0

    for it in range(iterations):
        beta = beta_max - (beta_max - beta_min) * (it / max(1, iterations - 1))
        mbest = pbest.mean(axis=0)
        prev_gbest_fitness = gbest_fitness

        for i in range(swarm_size):
            phi = rng.random(n)
            p = phi * pbest[i] + (1 - phi) * gbest
            u = np.clip(rng.random(n), 1e-9, 1 - 1e-9)
            sign = rng.choice([-1.0, 1.0], size=n)
            positions[i] = p + sign * beta * np.abs(mbest - positions[i]) * np.log(1.0 / u)

            fit = _fitness(positions[i], inst)
            if fit < pbest_fitness[i]:
                pbest[i] = positions[i].copy()
                pbest_fitness[i] = fit
                if fit < gbest_fitness:
                    gbest = positions[i].copy()
                    gbest_fitness = fit

        # Local search polish, periodic (2-opt is O(n^2) per route, too expensive
        # to run on the whole swarm every iteration)
        if (it + 1) % local_search_every == 0:
            candidate_routes = decode_particle(gbest, inst)
            candidate_routes = polish_routes(candidate_routes, inst)
            candidate_cost = route_set_cost(candidate_routes, inst)["cost"]
            if candidate_cost < polished_cost:
                polished_cost = candidate_cost
                polished_routes = candidate_routes

        # Stagnation control
        diversity = _swarm_diversity(positions)
        diversity_history.append(diversity)
        if gbest_fitness < prev_gbest_fitness - 1e-9:
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1

        if diversity < diversity_threshold or iters_since_improvement >= stagnation_window:
            n_reinit = max(1, int(swarm_size * reinit_fraction))
            worst_idx = np.argsort(pbest_fitness)[-n_reinit:]
            for idx in worst_idx:
                positions[idx] = rng.random(n)
                pbest[idx] = positions[idx].copy()
                pbest_fitness[idx] = _fitness(positions[idx], inst)
            reinit_events.append(it)
            iters_since_improvement = 0
            if verbose:
                print(f"iter {it}: stagnation triggered (diversity={diversity:.3f}), "
                      f"reinitialized {n_reinit} particles")

        history.append(gbest_fitness)
        if verbose and it % 25 == 0:
            print(f"iter {it}: gbest_cost={gbest_fitness:.2f} beta={beta:.2f}")

    # Report whichever is better: the swarm's own gbest, or the periodically
    # polished route set
    swarm_routes = decode_particle(gbest, inst)
    if polished_cost < gbest_fitness:
        final_routes, final_cost = polished_routes, polished_cost
    else:
        final_routes, final_cost = swarm_routes, gbest_fitness

    return final_routes, final_cost, {
        "gbest_history": history,
        "diversity_history": diversity_history,
        "reinit_events": reinit_events,
        "final_positions": positions,  # for warm-starting a future re-optimization
        "final_pbest": pbest,          # for explainability: alternative candidates
        "final_pbest_fitness": pbest_fitness,
    }
