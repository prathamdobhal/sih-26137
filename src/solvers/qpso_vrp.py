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
