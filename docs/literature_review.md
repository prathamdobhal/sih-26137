# Literature Review — SIH26137

## Foundational QPSO
- Sun, Feng, Xu (2004), *Particle Swarm Optimization with Particles Having Quantum Behavior* — introduces QPSO, replacing the velocity update with quantum-behaved particles sampled around a mean best position.
- Sun, Xu, Feng (2004), *A Global Search Strategy of Quantum-Behaved Particle Swarm Optimization* — establishes QPSO's global convergence property (vs. PSO's lack thereof).
- Sun, Lai, Xu, Ding, Chai (2007), *A Modified Quantum-Behaved Particle Swarm Optimization* — adds Gaussian disturbance to the mean best position specifically to prevent stagnation and help particles escape local optima. **Direct precedent for our Adaptive Stagnation Control feature.**

## QPSO applied to VRP
- Li, Li, Wang (2012), *Quantum-Behaved PSO Algorithm Based on Border Mutation and Chaos for VRP* — closest prior art. Uses chaotic search on best particles when trapped in local optima, plus a boundary-mutation strategy for diversity. **Gap: static graph only, no dynamic traffic handling.**

## Classical PSO/GA hybrids for VRPTW (baseline reference material)
- Xu, Liu, Zhang, Wang, Sun (2015), *A Combination of GA and PSO for VRPTW* — real-number particle encoding, linearly decreasing balance function, GA crossover to avoid premature convergence.
- (PMC10603129) Modified PSO for vehicle scheduling with soft time windows — elite-reverse initialization, adaptive inertia weight, jump-out mechanism.

## True quantum-hardware approaches (for the "why quantum-inspired, not quantum hardware" argument)
- QUBO formulation for multi-depot capacitated VRP + dynamic rerouting via quantum annealing (arXiv:2005.12478).
- QAOA applied to heterogeneous VRP via Ising Hamiltonian mapping (arXiv:2110.06799) — qubit requirement scales **quadratically** with customer count, confirming current quantum hardware cannot handle realistic city-scale VRP instances.

## Gap Statement
Prior QPSO-VRP work (Li et al. 2012) handles premature convergence via chaos/mutation but assumes a static graph. True quantum-hardware VRP approaches (QUBO/QAOA) address dynamism or theoretical quantum advantage but not scalability — QAOA's quadratic qubit scaling makes it impractical for real city-scale routing today. This project combines QPSO's classical scalability with dynamic traffic-aware re-optimization, predictive rerouting, and decision explainability — a combination not addressed by any single paper surveyed.
