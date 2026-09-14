# Problem Formulation — SIH26137

## 1. Graph Model

The road network is modeled as a directed weighted graph:

```
G = (V, E, W)
```

- **V** = set of nodes (intersections / delivery stops / depot)
- **E** = set of directed edges (road segments; directed because one-way streets and
  asymmetric congestion are common)
- **W** = edge weight function, time-varying:

```
w_ij(t) = f( distance_ij, speed_limit_ij, congestion_ij(t) )
```

Concretely, for the prototype:

```
travel_time_ij(t) = distance_ij / (speed_limit_ij × (1 − congestion_ij(t)))
```

`congestion_ij(t) ∈ [0, 0.9]` (capped below 1 to avoid division blow-up) — 0 = free flow,
0.9 = near-gridlock. This is the single value the incident-injection and predictive
modules will manipulate later.

## 2. Decision Variables

For a fleet of K vehicles serving a set of customers C from a single depot d:

```
R = { R_1, R_2, ..., R_K }
```

where each `R_k` is an ordered sequence of nodes (a route) starting and ending at `d`,
covering a disjoint subset of C, such that every customer in C is visited exactly once
across all routes.

## 3. Objective Function

Multi-objective, combined via weighted sum for the prototype (Pareto-front methods are
noted as future work — weighted sum is the right call for a judged demo since it lets you
show clean, explainable mode-switching):

```
minimize F(R) = α·T(R) + β·D(R) + γ·C(R) + δ·E(R)
```

| Term | Meaning | Normalization |
|---|---|---|
| T(R) | Total travel time across all routes | ÷ by a reference "all-Dijkstra" time |
| D(R) | Total distance across all routes | ÷ by a reference "all-Dijkstra" distance |
| C(R) | Congestion exposure = Σ (edge congestion × time spent on edge) | ÷ by worst-case congestion exposure |
| E(R) | Emissions estimate (Section 5) | ÷ by a reference max-emissions baseline |

**Always normalize each term to [0,1] before applying weights** — otherwise travel time
(minutes, large numbers) will dominate congestion (a 0–1 fraction) regardless of the
weights you pick. Do this once you have the baseline (Dijkstra) values for a given
instance, then normalize all four terms against that reference run.

### Mode weight presets (α, β, γ, δ — must sum to 1)

| Mode | α (time) | β (distance) | γ (congestion) | δ (emissions) | Notes |
|---|---|---|---|---|---|
| Fastest | 0.60 | 0.10 | 0.30 | 0.00 | Minimize time, congestion as a secondary time-proxy |
| Eco | 0.20 | 0.20 | 0.20 | 0.40 | Emissions dominate |
| Balanced | 0.35 | 0.15 | 0.25 | 0.25 | Default demo mode |
| Emergency | 0.90 | 0.05 | 0.05 | 0.00 | Congestion/eco effectively ignored; add a yield-flag (Section 6) |

These are starting points — you'll tune them once you see real convergence behavior in
Phase 5, but lock these as defaults now so every module built after this references the
same numbers.

## 4. Constraints

| Constraint | Definition | Handling |
|---|---|---|
| Capacity | `Σ demand_i (i ∈ R_k) ≤ Q_k` for every vehicle k | Penalty term + repair operator |
| Time window | Each customer i must be reached within `[e_i, l_i]` | Penalty term proportional to violation magnitude |
| Single depot | Every route starts and ends at `d` | Enforced structurally in the encoding, not penalized |
| Visit-once | Every customer appears in exactly one route | Enforced structurally in the encoding |

**Penalized fitness function actually optimized:**

```
Fitness(R) = F(R) + λ1·max(0, capacity_violation) + λ2·max(0, time_window_violation)
```

Start with `λ1 = λ2 = 5` (large relative to normalized F(R) ∈ [0,1], so infeasible
solutions are always ranked worse than feasible ones) — tune during Phase 5 testing.

## 5. Emissions Estimate (for Eco mode — Idea A)

Kept deliberately simple, no sensor data required:

```
E(R) = Σ over all route segments [ base_emission_rate × distance × speed_factor(v) ]
       + idle_emission_rate × num_stops
```

`speed_factor(v)` can be a simple U-shaped curve (emissions rise at very low speed from
idling/stop-start, and at very high speed from inefficiency) — a piecewise linear
approximation is enough for the prototype; cite standard traffic-emissions literature
(e.g., MOVES/COPERT-style speed-emission curves) for credibility in the report without
needing their full complexity.

## 6. Emergency Vehicle Profile (Idea B)

Same objective function, Emergency weights from the table above, plus a **yield flag**:
when an emergency route is active, edges shared with normal-vehicle routes get a
temporary congestion penalty added for *other* vehicles' re-optimization — modeling
real-world yielding behavior without needing true multi-agent simulation.

## 7. QPSO Particle Encoding Decision

VRP is a discrete/permutation problem; QPSO's position update (Day-1 doc) is continuous.
**Decision: use the Smallest-Position-Value (SPV) rule** — the standard technique for
applying continuous swarm algorithms to permutation problems:

1. Each particle's position is a real-valued vector `x ∈ R^n` (n = number of customers).
2. Sort the customers by their corresponding `x` values, ascending → gives a permutation
   (the "giant tour").
3. Split the giant tour into feasible routes using capacity/time-window constraints
   (greedy split, or Prins' optimal split algorithm if time permits in Phase 5).
4. Evaluate `Fitness(R)` on the resulting route set.
5. QPSO's mbest/potential-well update (Day-1 doc) operates on the continuous `x` vectors
   as normal — the SPV decode step only happens at evaluation time.

This is the cleanest way to keep QPSO's original continuous math intact (matching the
literature you're citing) while still solving a discrete routing problem — and it's easy
to explain to judges in one sentence: *"continuous particle positions are decoded into
route orderings via the smallest-position-value rule."**

## 8. Architecture (reference — full diagram in `docs/architecture.md`)

```
Graph + Traffic State → QPSO Solver (SPV-encoded, constraint-penalized)
                       → Local Search (2-opt) polish
                       → Route Set R
                       → Explainability Layer (perturb ±20%, compare alternatives)
                       → Dashboard
```
