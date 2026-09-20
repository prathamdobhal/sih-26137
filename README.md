# SIH26137 — Quantum-Inspired Intelligent Traffic Route Optimization

## Structure
```
sih26137/
├── docs/                  # formulation doc, literature review, architecture diagrams
├── src/
│   ├── network/            # graph modeling, OSM loader, traffic simulator
│   ├── solvers/             # Dijkstra, GA baseline, QPSO core, local search
│   ├── dynamic/             # incident injection, warm-started re-optimization, predictive rerouting
│   ├── explainability/      # alternative-route comparison, confidence scoring
│   └── api/                 # (later) FastAPI service if needed beyond Streamlit
├── app/                    # Streamlit or React dashboard
├── tests/                  # pytest unit tests per module
├── notebooks/               # exploratory analysis, algorithm tuning
├── data/                    # synthetic + OSM-cached graphs
└── reports/figures/         # benchmark charts, convergence plots for the report/PPT
```

## Status
- [x] Day 1 — Literature review + gap statement (`docs/literature_review.md`)
- [x] Day 2 — Problem formulation (`docs/formulation.md`) + architecture diagram (`docs/architecture.md`)
- [x] Day 3 — Network module: synthetic generator, OSM loader (Koramangala, Bengaluru — confirmed live: 2105 nodes / 5297 edges), traffic simulator (`src/network/`)
- [x] Day 4 — Baseline algorithms: hand-implemented Dijkstra (cross-validated vs. networkx), VRP instance generator, GA solver for CVRP (`src/solvers/`)
- [x] Day 5 — QPSO core engine, SPV-decoded, mbest/potential-well update (`src/solvers/qpso_vrp.py`)
- [x] Day 6 — 2-opt local search (`src/solvers/local_search.py`) + adaptive QPSO with stagnation control (`solve_qpso_adaptive`). Tuned via real hyperparameter sweep — final config: `reinit_fraction=0.10, stagnation_window=30, diversity_threshold=0.02, local_search_every=5`. Confirmed result: ties GA on small instances (15 customers, -0.1%), beats GA by ~2.4% on larger instances (45 customers) — matches VRP literature's expectation that metaheuristic gaps widen with problem size.
- [x] Day 7 — Full benchmarking suite across small/medium/large tiers (`src/benchmarking/run_benchmark.py`), doubling as the Phase 13 scalability report. Outputs: `reports/benchmark_results.csv`, `reports/benchmark_summary.md`, convergence + scalability charts in `reports/figures/`. Honest finding: QPSO has higher run-to-run variance than GA at larger sizes (occasional 19-21% wins, occasional losses), averaging out near parity — flagged as a real property, not a bug, with tuning noted as future work. Interpretation text is computed dynamically from actual run data, never hardcoded.
- [x] Day 8 — Dynamic rerouting (`src/dynamic/rerouting.py`): incident injection → distance matrix recompute → QPSO warm-started from previous swarm state. Proven, not just implemented: warm-start reaches within 5% of best cost in 76 iterations vs. cold-start's 101 (~25% faster recovery after an incident), and reached a better final answer in the same test run.
- [x] Day 9 — Predictive pre-emptive rerouting (`src/dynamic/predictor.py`, `src/dynamic/predictive_rerouting.py`): hand-implemented Holt's linear trend forecaster per edge, plus a reactive-vs-predictive A/B harness using real Dijkstra path reconstruction to measure congestion exposure (not an approximation). Confirmed across 3 seeds: predictive rerouting consistently triggers 3 steps earlier than reactive (step 10 vs. 13 on a 20-step jam ramp), with equal-or-lower congestion exposure every time.
- [x] Day 10 — Explainable route decision layer (`src/explainability/explain.py`): top-K distinct alternative routes pulled from the swarm's own personal-best particles (not synthetic perturbations), real Dijkstra-based distance/time/congestion breakdown per candidate, plain-language reason generation, and a Monte Carlo (±20% congestion, 20 trials) confidence score. Validated across 10 seeds: confidence correctly drops to 45% for near-tied candidates (20,980 vs 20,980) and reliably shows 100% for clearly-separated ones — the metric behaves sensibly across the full range, not just in one cherry-picked case.
- [x] Day 11 — Full multi-objective cost function wired in: `F(R) = α·T + β·D + γ·C + δ·E` with the four mode presets (Fastest/Eco/Balanced/Emergency) from `docs/formulation.md`, backed by real per-edge distance/congestion/emissions matrices (`build_full_metrics_matrices`), plus an Emergency yield mechanism (`src/dynamic/emergency.py`). Caught and fixed a real bug during testing: post-incident/perturbation instance recomputation was only refreshing the time matrix, leaving congestion/distance/emissions stale — now all four matrices + normalization scales recompute together everywhere (`rerouting.py`, `predictive_rerouting.py`, `explain.py`). Eco mode's emissions advantage was unreliable at low search budget (avg -0.35%, some seeds -4.5%) due to QPSO search variance, not a flawed objective — confirmed by doubling swarm size/iterations, which stabilized it to a consistent, if modest, average +0.4% saving.
- [x] Day 12 fix — Bug found by actually using the app: at the dashboard's fast interactive budget (40/70), a single QPSO run per mode could show Fastest with a HIGHER travel time than Balanced — confusing for a live demo. More iterations alone didn't fix it (tested to 80/200); the real fix was `solve_qpso_multistart` (3 restarts, keep the best), now the dashboard's default for the optimize button. Locked in with a regression test across 6 seeds (1/6 borderline failures, within tolerance).
- [x] Day 12 — Streamlit dashboard (`app/dashboard.py`, `app/styles.py`, `app/data_loader.py`, `app/map_utils.py`): "fleet dispatcher console" design — dark control-room palette with one accent color per mode used consistently across button/chip/route line, Space Grotesk + Inter + IBM Plex Mono (mono reserved for numeric readouts only). Cache-first data loading (real Koramangala if fetched, synthetic fallback otherwise, so the dashboard runs on any machine). Every control calls a real backend function — no mocked data. Tested via Streamlit's `AppTest` framework (not just a syntax check): all 8 controls (4 mode buttons, optimize, incident, explain, benchmark) clicked through in sequence with zero exceptions. Caught and fixed two real bugs during testing: CartoDB's dark map tiles now require an API key (switched to OpenStreetMap tiles before this could break the live demo), and a `width="stretch"` vs `use_container_width` mismatch on the map component specifically (third-party components use a different API than core Streamlit's newer widgets).

## Data source
Real road topology: [OpenStreetMap](https://www.openstreetmap.org) contributors, fetched via the
[`osmnx`](https://osmnx.readthedocs.io) Python library (Overpass API). Default demo area:
Koramangala, Bengaluru — change `DEFAULT_PLACE` in `src/network/osm_loader.py` to use a different area.
Traffic congestion is **simulated** on top of the real graph (`src/network/traffic_simulator.py`) —
this project does not use a live/real-time traffic feed.

## Setup
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
