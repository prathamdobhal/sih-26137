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
- [ ] Day 7+ — Full benchmarking suite across small/medium/large graphs + convergence charts

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
