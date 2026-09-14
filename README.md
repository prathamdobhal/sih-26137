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
- [x] Day 3 — Network module: synthetic generator, OSM loader (Koramangala, Bengaluru), traffic simulator (`src/network/`)
- [ ] Day 4+ — Baseline algorithms (Dijkstra + GA)

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
