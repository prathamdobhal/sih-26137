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
- [ ] Day 3+ — Network module (`src/network/`)

## Setup
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
