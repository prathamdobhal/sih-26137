# System Architecture — SIH26137

```mermaid
flowchart TB
    subgraph Frontend["Dashboard (Streamlit)"]
        MAP[Map View - folium]
        MODE[Mode Selector<br/>Fastest / Eco / Balanced / Emergency]
        INCIDENT[Inject Incident Button]
        BENCH[Benchmark & Convergence Charts]
        EXPLAIN[Explainability Panel]
    end

    subgraph Core["Core Engine (Python)"]
        GRAPH[Graph Model<br/>networkx / osmnx]
        SIM[Traffic Simulator<br/>baseline + spikes]
        PRED[Predictive Congestion<br/>exp-smoothing]
        QPSO[QPSO Solver<br/>SPV-encoded, constraint-penalized]
        LOCAL[Local Search<br/>2-opt polish]
        BASE[Baselines<br/>Dijkstra + GA]
        STAG[Stagnation Control<br/>diversity monitor + reinit]
        EXP[Explainability Engine<br/>±20% perturbation, confidence score]
    end

    subgraph Data["Data Layer"]
        SYN[Synthetic Graph Generator]
        OSM[OSM Loader]
    end

    MODE --> QPSO
    INCIDENT --> SIM
    SIM --> GRAPH
    PRED --> GRAPH
    GRAPH --> QPSO
    QPSO --> STAG
    STAG --> QPSO
    QPSO --> LOCAL
    LOCAL --> EXP
    EXP --> EXPLAIN
    LOCAL --> MAP
    BASE --> BENCH
    QPSO --> BENCH
    SYN --> GRAPH
    OSM --> GRAPH
```

## Data flow narrative (for the PPT slide)
1. **Input**: graph loaded (synthetic or real OSM area) + vehicle/demand data.
2. **Traffic state**: simulator sets baseline congestion; incident button or predictive
   layer can modify edge weights live.
3. **Optimization**: QPSO (with SPV decoding + stagnation control) searches for the
   best route set under the active mode's weights; 2-opt polishes the result.
4. **On traffic change**: QPSO is warm-started from its last swarm state, not restarted —
   this is the "incremental re-optimization" claim for the live demo.
5. **Explainability**: top-2 candidate route sets are re-scored under ±20% congestion
   perturbation to produce a confidence score and a plain-language reason.
6. **Output**: routes rendered on the map; benchmark tab shows QPSO vs. baselines.
