"""
Traffic simulator — sets baseline congestion on a graph (synthetic or OSM-loaded),
evolves it over simulated time, and supports the "Inject Incident" demo button.

Works identically on both synthetic.py and osm_loader.py graphs since both expose
the same edge attributes (length_m, speed_kph, congestion).
"""

import networkx as nx
import numpy as np


class TrafficSimulator:
    def __init__(self, G: nx.DiGraph, seed: int = 42):
        self.G = G
        self.rng = np.random.default_rng(seed)
        self._t = 0  # simulated time step
        self._init_baseline_congestion()

    def _init_baseline_congestion(self):
        """
        Assigns each edge a baseline congestion level using a simple archetype model:
        a handful of 'arterial' edges (high base load) and the rest 'local' roads
        (low base load) — mimics how real cities have a few busy corridors and many
        quiet side streets, which makes the demo's incident injection more visually
        meaningful (blocking an arterial matters more than blocking a side street).
        """
        edges = list(self.G.edges())
        n_arterial = max(1, len(edges) // 10)
        arterial_idx = self.rng.choice(len(edges), size=n_arterial, replace=False)
        arterial_edges = {edges[i] for i in arterial_idx}
        for u, v in edges:
            is_arterial = (u, v) in arterial_edges
            base = self.rng.uniform(0.25, 0.45) if is_arterial else self.rng.uniform(0.0, 0.15)
            self.G[u][v]["congestion"] = float(base)
            self.G[u][v]["is_arterial"] = is_arterial
            self.G[u][v]["_base_congestion"] = float(base)

    def step(self, dt: int = 1):
        """
        Advance simulated time. Congestion drifts with small random noise around
        each edge's baseline — call this repeatedly to simulate normal traffic
        fluctuation between optimization runs.
        """
        self._t += dt
        for u, v in self.G.edges():
            base = self.G[u][v]["_base_congestion"]
            noise = self.rng.normal(0, 0.03)
            new_val = np.clip(base + noise, 0.0, 0.9)
            self.G[u][v]["congestion"] = float(new_val)

    def inject_incident(self, u, v, severity: float = 0.85, radius_hops: int = 1):
        """
        The 'Inject Incident' demo button. Spikes congestion on edge (u, v) and
        partially on its neighbors (simulating a jam backing up onto nearby roads).
        Returns the list of affected edges so the caller can highlight them on the map.
        """
        if not self.G.has_edge(u, v):
            raise ValueError(f"No edge ({u}, {v}) in graph")

        affected = [(u, v)]
        self.G[u][v]["congestion"] = float(severity)

        # Spread a smaller spike to edges within `radius_hops` of the incident
        frontier = {v}
        for hop in range(radius_hops):
            next_frontier = set()
            decay = severity * (0.5 ** (hop + 1))
            for node in frontier:
                for nb in self.G.successors(node):
                    if (node, nb) not in affected:
                        current = self.G[node][nb]["congestion"]
                        self.G[node][nb]["congestion"] = float(max(current, decay))
                        affected.append((node, nb))
                        next_frontier.add(nb)
            frontier = next_frontier

        return affected

    def clear_incident(self, edges: list):
        """Reset the given edges back to their baseline congestion."""
        for u, v in edges:
            self.G[u][v]["congestion"] = self.G[u][v]["_base_congestion"]

    def travel_time(self, u, v) -> float:
        """
        travel_time_ij(t) = distance_ij / (speed_limit_ij * (1 - congestion_ij(t)))
        — matches docs/formulation.md Section 1 exactly.
        """
        data = self.G[u][v]
        length_m = data["length_m"]
        speed_kph = max(data["speed_kph"], 1.0)
        speed_mps = speed_kph * 1000 / 3600
        congestion = min(data["congestion"], 0.9)  # cap to avoid div-by-zero
        return length_m / (speed_mps * (1 - congestion))
