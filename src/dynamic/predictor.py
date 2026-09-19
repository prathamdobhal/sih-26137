"""
Congestion forecasting — Holt's linear trend method (double exponential
smoothing), hand-implemented for full transparency.

Why Holt's and not plain exponential smoothing: plain SES only smooths the
current observation, it has no notion of trend, so it can't extrapolate
forward — it would only ever tell you "congestion is currently X," which is
reactive, not predictive. Holt's method tracks a level AND a trend, so it can
answer "at the current rate of increase, congestion will hit threshold Y in Z
steps" — which is the actual claim behind "predictive pre-emptive rerouting."
"""


class HoltForecaster:
    """One instance per edge (or reuse across edges by re-fitting each time —
    kept per-edge here since each road's congestion trend is independent)."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        self.alpha = alpha  # level smoothing factor
        self.beta = beta    # trend smoothing factor
        self.level = None
        self.trend = 0.0
        self.history = []

    def update(self, value: float):
        self.history.append(value)
        if self.level is None:
            self.level = value
            return
        prev_level = self.level
        self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

    def forecast(self, steps_ahead: int = 1) -> float:
        if self.level is None:
            return 0.0
        value = self.level + steps_ahead * self.trend
        return max(0.0, min(0.9, value))  # clamp to valid congestion range

    def is_rising_toward(self, threshold: float, within_steps: int = 3) -> bool:
        """The actual pre-emptive trigger: will this edge cross `threshold`
        within `within_steps`, given its current trend?"""
        if self.level is None or self.trend <= 0:
            return False
        return self.forecast(within_steps) >= threshold
