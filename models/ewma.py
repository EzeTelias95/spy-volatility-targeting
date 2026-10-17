"""RiskMetrics EWMA variance, the no-estimation baseline."""

import numpy as np
import pandas as pd

from .base import VolatilityModel

# RiskMetrics daily value, ~11-day half-life. Fixed, not estimated.
LAMBDA = 0.94


class EWMA(VolatilityModel):
    """EWMA variance on squared log returns with lambda fixed at 0.94.

    Nothing is estimated, so fit only replays the recursion. The seed is the
    sample variance of the first training window and is kept across refits,
    which keeps fit(df up to t) equal to update() called for every row up to t.
    The forecast is flat across horizons, so horizon only labels the day it is
    stored under.
    """

    def __init__(self, lam: float = LAMBDA, horizon: int = 1):
        self.lam = lam
        self.horizon = horizon
        self.seed = None
        self.variance = None

    def fit(self, df: pd.DataFrame) -> None:
        returns = df["log_ret"].to_numpy(dtype=float)
        if self.seed is None:
            self.seed = float(np.var(returns, ddof=1))
        variance = self.seed
        for r in returns:
            variance = self.lam * variance + (1.0 - self.lam) * r * r
        self.variance = variance

    def update(self, row: pd.Series) -> None:
        if self.variance is None:
            raise RuntimeError("fit must be called before update")
        r = float(row["log_ret"])
        self.variance = self.lam * self.variance + (1.0 - self.lam) * r * r

    def predict(self) -> float:
        return float(np.sqrt(self.variance))
