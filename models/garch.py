"""GARCH(1,1) with a constant mean and Student-t innovations."""

import numpy as np
import pandas as pd
from arch import arch_model

from .base import VolatilityModel

# arch converges reliably on returns in percent, not in decimals.
SCALE = 100.0


class GARCH(VolatilityModel):
    """GARCH(1,1) fitted by arch, filtered by hand between refits.

    Returns are scaled by SCALE for the optimiser and the forecast is scaled
    back to daily volatility. arch cannot filter with parameters held fixed, so
    update runs the conditional variance recursion from the fitted omega, alpha
    and beta. The state is always the conditional variance of the next day, so
    fit leaves it where update would have, without ingesting the refit day twice.
    A horizon beyond one iterates the unconditional recursion from that state,
    the closed form of the multi-step GARCH(1,1) forecast.
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.params = None
        self.variance = None

    def fit(self, df: pd.DataFrame) -> None:
        returns = df["log_ret"].astype(float) * SCALE
        result = arch_model(
            returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t"
        ).fit(disp="off")
        self.params = result.params
        self.variance = float(
            result.forecast(horizon=1, reindex=False).variance.iloc[0, 0]
        )

    def update(self, row: pd.Series) -> None:
        if self.variance is None:
            raise RuntimeError("fit must be called before update")
        residual = float(row["log_ret"]) * SCALE - self.params["mu"]
        self.variance = (
            self.params["omega"]
            + self.params["alpha[1]"] * residual**2
            + self.params["beta[1]"] * self.variance
        )

    def predict(self) -> float:
        variance = self.variance
        persistence = self.params["alpha[1]"] + self.params["beta[1]"]
        for _ in range(self.horizon - 1):
            variance = self.params["omega"] + persistence * variance
        return float(np.sqrt(variance)) / SCALE
