"""Trailing mean of realized variance, the no-model benchmark."""

from collections import deque

import numpy as np
import pandas as pd

from .base import VolatilityModel

WINDOW = 21


class TrailingVol(VolatilityModel):
    """Square root of the mean realized variance over the last WINDOW days.

    Nothing is estimated: it isolates how much of the strategy comes from
    forecasting volatility rather than from volatility targeting itself. fit
    only refills the buffer from the tail of df, so it matches update() having
    been called for every row up to t. The forecast is flat across horizons, so
    horizon only labels the day it is stored under.
    """

    def __init__(self, window: int = WINDOW, horizon: int = 1):
        self.window = window
        self.horizon = horizon
        self.variances = deque(maxlen=window)

    def fit(self, df: pd.DataFrame) -> None:
        tail = df["realized_vol_gk"].to_numpy(dtype=float)[-self.window :]
        self.variances = deque(tail**2, maxlen=self.window)

    def update(self, row: pd.Series) -> None:
        self.variances.append(float(row["realized_vol_gk"]) ** 2)

    def predict(self) -> float:
        if not self.variances:
            raise RuntimeError("fit must be called before predict")
        return float(np.sqrt(np.mean(self.variances)))
