"""HAR-RV on log variance."""

from collections import deque

import numpy as np
import pandas as pd

from .base import VolatilityModel

WINDOWS = (1, 5, 22)
TRAIN_FRACTION = 0.8
EMBARGO = max(WINDOWS)
NAMES = ("const", "daily", "weekly", "monthly")


class HAR(VolatilityModel):
    """Log variance on the log of its 1, 5 and 22 day means.

    The windows end on day t, so every regressor is known at the close of t and
    the target is the log variance horizon days ahead, a direct regression as in
    Corsi (2009) for each of his horizons. Exponentiating a log forecast
    returns a median rather than a mean, so predict adds half the residual
    variance. That variance is measured out of sample on the last 20% of the
    training window, behind an embargo as long as the longest window plus the
    extra target days of a longer horizon, so the two sides share no raw day.
    It is the same split the GBM uses for its own correction, which keeps the
    procedure identical for both models in logs.
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.params = None
        self.residual_variance = None
        self.variances = deque(maxlen=EMBARGO)

    def fit(self, df: pd.DataFrame) -> None:
        variance = df["realized_vol_gk"].astype(float) ** 2
        design = _design(variance, self.horizon)
        cut = int(len(design) * TRAIN_FRACTION)
        x_train, y_train = _matrices(design.iloc[:cut])
        x_valid, y_valid = _matrices(design.iloc[cut + EMBARGO + self.horizon - 1 :])
        split = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        self.residual_variance = float(np.var(y_valid - x_valid @ split, ddof=1))
        x, y = _matrices(design)
        self.params = pd.Series(np.linalg.lstsq(x, y, rcond=None)[0], index=NAMES)
        self.variances = deque(variance.to_numpy()[-EMBARGO:], maxlen=EMBARGO)

    def update(self, row: pd.Series) -> None:
        self.variances.append(float(row["realized_vol_gk"]) ** 2)

    def predict(self) -> float:
        if self.params is None:
            raise RuntimeError("fit must be called before predict")
        buffer = np.asarray(self.variances)
        features = np.array([1.0] + [np.log(buffer[-w:].mean()) for w in WINDOWS])
        log_variance = features @ self.params.to_numpy() + self.residual_variance / 2.0
        return float(np.sqrt(np.exp(log_variance)))


def _design(variance: pd.Series, horizon: int = 1) -> pd.DataFrame:
    columns = {f"mean_{w}": np.log(variance.rolling(w).mean()) for w in WINDOWS}
    columns["target"] = np.log(variance).shift(-horizon)
    return pd.DataFrame(columns).dropna()


def _matrices(design: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = design[[f"mean_{w}" for w in WINDOWS]].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(design)), features])
    return x, design["target"].to_numpy(dtype=float)
