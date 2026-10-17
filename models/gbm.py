"""Gradient boosting on log variance with LightGBM."""

from collections import deque

import lightgbm as lgb
import numpy as np
import pandas as pd

from data.realized_vol import PARKINSON_SCALE

from .base import VolatilityModel

VAR_WINDOWS = (1, 5, 22)
RET_WINDOWS = (5, 22)
SEMIVARIANCE_FLOOR = 1e-8
TRAIN_FRACTION = 0.8
EMBARGO = 22
BUFFER = 22
GRID = tuple((rate, leaves) for rate in (0.025, 0.05, 0.1) for leaves in (3, 7, 15, 31))
MAX_TREES = 2000
EARLY_STOPPING = 50
BASE_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 2,
}
FEATURES = (
    *(f"mean_{w}" for w in VAR_WINDOWS),
    "lag_1",
    "ret",
    *(f"ret_sum_{w}" for w in RET_WINDOWS),
    "semivariance",
    "log_range",
)
DAILY = ("variance", "ret", "parkinson")


class GradientBoosting(VolatilityModel):
    """LightGBM regression on the log variance horizon days ahead.

    Nine features, the survivors of a validation-scored trim of a wider
    literature battery: the three HAR means, so the model nests the HAR's
    information, the previous day's log variance, the signed daily return and
    its 5 and 22 day sums, downside semivariance and the Parkinson range, all
    built from windows ending on day t. fit tunes learning rate and num_leaves
    on a single chronological 80/20 split with an embargo as long as the
    longest feature window plus the extra target days of a longer horizon, so
    the two sides share no raw day, keeping the tree count from early stopping, then
    refits on the full window with that count. Exponentiating a log forecast
    returns a median rather than a mean, so predict adds half the residual
    variance, measured on the validation side of the same split as in the HAR.
    """

    FEATURES = FEATURES
    DAILY = DAILY

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.model = None
        self.residual_variance = None
        self.best_iteration = None
        self.days = deque(maxlen=BUFFER)

    def daily(self, df: pd.DataFrame) -> pd.DataFrame:
        return _daily(df)

    def features(self, daily: pd.DataFrame) -> pd.DataFrame:
        return _features(daily)

    def design(self, daily: pd.DataFrame) -> pd.DataFrame:
        design = self.features(daily)
        design["target"] = np.log(daily["variance"]).shift(-self.horizon)
        return design.dropna()

    def fit(self, df: pd.DataFrame) -> None:
        daily = self.daily(df)
        design = self.design(daily)
        x_train, y_train, x_valid, y_valid = _split(design, self.FEATURES, self.horizon)
        best = None
        for rate, leaves in GRID:
            params = BASE_PARAMS | {"learning_rate": rate, "num_leaves": leaves}
            booster = lgb.train(
                params,
                lgb.Dataset(x_train, label=y_train),
                num_boost_round=MAX_TREES,
                valid_sets=[lgb.Dataset(x_valid, label=y_valid)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            score = booster.best_score["valid_0"]["l2"]
            if best is None or score < best[0]:
                best = (score, params, booster)
        _, params, booster = best
        residuals = y_valid - booster.predict(x_valid)
        self.residual_variance = float(np.var(residuals, ddof=1))
        self.best_iteration = booster.best_iteration
        x, y = _matrices(design, self.FEATURES)
        self.model = lgb.train(
            params, lgb.Dataset(x, label=y), num_boost_round=self.best_iteration
        )
        self.days = deque(
            daily.tail(BUFFER).itertuples(index=False, name=None), maxlen=BUFFER
        )

    def update(self, row: pd.Series) -> None:
        log_hl = float(np.log(row["high"] / row["low"]))
        self.days.append(
            (
                float(row["realized_vol_gk"]) ** 2,
                float(row["log_ret"]),
                max(log_hl**2 / PARKINSON_SCALE, SEMIVARIANCE_FLOOR),
            )
        )

    def predict(self) -> float:
        if self.model is None:
            raise RuntimeError("fit must be called before predict")
        daily = pd.DataFrame(list(self.days), columns=self.DAILY)
        row = self.features(daily).iloc[-1:].to_numpy(dtype=float)
        log_variance = float(self.model.predict(row)[0]) + self.residual_variance / 2.0
        return float(np.sqrt(np.exp(log_variance)))

    @property
    def params(self) -> pd.Series:
        chosen = self.model.params
        return pd.Series(
            {
                "learning_rate": chosen["learning_rate"],
                "num_leaves": chosen["num_leaves"],
                "trees": self.best_iteration,
            }
        )


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    log_hl = np.log(df["high"] / df["low"])
    return pd.DataFrame(
        {
            "variance": df["realized_vol_gk"].astype(float) ** 2,
            "ret": df["log_ret"].astype(float),
            "parkinson": (log_hl**2 / PARKINSON_SCALE).clip(lower=SEMIVARIANCE_FLOOR),
        }
    )


def _features(daily: pd.DataFrame) -> pd.DataFrame:
    variance, ret = daily["variance"], daily["ret"]
    columns = {f"mean_{w}": np.log(variance.rolling(w).mean()) for w in VAR_WINDOWS}
    columns["lag_1"] = np.log(variance).shift(1)
    columns["ret"] = ret
    columns |= {f"ret_sum_{w}": ret.rolling(w).sum() for w in RET_WINDOWS}
    downside = (ret.where(ret < 0.0, 0.0) ** 2).rolling(max(RET_WINDOWS)).mean()
    columns["semivariance"] = np.log(downside.clip(lower=SEMIVARIANCE_FLOOR))
    columns["log_range"] = np.log(daily["parkinson"])
    return pd.DataFrame(columns)[list(FEATURES)]


def _split(
    design: pd.DataFrame,
    features: tuple[str, ...],
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cut = int(len(design) * TRAIN_FRACTION)
    start = cut + EMBARGO + horizon - 1
    x, y = _matrices(design, features)
    return x[:cut], y[:cut], x[start:], y[start:]


def _matrices(
    design: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        design[list(features)].to_numpy(dtype=float),
        design["target"].to_numpy(dtype=float),
    )
