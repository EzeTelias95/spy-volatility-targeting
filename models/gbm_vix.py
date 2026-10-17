"""Gradient boosting on log variance with VIX predictors."""

import numpy as np
import pandas as pd

from data.realized_vol import PARKINSON_SCALE

from .gbm import (
    RET_WINDOWS,
    SEMIVARIANCE_FLOOR,
    VAR_WINDOWS,
    GradientBoosting,
    _features,
)

TRADING_DAYS = 252
VIX_SCALE = 100.0
VIX_FEATURES = ("vix_var", "vix_change_1", "vix_premium")
FEATURES = (
    *(f"mean_{w}" for w in VAR_WINDOWS),
    "lag_1",
    "ret",
    *(f"ret_sum_{w}" for w in RET_WINDOWS),
    "semivariance",
    "log_range",
    *VIX_FEATURES,
)
DAILY = ("variance", "ret", "parkinson", "vix")


class GradientBoostingVIX(GradientBoosting):
    """The base gradient boosting model plus three VIX predictors.

    vix_var is the VIX quote as a log daily variance, on the same scale as the
    other volatility features; vix_change_1 is its daily change, stationary
    where the level is not; vix_premium is vix_var minus the log of the one day
    realized variance, the variance risk premium the base model cannot see.
    Protocol, tuning grid, embargo and lognormal correction are inherited
    unchanged, so any difference against the base model comes from the
    predictors alone.
    """

    FEATURES = FEATURES
    DAILY = DAILY

    def daily(self, df: pd.DataFrame) -> pd.DataFrame:
        log_hl = np.log(df["high"] / df["low"])
        return pd.DataFrame(
            {
                "variance": df["realized_vol_gk"].astype(float) ** 2,
                "ret": df["log_ret"].astype(float),
                "parkinson": (log_hl**2 / PARKINSON_SCALE).clip(
                    lower=SEMIVARIANCE_FLOOR
                ),
                "vix": df["vix"].astype(float),
            }
        )

    def features(self, daily: pd.DataFrame) -> pd.DataFrame:
        columns = _features(daily)
        vix_var = np.log((daily["vix"] / VIX_SCALE) ** 2 / TRADING_DAYS)
        columns["vix_var"] = vix_var
        columns["vix_change_1"] = vix_var.diff()
        columns["vix_premium"] = vix_var - columns["mean_1"]
        return columns[list(FEATURES)]

    def update(self, row: pd.Series) -> None:
        log_hl = float(np.log(row["high"] / row["low"]))
        self.days.append(
            (
                float(row["realized_vol_gk"]) ** 2,
                float(row["log_ret"]),
                max(log_hl**2 / PARKINSON_SCALE, SEMIVARIANCE_FLOOR),
                float(row["vix"]),
            )
        )
