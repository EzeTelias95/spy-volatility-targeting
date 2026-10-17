"""Interface every volatility model implements."""

from abc import ABC, abstractmethod

import pandas as pd


class VolatilityModel(ABC):
    """Daily volatility forecaster driven by the walk-forward loop.

    Daily sequence: update(row for t); on a refit date fit(df up to t); predict().
    fit rebuilds the internal state from df, leaving it identical to having
    called update for every row up to t, so the refit day is not ingested twice.
    An initial fit on the training window precedes the first update.

    horizon is the number of trading days ahead predict looks. It is 1 unless a
    model is built for the two-day backtest, where the weight decided at the
    close of t earns the return of t+2 and the forecast must be for that day.
    """

    horizon = 1

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Re-estimate parameters on all data up to and including t."""

    @abstractmethod
    def update(self, row: pd.Series) -> None:
        """Ingest one new day and advance the state without re-estimating."""

    @abstractmethod
    def predict(self) -> float:
        """Total daily volatility for t+horizon, not annualized."""
