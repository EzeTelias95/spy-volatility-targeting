"""Capacity of the strategies against the dollar volume of SPY."""

import pandas as pd

from backtest.targeting import TRADING_DAYS
from data.download.snapshot import naive_index, verified_snapshot_paths

ADV_WINDOW = 21
PARTICIPATION = 0.01
REFERENCE_AUM = 1e9


def spy_dollar_volume() -> pd.Series:
    """Daily dollar volume of SPY from the raw snapshot."""
    snapshot = pd.read_parquet(verified_snapshot_paths()["spy"])
    return pd.Series(
        (snapshot["Volume"] * snapshot["Close"]).to_numpy(),
        index=naive_index(snapshot.index).normalize(),
        name="dollar_volume",
    )


def average_dollar_volume(
    dollar_volume: pd.Series, window: int = ADV_WINDOW
) -> pd.Series:
    """Trailing average dollar volume over the window ending the day before.

    A ledger row holds the trade made at the previous close, so the average
    aligned with a row covers the window ending on the trade day.
    """
    if window < 1:
        raise ValueError("window must be positive")
    return dollar_volume.rolling(window).mean().shift(1)


def capacity_table(
    turnover: pd.DataFrame,
    dollar_volume: pd.Series,
    participation: float = PARTICIPATION,
    reference_aum: float = REFERENCE_AUM,
) -> pd.DataFrame:
    """Capital at which each strategy's trades reach a share of SPY volume.

    On every day a strategy trades, the average dollar volume times the
    participation share over the turnover is the capital whose trade would
    take that share of the volume. The table reports the median, the fifth
    percentile and the minimum of that capital over trade days, plus the share
    of the volume a reference capital would take on its median and worst
    trade day. The first day builds the initial position and is left out. The
    volume is the average aligned with the ledger row, the window ending on
    the trade day.
    """
    if not 0.0 < participation <= 1.0:
        raise ValueError("participation must be in (0, 1]")
    if reference_aum <= 0.0:
        raise ValueError("reference_aum must be positive")
    turnover = turnover.iloc[1:].dropna(how="all")
    if turnover.empty:
        raise ValueError("turnover must cover days after the initial position")
    volume = dollar_volume.reindex(turnover.index)
    if volume.isna().any():
        raise ValueError("dollar volume must cover every strategy date")
    rows = {}
    for name in turnover.columns:
        trades = turnover[name].loc[turnover[name] > 0.0]
        if trades.empty:
            raise ValueError(f"{name} never trades after the initial position")
        capacity = participation * volume.loc[trades.index] / trades
        reference_share = reference_aum * trades / volume.loc[trades.index]
        rows[name] = {
            "trade_days_per_year": len(trades) / len(turnover) * TRADING_DAYS,
            "median_trade_size": trades.median(),
            "capacity_median": capacity.median(),
            "capacity_p05": capacity.quantile(0.05),
            "capacity_min": capacity.min(),
            "reference_share_median": reference_share.median(),
            "reference_share_max": reference_share.max(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")
