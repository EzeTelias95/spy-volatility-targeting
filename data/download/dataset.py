"""Build the shared dataset from a verified snapshot."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..realized_vol import realized_vol_gk
from .config import CACHE_DIR, DATASET_PATH, END, SCHEMA_COLUMNS, START, DataError
from .snapshot import naive_index, verified_snapshot_paths

log = logging.getLogger(__name__)

MAX_RF_STALE_DAYS = 5


def load_spy(path: Path) -> pd.DataFrame:
    spy = pd.read_parquet(path)
    spy.index = naive_index(spy.index).normalize()
    spy = spy.sort_index().loc[START:END]

    renames = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Dividends": "dividend",
    }
    missing = set(renames) - set(spy.columns)
    if missing:
        raise DataError(f"SPY snapshot missing columns: {sorted(missing)}")
    df = spy[list(renames)].rename(columns=renames)

    price_cols = ["open", "high", "low", "close", "adj_close"]
    if not (df[price_cols] > 0).all().all():
        raise DataError("SPY snapshot has NaN or non-positive prices")
    df["dividend"] = df["dividend"].fillna(0.0)
    return df


def check_ohlc_sanity(df: pd.DataFrame) -> None:
    """Refuse to build on inconsistent OHLC rows.

    Dropping them would leave the next day's return spanning two days while
    still labelled daily, and its overnight term spanning two nights.
    """
    valid = (df["high"] >= df[["open", "close"]].max(axis=1)) & (
        df["low"] <= df[["open", "close"]].min(axis=1)
    )
    bad = df.index[~valid]
    if len(bad) > 0:
        raise DataError(
            f"OHLC sanity check failed on {len(bad)} rows: "
            f"{[d.date().isoformat() for d in bad]}; "
            "inspect the snapshot and decide explicitly"
        )
    log.info("OHLC sanity check passed on %d rows", len(df))


def load_rf(path: Path, trading_index: pd.DatetimeIndex) -> pd.Series:
    """Daily cash return, already lagged one day.

    rf[t] is the return accrued from t-1 to t, priced off the DGS3MO yield
    known at t-1. Consumers use it as is and must not shift it again.
    """
    fred = pd.read_csv(path, na_values=".")
    date_col, value_col = fred.columns[0], fred.columns[1]
    yields = (
        pd.Series(
            fred[value_col].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(fred[date_col])),
        )
        .dropna()
        .sort_index()
    )

    daily = (1.0 + yields / 100.0) ** (1.0 / 252.0) - 1.0
    aligned = daily.reindex(trading_index, method="ffill", limit=MAX_RF_STALE_DAYS)
    if aligned.isna().any():
        gaps = aligned.index[aligned.isna()]
        raise DataError(
            f"no DGS3MO value within {MAX_RF_STALE_DAYS} trading days for "
            f"{len(gaps)} dates, first {gaps[0].date()}"
        )
    filled = int((~trading_index.isin(daily.index)).sum())
    log.info("rf: %d trading days forward-filled from a prior DGS3MO value", filled)
    return aligned.shift(1)


def load_vix(path: Path, trading_index: pd.DatetimeIndex) -> pd.Series:
    """VIX close on the SPY trading calendar.

    The two calendars agree except for rare CBOE-only closures, so a missing
    quote is carried forward from the prior session rather than interpolated.
    """
    vix = pd.read_parquet(path)
    vix.index = naive_index(vix.index).normalize()
    if "Close" not in vix.columns:
        raise DataError("VIX snapshot missing column: ['Close']")
    quotes = vix["Close"].astype(float).sort_index()
    if not (quotes > 0).all():
        raise DataError("VIX snapshot has NaN or non-positive quotes")
    aligned = quotes.reindex(trading_index, method="ffill", limit=MAX_RF_STALE_DAYS)
    if aligned.isna().any():
        gaps = aligned.index[aligned.isna()]
        raise DataError(
            f"no VIX quote within {MAX_RF_STALE_DAYS} trading days for "
            f"{len(gaps)} dates, first {gaps[0].date()}"
        )
    filled = int((~trading_index.isin(quotes.index)).sum())
    log.info("vix: %d trading days forward-filled from a prior quote", filled)
    return aligned


def build() -> None:
    paths = verified_snapshot_paths()
    df = load_spy(paths["spy"])
    check_ohlc_sanity(df)

    df["ret"] = df["adj_close"].pct_change()
    df["log_ret"] = np.log(df["adj_close"]).diff()
    df["rf"] = load_rf(paths["dgs3mo"], df.index)
    df["realized_vol_gk"] = realized_vol_gk(df)
    df["vix"] = load_vix(paths["vix"], df.index)
    df = df.iloc[1:]

    df = df[SCHEMA_COLUMNS]
    df.index.name = "date"
    if not np.isfinite(df.to_numpy()).all():
        raise DataError("built dataset contains NaN or inf")
    if not (df.index.is_monotonic_increasing and df.index.is_unique):
        raise DataError("built dataset index is not strictly increasing")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH)
    log.info(
        "dataset written: %s, %d rows, %s to %s, %d dividend dates",
        DATASET_PATH,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
        int((df["dividend"] > 0).sum()),
    )
