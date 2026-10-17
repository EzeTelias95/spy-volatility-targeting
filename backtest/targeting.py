"""Volatility targeting and gross SPY-cash accounting."""

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .returns import prepare_backtest_data
from .timing import apply_execution_lag

TRADING_DAYS = 252
BASIS_POINTS_PER_UNIT = 10_000
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Parameters for volatility targeting."""

    target_vol_annual: float = 0.10
    sizing_exponent: float = 1.0
    vol_cutoff_annual: float | None = None
    max_weight: float = 1.5
    funding_spread_annual: float = 0.005
    execution_lag: int = 2
    forecast_horizon: int = 1
    apply_cap: bool = True
    no_trade_band: float = 0.05
    transaction_cost_bps: float = 2.0
    apply_band: bool = True
    apply_costs: bool = True
    initial_capital: float = 1.0
    decision_frequency: str = "daily"
    weekly_anchor: str = "FRI"

    def __post_init__(self) -> None:
        values = {
            "target_vol_annual": self.target_vol_annual,
            "sizing_exponent": self.sizing_exponent,
            "max_weight": self.max_weight,
            "funding_spread_annual": self.funding_spread_annual,
            "no_trade_band": self.no_trade_band,
            "transaction_cost_bps": self.transaction_cost_bps,
            "initial_capital": self.initial_capital,
        }
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.target_vol_annual <= 0.0:
            raise ValueError("target_vol_annual must be positive")
        if self.sizing_exponent <= 0.0:
            raise ValueError("sizing_exponent must be positive")
        if self.vol_cutoff_annual is not None and (
            isinstance(self.vol_cutoff_annual, bool)
            or not isinstance(self.vol_cutoff_annual, Real)
            or not isfinite(self.vol_cutoff_annual)
            or self.vol_cutoff_annual <= 0.0
        ):
            raise ValueError("vol_cutoff_annual must be positive or None")
        if self.max_weight <= 0.0:
            raise ValueError("max_weight must be positive")
        if self.funding_spread_annual < 0.0:
            raise ValueError("funding_spread_annual must be non-negative")
        if self.no_trade_band < 0.0:
            raise ValueError("no_trade_band must be non-negative")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if (
            isinstance(self.execution_lag, bool)
            or not isinstance(self.execution_lag, Integral)
            or self.execution_lag not in {0, 1, 2}
        ):
            raise ValueError("execution_lag must be 0, 1 or 2")
        if (
            isinstance(self.forecast_horizon, bool)
            or not isinstance(self.forecast_horizon, Integral)
            or self.forecast_horizon not in {1, 2}
        ):
            raise ValueError("forecast_horizon must be 1 or 2")
        if self.forecast_horizon == 2 and self.execution_lag != 2:
            raise ValueError("forecast_horizon 2 requires execution_lag 2")
        toggles = {
            "apply_cap": self.apply_cap,
            "apply_band": self.apply_band,
            "apply_costs": self.apply_costs,
        }
        for name, value in toggles.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(
            self.decision_frequency, str
        ) or self.decision_frequency not in {
            "daily",
            "weekly",
            "monthly",
        }:
            raise ValueError("decision_frequency must be daily, weekly or monthly")
        if self.weekly_anchor not in WEEKDAYS:
            raise ValueError("weekly_anchor must be one of MON, TUE, WED, THU, FRI")
        if self.weekly_anchor != "FRI" and self.decision_frequency != "weekly":
            raise ValueError("weekly_anchor applies to weekly decisions")


def build_ledger(
    market_data: pd.DataFrame,
    forecasts: pd.Series,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Build daily portfolio state, returns, costs and values.

    The raw weight is target over forecast volatility raised to the sizing
    exponent: one is the headline rule, two is the variance scaling of Moreira
    and Muir, which agrees with the headline rule at the target and moves
    twice as far in log terms away from it. With a volatility cutoff the raw
    weight is zero on days whose annualized forecast exceeds it, a return
    timing rule rather than volatility targeting. Rows without an executable weight
    stay NaN rather than extending the curves silently: the first row under
    the headline lag of two, the final row under the look-ahead lag of zero.
    Trading starts on the first available weight.
    """
    config = BacktestConfig() if config is None else config
    ledger = prepare_backtest_data(market_data, forecasts)
    target_vol_daily = config.target_vol_annual / sqrt(TRADING_DAYS)
    raw_weight = (target_vol_daily / ledger["forecast_vol"]) ** config.sizing_exponent
    if config.vol_cutoff_annual is not None:
        cutoff_daily = config.vol_cutoff_annual / sqrt(TRADING_DAYS)
        raw_weight = raw_weight.where(ledger["forecast_vol"] <= cutoff_daily, 0.0)
    target_weight = (
        raw_weight.clip(lower=0.0, upper=config.max_weight)
        if config.apply_cap
        else raw_weight
    )
    return_weight = apply_execution_lag(
        target_weight, config.execution_lag, config.forecast_horizon
    )
    scheduled_rebalance = (
        apply_execution_lag(
            _scheduled_rebalances(
                market_data.index,
                target_weight.index,
                config.decision_frequency,
                config.forecast_horizon,
                config.weekly_anchor,
            ).astype(float),
            config.execution_lag,
            config.forecast_horizon,
        )
        .fillna(0.0)
        .astype(bool)
    )
    funding_spread_daily = (1.0 + config.funding_spread_annual) ** (
        1.0 / TRADING_DAYS
    ) - 1.0
    transaction_cost_rate = (
        config.transaction_cost_bps / BASIS_POINTS_PER_UNIT
        if config.apply_costs
        else 0.0
    )
    drifted_weight, held_weight, turnover = _portfolio_weights(
        return_weight,
        ledger["return_close_to_close"],
        ledger["rf"],
        funding_spread_daily,
        transaction_cost_rate,
        scheduled_rebalance,
        config,
    )
    spy_contribution = held_weight * ledger["return_close_to_close"]
    cash_contribution = _cash_contribution(
        held_weight,
        ledger["rf"],
        funding_spread_daily,
    )
    gross_return = spy_contribution + cash_contribution
    transaction_cost = turnover * transaction_cost_rate
    net_return = gross_return - transaction_cost

    ledger["raw_weight"] = raw_weight
    ledger["target_weight"] = target_weight
    ledger["return_weight"] = return_weight
    ledger["scheduled_rebalance"] = scheduled_rebalance
    ledger["drifted_weight"] = drifted_weight
    ledger["held_weight"] = held_weight
    ledger["turnover"] = turnover
    ledger["spy_contribution"] = spy_contribution
    ledger["cash_contribution"] = cash_contribution
    ledger["transaction_cost"] = transaction_cost
    ledger["gross_return"] = gross_return
    ledger["net_return"] = net_return
    ledger["portfolio_value_gross"] = _portfolio_value(
        gross_return, config.initial_capital
    )
    ledger["portfolio_value_net"] = _portfolio_value(net_return, config.initial_capital)
    return ledger


def _portfolio_weights(
    target_weight: pd.Series,
    spy_return: pd.Series,
    rf: pd.Series,
    funding_spread_daily: float,
    transaction_cost_rate: float,
    scheduled_rebalance: pd.Series,
    config: BacktestConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    drifted = pd.Series(float("nan"), index=target_weight.index, dtype=float)
    held = pd.Series(float("nan"), index=target_weight.index, dtype=float)
    turnover = pd.Series(float("nan"), index=target_weight.index, dtype=float)

    first = int(target_weight.notna().to_numpy().argmax())
    for position, desired in enumerate(target_weight):
        if pd.isna(desired):
            continue
        if position == first:
            before_trade = 0.0
        else:
            before_trade = _drift_weight(
                held.iloc[position - 1],
                spy_return.iloc[position - 1],
                rf.iloc[position - 1],
                funding_spread_daily,
                turnover.iloc[position - 1] * transaction_cost_rate,
            )
        cap_breached = config.apply_cap and before_trade > config.max_weight
        should_trade = (
            position == first
            or scheduled_rebalance.iloc[position]
            and (
                not config.apply_band
                or cap_breached
                or abs(desired - before_trade) > config.no_trade_band
            )
        )
        after_trade = desired if should_trade else before_trade
        drifted.iloc[position] = before_trade
        held.iloc[position] = after_trade
        turnover.iloc[position] = abs(after_trade - before_trade)
    return drifted, held, turnover


def _scheduled_rebalances(
    market_index: pd.DatetimeIndex,
    target_index: pd.DatetimeIndex,
    frequency: str,
    forecast_horizon: int = 1,
    weekly_anchor: str = "FRI",
) -> pd.Series:
    """Flag the forecasts decided on the last trading day of each period.

    Weekly periods end on the anchor weekday, so a weekly schedule decides on
    the anchor or on the last trading day before it when the anchor is a holiday.
    """
    if frequency == "daily":
        return pd.Series(True, index=target_index)
    positions = market_index.get_indexer(target_index)
    if (positions < forecast_horizon).any():
        raise ValueError("forecast dates must have a prior market date")
    decision_dates = market_index.take(positions - forecast_horizon)
    following_dates = market_index.take(positions - forecast_horizon + 1)
    period = f"W-{weekly_anchor}" if frequency == "weekly" else "M"
    scheduled = decision_dates.to_period(period) != following_dates.to_period(period)
    result = pd.Series(scheduled, index=target_index)
    result.iloc[0] = True
    return result


def _drift_weight(
    held_weight: float,
    spy_return: float,
    rf: float,
    funding_spread_daily: float,
    transaction_cost: float,
) -> float:
    cash = _cash_contribution(held_weight, rf, funding_spread_daily)
    gross_return = held_weight * spy_return + cash
    ending_equity = 1.0 + gross_return - transaction_cost
    if ending_equity <= 0.0:
        raise ValueError("portfolio equity must remain positive")
    return held_weight * (1.0 + spy_return) / ending_equity


def _portfolio_value(returns: pd.Series, initial_capital: float) -> pd.Series:
    value = pd.Series(float("nan"), index=returns.index, dtype=float)
    available = returns.notna()
    value.loc[available] = initial_capital * (1.0 + returns.loc[available]).cumprod()
    return value


def _cash_contribution(
    return_weight: float | pd.Series,
    rf: float | pd.Series,
    funding_spread_daily: float,
) -> float | pd.Series:
    cash_weight = np.maximum(1.0 - return_weight, 0.0)
    funding_weight = np.maximum(return_weight - 1.0, 0.0)
    return cash_weight * rf - funding_weight * (rf + funding_spread_daily)
