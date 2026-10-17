"""Temporal alignment and market returns for strategy backtests."""

from .returns import build_instrument_returns, prepare_backtest_data
from .targeting import BacktestConfig, build_ledger
from .timing import apply_execution_lag

__all__ = [
    "apply_execution_lag",
    "BacktestConfig",
    "build_ledger",
    "build_instrument_returns",
    "prepare_backtest_data",
]
