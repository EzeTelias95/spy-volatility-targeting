"""Regenerate every cached forecast, strategy table and report the thesis cites."""

import logging

from backtest.targeting import BacktestConfig
from data.download.dataset import build
from evaluation import run_forecasts, run_oracle, run_strategies
from evaluation.run_forecasts import ROBUSTNESS_CUTS, TRAIN_END, two_day_names
from reporting.__main__ import main as write_report

STRATEGY_CONFIGS = [
    BacktestConfig(),
    BacktestConfig(transaction_cost_bps=5.0),
    BacktestConfig(transaction_cost_bps=10.0),
    BacktestConfig(target_vol_annual=0.15),
    BacktestConfig(target_vol_annual=0.20),
    BacktestConfig(apply_cap=False),
    BacktestConfig(sizing_exponent=2.0),
    BacktestConfig(vol_cutoff_annual=0.30),
    BacktestConfig(vol_cutoff_annual=0.20),
    BacktestConfig(forecast_horizon=run_forecasts.TWO_DAY_HORIZON),
    BacktestConfig(decision_frequency="monthly"),
    *[
        BacktestConfig(decision_frequency="weekly", weekly_anchor=day)
        for day in ("MON", "TUE", "WED", "THU", "FRI")
    ],
]


def main() -> None:
    build()
    for train_end in (TRAIN_END, *ROBUSTNESS_CUTS):
        run_forecasts.run(train_end)
    run_forecasts.run(TRAIN_END, two_day_names())
    for config in STRATEGY_CONFIGS:
        run_strategies.run(TRAIN_END, config=config)
    run_strategies.frictions(TRAIN_END, BacktestConfig())
    run_oracle.run(TRAIN_END)
    for train_end in ROBUSTNESS_CUTS:
        run_strategies.run(train_end, config=BacktestConfig())
    write_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
