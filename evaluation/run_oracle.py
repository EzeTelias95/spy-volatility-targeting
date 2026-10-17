"""Cache the look-ahead ceilings and the sign-randomized counterfactual.

Both oracles size a day with volatility observed on that day or later, so they
run with the same-close order and only as a diagnostic. Sign randomization
keeps every absolute move and every forecast and breaks only the link between
the size of a move and its direction, so what each sizer keeps of its excess
over the passive is risk control and what it loses is return timing. The
draws take minutes, so the tables land next to the base strategy tables and
the report reads them from there.
"""

import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from backtest.targeting import BacktestConfig, build_ledger
from evaluation.oracle import daily_oracle, regime_oracle, sign_randomized_market
from evaluation.run_forecasts import TRAIN_END
from evaluation.run_strategies import BENCHMARK, _inputs, commit, strategy_run_dir
from evaluation.strategy import sharpe_ratio

ORACLE_CONFIG = BacktestConfig(execution_lag=1)
ORACLES = {"oracle_daily": "daily", "oracle_regime": "regime"}
QUINTILES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
QUINTILE_KEYS = {
    "realized": "oracle_daily",
    "regime": "oracle_regime",
    "gbm": "gbm",
}

log = logging.getLogger(__name__)


def sizers(market_data: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Oracles and model forecasts on the days every sizer covers."""
    regime = regime_oracle(market_data["realized_vol_gk"])
    common = forecasts.index.intersection(regime.index)
    return pd.DataFrame(
        {
            "oracle_daily": daily_oracle(market_data.loc[common, "realized_vol_gk"]),
            "oracle_regime": regime.loc[common],
            **{name: forecasts.loc[common, name] for name in forecasts},
        }
    )


def sharpes(market_data: pd.DataFrame, sizers: pd.DataFrame) -> pd.Series:
    row = {}
    for name, vol in sizers.items():
        ledger = build_ledger(market_data, vol.rename("forecast"), ORACLE_CONFIG)
        ledger = ledger.dropna(subset=["net_return"])
        row[name] = sharpe_ratio(ledger["net_return"], ledger["rf"])
    hold = market_data.loc[sizers.index]
    row[BENCHMARK] = sharpe_ratio(hold["ret"], hold["rf"])
    return pd.Series(row)


def ceiling(
    market_data: pd.DataFrame, sizers: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    actual = sharpes(market_data, sizers)
    rng = np.random.default_rng(seed)
    randomized = pd.DataFrame(
        [
            sharpes(sign_randomized_market(market_data, sizers.index, rng), sizers)
            for _ in range(draws)
        ]
    )
    excess = randomized.sub(randomized[BENCHMARK], axis=0)
    table = pd.DataFrame(
        {
            "sharpe": actual,
            "excess": actual - actual[BENCHMARK],
            "randomized_excess": excess.mean(),
            "randomized_excess_p05": excess.quantile(0.05),
            "randomized_excess_p95": excess.quantile(0.95),
        }
    )
    table["timing"] = table["excess"] - table["randomized_excess"]
    return table.rename_axis("strategy")


def return_by_quintile(market_data: pd.DataFrame, sizers: pd.DataFrame) -> pd.DataFrame:
    """Mean SPY return per quintile of each sorting key, in decimals."""
    ret = market_data.loc[sizers.index, "ret"]
    return pd.DataFrame(
        {
            label: ret.groupby(
                pd.qcut(sizers[key], len(QUINTILES), labels=QUINTILES), observed=True
            ).mean()
            for label, key in QUINTILE_KEYS.items()
        }
    ).rename_axis("quintile")


def run(train_end: str = TRAIN_END, draws: int = 100, seed: int = 42) -> None:
    market_data, forecasts, forecast_manifest = _inputs(train_end)
    keys = sizers(market_data, forecasts)
    log.info(
        "oracle ceiling on %d days to %s, %d draws",
        len(keys),
        keys.index[-1].date(),
        draws,
    )
    directory = strategy_run_dir(train_end, BacktestConfig())
    directory.mkdir(parents=True, exist_ok=True)
    ceiling(market_data, keys, draws, seed).to_csv(directory / "oracle_ceiling.csv")
    return_by_quintile(market_data, keys).to_csv(directory / "oracle_quintiles.csv")
    (directory / "oracle_manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(ORACLE_CONFIG),
                "draws": draws,
                "seed": seed,
                "days": len(keys),
                "first_day": str(keys.index[0].date()),
                "last_day": str(keys.index[-1].date()),
                "commit": commit(),
                "forecast_manifest": forecast_manifest,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
