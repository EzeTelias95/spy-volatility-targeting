"""Run and cache the economic strategy evaluation.

The base case writes the H2 and H3 tables for one configuration. The frictions
entry point writes the attribution table, one layer of friction at a time, the
no-trade band and leverage cap sensitivities and the aligned versus
implementable execution comparison for the same configuration.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from itertools import combinations
from math import sqrt
from pathlib import Path

import pandas as pd

from backtest.targeting import (
    BASIS_POINTS_PER_UNIT,
    TRADING_DAYS,
    WEEKDAYS,
    BacktestConfig,
)
from backtest.targeting import build_ledger as build_targeting_ledger
from data.download.config import DATASET_PATH, MANIFEST_PATH
from evaluation.run_forecasts import (
    MANIFEST_NAME,
    TRAIN_END,
    TWO_DAY_HORIZON,
    TWO_DAY_SUFFIX,
    commit,
    load_forecasts,
    run_dir,
    two_day_names,
)
from evaluation.strategy import (
    evaluate_strategy,
    sharpe_difference_test,
    spanning_test,
)

BENCHMARK = "buy_and_hold"
CACHE_DIR = DATASET_PATH.parent / "strategies"
EXECUTION_LAGS = (0, 1, 2)
COMPARISON_LAGS = (1, 2)
BANDS = (0.0, 0.025, 0.05, 0.10)
CAPS = (1.5, 2.0, 3.0, None)


def strategy_run_dir(
    train_end: str,
    config: BacktestConfig,
    bootstrap_reps: int = 1_000,
    seed: int = 42,
) -> Path:
    """Return a stable cache directory for one complete configuration.

    The name spells out the settings a reader looks for; the digest covers
    every field that differs from the defaults, so two configurations never
    share a directory and adding a field to the config leaves existing
    directories in place.
    """
    defaults = asdict(BacktestConfig())
    identity = {
        "config": {
            name: value
            for name, value in asdict(config).items()
            if value != defaults[name]
        },
        "bootstrap_reps": bootstrap_reps,
        "seed": seed,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    rule = "" if config.sizing_exponent == 1.0 else f"-pow{config.sizing_exponent:g}"
    if config.vol_cutoff_annual is not None:
        rule += f"-cut{config.vol_cutoff_annual * 100:g}pct"
    if config.forecast_horizon != 1:
        rule += f"-h{config.forecast_horizon}"
    frequency = config.decision_frequency
    if config.weekly_anchor != "FRI":
        frequency += f"-{config.weekly_anchor.lower()}"
    label = (
        f"{frequency}-target{config.target_vol_annual * 100:g}pct"
        f"{rule}-lag{config.execution_lag}-{config.transaction_cost_bps:g}bps"
        f"-band{config.no_trade_band * 100:g}pp-{digest}"
    )
    return CACHE_DIR / train_end / label


def build_ledgers(
    market_data: pd.DataFrame,
    forecasts: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every forecast and the passive benchmark through the base case."""
    config = BacktestConfig() if config is None else config
    if config.execution_lag not in COMPARISON_LAGS:
        raise ValueError("strategy execution_lag must be 1 or 2")
    if not isinstance(forecasts, pd.DataFrame) or forecasts.empty:
        raise ValueError("forecasts must be a non-empty DataFrame")
    if not forecasts.columns.is_unique:
        raise ValueError("forecast columns must be unique")
    if BENCHMARK in forecasts.columns:
        raise ValueError(f"forecast name is reserved: {BENCHMARK}")

    ledgers = {
        str(name): _available(
            build_targeting_ledger(market_data, forecasts[name], config)
        )
        for name in forecasts.columns
    }
    ledgers[BENCHMARK] = _buy_and_hold_ledger(market_data, forecasts.index, config)
    return ledgers


def _available(ledger: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows the execution lag leaves without a weight."""
    available = ledger.loc[ledger["return_weight"].notna()]
    if available.empty:
        raise ValueError("no executable weight survives the execution lag")
    return available


def evaluate_ledgers(
    gross_ledgers: dict[str, pd.DataFrame],
    net_ledgers: dict[str, pd.DataFrame],
    config: BacktestConfig,
    bootstrap_reps: int = 1_000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build the gross H2 and net H3 base-case tables."""
    if gross_ledgers.keys() != net_ledgers.keys():
        raise ValueError("gross and net ledgers must contain the same strategies")
    return {
        "gross_metrics": _metrics(gross_ledgers, "gross_return", config),
        "net_metrics": _metrics(net_ledgers, "net_return", config),
        "strategy_diagnostics": _strategy_diagnostics(net_ledgers, config),
        "h2_spanning": _spanning(gross_ledgers, "gross_return"),
        "h2_sharpe": _sharpe_pairs(gross_ledgers, "gross_return", bootstrap_reps, seed),
        "h3_spanning": _spanning(net_ledgers, "net_return"),
        "h3_sharpe": _sharpe_pairs(net_ledgers, "net_return", bootstrap_reps, seed),
    }


def friction_layers(config: BacktestConfig) -> dict[str, BacktestConfig]:
    """Configurations from the look-ahead counterfactual to the full case.

    The three lag rows carry no cap, band or costs: lag 0 applies the weight to
    the day whose volatility it already knows, lag 1 is the same-close order
    the literature assumes, lag 2 is the implementable case. The remaining rows
    add the cap, the band and the costs one at a time on the configured lag.
    """
    bare = replace(config, apply_cap=False, apply_band=False, apply_costs=False)
    layers = {f"lag_{lag}": replace(bare, execution_lag=lag) for lag in EXECUTION_LAGS}
    layers["cap"] = replace(bare, apply_cap=config.apply_cap)
    layers["band"] = replace(layers["cap"], apply_band=config.apply_band)
    layers["costs"] = config
    return layers


def two_day_layer(config: BacktestConfig) -> dict[str, BacktestConfig]:
    """The bare implementable lag sized on the two-day forecast of the return day."""
    bare = replace(config, apply_cap=False, apply_band=False, apply_costs=False)
    return {"two_day": replace(bare, forecast_horizon=TWO_DAY_HORIZON)}


def band_layers(config: BacktestConfig) -> dict[str, BacktestConfig]:
    """The configured case with the no-trade band at each width in BANDS."""
    return {f"band_{band:.3f}": replace(config, no_trade_band=band) for band in BANDS}


def cap_layers(config: BacktestConfig) -> dict[str, BacktestConfig]:
    """The configured case with the leverage cap at each level in CAPS.

    None removes the cap. The band and the costs stay as configured.
    """
    return {
        "cap_none" if cap is None else f"cap_{cap:.1f}": (
            replace(config, apply_cap=False)
            if cap is None
            else replace(config, apply_cap=True, max_weight=cap)
        )
        for cap in CAPS
    }


def layer_metrics(
    market_data: pd.DataFrame,
    forecasts: pd.DataFrame,
    layers: dict[str, BacktestConfig],
) -> pd.DataFrame:
    """Net strategy metrics of every forecast under every layer."""
    if not isinstance(forecasts, pd.DataFrame) or forecasts.empty:
        raise ValueError("forecasts must be a non-empty DataFrame")
    rows = {}
    for layer, config in layers.items():
        for name in forecasts.columns:
            ledger = _available(
                build_targeting_ledger(market_data, forecasts[name], config)
            )
            rows[(layer, str(name))] = evaluate_strategy(
                ledger["net_return"],
                ledger["rf"],
                ledger["turnover"],
                config.target_vol_annual,
            )
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = table.index.set_names(["layer", "strategy"])
    return table


def execution_timing_tables(
    market_data: pd.DataFrame,
    forecasts: pd.DataFrame,
    config: BacktestConfig,
    bootstrap_reps: int = 1_000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Compare aligned and implementable execution with identical frictions."""
    ledgers = {}
    for lag in COMPARISON_LAGS:
        lag_config = replace(config, execution_lag=lag)
        lag_forecasts = forecasts.iloc[1:] if lag == 1 else forecasts
        ledgers[lag] = (
            build_ledgers(
                market_data,
                lag_forecasts,
                replace(lag_config, apply_costs=False),
            ),
            build_ledgers(market_data, lag_forecasts, lag_config),
        )
    common_index = ledgers[2][1][BENCHMARK].index
    tables = {}
    for lag in COMPARISON_LAGS:
        gross, net = ledgers[lag]
        for ledger in (*gross.values(), *net.values()):
            if not ledger.index.equals(common_index):
                raise ValueError("timing comparison must share the return dates")
        evaluated = evaluate_ledgers(
            gross,
            net,
            replace(config, execution_lag=lag),
            bootstrap_reps,
            seed,
        )
        for name, table in evaluated.items():
            tables.setdefault(name, {})[f"lag_{lag}"] = table
    return {
        f"execution_timing_{name}": pd.concat(by_lag, names=["timing"])
        for name, by_lag in tables.items()
    }


def frictions(
    train_end: str = TRAIN_END,
    config: BacktestConfig | None = None,
    bootstrap_reps: int = 1_000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Write the attribution, the band and cap sensitivities and the timing tables.

    The tables land next to the base-case tables of the same configuration and
    bootstrap settings, with their own manifest. The attribution carries an
    two-day row after the lag rows: the implementable lag sized on the cached
    two-day forecasts, so the reader sees what the delay costs and what a
    forecast built for the return day wins back before the other frictions.
    The last attribution layer is the base case itself and matches its net
    table row by row; the lag rows each lose the day their own lag leaves
    without a weight, so they differ by one day at an end of the sample.
    """
    config = BacktestConfig() if config is None else config
    if config.forecast_horizon != 1:
        raise ValueError("frictions are attributed on the one-day forecasts")
    market_data, forecasts, forecast_manifest = _inputs(train_end)
    two_day_forecasts = _inputs(train_end, TWO_DAY_HORIZON)[1]
    layers = friction_layers(config)
    lags = {name: layer for name, layer in layers.items() if name.startswith("lag_")}
    rest = {name: layer for name, layer in layers.items() if name not in lags}
    tables = {
        "friction_attribution": pd.concat(
            [
                layer_metrics(market_data, forecasts, lags),
                layer_metrics(market_data, two_day_forecasts, two_day_layer(config)),
                layer_metrics(market_data, forecasts, rest),
            ]
        ),
        "band_sensitivity": layer_metrics(market_data, forecasts, band_layers(config)),
        "cap_sensitivity": layer_metrics(market_data, forecasts, cap_layers(config)),
        **execution_timing_tables(
            market_data,
            forecasts,
            config,
            bootstrap_reps,
            seed,
        ),
    }
    directory = strategy_run_dir(train_end, config, bootstrap_reps, seed)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv")
    (directory / "frictions_manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "layers": {
                    name: asdict(layer)
                    for name, layer in (
                        friction_layers(config)
                        | two_day_layer(config)
                        | band_layers(config)
                        | cap_layers(config)
                    ).items()
                },
                "execution_timing": {
                    f"lag_{lag}": asdict(replace(config, execution_lag=lag))
                    for lag in COMPARISON_LAGS
                },
                "commit": commit(),
                "strategies": [str(name) for name in forecasts.columns],
                "forecast_manifest": forecast_manifest,
            },
            indent=2,
        )
        + "\n"
    )
    return tables


def run(
    train_end: str = TRAIN_END,
    bootstrap_reps: int = 1_000,
    seed: int = 42,
    config: BacktestConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Load cached forecasts, evaluate one configuration and write its tables."""
    config = BacktestConfig() if config is None else config
    if config.execution_lag != 2:
        raise ValueError("base-case execution_lag must be 2")
    market_data, forecasts, forecast_manifest = _inputs(
        train_end, config.forecast_horizon
    )
    gross_ledgers = build_ledgers(
        market_data, forecasts, replace(config, apply_costs=False)
    )
    net_ledgers = build_ledgers(market_data, forecasts, config)
    tables = evaluate_ledgers(gross_ledgers, net_ledgers, config, bootstrap_reps, seed)
    directory = strategy_run_dir(train_end, config, bootstrap_reps, seed)
    _write_results(
        directory,
        gross_ledgers,
        net_ledgers,
        tables,
        config,
        bootstrap_reps,
        seed,
        forecast_manifest,
    )
    return tables


def _inputs(
    train_end: str, forecast_horizon: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Market data and cached forecasts, refusing forecasts from another snapshot.

    The two-day horizon loads its variants and drops their suffix, so
    the strategy tables carry the same names in both cases.
    """
    snapshot_manifest = json.loads(MANIFEST_PATH.read_text())
    forecast_manifest = json.loads((run_dir(train_end) / MANIFEST_NAME).read_text())
    if forecast_manifest.get("snapshot_manifest") != snapshot_manifest:
        raise ValueError("cached forecasts do not match the current snapshot")
    market_data = pd.read_parquet(DATASET_PATH)
    if forecast_horizon == TWO_DAY_HORIZON:
        forecasts = load_forecasts(two_day_names(), train_end=train_end)
        forecasts.columns = [name.removesuffix(TWO_DAY_SUFFIX) for name in forecasts]
    else:
        forecasts = load_forecasts(train_end=train_end)
    return market_data, forecasts, forecast_manifest


def _buy_and_hold_ledger(
    market_data: pd.DataFrame,
    index: pd.DatetimeIndex,
    config: BacktestConfig,
) -> pd.DataFrame:
    forecast = pd.Series(
        config.target_vol_annual / sqrt(TRADING_DAYS),
        index=index,
        name="forecast",
    )
    ledger = _available(
        build_targeting_ledger(
            market_data,
            forecast,
            replace(config, apply_cap=False, apply_band=True, apply_costs=False),
        )
    )
    cost = config.transaction_cost_bps / BASIS_POINTS_PER_UNIT
    transaction_cost = pd.Series(0.0, index=ledger.index)
    if config.apply_costs:
        transaction_cost.iloc[0] = cost
    ledger["turnover"] = 0.0
    ledger.iloc[0, ledger.columns.get_loc("turnover")] = 1.0
    ledger["transaction_cost"] = transaction_cost
    ledger["net_return"] = ledger["gross_return"] - transaction_cost
    ledger["portfolio_value_net"] = (
        config.initial_capital * (1.0 + ledger["net_return"]).cumprod()
    )
    return ledger


def _metrics(
    ledgers: dict[str, pd.DataFrame],
    return_column: str,
    config: BacktestConfig,
) -> pd.DataFrame:
    rows = {
        name: evaluate_strategy(
            ledger[return_column],
            ledger["rf"],
            ledger["turnover"],
            config.target_vol_annual,
        )
        for name, ledger in ledgers.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")


def _spanning(
    ledgers: dict[str, pd.DataFrame],
    return_column: str,
) -> pd.DataFrame:
    benchmark = ledgers[BENCHMARK]
    rows = {
        name: spanning_test(
            ledger[return_column], benchmark[return_column], ledger["rf"]
        )
        for name, ledger in ledgers.items()
        if name != BENCHMARK
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")


def _strategy_diagnostics(
    ledgers: dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Exposure, cap binding and trading intensity of every net ledger.

    The cap share counts days whose raw weight reaches the configured cap,
    whether or not the cap is applied, so an uncapped run still reports how
    often the headline cap would have bound.
    """
    rows = {}
    for name, ledger in ledgers.items():
        trades = ledger["turnover"] > 0.0
        rows[name] = {
            "average_held_weight": ledger["held_weight"].mean(),
            "maximum_held_weight": ledger["held_weight"].max(),
            "leverage_days_share": (ledger["held_weight"] > 1.0).mean(),
            "cap_days_share": (ledger["raw_weight"] >= config.max_weight).mean(),
            "trade_days_per_year": trades.mean() * TRADING_DAYS,
            "average_trade_size": ledger.loc[trades, "turnover"].mean(),
            "annualized_cost_drag": (
                ledger["gross_return"] - ledger["net_return"]
            ).mean()
            * TRADING_DAYS,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")


def _sharpe_pairs(
    ledgers: dict[str, pd.DataFrame],
    return_column: str,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for name_a, name_b in combinations(ledgers, 2):
        result = sharpe_difference_test(
            ledgers[name_a][return_column],
            ledgers[name_b][return_column],
            ledgers[name_a]["rf"],
            bootstrap_reps,
            seed,
        )
        rows.append({"strategy_a": name_a, "strategy_b": name_b, **result.to_dict()})
    return pd.DataFrame(rows).set_index(["strategy_a", "strategy_b"])


def _write_results(
    directory: Path,
    gross_ledgers: dict[str, pd.DataFrame],
    net_ledgers: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    config: BacktestConfig,
    bootstrap_reps: int,
    seed: int,
    forecast_manifest: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    returns = pd.concat(
        {
            name: pd.DataFrame(
                {
                    "gross_return": gross_ledgers[name]["gross_return"],
                    "gross_turnover": gross_ledgers[name]["turnover"],
                    "gross_held_weight": gross_ledgers[name]["held_weight"],
                    "net_return": ledger["net_return"],
                    "target_weight": ledger["target_weight"],
                    "scheduled_rebalance": ledger["scheduled_rebalance"],
                    "drifted_weight": ledger["drifted_weight"],
                    "held_weight": ledger["held_weight"],
                    "turnover": ledger["turnover"],
                    "transaction_cost": ledger["transaction_cost"],
                    "rf": ledger["rf"],
                }
            )
            for name, ledger in net_ledgers.items()
        },
        axis=1,
    )
    returns.to_parquet(directory / "returns.parquet")
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv")
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "gross_config": asdict(replace(config, apply_costs=False)),
                "net_config": asdict(config),
                "bootstrap_reps": bootstrap_reps,
                "seed": seed,
                "commit": commit(),
                "strategies": list(net_ledgers),
                "forecast_manifest": forecast_manifest,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--bootstrap-reps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decision-frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
    )
    parser.add_argument(
        "--weekly-anchor",
        choices=list(WEEKDAYS),
        default=BacktestConfig().weekly_anchor,
        help="weekday on which weekly decisions are taken",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=BacktestConfig().transaction_cost_bps,
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=BacktestConfig().target_vol_annual,
        help="annualized volatility target",
    )
    parser.add_argument(
        "--sizing-exponent",
        type=float,
        default=BacktestConfig().sizing_exponent,
        help="exponent on target over forecast volatility, 2 scales by variance",
    )
    parser.add_argument(
        "--vol-cutoff",
        type=float,
        default=None,
        help="annualized forecast volatility above which the weight is zero",
    )
    parser.add_argument(
        "--no-cap",
        action="store_true",
        help="remove the leverage cap",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        choices=[1, TWO_DAY_HORIZON],
        default=1,
        help="2 sizes on the cached two-day forecasts of the return day",
    )
    parser.add_argument(
        "--frictions",
        action="store_true",
        help="write the friction, sensitivity and timing tables instead",
    )
    arguments = parser.parse_args()
    config = BacktestConfig(
        target_vol_annual=arguments.target_vol,
        sizing_exponent=arguments.sizing_exponent,
        vol_cutoff_annual=arguments.vol_cutoff,
        apply_cap=not arguments.no_cap,
        decision_frequency=arguments.decision_frequency,
        weekly_anchor=arguments.weekly_anchor,
        transaction_cost_bps=arguments.transaction_cost_bps,
        forecast_horizon=arguments.forecast_horizon,
    )
    if arguments.frictions:
        frictions(arguments.train_end, config, arguments.bootstrap_reps, arguments.seed)
    else:
        run(arguments.train_end, arguments.bootstrap_reps, arguments.seed, config)
