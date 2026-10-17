import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest.targeting import BacktestConfig
from evaluation import run_strategies
from evaluation.run_strategies import (
    BANDS,
    BENCHMARK,
    CAPS,
    EXECUTION_LAGS,
    band_layers,
    build_ledgers,
    cap_layers,
    evaluate_ledgers,
    execution_timing_tables,
    friction_layers,
    layer_metrics,
    strategy_run_dir,
)


def market_data(periods: int = 40) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    returns = 0.001 + 0.01 * np.sin(np.arange(periods))
    adjusted_close = 100.0 * pd.Series(1.0 + returns, index=index).cumprod()
    previous_close = adjusted_close.shift(1).fillna(100.0)
    return pd.DataFrame(
        {
            "open": previous_close,
            "close": adjusted_close,
            "adj_close": adjusted_close,
            "ret": returns,
            "rf": 0.0001,
        },
        index=index,
    )


def forecasts(index: pd.DatetimeIndex) -> pd.DataFrame:
    target_daily = 0.10 / np.sqrt(252)
    return pd.DataFrame(
        {
            "constant": target_daily,
            "varying": target_daily * (1.0 + 0.2 * np.sin(np.arange(len(index)))),
        },
        index=index,
    )


def test_buy_and_hold_pays_only_the_initial_cost():
    data = market_data()
    forecast_frame = forecasts(data.index[1:])
    ledger = build_ledgers(data, forecast_frame)[BENCHMARK]

    assert ledger.index.equals(forecast_frame.index[1:])
    assert ledger["gross_return"].to_numpy() == pytest.approx(
        data.loc[ledger.index, "ret"].to_numpy()
    )
    assert ledger["turnover"].to_numpy() == pytest.approx(
        [1.0] + [0.0] * (len(ledger) - 1)
    )
    assert ledger["transaction_cost"].to_numpy() == pytest.approx(
        [0.0002] + [0.0] * (len(ledger) - 1)
    )


def test_base_case_builds_gross_h2_and_net_h3_tables():
    data = market_data()
    forecast_frame = forecasts(data.index[1:])
    config = BacktestConfig()
    gross_ledgers = build_ledgers(
        data, forecast_frame, replace(config, apply_costs=False)
    )
    net_ledgers = build_ledgers(data, forecast_frame, config)
    tables = evaluate_ledgers(gross_ledgers, net_ledgers, config, bootstrap_reps=9)

    assert list(tables["gross_metrics"].index) == [
        "constant",
        "varying",
        BENCHMARK,
    ]
    assert list(tables["h2_spanning"].index) == ["constant", "varying"]
    assert list(tables["h3_spanning"].index) == ["constant", "varying"]
    assert list(tables["h2_sharpe"].index) == [
        ("constant", "varying"),
        ("constant", BENCHMARK),
        ("varying", BENCHMARK),
    ]
    assert tables["h2_sharpe"].loc[("varying", BENCHMARK), "bootstrap_reps"] == 9
    days = len(net_ledgers[BENCHMARK])
    assert tables["net_metrics"].loc[BENCHMARK, "annualized_turnover"] == pytest.approx(
        252 / days
    )
    diagnostics = tables["strategy_diagnostics"]
    assert diagnostics.loc[BENCHMARK, "average_held_weight"] == pytest.approx(1.0)
    assert diagnostics["cap_days_share"].to_dict() == {
        "constant": 0.0,
        "varying": 0.0,
        BENCHMARK: 0.0,
    }
    assert diagnostics.loc[BENCHMARK, "trade_days_per_year"] == pytest.approx(
        252 / days
    )
    assert diagnostics.loc[BENCHMARK, "annualized_cost_drag"] == pytest.approx(
        0.0002 * 252 / days
    )
    assert gross_ledgers["constant"]["gross_return"].iloc[1] != pytest.approx(
        net_ledgers["constant"]["gross_return"].iloc[1]
    )
    expected_gross_turnover = gross_ledgers["constant"]["turnover"].mean() * 252
    assert tables["gross_metrics"].loc[
        "constant", "annualized_turnover"
    ] == pytest.approx(expected_gross_turnover)


def test_base_case_needs_more_than_one_forecast_date():
    data = market_data()
    forecast_frame = forecasts(data.index[1:2])

    with pytest.raises(ValueError, match="no executable weight"):
        build_ledgers(data, forecast_frame)


def test_strategy_tables_accept_aligned_execution_and_reject_look_ahead():
    data = market_data()
    forecast_frame = forecasts(data.index[1:])

    aligned = build_ledgers(data, forecast_frame, BacktestConfig(execution_lag=1))

    assert aligned["constant"].index.equals(forecast_frame.index)
    with pytest.raises(ValueError, match="execution_lag must be 1 or 2"):
        build_ledgers(data, forecast_frame, BacktestConfig(execution_lag=0))


def test_execution_timing_uses_full_friction_case_and_common_sample():
    data = market_data()
    forecast_frame = forecasts(data.index[1:])
    config = BacktestConfig()

    tables = execution_timing_tables(data, forecast_frame, config, bootstrap_reps=3)

    metrics = tables["execution_timing_net_metrics"]
    assert list(metrics.index.unique("timing")) == ["lag_1", "lag_2"]
    assert list(metrics.loc["lag_1"].index) == ["constant", "varying", BENCHMARK]
    assert tables["execution_timing_h2_spanning"].index.names == [
        "timing",
        "strategy",
    ]
    aligned = build_ledgers(
        data,
        forecast_frame.iloc[1:],
        replace(config, execution_lag=1),
    )
    implementable = build_ledgers(data, forecast_frame, config)
    assert aligned["constant"].index.equals(implementable["constant"].index)
    lag_2 = evaluate_ledgers(
        build_ledgers(data, forecast_frame, replace(config, apply_costs=False)),
        build_ledgers(data, forecast_frame, config),
        config,
        bootstrap_reps=3,
    )["net_metrics"]
    pd.testing.assert_frame_equal(metrics.loc["lag_2"], lag_2)
    aligned_config = replace(config, execution_lag=1)
    lag_1 = evaluate_ledgers(
        build_ledgers(
            data, forecast_frame.iloc[1:], replace(aligned_config, apply_costs=False)
        ),
        build_ledgers(data, forecast_frame.iloc[1:], aligned_config),
        aligned_config,
        bootstrap_reps=3,
    )["net_metrics"]
    pd.testing.assert_frame_equal(metrics.loc["lag_1"], lag_1)
    pd.testing.assert_series_equal(
        metrics.loc[("lag_1", BENCHMARK)],
        metrics.loc[("lag_2", BENCHMARK)],
        check_names=False,
    )
    assert metrics.loc[("lag_1", "varying"), "sharpe"] != pytest.approx(
        metrics.loc[("lag_2", "varying"), "sharpe"]
    )


def test_friction_layers_add_one_friction_at_a_time():
    config = BacktestConfig()
    layers = friction_layers(config)

    assert list(layers) == ["lag_0", "lag_1", "lag_2", "cap", "band", "costs"]
    for lag in EXECUTION_LAGS:
        bare = layers[f"lag_{lag}"]
        assert bare.execution_lag == lag
        assert (bare.apply_cap, bare.apply_band, bare.apply_costs) == (
            False,
            False,
            False,
        )
    assert layers["cap"] == replace(config, apply_band=False, apply_costs=False)
    assert layers["band"] == replace(config, apply_costs=False)
    assert layers["costs"] == config
    assert [layer.no_trade_band for layer in band_layers(config).values()] == list(
        BANDS
    )
    caps = cap_layers(config)
    assert list(caps) == ["cap_1.5", "cap_2.0", "cap_3.0", "cap_none"]
    assert [layer.max_weight for layer in caps.values()][:3] == list(CAPS[:3])
    assert caps["cap_none"] == replace(config, apply_cap=False)
    assert all(layer.apply_cap for name, layer in caps.items() if name != "cap_none")


def test_layer_metrics_end_on_the_base_case():
    data = market_data()
    forecast_frame = forecasts(data.index[1:])
    config = BacktestConfig()
    table = layer_metrics(data, forecast_frame, friction_layers(config))
    base = evaluate_ledgers(
        build_ledgers(data, forecast_frame, replace(config, apply_costs=False)),
        build_ledgers(data, forecast_frame, config),
        config,
        bootstrap_reps=3,
    )["net_metrics"]

    assert table.index.names == ["layer", "strategy"]
    assert len(table) == 6 * 2
    pd.testing.assert_series_equal(
        table.loc[("costs", "varying")],
        base.loc["varying"],
        check_names=False,
    )
    sharpe = table["sharpe"].xs("varying", level="strategy")
    assert sharpe["lag_0"] != pytest.approx(sharpe["lag_2"])
    assert table.loc[("lag_0", "varying"), "annualized_turnover"] > 0.0


def strategy_inputs(tmp_path, monkeypatch):
    data = market_data()
    forecast_frame = forecasts(data.index[1:])
    dataset_path = tmp_path / "dataset.parquet"
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    data.to_parquet(dataset_path)
    snapshot_manifest = {"download_date": "2026-08-24"}
    snapshot_path = tmp_path / "snapshot_manifest.json"
    snapshot_path.write_text(json.dumps(snapshot_manifest))
    source_manifest = {
        "train_end": "2009-12-31",
        "snapshot_manifest": snapshot_manifest,
        "models": {"constant": {}},
    }
    (forecast_dir / "run_manifest.json").write_text(json.dumps(source_manifest))
    monkeypatch.setattr(run_strategies, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(run_strategies, "MANIFEST_PATH", snapshot_path)
    monkeypatch.setattr(run_strategies, "CACHE_DIR", tmp_path / "strategies")
    monkeypatch.setattr(run_strategies, "run_dir", lambda train_end: forecast_dir)
    monkeypatch.setattr(run_strategies, "commit", lambda: "abc123")
    monkeypatch.setattr(
        run_strategies,
        "load_forecasts",
        lambda names=None, train_end=None: forecast_frame,
    )
    return source_manifest


def test_run_writes_tables_with_forecast_provenance(tmp_path, monkeypatch):
    source_manifest = strategy_inputs(tmp_path, monkeypatch)

    run_strategies.run(bootstrap_reps=3)

    output = strategy_run_dir("2009-12-31", BacktestConfig(), bootstrap_reps=3)
    assert (output / "returns.parquet").exists()
    assert (output / "h3_sharpe.csv").exists()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["forecast_manifest"] == source_manifest
    assert manifest["commit"] == "abc123"
    assert manifest["gross_config"]["apply_costs"] is False
    assert manifest["net_config"]["apply_costs"] is True
    saved_returns = pd.read_parquet(output / "returns.parquet")
    assert (BENCHMARK, "rf") in saved_returns.columns
    assert ("varying", "held_weight") in saved_returns.columns
    assert ("varying", "transaction_cost") in saved_returns.columns
    assert ("varying", "gross_turnover") in saved_returns.columns
    assert ("varying", "gross_held_weight") in saved_returns.columns
    assert ("varying", "scheduled_rebalance") in saved_returns.columns


def test_frictions_writes_both_tables_next_to_the_base_case(tmp_path, monkeypatch):
    strategy_inputs(tmp_path, monkeypatch)

    tables = run_strategies.frictions(bootstrap_reps=3)

    output = strategy_run_dir("2009-12-31", BacktestConfig(), bootstrap_reps=3)
    for name in (
        "friction_attribution",
        "band_sensitivity",
        "cap_sensitivity",
        "execution_timing_net_metrics",
        "execution_timing_h2_spanning",
        "execution_timing_h3_sharpe",
    ):
        index_columns = list(range(tables[name].index.nlevels))
        saved = pd.read_csv(output / f"{name}.csv", index_col=index_columns)
        assert saved.shape == tables[name].shape
    manifest = json.loads((output / "frictions_manifest.json").read_text())
    assert manifest["commit"] == "abc123"
    assert manifest["forecast_manifest"]["train_end"] == "2009-12-31"
    assert manifest["layers"]["lag_0"]["execution_lag"] == 0
    assert manifest["layers"]["two_day"]["forecast_horizon"] == 2
    assert manifest["layers"]["costs"]["apply_costs"] is True
    assert manifest["execution_timing"]["lag_1"]["apply_costs"] is True
    assert manifest["execution_timing"]["lag_2"]["execution_lag"] == 2
    assert list(tables["friction_attribution"].index.unique("layer")) == [
        "lag_0",
        "lag_1",
        "lag_2",
        "two_day",
        "cap",
        "band",
        "costs",
    ]
    assert list(tables["band_sensitivity"].index.unique("layer")) == [
        "band_0.000",
        "band_0.025",
        "band_0.050",
        "band_0.100",
    ]


def test_strategy_cache_directory_changes_with_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(run_strategies, "CACHE_DIR", tmp_path)
    base = BacktestConfig()

    paths = {
        strategy_run_dir("2009-12-31", base),
        strategy_run_dir("2009-12-31", replace(base, decision_frequency="weekly")),
        strategy_run_dir(
            "2009-12-31",
            replace(base, decision_frequency="weekly", weekly_anchor="MON"),
        ),
        strategy_run_dir("2009-12-31", replace(base, transaction_cost_bps=5.0)),
        strategy_run_dir("2009-12-31", replace(base, no_trade_band=0.10)),
        strategy_run_dir("2009-12-31", replace(base, target_vol_annual=0.15)),
        strategy_run_dir("2009-12-31", replace(base, sizing_exponent=2.0)),
        strategy_run_dir("2009-12-31", replace(base, apply_cap=False)),
        strategy_run_dir("2009-12-31", replace(base, vol_cutoff_annual=0.30)),
        strategy_run_dir("2009-12-31", base, bootstrap_reps=2_000),
        strategy_run_dir("2009-12-31", base, seed=7),
    }

    assert len(paths) == 11
    assert strategy_run_dir("2009-12-31", base) == strategy_run_dir("2009-12-31", base)
    assert strategy_run_dir("2009-12-31", base) == strategy_run_dir(
        "2009-12-31", replace(base, sizing_exponent=1.0, vol_cutoff_annual=None)
    )
    assert strategy_run_dir("2009-12-31", base).name.endswith("e31370cf09e0")


def test_run_rejects_forecasts_from_another_snapshot(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot_manifest.json"
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    snapshot_path.write_text(json.dumps({"download_date": "2026-08-24"}))
    (forecast_dir / "run_manifest.json").write_text(
        json.dumps({"snapshot_manifest": {"download_date": "2026-08-23"}})
    )
    monkeypatch.setattr(run_strategies, "MANIFEST_PATH", snapshot_path)
    monkeypatch.setattr(run_strategies, "run_dir", lambda train_end: forecast_dir)

    with pytest.raises(ValueError, match="do not match the current snapshot"):
        run_strategies.run()


def test_frictions_are_attributed_on_the_one_day_forecasts(tmp_path, monkeypatch):
    strategy_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="one-day"):
        run_strategies.frictions(config=BacktestConfig(forecast_horizon=2))
