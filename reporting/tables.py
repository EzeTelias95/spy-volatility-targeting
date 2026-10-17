"""Numbered tables of the final document, written as CSV ready to paste."""

from dataclasses import replace

import numpy as np
import pandas as pd

from backtest.targeting import WEEKDAYS, BacktestConfig
from data.download.config import DATASET_PATH
from evaluation.capacity import (
    average_dollar_volume,
    capacity_table,
    spy_dollar_volume,
)
from evaluation.forecasts import evaluate_forecasts, qlike_losses
from evaluation.run_forecasts import (
    ROBUSTNESS_CUTS,
    TRAIN_END,
    TWO_DAY_SUFFIX,
    load_forecasts,
    two_day_names,
)
from evaluation.run_strategies import BENCHMARK, strategy_run_dir
from evaluation.strategy import (
    evaluate_strategy,
    rolling_volatility,
    sharpe_difference_test,
)

ORDER = ["trailing", "ewma", "garch", "har", "gbm", "gbm_vix"]
LABELS = {
    "trailing": "Trailing 21 días",
    "ewma": "EWMA",
    "garch": "GARCH(1,1)",
    "har": "HAR-RV",
    "gbm": "GBM",
    "gbm_vix": "GBM + VIX",
    BENCHMARK: "Pasivo",
}
STRATEGIES = [*ORDER, BENCHMARK]
BASE = BacktestConfig()
YAHOO_ONLY = {
    "tabla_01_descriptivas",
    "tabla_02_pronosticos",
    "tabla_03_proxy_alternativo",
    "tabla_10_ganancia_por_quintil",
    "tabla_17_horizonte",
    "tabla_19_retorno_por_quintil",
    "grafico_01_precio_y_volatilidad",
    "grafico_02_pronosticos_2020",
}


def source(name: str) -> str:
    data = "Yahoo Finance" if name in YAHOO_ONLY else "Yahoo Finance y FRED"
    return f"Fuente: elaboración propia en base a datos de {data}."


def fixed(digits: int):
    return lambda x: f"{round(x, digits) + 0.0:.{digits}f}"


def percent(x: float) -> str:
    return f"{x * 100:.1f}%"


def points(x: float) -> str:
    return f"{x * 100:.2f}"


def times(x: float) -> str:
    return f"{x:.2f}x" if x < 1.0 else f"{x:.1f}x"


def pvalue(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.2f}"


def labelled(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(index=LABELS, columns=LABELS)


def cached(name: str, config: BacktestConfig = BASE, train_end: str = TRAIN_END):
    table = pd.read_csv(strategy_run_dir(train_end, config) / f"{name}.csv")
    keys = ["layer", "strategy", "strategy_a", "strategy_b"]
    return table.set_index([key for key in keys if key in table.columns])


def vs_passive(sharpe: pd.DataFrame) -> pd.DataFrame:
    return sharpe.xs(BENCHMARK, level="strategy_b")


def returns(config: BacktestConfig = BASE, train_end: str = TRAIN_END) -> pd.DataFrame:
    return pd.read_parquet(strategy_run_dir(train_end, config) / "returns.parquet")


def forecast_inputs(
    proxy_column: str = "realized_vol_gk", train_end: str = TRAIN_END
) -> tuple[dict[str, pd.Series], pd.Series]:
    forecasts = load_forecasts(train_end=train_end)
    df = pd.read_parquet(DATASET_PATH)
    proxy = df[proxy_column].reindex(forecasts.index)
    if proxy_column == "log_ret":
        proxy = proxy.abs()
    return {name: forecasts[name] for name in ORDER}, proxy


def forecast_scores(
    proxy_column: str = "realized_vol_gk", train_end: str = TRAIN_END
) -> pd.DataFrame:
    forecasts, proxy = forecast_inputs(proxy_column, train_end)
    return evaluate_forecasts(forecasts, proxy, benchmark="gbm").loc[ORDER]


def descriptive() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    samples = {
        "2000-2026": df,
        "Entrenamiento 2000-2009": df.loc[:"2009"],
        "Fuera de muestra 2010-2026": df.loc["2010":],
    }
    rows = {}
    for label, sample in samples.items():
        ret = sample["ret"].dropna()
        vol = sample["realized_vol_gk"].dropna() * np.sqrt(252)
        rows[label] = {
            "Días": f"{len(ret)}",
            "Retorno anual medio": percent(ret.mean() * 252),
            "Volatilidad anual": percent(ret.std() * np.sqrt(252)),
            "Asimetría": f"{ret.skew():.2f}",
            "Curtosis en exceso": f"{ret.kurt():.2f}",
            "Vol. GK + overnight media": percent(vol.mean()),
            "Vol. GK + overnight mediana": percent(vol.median()),
            "Vol. GK + overnight máxima": percent(vol.max()),
            "Raíz de la varianza GK + overnight media": percent(
                np.sqrt((vol**2).mean())
            ),
        }
    return pd.DataFrame(rows).rename_axis("Estadístico")


def forecast_evaluation() -> pd.DataFrame:
    scores = forecast_scores("realized_vol_gk")
    table = pd.DataFrame(
        {
            "QLIKE": scores["qlike"].map(fixed(3)),
            "RMSE (x10^4)": (scores["rmse"] * 1e4).map(fixed(2)),
            "DM vs GBM": scores["dm_stat"].map(fixed(2), na_action="ignore"),
            "p": scores["dm_pvalue"].map(pvalue, na_action="ignore"),
            "Utilidad realizada (% anual)": (scores["realized_utility"] * 100).map(
                fixed(2)
            ),
        }
    )
    return labelled(table).rename_axis("Modelo")


def alternative_proxy() -> pd.DataFrame:
    base, alt = forecast_scores("realized_vol_gk"), forecast_scores("log_ret")
    table = pd.DataFrame(
        {
            "QLIKE, GK + overnight": base["qlike"].map(fixed(3)),
            "QLIKE, retorno al cuadrado": alt["qlike"].map(fixed(3)),
            "DM vs GBM, retorno al cuadrado": alt["dm_stat"].map(
                fixed(2), na_action="ignore"
            ),
            "p": alt["dm_pvalue"].map(pvalue, na_action="ignore"),
            "Días en el piso, retorno al cuadrado": alt["proxy_floored_days"].map(
                lambda days: f"{days:.0f}"
            ),
        }
    )
    return labelled(table).rename_axis("Modelo")


def base_case() -> pd.DataFrame:
    metrics = cached("net_metrics").loc[[BENCHMARK, *ORDER]]
    qlike = forecast_scores("realized_vol_gk")["qlike"].map(fixed(3))
    table = pd.DataFrame(
        {
            "QLIKE": qlike.reindex(metrics.index).fillna(""),
            "Sharpe neto": metrics["sharpe"].map(fixed(3)),
            "Vol realizada": metrics["realized_volatility"].map(percent),
            "Vol-of-vol (pp)": metrics["vol_of_vol"].map(points),
            "Drawdown máx.": metrics["maximum_drawdown"].map(percent),
            "Turnover anual": metrics["annualized_turnover"].map(times),
        }
    )
    return labelled(table).rename_axis("Estrategia")


def pairwise_sharpe_tests() -> pd.DataFrame:
    columns = {}
    for path, label in (("h2", "bruto"), ("h3", "neto")):
        sharpe = cached(f"{path}_sharpe")
        columns[f"Dif. Sharpe, {label}"] = sharpe["sharpe_difference"].map(fixed(3))
        columns[f"p, {label}"] = sharpe["pvalue"].map(pvalue)
    table = pd.DataFrame(columns)
    table.index = pd.Index(
        [f"{LABELS[a]} vs {LABELS[b]}" for a, b in table.index], name="Par"
    )
    return table


def hypothesis_tests() -> pd.DataFrame:
    columns = {}
    for path, label in (("h2", "bruto"), ("h3", "neto")):
        spanning = cached(f"{path}_spanning").loc[ORDER]
        sharpe = vs_passive(cached(f"{path}_sharpe")).loc[ORDER]
        columns |= {
            f"Alfa anual, {label}": spanning["alpha_annual"].map(percent),
            f"t NW, {label}": spanning["alpha_stat"].map(fixed(2)),
            f"p alfa, {label}": spanning["alpha_pvalue"].map(pvalue),
            f"Dif. Sharpe vs pasivo, {label}": sharpe["sharpe_difference"].map(
                fixed(3)
            ),
            f"p Sharpe, {label}": sharpe["pvalue"].map(pvalue),
        }
    return labelled(pd.DataFrame(columns)).rename_axis("Modelo")


def by_sample(samples: dict[str, tuple]) -> pd.DataFrame:
    rows = {}
    for label, (metrics, pvalues, *scores) in samples.items():
        for score in scores:
            rows[(label, "QLIKE")] = score["qlike"].map(fixed(3))
            rows[(label, "DM vs GBM")] = score["dm_stat"].map(
                fixed(2), na_action="ignore"
            )
            rows[(label, "p DM")] = score["dm_pvalue"].map(pvalue, na_action="ignore")
        rows[(label, "Sharpe")] = metrics["sharpe"].map(fixed(3))
        rows[(label, "Vol-of-vol (pp)")] = metrics["vol_of_vol"].map(points)
        rows[(label, "Sharpe")]["p mínimo"] = pvalue(pvalues.min())
    table = pd.DataFrame(rows).T.reindex(columns=[*STRATEGIES, "p mínimo"])
    return labelled(table.fillna("")).rename_axis(["Muestra", "Medida"])


def training_cut() -> pd.DataFrame:
    labels = {"2008-12-31": "Entrenamiento hasta 2008"}
    labels |= {TRAIN_END: "Entrenamiento hasta 2009 (base)"}
    labels |= {"2012-12-31": "Entrenamiento hasta 2012"}
    cuts = sorted((TRAIN_END, *ROBUSTNESS_CUTS))
    return by_sample(
        {
            labels[cut]: (
                cached("net_metrics", train_end=cut),
                vs_passive(cached("h3_sharpe", train_end=cut))["pvalue"],
                forecast_scores(train_end=cut),
            )
            for cut in cuts
        }
    )


def gain_by_quintile() -> pd.DataFrame:
    """Mean QLIKE per realized-vol quintile and where each GBM gain sits.

    The gain is the loss differential against the GBM (against GBM + VIX for
    the GBM itself); its share per quintile sums to one across quintiles.
    """
    quintiles = ["Q1 (más calmo)", "Q2", "Q3", "Q4", "Q5 (más volátil)"]
    forecasts, proxy = forecast_inputs()
    losses = qlike_losses(forecasts, proxy)
    quintile = pd.qcut(proxy, len(quintiles), labels=quintiles)
    gain = losses.drop(columns="gbm").sub(losses["gbm"], axis=0)
    share = gain.groupby(quintile, observed=True).sum() / gain.sum()
    rows = {}
    for name, mean in losses.groupby(quintile, observed=True).mean().items():
        rows[("QLIKE medio", LABELS[name])] = mean.map(fixed(3))
    pairs = {name: f"GBM sobre {LABELS[name]}" for name in share.columns}
    pairs["gbm_vix"] = "GBM + VIX sobre GBM"
    for name, pair in pairs.items():
        rows[("Participación en la ganancia", pair)] = share[name].map(percent)
    return pd.DataFrame(rows).T.rename_axis(["Medida", "Modelo"])


def split_vol_of_vol(returns: pd.Series, segments: list) -> float:
    """Vol-of-vol pooled over segments, so no rolling window spans a gap."""
    deviations = pd.concat(
        rolling_volatility(returns.loc[segment]) - BASE.target_vol_annual
        for segment in segments
    )
    return float(np.sqrt((deviations**2).mean()))


def sub_periods() -> pd.DataFrame:
    ledger = returns()
    net = ledger.xs("net_return", axis=1, level=1)
    turnover = ledger.xs("turnover", axis=1, level=1)
    rf = ledger.xs("rf", axis=1, level=1)
    cuts = {
        "2010-2017": net.index.year <= 2017,
        "2018-2026": net.index.year >= 2018,
        "Sin 2020": net.index.year != 2020,
    }
    samples = {}
    for label, cut in cuts.items():
        metrics = pd.DataFrame(
            {
                name: evaluate_strategy(
                    net.loc[cut, name], rf.loc[cut, name], turnover.loc[cut, name]
                )
                for name in STRATEGIES
            }
        ).T
        if label == "Sin 2020":
            segments = [net.index.year < 2020, net.index.year > 2020]
            metrics["vol_of_vol"] = [
                split_vol_of_vol(net[name], segments) for name in STRATEGIES
            ]
        pvalues = pd.Series(
            {
                name: sharpe_difference_test(
                    net.loc[cut, name], net.loc[cut, BENCHMARK], rf.loc[cut, name]
                )["pvalue"]
                for name in ORDER
            }
        )
        samples[label] = (metrics, pvalues)
    return by_sample(samples)


def weekday() -> pd.DataFrame:
    names = {"MON": "Lunes", "TUE": "Martes", "WED": "Miércoles"}
    names |= {"THU": "Jueves", "FRI": "Viernes"}
    samples = {}
    for anchor in WEEKDAYS:
        config = replace(BASE, decision_frequency="weekly", weekly_anchor=anchor)
        samples[names[anchor]] = (
            cached("net_metrics", config),
            vs_passive(cached("h3_sharpe", config))["pvalue"],
        )
    return by_sample(samples).drop(columns=LABELS[BENCHMARK])


def execution() -> pd.DataFrame:
    labels = {
        "lag_0": "Peso aplicado al retorno del día de la decisión (contrafactual)",
        "lag_1": "Orden al mismo cierre de la decisión",
        "lag_2": "Orden al cierre siguiente (implementable)",
        "cap": "+ cap",
        "band": "+ banda",
        "costs": "+ costos (caso base)",
        "lag_1_frictions": "Orden al mismo cierre con cap, banda y costos",
        "two_day": "Orden al cierre siguiente, pronóstico a dos días, sin fricciones",
    }
    sharpe = cached("friction_attribution")["sharpe"].unstack("strategy")
    timing = pd.read_csv(
        strategy_run_dir(TRAIN_END, BASE) / "execution_timing_net_metrics.csv"
    )
    timing = timing.set_index(["timing", "strategy"])["sharpe"].unstack("strategy")
    sharpe.loc["lag_1_frictions"] = timing.loc["lag_1"]
    table = sharpe.reindex(index=list(labels), columns=ORDER).map(fixed(3))
    return labelled(table.rename(index=labels)).rename_axis("Convención")


def costs() -> pd.DataFrame:
    rows = {
        f"{bps:g} bps": cached("net_metrics", replace(BASE, transaction_cost_bps=bps))[
            "sharpe"
        ]
        for bps in (2.0, 5.0, 10.0)
    }
    table = pd.DataFrame(rows).T.reindex(columns=STRATEGIES).map(fixed(3))
    return labelled(table).rename_axis("Costo por operación")


def sensitivities() -> pd.DataFrame:
    variants = {
        "Caso base": BASE,
        "Vol objetivo 15%": replace(BASE, target_vol_annual=0.15),
        "Vol objetivo 20%": replace(BASE, target_vol_annual=0.20),
        "Sin cap": replace(BASE, apply_cap=False),
        "Regla por varianza": replace(BASE, sizing_exponent=2.0),
        "Corte de vol 30%": replace(BASE, vol_cutoff_annual=0.30),
        "Corte de vol 20%": replace(BASE, vol_cutoff_annual=0.20),
        "Pronóstico a dos días": replace(BASE, forecast_horizon=2),
        "Decisión semanal": replace(BASE, decision_frequency="weekly"),
        "Decisión mensual": replace(BASE, decision_frequency="monthly"),
    }
    metrics = {
        label: (
            cached("net_metrics", config),
            vs_passive(cached("h3_sharpe", config))["pvalue"].min(),
        )
        for label, config in variants.items()
    }
    friction_rows = {
        "Banda 0 pp": ("band_0.000", "band_sensitivity"),
        "Banda 2.5 pp": ("band_0.025", "band_sensitivity"),
        "Banda 10 pp": ("band_0.100", "band_sensitivity"),
        "Cap 2x": ("cap_2.0", "cap_sensitivity"),
        "Cap 3x": ("cap_3.0", "cap_sensitivity"),
    }
    for label, (layer, name) in friction_rows.items():
        metrics[label] = (cached(name).xs(layer, level="layer"), np.nan)
    passive = metrics["Caso base"][0].loc[BENCHMARK]
    rows = {}
    for label, (frame, minimum_p) in metrics.items():
        frame = frame.reindex(STRATEGIES)
        frame.loc[BENCHMARK] = frame.loc[BENCHMARK].fillna(passive)
        rows[(label, "Sharpe neto")] = frame["sharpe"].map(fixed(3))
        rows[(label, "Turnover anual")] = frame["annualized_turnover"].map(times)
        if not np.isnan(minimum_p):
            rows[(label, "Sharpe neto")]["p mínimo"] = pvalue(minimum_p)
    table = pd.DataFrame(rows).T.reindex(columns=[*STRATEGIES, "p mínimo"])
    return labelled(table.fillna("")).rename_axis(["Variante", "Medida"])


def forecast_horizon() -> pd.DataFrame:
    """QLIKE of the forecast on time, one day stale and built for the return day.

    Everything is indexed by the return day: the weight decided with the close
    of t carries the forecast made for t+1 into the return of t+2, so the stale
    column is the forecast the strategy actually uses, and the two-day column
    is the cached two-day forecast made on the decision day for the return day.
    """
    one_day = load_forecasts()
    two_day = load_forecasts(two_day_names(ORDER)).rename(
        columns=lambda name: name.removesuffix(TWO_DAY_SUFFIX)
    )
    proxy = pd.read_parquet(DATASET_PATH)["realized_vol_gk"]
    frames = {"on_time": one_day, "stale": one_day.shift(1), "two_day": two_day}
    index = one_day.index[1:]
    proxy = proxy.reindex(index)
    evaluated = {
        key: evaluate_forecasts(
            {name: frame.loc[index, name] for name in ORDER}, proxy, benchmark="gbm"
        )
        for key, frame in frames.items()
    }
    scores = {key: table["qlike"] for key, table in evaluated.items()}
    recovered = (scores["stale"] - scores["two_day"]) / (
        scores["stale"] - scores["on_time"]
    )
    against_stale = pd.Series(
        {
            name: evaluate_forecasts(
                {
                    "two_day": two_day.loc[index, name],
                    "stale": frames["stale"].loc[index, name],
                },
                proxy,
                benchmark="stale",
            ).loc["two_day", "dm_pvalue"]
            for name in ORDER
        }
    )
    table = pd.DataFrame(
        {
            "QLIKE a tiempo": scores["on_time"].map(fixed(3)),
            "QLIKE con un día de atraso": scores["stale"].map(fixed(3)),
            "Aumento por el atraso": (scores["stale"] / scores["on_time"] - 1).map(
                percent
            ),
            "DM vs GBM, con atraso": evaluated["stale"]["dm_stat"].map(
                fixed(2), na_action="ignore"
            ),
            "p DM, con atraso": evaluated["stale"]["dm_pvalue"].map(
                pvalue, na_action="ignore"
            ),
            "QLIKE a dos días": scores["two_day"].map(fixed(3)),
            "Recuperado por el pronóstico a dos días": recovered.map(percent),
            "p, dos días contra atrasado": against_stale.map(pvalue),
        }
    )
    return labelled(table.loc[ORDER]).rename_axis("Modelo")


ORACLE_LABELS = {
    "oracle_daily": "Oráculo diario",
    "oracle_regime": "Oráculo de 21 días",
}


def oracle_ceiling() -> pd.DataFrame:
    """Look-ahead ceilings and the sign-randomized counterfactual, from the cache."""
    ceiling = cached("oracle_ceiling")
    table = pd.DataFrame(
        {
            "Sharpe neto": ceiling["sharpe"].map(fixed(3)),
            "Exceso sobre el pasivo": ceiling["excess"].map(fixed(3)),
            "Exceso con signos aleatorizados": ceiling["randomized_excess"].map(
                fixed(3)
            ),
            "Diferencia entre ambos excesos": ceiling["timing"].map(fixed(3)),
        }
    )
    order = [*ORACLE_LABELS, *ORDER, BENCHMARK]
    table = table.loc[order].rename(index=ORACLE_LABELS)
    return labelled(table).rename_axis("Estrategia")


def return_by_quintile() -> pd.DataFrame:
    """Mean SPY return per quintile of three sorting keys, in basis points."""
    labels = {
        "realized": "Vol realizada del día",
        "regime": "Oráculo de 21 días",
        "gbm": "Pronóstico del GBM",
    }
    quintiles = {"Q1": "Q1 (más calmo)", "Q5": "Q5 (más volátil)"}
    table = pd.read_csv(
        strategy_run_dir(TRAIN_END, BASE) / "oracle_quintiles.csv", index_col="quintile"
    )
    table = (table * 1e4).map(fixed(1)).rename(columns=labels, index=quintiles)
    return table.rename_axis("Quintil")


def tails_and_trading() -> pd.DataFrame:
    gross = cached("gross_metrics").loc[[BENCHMARK, *ORDER]]
    net = cached("net_metrics").loc[gross.index]
    activity = cached("strategy_diagnostics").loc[gross.index]
    table = pd.DataFrame(
        {
            "Sharpe bruto": gross["sharpe"].map(fixed(3)),
            "Sharpe neto": net["sharpe"].map(fixed(3)),
            "Costo en Sharpe": (gross["sharpe"] - net["sharpe"]).map(fixed(3)),
            "Peor día": net["worst_return_1"].map(percent),
            "Asimetría": net["skew"].map(fixed(2)),
            "Curtosis en exceso": net["kurtosis"].map(fixed(2)),
            "Días operados por año": activity["trade_days_per_year"].map(fixed(1)),
            "Cambio medio de peso (pp)": activity["average_trade_size"].map(points),
        }
    )
    return labelled(table).rename_axis("Estrategia")


SIGNIFICANCE_LEVEL = 0.05


def pvalue3(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def pairwise_tests(net: pd.DataFrame, rf: pd.DataFrame, cut) -> pd.DataFrame:
    rows = {}
    for i, a in enumerate(ORDER):
        for b in [*ORDER[i + 1 :], BENCHMARK]:
            rows[(a, b)] = sharpe_difference_test(
                net.loc[cut, a], net.loc[cut, b], rf.loc[cut, a]
            )
    table = pd.DataFrame(rows).T
    return table.rename_axis(["strategy_a", "strategy_b"])


def summarize_significance(sharpe: pd.DataFrame) -> dict[str, str]:
    against = sharpe.xs(BENCHMARK, level="strategy_b")["pvalue"].loc[ORDER]
    best = against.idxmin()
    between = sharpe[sharpe.index.get_level_values("strategy_b") != BENCHMARK]
    hits = between.loc[between["pvalue"] < SIGNIFICANCE_LEVEL, "pvalue"]
    pairs = ", ".join(
        f"{LABELS[a]} vs {LABELS[b]} ({pvalue3(p)})"
        for (a, b), p in hits.sort_values().items()
    )
    return {
        "p mínimo contra el pasivo": f"{pvalue3(against[best])} ({LABELS[best]})",
        "Pares entre modelos con p < 0.05": pairs or "ninguno",
    }


def significance_by_variant() -> pd.DataFrame:
    """Net-of-costs Sharpe tests for every variant of chapters 6 and 7."""
    rows = {"Caso base": cached("h3_sharpe")}
    for cut in ROBUSTNESS_CUTS:
        rows[f"Entrenamiento hasta {cut[:4]}"] = cached("h3_sharpe", train_end=cut)
    ledger = returns()
    net = ledger.xs("net_return", axis=1, level=1)
    rf = ledger.xs("rf", axis=1, level=1)
    cuts = {
        "2010-2017": net.index.year <= 2017,
        "2018-2026": net.index.year >= 2018,
        "Sin 2020": net.index.year != 2020,
    }
    for label, cut in cuts.items():
        rows[label] = pairwise_tests(net, rf, cut)
    names = {"MON": "lunes", "TUE": "martes", "WED": "miércoles"}
    names |= {"THU": "jueves", "FRI": "viernes"}
    for anchor in WEEKDAYS:
        config = replace(BASE, decision_frequency="weekly", weekly_anchor=anchor)
        rows[f"Decisión semanal, {names[anchor]}"] = cached("h3_sharpe", config)
    variants = {
        "Decisión mensual": replace(BASE, decision_frequency="monthly"),
        "Vol objetivo 15%": replace(BASE, target_vol_annual=0.15),
        "Vol objetivo 20%": replace(BASE, target_vol_annual=0.20),
        "Sin cap": replace(BASE, apply_cap=False),
        "Regla por varianza": replace(BASE, sizing_exponent=2.0),
        "Corte de vol 30%": replace(BASE, vol_cutoff_annual=0.30),
        "Corte de vol 20%": replace(BASE, vol_cutoff_annual=0.20),
        "Pronóstico a dos días": replace(BASE, forecast_horizon=2),
        "5 bps": replace(BASE, transaction_cost_bps=5.0),
        "10 bps": replace(BASE, transaction_cost_bps=10.0),
    }
    for label, config in variants.items():
        rows[label] = cached("h3_sharpe", config)
    timing = pd.read_csv(
        strategy_run_dir(TRAIN_END, BASE) / "execution_timing_h3_sharpe.csv"
    ).set_index(["timing", "strategy_a", "strategy_b"])
    rows["Orden al mismo cierre con cap, banda y costos"] = timing.xs("lag_1")
    table = pd.DataFrame(
        {label: summarize_significance(s) for label, s in rows.items()}
    )
    return table.T.rename_axis("Variante")


def diagnostics() -> pd.DataFrame:
    """Figures cited in chapters 6 and 7 that come from the cache and the raw snapshot.

    Days above the cap and the maximum weight held between weekly and monthly
    decisions, the share of days with the cap active by volatility target, the
    mean SPY return on the days the exit rules leave out and keep, and the
    median capital at which a trade takes one percent of SPY's average dollar
    volume.
    """
    rows = {}
    tolerance = 1e-9
    frequencies = {"semanal": "weekly", "mensual": "monthly"}
    for label, frequency in frequencies.items():
        held = returns(replace(BASE, decision_frequency=frequency))
        held = held.xs("held_weight", axis=1, level=1)[ORDER]
        above = held.gt(BASE.max_weight + tolerance).sum()
        rows[f"Días sobre el cap, decisión {label}"] = above.map(str)
        rows[f"Peso máximo, decisión {label}"] = held.max().map(fixed(2))
    for target in (0.10, 0.15, 0.20):
        share = cached("strategy_diagnostics", replace(BASE, target_vol_annual=target))[
            "cap_days_share"
        ]
        rows[f"Días con el cap activo, objetivo {target:.0%}"] = share.map(percent)
    spy = pd.read_parquet(DATASET_PATH)["ret"]
    for cutoff in (0.30, 0.20):
        held = returns(replace(BASE, vol_cutoff_annual=cutoff))
        held = held.xs("held_weight", axis=1, level=1)[ORDER]
        aligned = spy.reindex(held.index)
        out = held.eq(0.0)
        outside = {name: aligned[out[name]].mean() * 1e4 for name in ORDER}
        inside = {name: aligned[~out[name]].mean() * 1e4 for name in ORDER}
        label = f"corte de vol {cutoff:.0%} (pb)"
        rows[f"Retorno medio de SPY fuera del mercado, {label}"] = pd.Series(
            outside
        ).map(fixed(1))
        rows[f"Retorno medio de SPY dentro del mercado, {label}"] = pd.Series(
            inside
        ).map(fixed(1))
    turnover = returns().xs("turnover", axis=1, level=1)[ORDER]
    capacity = capacity_table(turnover, average_dollar_volume(spy_dollar_volume()))
    rows["Capacidad mediana (miles de millones de dólares)"] = (
        capacity["capacity_median"] / 1e9
    ).map(fixed(2))
    table = pd.DataFrame(rows).T.reindex(columns=ORDER)
    return labelled(table).rename_axis("Medida")


TABLES = {
    "tabla_01_descriptivas": (
        "Estadísticas descriptivas de SPY y de la volatilidad realizada",
        descriptive,
    ),
    "tabla_02_pronosticos": (
        "Evaluación de pronósticos contra la varianza Garman-Klass más overnight",
        forecast_evaluation,
    ),
    "tabla_03_proxy_alternativo": (
        "Ranking de pronósticos con el retorno al cuadrado como proxy alternativo",
        alternative_proxy,
    ),
    "tabla_04_caso_base": ("Estrategias en el caso base", base_case),
    "tabla_05_tests": (
        "Regresión de spanning y diferencia de Sharpe contra el pasivo",
        hypothesis_tests,
    ),
    "tabla_06_corte_entrenamiento": (
        "QLIKE, Sharpe y vol-of-vol por corte de entrenamiento",
        training_cut,
    ),
    "tabla_07_subperiodos": ("Sharpe y vol-of-vol por subperíodo", sub_periods),
    "tabla_08_sensibilidades": (
        "Sharpe neto y turnover por variante de la regla",
        sensitivities,
    ),
    "tabla_09_dia_semanal": ("Decisión semanal según el día ancla", weekday),
    "tabla_10_ganancia_por_quintil": (
        "QLIKE y ganancia del GBM por quintil de volatilidad realizada",
        gain_by_quintile,
    ),
    "tabla_11_ejecucion": (
        "Sharpe según la convención de ejecución y las fricciones",
        execution,
    ),
    "tabla_12_costos": ("Sharpe neto por costo de transacción", costs),
    "tabla_13_colas_y_actividad": (
        "Colas y actividad de trading en el caso base, bruto y neto de costos",
        tails_and_trading,
    ),
    "tabla_14_pares": (
        "Diferencia de Sharpe entre pares de estrategias, bruta y neta de costos",
        pairwise_sharpe_tests,
    ),
    "tabla_15_diagnosticos": (
        "Diagnósticos del cap, de las reglas de salida y de la capacidad",
        diagnostics,
    ),
    "tabla_16_significancia": (
        (
            "Tests de diferencia de Sharpe neta por variante, p mínimo contra el "
            "pasivo y pares entre modelos con p menor que 0.05"
        ),
        significance_by_variant,
    ),
    "tabla_17_horizonte": (
        (
            "QLIKE del pronóstico a tiempo, con un día de atraso y a dos días, "
            "4146 días desde el 5 de enero de 2010"
        ),
        forecast_horizon,
    ),
    "tabla_18_oraculos": (
        (
            "Oráculos con información futura y aleatorización de signos, "
            "orden al mismo cierre, 100 réplicas, 4127 días hasta el 1 de junio de 2026"
        ),
        oracle_ceiling,
    ),
    "tabla_19_retorno_por_quintil": (
        "Retorno medio de SPY por quintil de volatilidad, en puntos básicos",
        return_by_quintile,
    ),
}
