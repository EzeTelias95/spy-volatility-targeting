"""Numbered charts of the final document, one palette and one frame style."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from data.download.config import DATASET_PATH
from evaluation.run_forecasts import load_forecasts
from evaluation.run_strategies import BENCHMARK
from evaluation.strategy import rolling_volatility
from reporting.tables import BASE, LABELS, returns

ANNUAL = np.sqrt(252)
COLORS = {
    "ewma": "#2a78d6",
    "garch": "#eb6834",
    "har": "#3f8f5b",
    "gbm": "#8657b5",
    "gbm_vix": "#b5453a",
    "trailing": "#c49a2c",
    BENCHMARK: "#0b0b0b",
    "realized": "#a9a8a2",
}
STYLE = {
    "figure.figsize": (8, 4),
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#898781",
    "legend.frameon": False,
}


def price_and_volatility():
    df = pd.read_parquet(DATASET_PATH)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(df.index, df["adj_close"], lw=0.9, color=COLORS[BENCHMARK])
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(mticker.FixedLocator([50, 100, 200, 400, 600]))
    axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0].set_ylabel("SPY ajustado (escala log)")
    vol = df["realized_vol_gk"] * ANNUAL * 100
    monthly = np.sqrt((df["realized_vol_gk"] ** 2).rolling(21).mean()) * ANNUAL * 100
    axes[1].plot(df.index, vol, lw=0.4, color=COLORS["realized"], label="Diaria")
    axes[1].plot(df.index, monthly, lw=1.0, color=COLORS["ewma"], label="Media 21 días")
    axes[1].set_ylabel("Vol. anualizada (%)")
    axes[1].legend()
    for ax in axes:
        ax.axvline(pd.Timestamp("2010-01-01"), ls="--", lw=0.8, color="#898781")
    return fig


def forecasts_2020():
    forecasts = load_forecasts().loc["2020-01":"2020-06"]
    realized = pd.read_parquet(DATASET_PATH)["realized_vol_gk"].reindex(forecasts.index)
    fig, ax = plt.subplots()
    ax.plot(
        realized.index,
        realized * ANNUAL * 100,
        lw=0.8,
        color=COLORS["realized"],
        label="Realizada (GK + overnight)",
    )
    for name in ("garch", "gbm"):
        ax.plot(
            forecasts.index,
            forecasts[name] * ANNUAL * 100,
            lw=1.4,
            color=COLORS[name],
            label=LABELS[name],
        )
    ax.set_ylabel("Vol. anualizada (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.legend()
    return fig


def wealth_and_drawdown():
    net = returns().xs("net_return", axis=1, level=1)
    names = [BENCHMARK, "trailing", "har", "gbm"]
    wealth = (1.0 + net[names]).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for name in names:
        axes[0].plot(
            wealth.index, wealth[name], lw=1.0, color=COLORS[name], label=LABELS[name]
        )
        axes[1].plot(drawdown.index, drawdown[name] * 100, lw=0.8, color=COLORS[name])
    axes[0].set_ylabel("Riqueza neta (inicio = 1)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[0].legend(ncol=2)
    return fig


def volatility_control():
    net = returns().xs("net_return", axis=1, level=1)
    fig, ax = plt.subplots()
    for name in (BENCHMARK, "trailing", "gbm"):
        rolling = rolling_volatility(net[name]) * 100
        ax.plot(rolling.index, rolling, lw=0.8, color=COLORS[name], label=LABELS[name])
    ax.axhline(
        BASE.target_vol_annual * 100,
        ls="--",
        lw=1.0,
        color="#898781",
        label="Objetivo 10%",
    )
    ax.set_ylabel("Vol. anualizada 21 días (%)")
    ax.legend(ncol=2)
    return fig


FIGURES = {
    "grafico_01_precio_y_volatilidad": (
        "Precio de SPY y volatilidad realizada Garman-Klass más overnight, 2000-2026",
        price_and_volatility,
    ),
    "grafico_02_pronosticos_2020": (
        "GARCH y GBM contra la volatilidad realizada, enero a junio de 2020",
        forecasts_2020,
    ),
    "grafico_03_riqueza_y_drawdown": (
        "Riqueza acumulada neta y drawdown, caso base",
        wealth_and_drawdown,
    ),
    "grafico_04_control_de_volatilidad": (
        "Volatilidad móvil de 21 días contra el objetivo, caso base",
        volatility_control,
    ),
}
