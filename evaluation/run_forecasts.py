"""Run the walk-forward for every model once and cache the forecasts.

The two-day variants, suffixed _h2, forecast two days ahead for the backtest
where the weight decided at the close of t earns the return of t+2. They are
cached next to the one-day forecasts and are run only when named.
"""

import argparse
import json
import logging
import subprocess
from datetime import date
from functools import partial

import pandas as pd

from data.download.config import DATASET_PATH, MANIFEST_PATH
from models.ewma import EWMA
from models.garch import GARCH
from models.gbm import GradientBoosting
from models.gbm_vix import GradientBoostingVIX
from models.har import HAR
from models.trailing import TrailingVol

from .walk_forward import walk_forward

log = logging.getLogger(__name__)

TRAIN_END = "2009-12-31"
ROBUSTNESS_CUTS = ("2008-12-31", "2012-12-31")
CACHE_DIR = DATASET_PATH.parent / "forecasts"
MANIFEST_NAME = "run_manifest.json"
MODELS = {
    "ewma": EWMA,
    "garch": GARCH,
    "har": HAR,
    "gbm": GradientBoosting,
    "gbm_vix": GradientBoostingVIX,
    "trailing": TrailingVol,
}
TWO_DAY_HORIZON = 2
TWO_DAY_SUFFIX = f"_h{TWO_DAY_HORIZON}"


def factories() -> dict:
    """Every runnable name: the one-day models and their two-day variants."""
    two_day = {
        f"{name}{TWO_DAY_SUFFIX}": partial(cls, horizon=TWO_DAY_HORIZON)
        for name, cls in MODELS.items()
    }
    return MODELS | two_day


def run_dir(train_end: str = TRAIN_END):
    """One directory per training cut, so a robustness run cannot overwrite another."""
    return CACHE_DIR / train_end


def commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def kept_entries(path, snapshot: dict) -> dict:
    """Manifest entries a partial run can still vouch for.

    A snapshot that moved invalidates every cached model, not only the ones
    being rerun, so in that case nothing is kept.
    """
    if not path.exists():
        return {}
    previous = json.loads(path.read_text())
    if previous.get("snapshot_manifest") != snapshot:
        return {}
    return previous.get("models", {})


def write_manifest(path, train_end: str, snapshot: dict, models: dict) -> None:
    """Record exactly the models whose parquets are complete right now."""
    path.write_text(
        json.dumps(
            {
                "train_end": train_end,
                "snapshot_manifest": snapshot,
                "models": models,
            },
            indent=2,
        )
        + "\n"
    )


def run(train_end: str = TRAIN_END, names: list[str] | None = None) -> None:
    """Cache the forecasts of names, or of every one-day model when names is None."""
    names = names or list(MODELS)
    runnable = factories()
    unknown = [name for name in names if name not in runnable]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    df = pd.read_parquet(DATASET_PATH)
    snapshot = json.loads(MANIFEST_PATH.read_text())
    directory = run_dir(train_end)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    models = kept_entries(path, snapshot)
    for name in names:
        models.pop(name, None)
    write_manifest(path, train_end, snapshot, models)
    stamp = {"run_date": date.today().isoformat(), "commit": commit()}
    for name in names:
        forecast, params = walk_forward(runnable[name](), df, train_end)
        forecast.to_frame().to_parquet(directory / f"{name}.parquet")
        params_path = directory / f"{name}_params.parquet"
        params_path.unlink(missing_ok=True)
        if not params.empty:
            params.to_parquet(params_path)
        models[name] = {"forecasts": len(forecast), "refits": len(params)} | stamp
        write_manifest(path, train_end, snapshot, models)
        log.info("%s: %d forecasts, %d refits", name, len(forecast), len(params))


def cached_models(train_end: str) -> dict:
    """What the manifest of a run directory currently vouches for."""
    path = run_dir(train_end) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no run for train_end {train_end}; run python -m evaluation.run_forecasts"
        )
    return json.loads(path.read_text()).get("models", {})


def load_forecasts(
    names: list[str] | None = None, train_end: str = TRAIN_END
) -> pd.DataFrame:
    """Cached forecasts as one frame, a column per model.

    Defaults to the one-day models; pass two_day_names() for the two-day ones.
    """
    names = names or list(MODELS)
    missing = [name for name in names if name not in cached_models(train_end)]
    if missing:
        raise FileNotFoundError(
            f"no cached forecasts for {missing}; "
            f"run python -m evaluation.run_forecasts {' '.join(missing)}"
        )
    directory = run_dir(train_end)
    columns = {
        name: pd.read_parquet(directory / f"{name}.parquet")["forecast"]
        for name in names
    }
    index = next(iter(columns.values())).index
    stale = [name for name, s in columns.items() if not s.index.equals(index)]
    if stale:
        raise ValueError(f"cached forecasts disagree on dates: {stale}")
    return pd.DataFrame(columns)


def two_day_names(names: list[str] | None = None) -> list[str]:
    """Cache names of the two-day variants of names, every model by default."""
    return [f"{name}{TWO_DAY_SUFFIX}" for name in names or MODELS]


def load_params(name: str, train_end: str = TRAIN_END) -> pd.DataFrame:
    """Per-refit parameters cached for one model."""
    if name not in cached_models(train_end):
        raise FileNotFoundError(
            f"no cached forecasts for {name}; "
            f"run python -m evaluation.run_forecasts {name}"
        )
    path = run_dir(train_end) / f"{name}_params.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{name} caches no per-refit params")
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=list(factories()))
    parser.add_argument("--train-end", default=TRAIN_END)
    args = parser.parse_args()
    run(train_end=args.train_end, names=args.names or None)
