"""Partial runs keep the manifest honest about what is cached."""

import json

import numpy as np
import pandas as pd
import pytest

from evaluation import run_forecasts
from models.base import VolatilityModel

SNAPSHOT = {"download_date": "2026-08-24"}
TRAIN_END = "2001-12-31"


class Constant(VolatilityModel):
    """Forecasts a fixed level, so a run is cheap and its output identifiable."""

    def __init__(self, level=0.01, horizon=1):
        self.level = level
        self.horizon = horizon

    def fit(self, df):
        pass

    def update(self, row):
        pass

    def predict(self):
        return self.level

    @property
    def params(self):
        return pd.Series({"level": self.level})


class Broken(Constant):
    """Fails the way a real model would, part way through a run."""

    def fit(self, df):
        raise RuntimeError("fit failed")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    days = pd.bdate_range("2000-01-03", "2002-12-31")
    frame = pd.DataFrame({"log_ret": np.zeros(len(days))}, index=days)
    dataset = tmp_path / "spy_dataset.parquet"
    frame.to_parquet(dataset)
    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text(json.dumps(SNAPSHOT))
    monkeypatch.setattr(run_forecasts, "DATASET_PATH", dataset)
    monkeypatch.setattr(run_forecasts, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(run_forecasts, "CACHE_DIR", tmp_path / "forecasts")
    monkeypatch.setattr(
        run_forecasts,
        "MODELS",
        {
            "one": lambda horizon=1: Constant(0.01, horizon),
            "two": lambda horizon=1: Constant(0.02, horizon),
        },
    )
    return manifest


def manifest_path(train_end=TRAIN_END):
    return run_forecasts.run_dir(train_end) / run_forecasts.MANIFEST_NAME


def models(train_end=TRAIN_END):
    return json.loads(manifest_path(train_end).read_text())["models"]


def stamp(name, value, train_end=TRAIN_END):
    written = json.loads(manifest_path(train_end).read_text())
    written["models"][name]["commit"] = value
    manifest_path(train_end).write_text(json.dumps(written))


def test_a_partial_run_leaves_the_other_models_cached(cache):
    run_forecasts.run(TRAIN_END)
    run_forecasts.run(TRAIN_END, ["two"])
    assert set(models()) == {"one", "two"}


def test_a_partial_run_keeps_the_other_entries_verbatim(cache):
    run_forecasts.run(TRAIN_END)
    stamp("one", "sentinel")
    run_forecasts.run(TRAIN_END, ["two"])
    assert models()["one"]["commit"] == "sentinel"


def test_a_failed_model_leaves_the_others_cached(cache, monkeypatch):
    run_forecasts.run(TRAIN_END)
    monkeypatch.setitem(run_forecasts.MODELS, "two", Broken)
    with pytest.raises(RuntimeError):
        run_forecasts.run(TRAIN_END, ["two"])
    assert set(models()) == {"one"}
    assert run_forecasts.load_forecasts(["one"], TRAIN_END).columns.tolist() == ["one"]


def test_a_model_being_rerun_is_uncached_until_it_finishes(cache, monkeypatch):
    run_forecasts.run(TRAIN_END)
    monkeypatch.setitem(run_forecasts.MODELS, "two", Broken)
    with pytest.raises(RuntimeError):
        run_forecasts.run(TRAIN_END, ["two"])
    with pytest.raises(FileNotFoundError, match="two"):
        run_forecasts.load_forecasts(["two"], TRAIN_END)


def test_a_moved_snapshot_drops_every_cached_model(cache):
    run_forecasts.run(TRAIN_END)
    cache.write_text(json.dumps({"download_date": "2026-09-01"}))
    run_forecasts.run(TRAIN_END, ["two"])
    assert set(models()) == {"two"}


def test_loading_an_uncached_model_names_it(cache):
    run_forecasts.run(TRAIN_END, ["one"])
    with pytest.raises(FileNotFoundError, match="two"):
        run_forecasts.load_forecasts(["one", "two"], TRAIN_END)


def test_params_of_an_uncached_model_are_refused(cache):
    run_forecasts.run(TRAIN_END)
    cache.write_text(json.dumps({"download_date": "2026-09-01"}))
    run_forecasts.run(TRAIN_END, ["two"])
    with pytest.raises(FileNotFoundError, match="one"):
        run_forecasts.load_params("one", TRAIN_END)


def test_an_unknown_model_is_rejected(cache):
    with pytest.raises(ValueError, match="three"):
        run_forecasts.run(TRAIN_END, ["three"])


def test_a_two_day_variant_is_run_only_when_named(cache):
    run_forecasts.run(TRAIN_END)
    assert "one_h2" not in models()
    run_forecasts.run(TRAIN_END, ["one_h2"])
    assert set(models()) == {"one", "two", "one_h2"}
    assert run_forecasts.two_day_names(["one"]) == ["one_h2"]


def test_a_two_day_variant_starts_one_day_later(cache):
    run_forecasts.run(TRAIN_END, ["one", "one_h2"])
    base = run_forecasts.load_forecasts(["one"], TRAIN_END)
    two_day = run_forecasts.load_forecasts(["one_h2"], TRAIN_END)
    assert two_day.index[0] == base.index[1]
    assert two_day.index[-1] == base.index[-1]
