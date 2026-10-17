"""Freeze raw downloads into a snapshot and verify it against the manifest."""

import hashlib
import json
import logging
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance

from .config import (
    END,
    FRED_URL,
    MANIFEST_PATH,
    SNAPSHOT_DIR,
    START,
    VIX_TICKER,
    DataError,
)

log = logging.getLogger(__name__)

COVERAGE_TOLERANCE = pd.Timedelta(days=7)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def naive_index(index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    return index.tz_localize(None) if index.tz is not None else index


def history(ticker: str, label: str) -> pd.DataFrame:
    """Download a full history, refusing anything that misses the sample."""
    end_exclusive = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frame = yfinance.Ticker(ticker).history(
        start=START, end=end_exclusive, auto_adjust=False, actions=True
    )
    if frame.empty:
        raise DataError(f"{label} download returned no data")
    index = naive_index(frame.index)
    if (
        index[0] > pd.Timestamp(START) + COVERAGE_TOLERANCE
        or index[-1] < pd.Timestamp(END) - COVERAGE_TOLERANCE
    ):
        raise DataError(
            f"{label} download covers {index[0].date()}..{index[-1].date()}, "
            f"expected {START}..{END}; refusing to freeze a partial download"
        )
    log.info(
        "%s downloaded: %d rows, %s to %s",
        label,
        len(frame),
        index[0].date(),
        index[-1].date(),
    )
    return frame


def freeze() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    spy_path = SNAPSHOT_DIR / f"spy_raw_{stamp}.parquet"
    history("SPY", "SPY").to_parquet(spy_path)
    vix_path = SNAPSHOT_DIR / f"vix_raw_{stamp}.parquet"
    history(VIX_TICKER, "VIX").to_parquet(vix_path)

    with urllib.request.urlopen(FRED_URL, timeout=60) as response:
        fred_bytes = response.read()
    header = fred_bytes.split(b"\n", 1)[0]
    if b"DGS3MO" not in header:
        raise DataError(
            f"DGS3MO download is not the expected CSV: header {header[:80]!r}"
        )
    fred_path = SNAPSHOT_DIR / f"dgs3mo_raw_{stamp}.csv"
    fred_path.write_bytes(fred_bytes)
    log.info("DGS3MO downloaded: %d bytes", len(fred_bytes))

    if MANIFEST_PATH.exists():
        log.warning("replacing existing manifest %s", MANIFEST_PATH)
    manifest = {
        "download_date": stamp,
        "start": START,
        "end": END,
        "yfinance_version": yfinance.__version__,
        "files": {p.name: sha256(p) for p in (spy_path, vix_path, fred_path)},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, digest in manifest["files"].items():
        log.info("snapshot %s sha256=%s", name, digest)
    log.info("manifest written; commit it together with the snapshot files")


def verified_snapshot_paths() -> dict[str, Path]:
    if not MANIFEST_PATH.exists():
        raise DataError("no snapshot manifest found; run with --freeze to create one")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (manifest["start"], manifest["end"]) != (START, END):
        raise DataError(
            f"configured sample {START}..{END} differs from the manifest "
            f"{manifest['start']}..{manifest['end']}; "
            "the frozen snapshot does not cover the configured sample"
        )
    paths = {}
    for name, expected in manifest["files"].items():
        path = SNAPSHOT_DIR / name
        if not path.exists():
            raise DataError(
                f"snapshot file missing: {path}\n"
                "restore the committed snapshot in data/snapshots"
            )
        if sha256(path) != expected:
            raise DataError(
                f"checksum mismatch for {name}\n"
                "local snapshot differs from the manifest; "
                "restore the committed snapshot in data/snapshots"
            )
        key = name.split("_raw_")[0]
        paths[key] = path
    for key in ("spy", "vix", "dgs3mo"):
        if key not in paths:
            raise DataError(f"manifest has no {key} snapshot file")
    log.info("snapshot verified against manifest (%s)", manifest["download_date"])
    return paths
