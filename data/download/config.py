"""Constants, paths and errors shared by the download package."""

from pathlib import Path


class DataError(Exception):
    """Snapshot or dataset validation failed; the build must not proceed."""


START = "2000-01-01"
END = "2026-06-30"
FRED_START = "1999-12-01"
FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    f"?id=DGS3MO&cosd={FRED_START}&coed={END}"
)

DATA_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = DATA_DIR / "snapshots"
CACHE_DIR = DATA_DIR / "cache"
MANIFEST_PATH = SNAPSHOT_DIR / "snapshot_manifest.json"
DATASET_PATH = CACHE_DIR / "spy_dataset.parquet"

VIX_TICKER = "^VIX"

SCHEMA_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "ret",
    "log_ret",
    "rf",
    "dividend",
    "realized_vol_gk",
    "vix",
]
