"""Download SPY, VIX and DGS3MO, freeze a raw snapshot, and build the shared dataset.

Usage:
    python -m data.download --freeze   # download fresh data, snapshot + manifest, build
    python -m data.download            # verify snapshot against manifest, build cache

The snapshot files and their manifest (data/snapshots/snapshot_manifest.json) are
tracked in git. Every build verifies checksums so every machine uses identical raw data.
"""
