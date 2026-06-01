"""Build docs/archive/declass3.pmtiles from assets/declass3_metadata.parquet.

Reads the master frame catalog, normalizes column names to lower-snake-case,
converts WKB footprint polygons to GeoJSON, writes line-delimited GeoJSON to a
temp file, then shells out to `tippecanoe` to produce a PMTiles archive that
the Image Archive page (docs/archive/index.html via overrides/archive.html)
loads with the pmtiles JS protocol handler.

Tippecanoe is Linux/macOS only. On Windows, run via WSL or skip — CI runs on
Ubuntu and handles the build there.

Usage:
    python scripts/web/build_pmtiles.py [--parquet PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from shapely import wkb

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = REPO_ROOT / "assets" / "declass3_metadata.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "archive" / "declass3.pmtiles"

# Normalized property keys the map JS expects, and the prefixes (case-insensitive)
# we'll use to find the matching source column. Source column names in the
# parquet are often truncated (e.g. "Acquisitio" for "Acquisition Date"), so
# prefix-matching is more robust than exact-name matching.
#
# Only `entity_id` and `geometry` are required; the rest are best-effort.
FIELDS = [
    ("entity_id", ["entity"]),
    ("mission", ["mission"]),
    ("frame", ["frame"]),
    ("camera", ["camera"]),
    ("acq_date", ["acquisit"]),
    ("resolution", ["resolut", "camera res"]),
    ("download_avail", ["download"]),
]


def resolve_columns(src_cols: list[str]) -> dict[str, str]:
    """Map normalized field name → actual source column name, by prefix match."""
    lower_to_actual = {c.lower(): c for c in src_cols}
    resolved: dict[str, str] = {}
    for dst, prefixes in FIELDS:
        for c_lower, c_actual in lower_to_actual.items():
            if any(c_lower.startswith(p) for p in prefixes):
                resolved[dst] = c_actual
                break
    return resolved


def parquet_to_geojsonl(parquet_path: Path, out_path: Path) -> int:
    """Stream parquet → line-delimited GeoJSON. Returns feature count written."""
    table = pq.read_table(parquet_path)
    src_cols = table.column_names

    if "geometry" not in src_cols:
        sys.exit(f"Parquet is missing required 'geometry' column. Available: {src_cols}")

    resolved = resolve_columns(src_cols)
    if "entity_id" not in resolved:
        sys.exit(
            f"Parquet is missing a column starting with 'entity'. Available: {src_cols}"
        )

    print(f"Available columns ({len(src_cols)}): {src_cols}")
    print("Resolved field mapping:")
    for dst, _ in FIELDS:
        print(f"  {dst:16s} ← {resolved.get(dst, '(not found, will skip)')}")

    df = table.to_pandas()

    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            row_d = row._asdict()
            wkb_bytes = row_d.get("geometry")
            if not wkb_bytes:
                skipped += 1
                continue
            try:
                geom = wkb.loads(bytes(wkb_bytes))
            except Exception:
                skipped += 1
                continue
            if geom.is_empty or not geom.is_valid:
                skipped += 1
                continue

            props = {}
            for dst, src in resolved.items():
                v = row_d.get(src)
                if v is None or (isinstance(v, float) and v != v):  # NaN
                    continue
                if dst == "acq_date" and hasattr(v, "isoformat"):
                    v = v.date().isoformat()
                props[dst] = v if isinstance(v, (str, int, float, bool)) else str(v)

            feature = {
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": props,
            }
            fh.write(json.dumps(feature, separators=(",", ":")))
            fh.write("\n")
            written += 1

    print(f"Wrote {written:,} features ({skipped:,} skipped) → {out_path}")
    return written


def run_tippecanoe(geojsonl: Path, output: Path) -> None:
    if shutil.which("tippecanoe") is None:
        sys.exit(
            "tippecanoe not found on PATH. Install via:\n"
            "  Ubuntu: apt-get install tippecanoe\n"
            "  macOS:  brew install tippecanoe\n"
            "  Windows: use WSL"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tippecanoe",
        "-l", "declass3",
        "-o", str(output),
        "--force",
        # Zoom range: globe view down to ~city scale. KH-9 footprints are
        # large (~hundreds of km across) so we don't need zoom > 8.
        "-Z", "0",
        "-z", "8",
        # Polygon-friendly defaults.
        "--drop-densest-as-needed",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
        "--no-tile-size-limit",
        "--no-feature-limit",
        # Visual fidelity.
        "--simplification=10",
        str(geojsonl),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if not args.parquet.exists():
        sys.exit(f"Parquet not found: {args.parquet}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".geojsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        n = parquet_to_geojsonl(args.parquet, tmp_path)
        if n == 0:
            sys.exit("No valid features written — aborting.")
        run_tippecanoe(tmp_path, args.output)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
