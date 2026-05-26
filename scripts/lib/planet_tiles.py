"""Select Planet panchromatic tiles intersecting a bbox and crop to a working mosaic.

The Planet directory holds an arbitrary number of GeoTIFFs (often a single
large mosaic per AOI). We probe each .tif's bounds, keep the ones that
intersect the padded bbox, and gdalbuildvrt + gdal_translate them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import config as cfg_mod
from . import footprint as fp_mod
from . import manifest as manifest_mod


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _bounds_lonlat(path: Path) -> tuple[float, float, float, float] | None:
    """Return (minx, miny, maxx, maxy) in EPSG:4326 for the GeoTIFF at ``path``."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as e:
        raise RuntimeError("rasterio is required for Planet tile selection") from e

    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            return tuple(b)
    except Exception as e:
        log(f"[planet_tiles] WARNING: could not read bounds for {path.name}: {e}")
        return None


def select_tiles(tiles_dir: Path, bbox: fp_mod.BBox) -> list[Path]:
    if not tiles_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(tiles_dir.glob("*.tif")):
        b = _bounds_lonlat(p)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        if x0 < bbox.max_lon and x1 > bbox.min_lon and y0 < bbox.max_lat and y1 > bbox.min_lat:
            out.append(p)
    return out


def _run(cmd: list[str]) -> None:
    log("[planet_tiles] $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def build_planet_mosaic(tiles_dir: Path, bbox: fp_mod.BBox, out_path: Path, required: bool = False) -> Path | None:
    """Build a bbox-cropped Planet mosaic. Returns the output path or None if no tiles
    were available (and ``required`` is False).
    """
    tiles = select_tiles(tiles_dir, bbox)
    if not tiles:
        msg = f"no Planet tiles in {tiles_dir} intersect bbox {bbox.as_tuple()}"
        if required:
            raise RuntimeError(msg)
        log(f"[planet_tiles] WARNING: {msg} (skipping — S2 will fail until populated)")
        return None
    log(f"[planet_tiles] {len(tiles)} tile(s) selected: " + ", ".join(t.name for t in tiles[:6])
        + (" ..." if len(tiles) > 6 else ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vrt_path = out_path.with_suffix(".vrt")

    _run(["gdalbuildvrt", "-overwrite", str(vrt_path), *[str(t) for t in tiles]])
    _run([
        "gdal_translate",
        "-projwin", str(bbox.min_lon), str(bbox.max_lat), str(bbox.max_lon), str(bbox.min_lat),
        "-projwin_srs", "EPSG:4326",
        "-co", "COMPRESS=LZW", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
        str(vrt_path), str(out_path),
    ])
    try:
        vrt_path.unlink()
    except OSError:
        pass
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the bbox Planet mosaic for the configured manifest.")
    p.add_argument("config", nargs="?", default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="output mosaic path (default: <repo>/inputs/planet.tif)")
    p.add_argument("--required", action="store_true",
                   help="fail hard if no overlapping Planet tiles are found")
    args = p.parse_args(argv)

    cfg = cfg_mod.load_config(args.config)
    strips = manifest_mod.resolve(cfg)
    info = fp_mod.compute_footprint(cfg, strips)
    out = args.out or (cfg_mod.repo_root() / "inputs" / "planet.tif")
    build_planet_mosaic(cfg.paths.planet_tiles_dir, info["union_bbox"], out, required=args.required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
