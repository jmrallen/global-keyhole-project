"""Select SRTM GL1 tiles intersecting a bbox, mosaic, crop, and adjust to ellipsoid."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from . import config as cfg_mod
from . import footprint as fp_mod
from . import manifest as manifest_mod


TILE_RE = re.compile(r"^(N|S)(\d{1,2})(E|W)(\d{1,3})\.tif$", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _tile_bounds(name: str) -> tuple[int, int, int, int] | None:
    """Return (min_lon, min_lat, max_lon, max_lat) for a SRTM tile filename, or None."""
    m = TILE_RE.fullmatch(name)
    if m is None:
        return None
    ns, lat_s, ew, lon_s = m.groups()
    lat = int(lat_s)
    lon = int(lon_s)
    if ns.upper() == "S":
        lat = -lat
    if ew.upper() == "W":
        lon = -lon
    # SRTM tiles are named by the integer SW corner.
    return (lon, lat, lon + 1, lat + 1)


def select_tiles(tiles_dir: Path, bbox: fp_mod.BBox) -> list[Path]:
    """Return all .tif files in ``tiles_dir`` whose 1° footprint intersects ``bbox``."""
    out: list[Path] = []
    for p in sorted(tiles_dir.glob("*.tif")):
        b = _tile_bounds(p.name)
        if b is None:
            continue
        tx0, ty0, tx1, ty1 = b
        if tx0 < bbox.max_lon and tx1 > bbox.min_lon and ty0 < bbox.max_lat and ty1 > bbox.min_lat:
            out.append(p)
    return out


def _run(cmd: list[str]) -> None:
    log("[dem_tiles] $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def build_dem(tiles_dir: Path, bbox: fp_mod.BBox, out_path: Path, adjust_ellipsoid: bool = True) -> Path:
    """Build a bbox-cropped DEM mosaic and (optionally) convert orthometric -> ellipsoid heights.

    Returns the final DEM path.
    """
    tiles = select_tiles(tiles_dir, bbox)
    if not tiles:
        raise RuntimeError(
            f"no SRTM tiles in {tiles_dir} intersect bbox {bbox.as_tuple()}"
        )
    log(f"[dem_tiles] {len(tiles)} tile(s) selected: " + ", ".join(t.name for t in tiles[:6])
        + (" ..." if len(tiles) > 6 else ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vrt_path = out_path.with_suffix(".vrt")

    _run(["gdalbuildvrt", "-overwrite", str(vrt_path), *[str(t) for t in tiles]])
    _run([
        "gdal_translate",
        "-projwin", str(bbox.min_lon), str(bbox.max_lat), str(bbox.max_lon), str(bbox.min_lat),
        "-co", "COMPRESS=LZW", "-co", "TILED=YES",
        str(vrt_path), str(out_path),
    ])
    try:
        vrt_path.unlink()
    except OSError:
        pass

    if adjust_ellipsoid:
        stem = out_path.with_suffix("")  # dem_geoid appends -adj.tif when called with no .tif suffix
        _run([
            "dem_geoid", "--geoid", "egm96", "--reverse-adjustment",
            str(out_path), "-o", str(stem),
        ])
        adj_path = stem.parent / (stem.name + "-adj.tif")
        if adj_path.exists():
            adj_path.replace(out_path)
            log(f"[dem_tiles] ellipsoid-adjusted -> {out_path}")
        else:
            raise RuntimeError(
                f"dem_geoid did not produce expected output {adj_path}; check ASP install"
            )

    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the bbox DEM for the configured manifest.")
    p.add_argument("config", nargs="?", default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="output DEM path (default: <repo>/inputs/dem.tif)")
    p.add_argument("--no-adjust", action="store_true",
                   help="skip the dem_geoid ellipsoid adjustment")
    args = p.parse_args(argv)

    cfg = cfg_mod.load_config(args.config)
    strips = manifest_mod.resolve(cfg)
    info = fp_mod.compute_footprint(cfg, strips)
    out = args.out or (cfg_mod.repo_root() / "inputs" / "dem.tif")
    build_dem(cfg.paths.dem_tiles_dir, info["union_bbox"], out, adjust_ellipsoid=not args.no_adjust)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
