"""raw_npz_to_gcp.py — Convert raw_tile_match NPZ to ASP .gcp file.

Bypasses sift_to_gcp.py's mapped->raw projection step entirely (we already
have raw KH-9 pixel coords). For each match in the NPZ:
  - lat/lon = inverse-project xs_ref/ys_ref (UTM) -> WGS84
  - h = bilinear DEM sample at (xs_ref, ys_ref)
  - col, row = col_kh9_raw, row_kh9_raw
Then write ASP GCP lines in the production format.

Usage:
  python raw_npz_to_gcp.py \\
      --npz       work/npz/011_fwd.stage1.sift.npz \\
      --raw-image multi-track-rig/raw/D3C1216-300814-011_fwd_sub4.tif \\
      --dem       dem-adj-43.tif \\
      --output    work/gcps/011_fwd.stage1.sift.gcp \\
      --utm-zone-epsg 32643 \\
      [--max-residual 3.0] \\
      [--sig-horiz 5.0] [--sig-vert 10.0] [--sig-pix 1.0]
"""
import argparse
import os
import sys

import numpy as np
import rasterio
from pyproj import Transformer


def sample_dem_bilinear(dem_arr, dem_tf, xs, ys, src_crs, dem_crs):
    """Bilinear DEM sample at (xs, ys) in src_crs. Returns array of heights.
    Out-of-bounds or nodata -> NaN.
    """
    if str(src_crs).split(":")[-1] != str(dem_crs).split(":")[-1]:
        to_dem = Transformer.from_crs(src_crs, dem_crs, always_xy=True)
        xs, ys = to_dem.transform(np.asarray(xs), np.asarray(ys))
    cols = (np.asarray(xs) - dem_tf.c) / dem_tf.a
    rows = (np.asarray(ys) - dem_tf.f) / dem_tf.e
    H, W = dem_arr.shape
    h_out = np.full(cols.shape, np.nan, dtype=np.float64)
    valid = ((cols >= 0) & (cols <= W - 1) & (rows >= 0) & (rows <= H - 1))
    if not valid.any():
        return h_out
    ci0 = np.floor(cols[valid]).astype(np.int64)
    ci1 = np.clip(ci0 + 1, 0, W - 1)
    ri0 = np.floor(rows[valid]).astype(np.int64)
    ri1 = np.clip(ri0 + 1, 0, H - 1)
    fc = cols[valid] - ci0
    fr = rows[valid] - ri0
    v00 = dem_arr[ri0, ci0]; v01 = dem_arr[ri0, ci1]
    v10 = dem_arr[ri1, ci0]; v11 = dem_arr[ri1, ci1]
    h = (v00 * (1 - fc) * (1 - fr) + v01 * fc * (1 - fr)
         + v10 * (1 - fc) * fr + v11 * fc * fr)
    h_out[valid] = h
    return h_out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--raw-image", required=True,
                    help="Rotated sub4 raw image path (goes into gcp lines verbatim)")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--utm-zone-epsg", type=int, default=32643)
    ap.add_argument("--max-residual", type=float, default=None,
                    help="Drop GCPs with residual_px > this (default: no filter)")
    ap.add_argument("--sig-horiz", type=float, default=5.0,
                    help="GCP horizontal sigma in meters (default: 5)")
    ap.add_argument("--sig-vert", type=float, default=10.0,
                    help="GCP vertical sigma in meters (default: 10)")
    ap.add_argument("--sig-pix", type=float, default=1.0,
                    help="GCP pixel sigma (default: 1)")
    args = ap.parse_args()

    if not os.path.isfile(args.npz):
        sys.exit(f"ERROR: NPZ not found: {args.npz}")
    # Resolve the path for existence check + raster reads, but preserve the
    # caller's path verbatim for the GCP file. jitter_solve compares paths
    # against its own positional args (relative or absolute), and embedding an
    # abspath here breaks the match when the pipeline passes relative paths
    # — and breaks cross-machine portability (absolute WSL paths don't exist
    # on a copied tree).
    raw_image_path = args.raw_image
    raw_image_abs = os.path.abspath(raw_image_path)
    if not os.path.isfile(raw_image_abs):
        sys.exit(f"ERROR: raw image not found: {raw_image_abs}")

    d = np.load(args.npz)
    needed = ["xs_ref", "ys_ref", "col_kh9_raw", "row_kh9_raw", "residual_px"]
    for k in needed:
        if k not in d.files:
            sys.exit(f"ERROR: NPZ missing key '{k}'. Got: {d.files}")
    xs_ref = d["xs_ref"].astype(np.float64)
    ys_ref = d["ys_ref"].astype(np.float64)
    col_raw = d["col_kh9_raw"].astype(np.float64)
    row_raw = d["row_kh9_raw"].astype(np.float64)
    residual = d["residual_px"].astype(np.float32)
    n_in = len(xs_ref)
    print(f"Loaded {n_in} matches from {args.npz}", flush=True)

    # Filter by residual
    if args.max_residual is not None:
        keep = residual <= args.max_residual
        xs_ref = xs_ref[keep]; ys_ref = ys_ref[keep]
        col_raw = col_raw[keep]; row_raw = row_raw[keep]
        residual = residual[keep]
        print(f"  residual filter (<= {args.max_residual}): "
              f"kept {keep.sum()} / {n_in}", flush=True)

    # Get raw image dims to filter out-of-bounds pixels (defensive)
    with rasterio.open(raw_image_abs) as src:
        raw_W = src.width
        raw_H = src.height
    in_bounds = ((col_raw >= 0) & (col_raw <= raw_W - 1)
                 & (row_raw >= 0) & (row_raw <= raw_H - 1))
    if (~in_bounds).any():
        print(f"  Dropping {(~in_bounds).sum()} out-of-bounds raw pixels",
              flush=True)
        xs_ref = xs_ref[in_bounds]; ys_ref = ys_ref[in_bounds]
        col_raw = col_raw[in_bounds]; row_raw = row_raw[in_bounds]

    # UTM -> WGS84
    utm_crs = f"EPSG:{args.utm_zone_epsg}"
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    lons, lats = to_wgs.transform(xs_ref, ys_ref)

    # Sample DEM heights (DEM in EPSG:32643 per CLAUDE.md; reproject if needed)
    with rasterio.open(args.dem) as dem_src:
        dem_arr = dem_src.read(1).astype(np.float32)
        dem_tf = dem_src.transform
        dem_crs = dem_src.crs
        nd = dem_src.nodata
    if nd is not None:
        dem_arr = np.where(dem_arr == nd, np.nan, dem_arr)
    heights = sample_dem_bilinear(dem_arr, dem_tf, xs_ref, ys_ref,
                                  utm_crs, str(dem_crs))
    valid_h = np.isfinite(heights)
    if (~valid_h).any():
        print(f"  Dropping {(~valid_h).sum()} GCPs with NaN DEM height",
              flush=True)
        xs_ref = xs_ref[valid_h]; ys_ref = ys_ref[valid_h]
        col_raw = col_raw[valid_h]; row_raw = row_raw[valid_h]
        lons = np.asarray(lons)[valid_h]
        lats = np.asarray(lats)[valid_h]
        heights = heights[valid_h]

    # Write
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    sh = args.sig_horiz
    sv = args.sig_vert
    sp = args.sig_pix
    n_out = 0
    with open(args.output, "w") as f:
        f.write('# WKT: GEOGCS["WGS 84",DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433],'
                'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]\n')
        for i in range(len(xs_ref)):
            f.write(f"{n_out} {lats[i]:.15f} {lons[i]:.15f} {heights[i]:.6f} "
                    f"{sh} {sh} {sv} "
                    f"{raw_image_path} "
                    f"{col_raw[i]:.3f} {row_raw[i]:.3f} "
                    f"{sp} {sp}\n")
            n_out += 1
    print(f"Wrote {n_out} GCPs -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
