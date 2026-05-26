"""batch_npz_to_gcp.py — Convert many raw_tile_match NPZs to ASP .gcp files.

Same per-file logic as raw_npz_to_gcp.py, but loads the DEM exactly once
across the whole batch (DEM read dominates per-call cost for the 5.7 GB
Pamir DEM). Designed for the production pipeline's GCP-regen step.

Usage:
  python batch_npz_to_gcp.py \\
      --npz-dir multi-track-rig/raw_match/stage1/npz \\
      --raw-dir multi-track-rig/raw \\
      --out-dir multi-track-rig/raw_match/stage1/gcps \\
      --dem dem-adj-43.tif \\
      --utm-zone-epsg 32643 \\
      [--res-suffix .10m]            # NPZs are {STEM}{suffix}.npz; output is {STEM}.gcp
      [--max-residual 3.0]           # default: no filter
      [--skip-existing]              # default: regenerate all
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio
from pyproj import Transformer


def sample_dem_bilinear(dem_arr, dem_tf, xs, ys):
    """Bilinear DEM sample at (xs, ys) in the DEM's CRS. Returns array of heights.
    Out-of-bounds or nodata -> NaN.
    """
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


def process_one(npz_path, raw_image_path, out_path,
                dem_arr, dem_tf, utm_to_dem, utm_to_wgs,
                max_residual=None,
                sig_horiz=5.0, sig_vert=10.0, sig_pix=1.0):
    d = np.load(npz_path)
    xs_ref = d["xs_ref"].astype(np.float64)
    ys_ref = d["ys_ref"].astype(np.float64)
    col_raw = d["col_kh9_raw"].astype(np.float64)
    row_raw = d["row_kh9_raw"].astype(np.float64)
    residual = d["residual_px"].astype(np.float32)
    n_in = len(xs_ref)

    if max_residual is not None:
        keep = residual <= max_residual
        xs_ref = xs_ref[keep]; ys_ref = ys_ref[keep]
        col_raw = col_raw[keep]; row_raw = row_raw[keep]
        residual = residual[keep]

    with rasterio.open(raw_image_path) as src:
        raw_W = src.width
        raw_H = src.height
    in_bounds = ((col_raw >= 0) & (col_raw <= raw_W - 1)
                 & (row_raw >= 0) & (row_raw <= raw_H - 1))
    if (~in_bounds).any():
        xs_ref = xs_ref[in_bounds]; ys_ref = ys_ref[in_bounds]
        col_raw = col_raw[in_bounds]; row_raw = row_raw[in_bounds]

    lons, lats = utm_to_wgs.transform(xs_ref, ys_ref)
    xs_dem, ys_dem = utm_to_dem.transform(xs_ref, ys_ref)
    heights = sample_dem_bilinear(dem_arr, dem_tf, xs_dem, ys_dem)
    valid_h = np.isfinite(heights)
    if (~valid_h).any():
        xs_ref = xs_ref[valid_h]; ys_ref = ys_ref[valid_h]
        col_raw = col_raw[valid_h]; row_raw = row_raw[valid_h]
        lons = np.asarray(lons)[valid_h]; lats = np.asarray(lats)[valid_h]
        heights = heights[valid_h]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write('# WKT: GEOGCS["WGS 84",DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433],'
                'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]\n')
        for i in range(len(xs_ref)):
            f.write(f"{i} {lats[i]:.15f} {lons[i]:.15f} {heights[i]:.6f} "
                    f"{sig_horiz} {sig_horiz} {sig_vert} "
                    f"{raw_image_path} "
                    f"{col_raw[i]:.3f} {row_raw[i]:.3f} "
                    f"{sig_pix} {sig_pix}\n")
    return len(xs_ref), n_in


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--utm-zone-epsg", type=int, default=32643)
    ap.add_argument("--res-suffix", default=".10m",
                    help="NPZ stem suffix to strip (e.g. .10m). Default: .10m")
    ap.add_argument("--raw-suffix", default="_sub4.tif",
                    help="Raw image suffix appended to image stem. Default: _sub4.tif")
    ap.add_argument("--max-residual", type=float, default=None)
    ap.add_argument("--sig-horiz", type=float, default=5.0)
    ap.add_argument("--sig-vert", type=float, default=10.0)
    ap.add_argument("--sig-pix", type=float, default=1.0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    npzs = sorted(glob.glob(os.path.join(args.npz_dir, f"*{args.res_suffix}.npz")))
    if not npzs:
        sys.exit(f"ERROR: no NPZs in {args.npz_dir} (suffix {args.res_suffix}.npz)")

    print(f"Loading DEM: {args.dem}", flush=True)
    with rasterio.open(args.dem) as dem_src:
        dem_arr = dem_src.read(1).astype(np.float32)
        dem_tf = dem_src.transform
        dem_crs = str(dem_src.crs)
        nd = dem_src.nodata
    if nd is not None:
        dem_arr = np.where(dem_arr == nd, np.nan, dem_arr)
    print(f"  shape={dem_arr.shape} crs={dem_crs}", flush=True)

    utm_crs = f"EPSG:{args.utm_zone_epsg}"
    utm_to_dem = Transformer.from_crs(utm_crs, dem_crs, always_xy=True)
    utm_to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    os.makedirs(args.out_dir, exist_ok=True)
    total_in = 0
    total_out = 0
    n_processed = 0
    for npz in npzs:
        stem = os.path.basename(npz).replace(f"{args.res_suffix}.npz", "")
        out = os.path.join(args.out_dir, f"{stem}.gcp")
        if args.skip_existing and os.path.isfile(out):
            print(f"  [skip] {out}", flush=True)
            continue
        raw = os.path.join(args.raw_dir, f"{stem}{args.raw_suffix}")
        if not os.path.isfile(raw):
            print(f"  WARNING: raw image missing for {stem}: {raw}", flush=True)
            continue
        # jitter_solve string-compares the GCP image path against its positional
        # args. The pipeline passes "./multi-track-rig/raw/..." so prepend "./"
        # for relative paths to ensure byte-identical match.
        if not os.path.isabs(raw) and not raw.startswith("./") and not raw.startswith("../"):
            raw = "./" + raw
        n_out, n_in = process_one(
            npz, raw, out, dem_arr, dem_tf, utm_to_dem, utm_to_wgs,
            max_residual=args.max_residual,
            sig_horiz=args.sig_horiz, sig_vert=args.sig_vert, sig_pix=args.sig_pix,
        )
        total_in += n_in; total_out += n_out
        n_processed += 1
        print(f"  {stem}: {n_out}/{n_in} -> {out}", flush=True)

    print(f"\nProcessed {n_processed} NPZs. Kept {total_out}/{total_in} matches.")


if __name__ == "__main__":
    main()
