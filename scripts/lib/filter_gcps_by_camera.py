#!/usr/bin/env python3
"""filter_gcps_by_camera.py — drop GCPs whose ground point doesn't fit their solved camera.

After a first pass of jitter_solve, Cauchy-attenuated outlier GCPs survive in the
solution with astronomical residuals (we measured 1.5% of GCPs with residual > 100 px,
0.8% with > 1M px). They drag a single camera (014_fwd) to a 1e+76 mean residual and
inflate its mapproject envelope by ~60%, even though the rest of the cameras converged
cleanly. `jitter_solve --max-initial-reprojection-error` does NOT filter GCPs (it only
filters tie-points), so the only way to drop them is to re-project each GCP through the
solved camera, compute the residual, and trim by it.

Workflow:
  1. Run Phase 7 with the original planet_gcps/ once.
  2. Run this script to produce planet_gcps_clean/.
  3. Re-run Phase 7 pointing at planet_gcps_clean/.

Default threshold of 50 px sits in the obvious-outlier gap (the per-point residual
distribution has p95=10.5 px, p99=1.3e+4 px — anything > ~20 px is a real outlier).
"""
import argparse
import os
import sys
import glob

# Reuse the CSM loader from sift_to_gcp.py — same plugin discovery path and quirks.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from sift_to_gcp import load_csm_model  # noqa: E402

from pyproj import Transformer  # noqa: E402


def gcp_stem_to_cam_path(stem, cam_dir, cam_prefix, cam_suffix):
    """e.g. 'D3C1216-300814-014_fwd' -> '<cam_dir>/run-D3C1216-300814-014_fwd_sub4.adjusted_state.json'"""
    return os.path.join(cam_dir, f"{cam_prefix}{stem}_sub4{cam_suffix}")


def filter_one(gcp_path, model, csmapi, ecef_tf, max_residual):
    """Returns (n_kept, n_dropped, list_of_kept_lines, list_of_header_lines)."""
    kept = []
    headers = []
    n_drop = 0
    with open(gcp_path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                kept.append(line)
                continue
            if s.lstrip().startswith("#"):
                headers.append(line)
                kept.append(line)
                continue
            parts = s.split()
            # Format: id lat lon h sig_h sig_h sig_v img_path col row sig_pix [sig_pix]
            if len(parts) < 11:
                kept.append(line)
                continue
            try:
                lat = float(parts[1])
                lon = float(parts[2])
                h = float(parts[3])
                obs_col = float(parts[8])
                obs_row = float(parts[9])
            except ValueError:
                kept.append(line)
                continue

            x, y, z = ecef_tf.transform(lon, lat, h)
            try:
                pt = model.groundToImage(csmapi.EcefCoord(x, y, z))
            except Exception:
                n_drop += 1
                continue
            pred_col, pred_row = pt.samp, pt.line
            # Out-of-bounds projections (huge/NaN) are guaranteed outliers.
            import math
            if not (math.isfinite(pred_col) and math.isfinite(pred_row)):
                n_drop += 1
                continue
            d = math.hypot(pred_col - obs_col, pred_row - obs_row)
            if d <= max_residual:
                kept.append(line)
            else:
                n_drop += 1

    return len(kept) - len(headers), n_drop, kept, headers


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gcp-dir", default="multi-track-rig/planet_gcps",
                    help="directory of input *.gcp files")
    ap.add_argument("--cam-dir", default="multi-track-rig/jitter_solve",
                    help="directory of solved adjusted_state.json files")
    ap.add_argument("--cam-prefix", default="run-",
                    help="prefix on camera filenames (default 'run-')")
    ap.add_argument("--cam-suffix", default=".adjusted_state.json",
                    help="suffix on camera filenames")
    ap.add_argument("--out-dir", default="multi-track-rig/planet_gcps_clean",
                    help="output directory for cleaned .gcp files")
    ap.add_argument("--max-residual", type=float, default=50.0,
                    help="drop GCPs whose camera-reprojection residual exceeds this (px)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gcp_files = sorted(glob.glob(os.path.join(args.gcp_dir, "*.gcp")))
    if not gcp_files:
        print(f"ERROR: no .gcp files in {args.gcp_dir}", file=sys.stderr)
        sys.exit(1)

    ecef_tf = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    total_kept = 0
    total_dropped = 0
    print(f"{'gcp file':<40} {'kept':>6} {'dropped':>8}  pct")
    for gcp_path in gcp_files:
        stem = os.path.splitext(os.path.basename(gcp_path))[0]
        cam_path = gcp_stem_to_cam_path(stem, args.cam_dir, args.cam_prefix, args.cam_suffix)
        if not os.path.isfile(cam_path):
            print(f"  WARNING: no camera for {stem} (looked for {cam_path}) — skipping", file=sys.stderr)
            continue

        model, csmapi = load_csm_model(cam_path)
        kept, dropped, kept_lines, _ = filter_one(
            gcp_path, model, csmapi, ecef_tf, args.max_residual)
        total = kept + dropped
        pct = (100.0 * dropped / total) if total else 0.0
        print(f"{stem:<40} {kept:>6} {dropped:>8}  {pct:5.1f}%")

        out_path = os.path.join(args.out_dir, os.path.basename(gcp_path))
        with open(out_path, "w") as f:
            f.writelines(kept_lines)
        total_kept += kept
        total_dropped += dropped

    overall = total_kept + total_dropped
    pct = (100.0 * total_dropped / overall) if overall else 0.0
    print(f"\nTotal: {total_kept} kept, {total_dropped} dropped  ({pct:.1f}%)")
    print(f"Cleaned files in: {args.out_dir}")


if __name__ == "__main__":
    main()
