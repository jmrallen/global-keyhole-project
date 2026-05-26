"""raw_tile_match.py — Raw-space tile matcher for KH-9 jitter solve.

Per UTM tile:
  1. Project UTM corners through synth.groundToImage -> raw bbox.
  2. Crop the (rotated) raw KH-9 to that bbox.
  3. Build a sparse imageToGround warp over the raw bbox (every N raw px),
     iterating DEM heights to convergence; bilinear-fill to dense (Hr, Wr).
  4. Convert dense UTM grid -> Planet pixel coords, cv2.remap Planet into
     raw-pixel frame -> warped_planet aligned with raw_kh9 crop.
  5. Downsample both to match_res (e.g. 10 m/px), CLAHE-enhance, dispatch to
     matcher adapter (SIFT or DKM) -> matches in tile-pixel coords.
  6. Convert matches back to raw-pixel coords (c1_raw, r1_raw) for KH-9 and
     (c2_raw, r2_raw) for warped-planet. Look up U_real = dense_UTM(c2, r2).
  7. Optional camera-consistency filter |groundToImage(U_real) - (c1, r1)| > k.

Output NPZ schema (superset of bake-off schema):
    xs_ref          : (N,) float64  U_real easting (UTM, ground truth)
    ys_ref          : (N,) float64  U_real northing
    col_kh9_raw     : (N,) float32  raw KH-9 col (observation)
    row_kh9_raw     : (N,) float32  raw KH-9 row
    col_kh9_mapped  : (N,) float32  synth.groundToImage(U_real) col (for shp gen)
    row_kh9_mapped  : (N,) float32  synth.groundToImage(U_real) row
    residual_px     : (N,) float32  1/confidence (or RANSAC residual at match_res)
    confidence      : (N,) float32  matcher confidence in [0, 1]

The raw KH-9 file MUST be the rotated sub4 from multi-track-rig/raw/ —
the CSM camera in multi-track-rig/cameras/ is built for that orientation
(Phase 3 image_mosaic --rotate-90). A startup round-trip check confirms.

Usage:
  python raw_tile_match.py \\
      --raw      multi-track-rig/raw/D3C1216-300814-011_fwd_sub4.tif \\
      --planet   planet_pan_full.tif \\
      --camera   multi-track-rig/cameras/D3C1216-300814-011_fwd_sub4.json \\
      --dem      dem-adj-43.tif \\
      --matcher  {sift|dkm} \\
      --match-res 10 \\
      --tile-utm-km 8 --tile-step-frac 0.5 \\
      --output   work/npz/011_fwd.stage1.sift.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

# Project-root helpers
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from sift_to_gcp import load_csm_model  # noqa: E402
from run_sift_per_strip import apply_clahe, _resolve_device  # noqa: E402

# Bake-off adapters on path
_BAKEOFF = os.path.abspath(os.path.join(_ROOT, "experiments", "bakeoff"))
if _BAKEOFF not in sys.path:
    sys.path.insert(0, _BAKEOFF)


# ----------------------------------------------------------------------------
# Matcher dispatch
# ----------------------------------------------------------------------------

def _sift_match_tile(tile_kh9, tile_plan,
                     device=None,
                     max_keypoints=8000,
                     min_confidence=0.0,
                     ratio=0.75,
                     ransac_thresh=3.0,
                     **kwargs):
    """SIFT + FLANN + Lowe-ratio + RANSAC; same conventions as run_sift_per_strip."""
    sift = cv2.SIFT_create(nfeatures=max_keypoints,
                           contrastThreshold=0.03,
                           edgeThreshold=10)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))

    enh_kh9 = apply_clahe(tile_kh9)
    enh_plan = apply_clahe(tile_plan)
    kp1, des1 = sift.detectAndCompute(enh_kh9, None)
    kp2, des2 = sift.detectAndCompute(enh_plan, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return (np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32))
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return (np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32))
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    if len(good) < 4:
        return (np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32))
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src_pts.reshape(-1, 1, 2),
                                 dst_pts.reshape(-1, 1, 2),
                                 cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return (np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32))
    inlier = mask.ravel().astype(bool)
    src = src_pts[inlier]
    dst = dst_pts[inlier]
    if len(src) < 4:
        return (np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32))
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(proj - dst, axis=1).astype(np.float32)
    # Convert RANSAC residual -> pseudo-confidence in [0, 1] (lower res = higher conf)
    confs = np.clip(1.0 - residuals / (4.0 * ransac_thresh + 1e-6), 0.05, 1.0).astype(np.float32)
    return src, dst, confs


def load_match_fn(name):
    if name == "sift":
        return _sift_match_tile
    if name == "dkm":
        from adapters.matcher_dkm import match_tile  # noqa: WPS433
        return match_tile
    raise ValueError(f"Unknown matcher '{name}' (allowed: sift, dkm)")


# ----------------------------------------------------------------------------
# csmapi wrappers
# ----------------------------------------------------------------------------

class CsmGeom:
    """Cached CSM model + projection helpers."""

    def __init__(self, model, csmapi_mod, dem_arr, dem_tf, dem_crs,
                 utm_crs, h_init=4000.0):
        self.model = model
        self.csm = csmapi_mod
        self.dem = dem_arr
        self.dem_tf = dem_tf
        self.h_init = float(h_init)

        self.to_wgs = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
        self.to_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        self.ecef_to_utm = Transformer.from_crs("EPSG:4978", utm_crs, always_xy=True)
        self.utm_to_ecef = Transformer.from_crs(utm_crs, "EPSG:4978", always_xy=True)

    def _dem_sample(self, lon, lat):
        dx, dy = self.to_dem.transform(lon, lat)
        col_d = (dx - self.dem_tf.c) / self.dem_tf.a
        row_d = (dy - self.dem_tf.f) / self.dem_tf.e
        if not (0 <= col_d < self.dem.shape[1] and 0 <= row_d < self.dem.shape[0]):
            return None
        return float(self.dem[int(row_d), int(col_d)])

    def dem_sample_utm(self, x_utm, y_utm):
        """Direct DEM lookup at a UTM point. Assumes DEM CRS == self.utm_crs."""
        col_d = (x_utm - self.dem_tf.c) / self.dem_tf.a
        row_d = (y_utm - self.dem_tf.f) / self.dem_tf.e
        if not (0 <= col_d < self.dem.shape[1] and 0 <= row_d < self.dem.shape[0]):
            return None
        return float(self.dem[int(row_d), int(col_d)])

    def imageToGround_utm(self, col, row, max_iter=5, tol=0.1):
        """Iterate imageToGround against DEM, return (x_utm, y_utm) or (nan, nan)."""
        h = self.h_init
        ecef = None
        for _ in range(max_iter):
            try:
                ic = self.csm.ImageCoord(float(row), float(col))
                ecef = self.model.imageToGround(ic, float(h))
            except Exception:
                return np.nan, np.nan
            lon, lat, _ = self.to_wgs.transform(ecef.x, ecef.y, ecef.z)
            h_dem = self._dem_sample(lon, lat)
            if h_dem is None:
                break
            if abs(h_dem - h) < tol:
                h = h_dem
                break
            h = h_dem
        if ecef is None:
            return np.nan, np.nan
        x, y, _ = self.ecef_to_utm.transform(ecef.x, ecef.y, ecef.z)
        return float(x), float(y)

    def groundToImage_utm(self, x_utm, y_utm, h=None):
        """Project UTM point to raw pixel. If h is None, look up DEM at (x_utm, y_utm)."""
        if h is None:
            h_dem = self.dem_sample_utm(x_utm, y_utm)
            if h_dem is None:
                return np.nan, np.nan
            h = h_dem
        try:
            ex, ey, ez = self.utm_to_ecef.transform(float(x_utm), float(y_utm), float(h))
            pt = self.model.groundToImage(self.csm.EcefCoord(ex, ey, ez))
            return float(pt.samp), float(pt.line)  # col, row
        except Exception:
            return np.nan, np.nan


# ----------------------------------------------------------------------------
# Tile geometry
# ----------------------------------------------------------------------------

def compute_raw_utm_bbox(geom, W, H):
    """Project raw image corners through imageToGround_utm, return bbox."""
    pts = []
    for c, r in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1),
                 (W // 2, 0), (W // 2, H - 1), (0, H // 2), (W - 1, H // 2)]:
        x, y = geom.imageToGround_utm(c, r)
        if np.isfinite(x):
            pts.append((x, y))
    if len(pts) < 3:
        raise RuntimeError("Could not project enough raw image corners to UTM.")
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def iterate_utm_tiles(extent_utm, tile_m, step_m):
    x0, y0, x1, y1 = extent_utm
    xs = np.arange(x0, x1 - tile_m + step_m * 0.5, step_m)
    ys = np.arange(y0, y1 - tile_m + step_m * 0.5, step_m)
    for x in xs:
        for y in ys:
            yield (float(x), float(y), float(x + tile_m), float(y + tile_m))


def utm_bbox_to_raw_bbox(geom, utm_bbox, W, H, margin_px=64):
    """Project UTM-bbox corners + center through groundToImage (auto-DEM) to find raw bbox."""
    x0, y0, x1, y1 = utm_bbox
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1),
               ((x0 + x1) / 2, (y0 + y1) / 2),
               ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1),
               (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)]
    cs, rs = [], []
    for (xx, yy) in corners:
        c, r = geom.groundToImage_utm(xx, yy)  # auto-DEM
        if np.isfinite(c) and np.isfinite(r):
            cs.append(c)
            rs.append(r)
    if len(cs) < 3:
        return None
    c0 = max(0, int(np.floor(min(cs))) - margin_px)
    r0 = max(0, int(np.floor(min(rs))) - margin_px)
    c1 = min(W, int(np.ceil(max(cs))) + margin_px)
    r1 = min(H, int(np.ceil(max(rs))) + margin_px)
    if c1 - c0 < 32 or r1 - r0 < 32:
        return None
    return (c0, r0, c1, r1)


# ----------------------------------------------------------------------------
# Warp building
# ----------------------------------------------------------------------------

def build_sparse_warp(geom, raw_bbox, sparse_step):
    """imageToGround_utm on a sparse grid; return (sparse_rows, sparse_cols, X, Y)."""
    c0, r0, c1, r1 = raw_bbox
    sparse_cols = np.arange(c0, c1, sparse_step, dtype=np.float64)
    sparse_rows = np.arange(r0, r1, sparse_step, dtype=np.float64)
    # Ensure last sample reaches the bbox edge for clean interp coverage
    if sparse_cols[-1] < c1 - 1:
        sparse_cols = np.append(sparse_cols, float(c1 - 1))
    if sparse_rows[-1] < r1 - 1:
        sparse_rows = np.append(sparse_rows, float(r1 - 1))

    X = np.zeros((len(sparse_rows), len(sparse_cols)), dtype=np.float64)
    Y = np.zeros_like(X)
    for ri, r in enumerate(sparse_rows):
        for ci, c in enumerate(sparse_cols):
            x, y = geom.imageToGround_utm(c, r)
            X[ri, ci] = x
            Y[ri, ci] = y
    return sparse_rows, sparse_cols, X, Y


def dense_warp_from_sparse(sparse_rows, sparse_cols, X, Y, raw_bbox):
    """Bilinearly interpolate sparse (X, Y) to dense (Hr, Wr) grid."""
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        return None, None
    c0, r0, c1, r1 = raw_bbox
    Wr = c1 - c0
    Hr = r1 - r0
    interp_x = RegularGridInterpolator((sparse_rows, sparse_cols), X,
                                       bounds_error=False, fill_value=np.nan)
    interp_y = RegularGridInterpolator((sparse_rows, sparse_cols), Y,
                                       bounds_error=False, fill_value=np.nan)
    rr, cc = np.meshgrid(np.arange(r0, r1, dtype=np.float64),
                         np.arange(c0, c1, dtype=np.float64),
                         indexing="ij")
    pts = np.stack([rr.ravel(), cc.ravel()], axis=1)
    dense_X = interp_x(pts).reshape(Hr, Wr)
    dense_Y = interp_y(pts).reshape(Hr, Wr)
    return dense_X, dense_Y


def crop_planet_for_utm(planet_ds, utm_xy_extent, margin_m=200.0):
    """Read a Planet window covering utm_xy_extent + margin. Returns (uint8, transform)."""
    x0, y0, x1, y1 = utm_xy_extent
    x0 -= margin_m; x1 += margin_m
    y0 -= margin_m; y1 += margin_m
    win = from_bounds(x0, y0, x1, y1, planet_ds.transform)
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    # Clip to dataset
    win_off_c = max(0, win.col_off)
    win_off_r = max(0, win.row_off)
    win_w = max(1, win.width - (win_off_c - win.col_off))
    win_h = max(1, win.height - (win_off_r - win.row_off))
    win_w = min(win_w, planet_ds.width - win_off_c)
    win_h = min(win_h, planet_ds.height - win_off_r)
    if win_w < 4 or win_h < 4:
        return None, None
    safe_win = rasterio.windows.Window(win_off_c, win_off_r, win_w, win_h)
    arr = planet_ds.read(1, window=safe_win).astype(np.float32)
    tf = rasterio.windows.transform(safe_win, planet_ds.transform)
    nd = planet_ds.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    if not np.isfinite(arr).any():
        return None, None
    lo, hi = np.nanpercentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1.0), 0, 1)
    arr = np.nan_to_num(arr, nan=0.0)
    return (arr * 255).astype(np.uint8), tf


# ----------------------------------------------------------------------------
# Main per-tile processing
# ----------------------------------------------------------------------------

def process_tile(geom, raw_arr, raw_W, raw_H, planet_ds,
                 utm_bbox, match_res, native_res,
                 match_fn, match_kwargs,
                 sparse_step, cam_filter_thresh):
    """Return list of (xs_ref, ys_ref, col_kh9_raw, row_kh9_raw,
       col_kh9_mapped, row_kh9_mapped, residual_px, confidence)."""
    raw_bbox = utm_bbox_to_raw_bbox(geom, utm_bbox, raw_W, raw_H)
    if raw_bbox is None:
        return None, "raw_bbox_invalid"

    c0, r0, c1, r1 = raw_bbox
    raw_crop = raw_arr[r0:r1, c0:c1]
    if raw_crop.mean() < 5:
        return None, "raw_crop_nodata"

    # 1. Sparse imageToGround warp
    sparse_rows, sparse_cols, sX, sY = build_sparse_warp(geom, raw_bbox, sparse_step)
    if not (np.all(np.isfinite(sX)) and np.all(np.isfinite(sY))):
        return None, "sparse_warp_nonfinite"

    # 2. Dense interp
    dense_X, dense_Y = dense_warp_from_sparse(sparse_rows, sparse_cols, sX, sY, raw_bbox)
    if dense_X is None:
        return None, "dense_warp_failed"

    finite_mask = np.isfinite(dense_X) & np.isfinite(dense_Y)
    if finite_mask.sum() < 0.5 * dense_X.size:
        return None, "dense_warp_too_few_finite"

    # 3. UTM extent of valid dense warp, crop Planet
    utm_xy_extent = (
        float(np.nanmin(dense_X)), float(np.nanmin(dense_Y)),
        float(np.nanmax(dense_X)), float(np.nanmax(dense_Y)),
    )
    # Sanity gate BEFORE the expensive Planet crop: a corrupt camera can blow
    # the dense-warp UTM extent up to hundreds of km, making crop_planet_for_utm
    # read enormous Planet regions only to be rejected by SHRT_MAX downstream.
    # 50 km is ~4x a normal 12 km tile — well above legitimate ranges, well
    # below the ~98 km SHRT_MAX limit at Planet's 3 m/px.
    if (utm_xy_extent[2] - utm_xy_extent[0] > 50000.0
            or utm_xy_extent[3] - utm_xy_extent[1] > 50000.0):
        return None, "utm_extent_oversize"
    planet_crop, planet_tf = crop_planet_for_utm(planet_ds, utm_xy_extent)
    if planet_crop is None:
        return None, "planet_crop_empty"
    # cv2.remap requires every map dim and the source < SHRT_MAX (32767).
    # The utm_extent_oversize gate above catches most pathological cases; this
    # is the defense-in-depth backstop for borderline crops.
    if planet_crop.shape[0] >= 32760 or planet_crop.shape[1] >= 32760:
        return None, "planet_crop_oversize"

    # 4. UTM -> local planet pixel, cv2.remap
    map_x = ((dense_X - planet_tf.c) / planet_tf.a).astype(np.float32)
    map_y = ((dense_Y - planet_tf.f) / planet_tf.e).astype(np.float32)
    if map_x.shape[0] >= 32760 or map_x.shape[1] >= 32760:
        return None, "raw_crop_oversize"
    # Replace NaN with -1 so cv2.remap fills borderValue (=0)
    map_x = np.where(np.isfinite(map_x), map_x, -1).astype(np.float32)
    map_y = np.where(np.isfinite(map_y), map_y, -1).astype(np.float32)
    warped_planet = cv2.remap(planet_crop, map_x, map_y,
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if warped_planet.mean() < 5:
        return None, "warped_planet_nodata"

    # 5. Stretch raw_crop to uint8 [0, 255] via per-tile p1-p99
    raw_f = raw_crop.astype(np.float32)
    lo, hi = np.percentile(raw_f, [1, 99])
    raw_u8 = np.clip((raw_f - lo) / max(hi - lo, 1.0), 0, 1)
    raw_u8 = (raw_u8 * 255).astype(np.uint8)

    # 6. Downsample to match_res
    scale = native_res / match_res  # e.g. 4/10 = 0.4 -> shrink
    if abs(scale - 1.0) > 1e-3:
        new_w = max(8, int(round(raw_u8.shape[1] * scale)))
        new_h = max(8, int(round(raw_u8.shape[0] * scale)))
        raw_match = cv2.resize(raw_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        warp_match = cv2.resize(warped_planet, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        raw_match = raw_u8
        warp_match = warped_planet

    # 7. Dispatch matcher
    try:
        src_pts, dst_pts, confs = match_fn(raw_match, warp_match, **match_kwargs)
    except Exception as e:
        return None, f"matcher_error:{type(e).__name__}:{e}"

    if len(src_pts) == 0:
        return None, "zero_matches"

    # 8. Recover raw KH-9 pixel + warped planet raw-pixel
    #    Both are in tile-pixel @ match_res. raw_pixel = tile_pixel / scale + crop_origin
    inv_scale = 1.0 / scale
    c1_match, r1_match = src_pts[:, 0], src_pts[:, 1]
    c2_match, r2_match = dst_pts[:, 0], dst_pts[:, 1]
    c1_raw = c1_match * inv_scale + c0
    r1_raw = r1_match * inv_scale + r0
    c2_raw_off = c2_match * inv_scale  # offset within crop
    r2_raw_off = r2_match * inv_scale

    # 9. Look up U_real from dense_X / dense_Y at (c2_raw_off, r2_raw_off)
    valid = ((c2_raw_off >= 0) & (c2_raw_off < dense_X.shape[1] - 1)
             & (r2_raw_off >= 0) & (r2_raw_off < dense_X.shape[0] - 1))
    if not valid.any():
        return None, "no_valid_matches_in_dense_warp"
    c2x = c2_raw_off[valid]
    r2y = r2_raw_off[valid]

    ci0 = np.floor(c2x).astype(np.int32); ci1 = ci0 + 1
    ri0 = np.floor(r2y).astype(np.int32); ri1 = ri0 + 1
    fc = c2x - ci0
    fr = r2y - ri0
    X00 = dense_X[ri0, ci0]; X01 = dense_X[ri0, ci1]
    X10 = dense_X[ri1, ci0]; X11 = dense_X[ri1, ci1]
    Y00 = dense_Y[ri0, ci0]; Y01 = dense_Y[ri0, ci1]
    Y10 = dense_Y[ri1, ci0]; Y11 = dense_Y[ri1, ci1]
    U_real_x = (X00 * (1 - fc) * (1 - fr) + X01 * fc * (1 - fr)
                + X10 * (1 - fc) * fr + X11 * fc * fr)
    U_real_y = (Y00 * (1 - fc) * (1 - fr) + Y01 * fc * (1 - fr)
                + Y10 * (1 - fc) * fr + Y11 * fc * fr)
    finite = np.isfinite(U_real_x) & np.isfinite(U_real_y)

    c1_raw = c1_raw[valid][finite]
    r1_raw = r1_raw[valid][finite]
    c2_raw = (c2_raw_off + c0)[valid][finite]
    r2_raw = (r2_raw_off + r0)[valid][finite]
    U_real_x = U_real_x[finite]
    U_real_y = U_real_y[finite]
    confs_kept = confs[valid][finite]

    if len(U_real_x) == 0:
        return None, "no_valid_U_real"

    # 10. Optional cam-consistency filter:
    #     Project U_real through groundToImage at DEM h, compare to (c1_raw, r1_raw).
    if cam_filter_thresh is not None:
        keep = np.ones(len(U_real_x), dtype=bool)
        for i in range(len(U_real_x)):
            c_proj, r_proj = geom.groundToImage_utm(U_real_x[i], U_real_y[i])  # auto-DEM
            if not np.isfinite(c_proj):
                keep[i] = False
                continue
            dr = np.hypot(c_proj - c1_raw[i], r_proj - r1_raw[i])
            if dr > cam_filter_thresh:
                keep[i] = False
        c1_raw = c1_raw[keep]; r1_raw = r1_raw[keep]
        c2_raw = c2_raw[keep]; r2_raw = r2_raw[keep]
        U_real_x = U_real_x[keep]; U_real_y = U_real_y[keep]
        confs_kept = confs_kept[keep]

    if len(U_real_x) == 0:
        return None, "all_filtered_by_cam_consistency"

    # 11. col_kh9_mapped / row_kh9_mapped: synth.groundToImage at U_real
    #     (used by npz_to_match_shp.py for diagnostic shapefiles; here we
    #     just store c2_raw / r2_raw since they ARE the synth.groundToImage
    #     projection of U_real by construction).
    col_kh9_mapped = c2_raw.astype(np.float32)
    row_kh9_mapped = r2_raw.astype(np.float32)
    residual_px = (1.0 / np.maximum(confs_kept, 1e-6)).astype(np.float32)

    rows_out = np.stack([
        U_real_x.astype(np.float64),
        U_real_y.astype(np.float64),
        c1_raw.astype(np.float32),
        r1_raw.astype(np.float32),
        col_kh9_mapped,
        row_kh9_mapped,
        residual_px,
        confs_kept.astype(np.float32),
    ], axis=1)
    return rows_out, "ok"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="rotated sub4 raw KH-9 TIF")
    ap.add_argument("--planet", required=True, help="Planet pan TIF")
    ap.add_argument("--camera", required=True, help="CSM .json (synthetic or adjusted)")
    ap.add_argument("--dem", required=True, help="DEM (ellipsoid heights)")
    ap.add_argument("--matcher", required=True, choices=("sift", "dkm"))
    ap.add_argument("--match-res", type=float, default=10.0,
                    help="Matcher input resolution in m/px (default: 10)")
    ap.add_argument("--native-res", type=float, default=4.0,
                    help="Raw image's native ground resolution in m/px. "
                         "Default 4.0 for sub4. The rotated raw TIFs have no "
                         "GIS transform so we can't infer this.")
    ap.add_argument("--tile-utm-km", type=float, default=8.0)
    ap.add_argument("--tile-step-frac", type=float, default=0.5)
    ap.add_argument("--sparse-step", type=int, default=64,
                    help="Raw-pixel spacing of imageToGround warp samples (default: 64)")
    ap.add_argument("--cam-filter-thresh", type=float, default=None,
                    help="Drop matches where |groundToImage(U_real) - raw_pixel| > this (raw px)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--max-keypoints", type=int, default=8000)
    ap.add_argument("--min-confidence", type=float, default=0.3)
    ap.add_argument("--utm-zone-epsg", type=int, default=32643,
                    help="UTM EPSG for the strip (default: 32643)")
    ap.add_argument("--h-init", type=float, default=4000.0,
                    help="Initial height guess for imageToGround DEM iteration (m)")
    ap.add_argument("--output", required=True, help="output NPZ path")
    args = ap.parse_args()

    print(f"=== raw_tile_match: matcher={args.matcher} match_res={args.match_res} "
          f"tile={args.tile_utm_km}km ===", flush=True)

    # --- Load CSM model + DEM + raw image header ---------------------------
    model, csmapi_mod = load_csm_model(args.camera)
    print(f"  CSM model: {type(model).__name__}", flush=True)

    with rasterio.open(args.raw) as src:
        raw_W = src.width
        raw_H = src.height
    raw_native_res = float(args.native_res)
    print(f"  Raw image: {raw_W} x {raw_H}  native_res={raw_native_res} m/px "
          f"(rotated sub4 expected; --native-res flag controls)", flush=True)

    with rasterio.open(args.dem) as dem_src:
        dem_arr = dem_src.read(1)
        dem_tf = dem_src.transform
        dem_crs = dem_src.crs
    print(f"  DEM: {dem_arr.shape}  CRS={dem_crs.to_string() if dem_crs else '(none)'}",
          flush=True)

    utm_crs = f"EPSG:{args.utm_zone_epsg}"
    geom = CsmGeom(model, csmapi_mod, dem_arr, dem_tf, dem_crs, utm_crs,
                   h_init=args.h_init)

    # --- Startup round-trip sanity check -----------------------------------
    print("  Sanity check: groundToImage(imageToGround) round-trip ...", flush=True)
    tc, tr = raw_W // 2, raw_H // 2
    tx, ty = geom.imageToGround_utm(tc, tr)
    if not np.isfinite(tx):
        print("FATAL: middle-pixel imageToGround_utm returned non-finite. "
              "Camera/image rotation mismatch?", file=sys.stderr)
        sys.exit(2)
    cb, rb = geom.groundToImage_utm(tx, ty)  # auto-DEM
    err = max(abs(cb - tc), abs(rb - tr))
    if err > 1.0:
        print(f"FATAL: round-trip error {err:.4f} px > 1.0. "
              f"Camera/image mismatch likely.", file=sys.stderr)
        sys.exit(2)
    print(f"    OK (round-trip error {err:.4f} px at center)", flush=True)

    # --- Raw image UTM extent ----------------------------------------------
    raw_utm_bbox = compute_raw_utm_bbox(geom, raw_W, raw_H)
    print(f"  Raw UTM extent: x[{raw_utm_bbox[0]:.0f}, {raw_utm_bbox[2]:.0f}]  "
          f"y[{raw_utm_bbox[1]:.0f}, {raw_utm_bbox[3]:.0f}]", flush=True)

    # --- Tile loop ---------------------------------------------------------
    tile_m = args.tile_utm_km * 1000.0
    step_m = tile_m * args.tile_step_frac
    tiles = list(iterate_utm_tiles(raw_utm_bbox, tile_m, step_m))
    print(f"  Tiling: {len(tiles)} UTM tiles "
          f"({args.tile_utm_km} km @ step_frac {args.tile_step_frac})", flush=True)

    # --- Read raw image into memory (sub4 5000×85000 uint8 ~ 0.4 GB) -------
    with rasterio.open(args.raw) as src:
        raw_arr = src.read(1)
    print(f"  Raw image loaded: dtype={raw_arr.dtype}  mean={raw_arr.mean():.1f}", flush=True)

    # --- Planet dataset (kept open for windowed reads). If its CRS differs
    # from the strip's UTM, wrap with WarpedVRT so windowed reads are in
    # our UTM zone (matches run_sift_per_strip's read_at_res behavior).
    planet_src = rasterio.open(args.planet)
    print(f"  Planet (raw): {planet_src.width} x {planet_src.height}  "
          f"CRS={planet_src.crs.to_string()}", flush=True)
    if str(planet_src.crs).split(":")[-1] != str(args.utm_zone_epsg):
        from rasterio.vrt import WarpedVRT
        planet_ds = WarpedVRT(planet_src, crs=utm_crs)
        print(f"  Planet wrapped via WarpedVRT into {utm_crs}: "
              f"{planet_ds.width} x {planet_ds.height}", flush=True)
    else:
        planet_ds = planet_src

    # --- Matcher dispatch + adapter kwargs ---------------------------------
    match_fn = load_match_fn(args.matcher)
    device = _resolve_device(args.device)
    if args.matcher == "dkm":
        match_kwargs = dict(device=device,
                            max_keypoints=args.max_keypoints,
                            min_confidence=args.min_confidence,
                            n_sample=4096)
    else:
        match_kwargs = dict(max_keypoints=args.max_keypoints,
                            min_confidence=args.min_confidence)
    print(f"  Matcher: {args.matcher}  device={device}  "
          f"max_kp={args.max_keypoints}  min_conf={args.min_confidence}", flush=True)

    # --- Loop --------------------------------------------------------------
    all_rows = []
    skip_reasons = {}
    t0 = time.time()
    for idx, utm_bbox in enumerate(tiles):
        rows, status = process_tile(
            geom, raw_arr, raw_W, raw_H, planet_ds,
            utm_bbox, args.match_res, raw_native_res,
            match_fn, match_kwargs,
            args.sparse_step,
            args.cam_filter_thresh,
        )
        if rows is None:
            skip_reasons[status] = skip_reasons.get(status, 0) + 1
            if (idx + 1) % 5 == 0:
                print(f"  Tile {idx+1:3d}/{len(tiles)}: skip ({status})", flush=True)
            continue
        all_rows.append(rows)
        print(f"  Tile {idx+1:3d}/{len(tiles)}: kept {len(rows)}  "
              f"(running total {sum(len(a) for a in all_rows)})", flush=True)

    dt = time.time() - t0
    print(f"\n  Tile loop done in {dt:.1f}s. Skip counts:", flush=True)
    for k, v in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}", flush=True)

    try:
        planet_ds.close()
    except Exception:
        pass
    try:
        planet_src.close()
    except Exception:
        pass

    if not all_rows:
        print("ERROR: no matches kept; nothing to save.", file=sys.stderr)
        sys.exit(1)

    arr = np.concatenate(all_rows, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output,
             xs_ref=arr[:, 0],
             ys_ref=arr[:, 1],
             col_kh9_raw=arr[:, 2].astype(np.float32),
             row_kh9_raw=arr[:, 3].astype(np.float32),
             col_kh9_mapped=arr[:, 4].astype(np.float32),
             row_kh9_mapped=arr[:, 5].astype(np.float32),
             residual_px=arr[:, 6].astype(np.float32),
             confidence=arr[:, 7].astype(np.float32))
    print(f"  Saved {arr.shape[0]} matches -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
