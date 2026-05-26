"""Auto-detect the image rectangle inside a KH-9 HEXAGON mosaic.

The mosaic produced by ASP's ``image_mosaic`` surrounds the imagery with black
film rebate (and extra empty canvas where the strips don't cover). This script
finds the imagery rectangle, applies a conservative inward margin, and prints
``xoff yoff xsize ysize`` ready to drop into ``gdal_translate -srcwin``.

Usage
-----
    python crop_detect.py mosaics/D3C1206-400556F016.tif
        -> stdout: "<xoff> <yoff> <xsize> <ysize>"
        -> writes: mosaics/D3C1206-400556F016_crop_preview.png

Algorithm
---------
1. Read a downsampled version of the mosaic with rasterio.
2. Morphological opening to erase punch holes / text inside the rebate.
3. Binary threshold -> "fraction of bright pixels" per row and per column.
4. Longest contiguous run above a density threshold = imagery span on that axis.
5. Refine each of the four edges by re-running the density projection on a
   thin strip at full resolution straddling the coarse edge estimate.
6. Apply a fixed inward margin and clip to image bounds.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

CROP_COLUMNS = ("crop_xoff", "crop_yoff", "crop_xsize", "crop_ysize")
ENTITY_ID_COLUMN = "Entity ID"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class CropResult:
    xoff: int
    yoff: int
    xsize: int
    ysize: int
    raw_left: int
    raw_right: int
    raw_top: int
    raw_bottom: int
    image_width: int
    image_height: int


def _longest_run_above(profile: np.ndarray, threshold: float) -> tuple[int, int]:
    """Return (start, stop) of the longest contiguous run with profile >= threshold.

    ``stop`` is exclusive. Raises ValueError if no run exists.
    """
    mask = profile >= threshold
    if not mask.any():
        raise ValueError("density profile never crosses the threshold")

    # Find run boundaries by diffing the padded mask.
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    stops = np.flatnonzero(diffs == -1)
    lengths = stops - starts
    i = int(np.argmax(lengths))
    return int(starts[i]), int(stops[i])


def _density_profile(gray: np.ndarray, axis: int, threshold: int, open_ksize: int) -> np.ndarray:
    """Fraction of pixels above ``threshold`` per row (axis=1) or column (axis=0),
    after a morphological opening that suppresses small bright artifacts."""
    if open_ksize >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    else:
        cleaned = gray
    mask = cleaned > threshold
    # axis=1 -> mean along columns -> one value per row (row profile).
    # axis=0 -> mean along rows -> one value per column (column profile).
    return mask.mean(axis=axis)


def _refine_edge(
    src: rasterio.io.DatasetReader,
    edge: str,
    coarse: int,
    halfwidth: int,
    threshold: int,
    density: float,
    open_ksize: int,
) -> int:
    """Refine a single edge by re-running the density transition test on a thin
    full-resolution strip straddling ``coarse``.

    ``edge`` is one of {"left", "right", "top", "bottom"}.
    Returns the refined edge coordinate (in full-res pixels).
    """
    W, H = src.width, src.height
    if edge in ("left", "right"):
        x0 = max(0, coarse - halfwidth)
        x1 = min(W, coarse + halfwidth)
        # Take a slab spanning most of the image height to get a stable profile.
        y0 = int(H * 0.15)
        y1 = int(H * 0.85)
        window = Window(x0, y0, x1 - x0, y1 - y0)
        strip = src.read(1, window=window)
        # Column-wise density across the strip.
        profile = _density_profile(strip, axis=0, threshold=threshold, open_ksize=open_ksize)
        binary = profile >= density
        if edge == "left":
            idxs = np.flatnonzero(binary)
            return int(x0 + idxs[0]) if idxs.size else coarse
        else:  # right
            idxs = np.flatnonzero(binary)
            return int(x0 + idxs[-1] + 1) if idxs.size else coarse
    else:
        y0 = max(0, coarse - halfwidth)
        y1 = min(H, coarse + halfwidth)
        x0 = int(W * 0.15)
        x1 = int(W * 0.85)
        window = Window(x0, y0, x1 - x0, y1 - y0)
        strip = src.read(1, window=window)
        profile = _density_profile(strip, axis=1, threshold=threshold, open_ksize=open_ksize)
        binary = profile >= density
        if edge == "top":
            idxs = np.flatnonzero(binary)
            return int(y0 + idxs[0]) if idxs.size else coarse
        else:  # bottom
            idxs = np.flatnonzero(binary)
            return int(y0 + idxs[-1] + 1) if idxs.size else coarse


def _write_preview(
    out_path: Path,
    preview: np.ndarray,
    factor: float,
    raw: tuple[int, int, int, int],
    final: tuple[int, int, int, int],
    row_profile: np.ndarray,
    col_profile: np.ndarray,
    density_threshold: float,
) -> None:
    """raw and final are (left, top, right, bottom) at full resolution.

    Composites a four-panel preview: top strip = column density profile,
    left strip = row density profile, centre = the downsampled image with
    both crop boxes drawn, bottom and right are blank for alignment.
    """
    h, w = preview.shape
    plot_h = 80  # height of the column-density strip
    plot_w = 80  # width of the row-density strip
    pad = 4

    canvas = np.full((plot_h + pad + h, plot_w + pad + w, 3), 240, dtype=np.uint8)
    img_y0, img_x0 = plot_h + pad, plot_w + pad

    rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    rl, rt, rr, rb = [int(round(v / factor)) for v in raw]
    fl, ft, fr, fb = [int(round(v / factor)) for v in final]
    cv2.rectangle(rgb, (rl, rt), (rr - 1, rb - 1), (0, 255, 255), 2)  # raw = yellow
    cv2.rectangle(rgb, (fl, ft), (fr - 1, fb - 1), (0, 0, 255), 2)    # final = red
    canvas[img_y0:img_y0 + h, img_x0:img_x0 + w] = rgb

    # Column-density strip across the top.
    col_strip = np.full((plot_h, w, 3), 255, dtype=np.uint8)
    for x in range(w):
        bar = int(round(col_profile[x] * (plot_h - 2)))
        cv2.line(col_strip, (x, plot_h - 1), (x, plot_h - 1 - bar), (90, 90, 90), 1)
    thresh_y = plot_h - 1 - int(round(density_threshold * (plot_h - 2)))
    cv2.line(col_strip, (0, thresh_y), (w - 1, thresh_y), (0, 0, 255), 1)
    canvas[0:plot_h, img_x0:img_x0 + w] = col_strip

    # Row-density strip down the left.
    row_strip = np.full((h, plot_w, 3), 255, dtype=np.uint8)
    for y in range(h):
        bar = int(round(row_profile[y] * (plot_w - 2)))
        cv2.line(row_strip, (plot_w - 1, y), (plot_w - 1 - bar, y), (90, 90, 90), 1)
    thresh_x = plot_w - 1 - int(round(density_threshold * (plot_w - 2)))
    cv2.line(row_strip, (thresh_x, 0), (thresh_x, h - 1), (0, 0, 255), 1)
    canvas[img_y0:img_y0 + h, 0:plot_w] = row_strip

    cv2.imwrite(str(out_path), canvas)


def detect_crop(
    path: str | Path,
    downsample: int = 16,
    threshold: int = 30,
    density: float = 0.5,
    margin: int = 500,
    open_ksize: int = 7,
    refine: bool = True,
    preview_path: str | Path | None = None,
) -> CropResult:
    """Detect the imagery rectangle in ``path`` and return crop coordinates."""
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(f"{path} has no bands")
        W, H = src.width, src.height
        out_h = max(1, H // downsample)
        out_w = max(1, W // downsample)
        log(f"[crop_detect] {path.name}: {W}x{H} -> downsample {downsample}x = {out_w}x{out_h}")
        preview = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
        )
        eff_fx = W / out_w
        eff_fy = H / out_h

        row_prof = _density_profile(preview, axis=1, threshold=threshold, open_ksize=open_ksize)
        col_prof = _density_profile(preview, axis=0, threshold=threshold, open_ksize=open_ksize)

        try:
            r0, r1 = _longest_run_above(row_prof, density)
            c0, c1 = _longest_run_above(col_prof, density)
        except ValueError as e:
            raise RuntimeError(
                f"could not find image region in {path.name}: {e}. "
                "Try lowering --threshold or --density."
            ) from e

        # Sanity check: the detected span should cover a meaningful fraction of the image.
        if (r1 - r0) < 0.5 * out_h or (c1 - c0) < 0.5 * out_w:
            raise RuntimeError(
                f"detected region too small in {path.name}: "
                f"rows {r0}-{r1} of {out_h}, cols {c0}-{c1} of {out_w}. "
                "Probably mis-detected; not returning a degenerate crop."
            )

        # Map coarse span back to full resolution.
        left = int(round(c0 * eff_fx))
        right = int(round(c1 * eff_fx))
        top = int(round(r0 * eff_fy))
        bottom = int(round(r1 * eff_fy))
        log(f"[crop_detect] coarse box (full-res px): L={left} T={top} R={right} B={bottom}")

        if refine:
            halfwidth = max(32, int(round(4 * eff_fx)))
            left_r = _refine_edge(src, "left", left, halfwidth, threshold, density, open_ksize)
            right_r = _refine_edge(src, "right", right, halfwidth, threshold, density, open_ksize)
            halfwidth_y = max(32, int(round(4 * eff_fy)))
            top_r = _refine_edge(src, "top", top, halfwidth_y, threshold, density, open_ksize)
            bottom_r = _refine_edge(src, "bottom", bottom, halfwidth_y, threshold, density, open_ksize)
            log(
                f"[crop_detect] refined box (full-res px): "
                f"L={left_r} T={top_r} R={right_r} B={bottom_r}"
            )
            left, right, top, bottom = left_r, right_r, top_r, bottom_r

        raw = (left, top, right, bottom)

        # Conservative inward shrink, clipped to image bounds.
        left_m = max(0, left + margin)
        right_m = min(W, right - margin)
        top_m = max(0, top + margin)
        bottom_m = min(H, bottom - margin)
        if right_m <= left_m or bottom_m <= top_m:
            raise RuntimeError(
                f"margin of {margin} px collapsed the crop region. "
                "Lower --margin or check the image."
            )

        xoff, yoff = left_m, top_m
        xsize, ysize = right_m - left_m, bottom_m - top_m
        final = (left_m, top_m, right_m, bottom_m)
        log(f"[crop_detect] final (post-margin): xoff={xoff} yoff={yoff} xsize={xsize} ysize={ysize}")

        if preview_path is not None:
            _write_preview(
                Path(preview_path),
                preview,
                eff_fx,
                raw,
                final,
                row_profile=row_prof,
                col_profile=col_prof,
                density_threshold=density,
            )
            log(f"[crop_detect] wrote preview {preview_path}")

        return CropResult(
            xoff=xoff,
            yoff=yoff,
            xsize=xsize,
            ysize=ysize,
            raw_left=left,
            raw_right=right,
            raw_top=top,
            raw_bottom=bottom,
            image_width=W,
            image_height=H,
        )


def _default_preview_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + "_crop_preview.png")


def update_metadata_parquet(parquet_path: Path, entity_id: str, result: CropResult) -> None:
    """Add/update crop columns for ``entity_id`` in the metadata parquet.

    Writes atomically (temp file in the same directory, then os.replace).
    No-ops with a warning if the entity isn't present in the table.
    """
    import pandas as pd

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"metadata parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if ENTITY_ID_COLUMN not in df.columns:
        raise ValueError(
            f"metadata parquet {parquet_path} has no {ENTITY_ID_COLUMN!r} column; "
            f"got: {list(df.columns)[:8]}..."
        )

    # Ensure the crop columns exist with a nullable integer dtype.
    for col in CROP_COLUMNS:
        if col not in df.columns:
            df[col] = pd.array([pd.NA] * len(df), dtype="Int64")
        else:
            df[col] = df[col].astype("Int64")

    mask = df[ENTITY_ID_COLUMN] == entity_id
    n_match = int(mask.sum())
    if n_match == 0:
        log(
            f"[crop_detect] WARNING: entity_id {entity_id!r} not found in "
            f"{parquet_path.name}; parquet not modified."
        )
        return

    df.loc[mask, "crop_xoff"] = result.xoff
    df.loc[mask, "crop_yoff"] = result.yoff
    df.loc[mask, "crop_xsize"] = result.xsize
    df.loc[mask, "crop_ysize"] = result.ysize

    # Atomic write via temp file in the same directory.
    fd, tmp = tempfile.mkstemp(
        prefix=parquet_path.stem + ".",
        suffix=parquet_path.suffix + ".tmp",
        dir=str(parquet_path.parent),
    )
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, parquet_path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

    log(
        f"[crop_detect] updated {n_match} row(s) for {entity_id} in "
        f"{parquet_path.name}: xoff={result.xoff} yoff={result.yoff} "
        f"xsize={result.xsize} ysize={result.ysize}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", type=Path, help="path to the input mosaic TIFF")
    p.add_argument("--downsample", type=int, default=16, help="downsample factor for coarse pass (default: 16)")
    p.add_argument("--threshold", type=int, default=30, help="brightness threshold (0-255, default: 30)")
    p.add_argument("--density", type=float, default=0.5, help="row/column density threshold (default: 0.5)")
    p.add_argument("--margin", type=int, default=500, help="inward safety margin in full-res px (default: 500)")
    p.add_argument("--open-ksize", type=int, default=7, help="morphological opening kernel size on the coarse pass (default: 7)")
    p.add_argument("--no-refine", action="store_true", help="skip full-resolution edge refinement")
    p.add_argument("--no-preview", action="store_true", help="do not write a QA preview PNG")
    p.add_argument("--preview-path", type=Path, default=None, help="override path of the QA preview PNG")
    p.add_argument(
        "--metadata-parquet",
        type=Path,
        default=None,
        help=(
            "path to a metadata parquet (e.g. <root>/assets/file.parquet) to update with "
            "crop_xoff/crop_yoff/crop_xsize/crop_ysize for the matching Entity ID. "
            "Entity ID is derived from the image filename stem."
        ),
    )
    p.add_argument(
        "--entity-id",
        type=str,
        default=None,
        help="override the Entity ID used to locate the parquet row (defaults to the image filename stem)",
    )
    args = p.parse_args(argv)

    preview_path: Path | None
    if args.no_preview:
        preview_path = None
    else:
        preview_path = args.preview_path or _default_preview_path(args.image)

    res = detect_crop(
        args.image,
        downsample=args.downsample,
        threshold=args.threshold,
        density=args.density,
        margin=args.margin,
        open_ksize=args.open_ksize,
        refine=not args.no_refine,
        preview_path=preview_path,
    )
    print(f"{res.xoff} {res.yoff} {res.xsize} {res.ysize}")

    if args.metadata_parquet is not None:
        entity_id = args.entity_id or args.image.stem
        update_metadata_parquet(args.metadata_parquet, entity_id, res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
