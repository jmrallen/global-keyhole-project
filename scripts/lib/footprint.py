"""Footprint math: per-strip polygons, union bbox, UTM zone, contiguity gate."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config as cfg_mod
from . import manifest as manifest_mod
from . import metadata as md_mod


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def pad(self, deg: float) -> "BBox":
        return BBox(
            self.min_lon - deg,
            self.min_lat - deg,
            self.max_lon + deg,
            self.max_lat + deg,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


def utm_zone_for(lon: float, lat: float) -> tuple[int, str, int]:
    """Return (zone_number, hemisphere, epsg) for a (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    hemisphere = "N" if lat >= 0 else "S"
    epsg = 32600 + zone if hemisphere == "N" else 32700 + zone
    return zone, hemisphere, epsg


def _polygon_from_corners(corners: md_mod.Corners):
    """Build a shapely polygon from NW/NE/SE/SW corners (in (lon, lat))."""
    from shapely.geometry import Polygon
    return Polygon([corners.nw, corners.ne, corners.se, corners.sw, corners.nw])


def compute_footprint(cfg: cfg_mod.Config, strips: list[manifest_mod.Strip], pad_deg: float | None = None) -> dict:
    """Compute polygons + union bbox + UTM zone for a strip list.

    Returns a dict with keys: polygons (per-entity), union_bbox (padded),
    raw_bbox (unpadded), utm_zone, utm_hemisphere, utm_epsg, median_lon, median_lat.
    Raises RuntimeError if the union is a MultiPolygon (contiguity gate fails).
    """
    from shapely.ops import unary_union

    pad = cfg.bbox_pad_deg if pad_deg is None else pad_deg

    polys = {}
    centers_lon: list[float] = []
    centers_lat: list[float] = []
    for s in strips:
        for ent in (s.fwd, s.aft, *s.extras):
            if ent is None:
                continue
            meta = md_mod.lookup(cfg.paths.metadata_parquet, ent.entity_id)
            poly = _polygon_from_corners(meta.corners)
            polys[ent.entity_id] = poly
            cx, cy = poly.centroid.x, poly.centroid.y
            centers_lon.append(cx)
            centers_lat.append(cy)

    if not polys:
        raise RuntimeError("no entities to compute a footprint from")

    union = unary_union(list(polys.values()))
    geom_type = union.geom_type

    if geom_type == "MultiPolygon":
        # Identify disjoint groups by reverse-mapping each entity to a connected component.
        components = list(union.geoms)
        groups: list[list[str]] = [[] for _ in components]
        for eid, p in polys.items():
            for i, c in enumerate(components):
                if c.intersects(p):
                    groups[i].append(eid)
                    break
        groups_repr = "; ".join("[" + ", ".join(g) + "]" for g in groups)
        raise RuntimeError(
            f"contiguity gate failed: union is a MultiPolygon with "
            f"{len(components)} disjoint groups: {groups_repr}"
        )

    raw_bounds = union.bounds  # (minx, miny, maxx, maxy)
    raw_bbox = BBox(*raw_bounds)
    padded = raw_bbox.pad(pad)

    median_lon = float(sorted(centers_lon)[len(centers_lon) // 2])
    median_lat = float(sorted(centers_lat)[len(centers_lat) // 2])
    zone, hemi, epsg = utm_zone_for(median_lon, median_lat)

    return {
        "polygons": polys,
        "raw_bbox": raw_bbox,
        "union_bbox": padded,
        "utm_zone": zone,
        "utm_hemisphere": hemi,
        "utm_epsg": epsg,
        "median_lon": median_lon,
        "median_lat": median_lat,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute footprint + UTM zone for the configured manifest.")
    p.add_argument("config", nargs="?", default=None)
    args = p.parse_args(argv)

    cfg = cfg_mod.load_config(args.config)
    strips = manifest_mod.resolve(cfg)
    info = compute_footprint(cfg, strips)
    bbox = info["union_bbox"]
    log(f"[footprint] {len(info['polygons'])} entities, {len(strips)} strips")
    log(f"[footprint] raw bbox    : {info['raw_bbox'].as_tuple()}")
    log(f"[footprint] padded bbox : {bbox.as_tuple()} (pad={cfg.bbox_pad_deg}°)")
    log(f"[footprint] median lon/lat: ({info['median_lon']:.4f}, {info['median_lat']:.4f})")
    log(f"[footprint] UTM zone     : {info['utm_zone']}{info['utm_hemisphere']} (EPSG:{info['utm_epsg']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
