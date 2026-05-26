"""Resolve the run manifest into a normalized list of strips.

Entity ID format
----------------
KH-9 declass3 IDs look like ``D3C<mission>-<frame><camera><seq>`` where:
  * <mission> is 4 digits (e.g. 1216)
  * <frame> is 6 digits (e.g. 300814)
  * <camera> is one of {F, A, T} (Forward, Aft, Transit)
  * <seq> is 3 digits (e.g. 011)

A "strip" is the pair (fwd, aft) sharing the same <mission>-<frame>-<seq>.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from . import config as cfg_mod


ENTITY_RE = re.compile(r"(D3C\d{4}-\d{6})([FAT])(\d{3})")
PIECE_RE = re.compile(r"(D3C\d{4}-\d{6}[FAT]\d{3})_([a-z])\.tif$", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Entity:
    entity_id: str       # D3C1216-300814F011
    camera: str          # "fwd" | "aft" | "transit"
    pieces: list[Path]   # *_a.tif, *_b.tif, ...
    raw_dir: Path        # directory containing the pieces


@dataclass
class Strip:
    strip_id: str        # D3C1216-300814-011
    fwd: Entity | None = None
    aft: Entity | None = None
    extras: list[Entity] = field(default_factory=list)  # transit, etc.


CAMERA_LETTER_TO_NAME = {"F": "fwd", "A": "aft", "T": "transit"}


def _parse_entity_id(eid: str) -> tuple[str, str, str]:
    """Return (mission_frame, camera_letter, seq) or raise ValueError."""
    m = ENTITY_RE.fullmatch(eid)
    if m is None:
        raise ValueError(f"not a valid declass3 entity id: {eid!r}")
    return m.group(1), m.group(2), m.group(3)


def strip_id_for(entity_id: str) -> str:
    mf, _cam, seq = _parse_entity_id(entity_id)
    return f"{mf}-{seq}"


def _find_pieces(entity_id: str, search_dir: Path) -> tuple[list[Path], Path]:
    """Find ``<entity_id>_{a,b,c,...}.tif`` pieces.

    Search order:
      1) ``<search_dir>/<entity_id>/*.tif``
      2) ``<search_dir>/*.tif`` matching ``<entity_id>_<letter>.tif``
      3) extract ``<search_dir>/<entity_id>.tgz`` (or ``.tar.gz``) in place then re-search

    Returns (sorted_pieces, dir_holding_them).
    """
    # 1) per-entity subdir
    sub = search_dir / entity_id
    if sub.is_dir():
        pieces = sorted(p for p in sub.glob(f"{entity_id}_*.tif"))
        if pieces:
            return pieces, sub

    # 2) flat
    flat = sorted(p for p in search_dir.glob(f"{entity_id}_*.tif"))
    if flat:
        return flat, search_dir

    # 3) tarball
    for ext in (".tgz", ".tar.gz"):
        tarball = search_dir / f"{entity_id}{ext}"
        if tarball.exists():
            target = search_dir / entity_id
            target.mkdir(exist_ok=True)
            log(f"[manifest] extracting {tarball.name} -> {target}/")
            with tarfile.open(tarball) as tf:
                tf.extractall(target)
            pieces = sorted(p for p in target.glob(f"{entity_id}_*.tif"))
            if pieces:
                return pieces, target

    raise FileNotFoundError(
        f"no raw pieces for {entity_id} under {search_dir} "
        f"(looked for {entity_id}_*.tif, {entity_id}/*.tif, and {entity_id}.tgz)"
    )


def _entity_ids_from_subdir(subdir: Path) -> list[str]:
    """Scan a directory for unique declass3 entity IDs across .tif, .tgz, and subdirs."""
    seen: set[str] = set()

    # Files in the directory
    for p in subdir.iterdir():
        if p.is_file():
            name = p.name
            for stem in (name, name.replace(".tar.gz", "").replace(".tgz", "")):
                m = ENTITY_RE.search(stem)
                if m:
                    seen.add(m.group(0))
                    break
        elif p.is_dir():
            m = ENTITY_RE.search(p.name)
            if m:
                seen.add(m.group(0))

    return sorted(seen)


def resolve(cfg: cfg_mod.Config) -> list[Strip]:
    """Resolve the configured manifest into a list of strips with located pieces."""
    if cfg.manifest.entity_ids:
        entity_ids = list(cfg.manifest.entity_ids)
        search_dir = cfg.raw_dir
    else:
        subdir = cfg.raw_dir / cfg.manifest.subdir
        if not subdir.is_dir():
            raise FileNotFoundError(f"manifest.subdir not found: {subdir}")
        entity_ids = _entity_ids_from_subdir(subdir)
        if not entity_ids:
            raise RuntimeError(f"no declass3 entity IDs discovered under {subdir}")
        search_dir = subdir

    # Group by strip
    strips: dict[str, Strip] = {}
    for eid in entity_ids:
        mf, cam, seq = _parse_entity_id(eid)
        sid = f"{mf}-{seq}"
        cam_name = CAMERA_LETTER_TO_NAME[cam]

        pieces, holder = _find_pieces(eid, search_dir)
        ent = Entity(entity_id=eid, camera=cam_name, pieces=pieces, raw_dir=holder)

        strip = strips.setdefault(sid, Strip(strip_id=sid))
        if cam_name == "fwd":
            strip.fwd = ent
        elif cam_name == "aft":
            strip.aft = ent
        else:
            strip.extras.append(ent)

    # Strict: every strip must have both fwd and aft for the pipeline to be useful.
    incomplete = [s.strip_id for s in strips.values() if s.fwd is None or s.aft is None]
    if incomplete:
        raise RuntimeError(
            f"strips missing fwd or aft: {incomplete}. "
            "Pipeline requires both cameras for each strip."
        )

    return sorted(strips.values(), key=lambda s: s.strip_id)


def to_jsonable(strips: list[Strip]) -> list[dict]:
    out = []
    for s in strips:
        out.append({
            "strip_id": s.strip_id,
            "fwd": _entity_to_dict(s.fwd),
            "aft": _entity_to_dict(s.aft),
            "extras": [_entity_to_dict(e) for e in s.extras],
        })
    return out


def _entity_to_dict(e: Entity | None) -> dict | None:
    if e is None:
        return None
    return {
        "entity_id": e.entity_id,
        "camera": e.camera,
        "raw_dir": str(e.raw_dir),
        "pieces": [str(p) for p in e.pieces],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resolve and print the manifest as JSON.")
    p.add_argument("config", nargs="?", default=None)
    args = p.parse_args(argv)

    cfg = cfg_mod.load_config(args.config)
    strips = resolve(cfg)
    log(f"[manifest] resolved {len(strips)} strip(s):")
    for s in strips:
        log(f"  {s.strip_id}: fwd={s.fwd.entity_id} ({len(s.fwd.pieces)} pcs) "
            f"aft={s.aft.entity_id} ({len(s.aft.pieces)} pcs)")
    json.dump(to_jsonable(strips), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
