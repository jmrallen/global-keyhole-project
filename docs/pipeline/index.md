# Pipeline overview

The pipeline is organized as three sequential stages. Each stage has a shell driver in `scripts/` and Python helpers in `scripts/lib/`.

```
S0  →  S1  →  S2
pre-     per-pair    multi-strip
process  cameras     jitter solve
```

## Stage summaries

### [S0 — Pre-processing](s0-preprocessing.md)
Resolve frame metadata from the declassified catalog; build DEM and modern-imagery mosaics (e.g., Planet basemap) over the target region; auto-detect per-frame crop windows.

### [S1 — Per-pair camera optimization](s1-cameras.md)
An 11-phase cascade run per stereo pair: corner-seeded `cam_gen`, mapproject, bundle_adjust, stereo, hillshade-matched correlator, `dem2gcp` refinement, output. The canonical KH-9 path uses hillshade + correlator + `dem2gcp`; see [troubleshooting](../reference/troubleshooting.md) for variations.

### [S2 — Multi-strip jitter solve](s2-jitter-solve.md)
A 17-phase raw-image-space cascade that jointly solves jitter across overlapping strips, producing a globally consistent reconstruction over a region rather than a per-pair patchwork.

## Code map

| File | Role |
|------|------|
| `scripts/S0.sh` | S0 driver |
| `scripts/S1.sh` | S1 driver |
| `scripts/S2.sh` | S2 driver |
| `scripts/lib/` | Python helpers (see [lib/ reference](lib-reference.md)) |
| `config/config.yaml` | Per-machine paths and run options |
| `cameras/` | Camera model templates per KH series |
| `assets/declass3_metadata.parquet` | Master frame catalog (rendered on the [Image Archive](../archive/index.md)) |
