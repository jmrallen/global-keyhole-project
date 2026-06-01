# S2 — Multi-strip jitter solve

Driver: `scripts/S2.sh`. Purpose: take multiple S1-optimized strips that overlap over a region and jointly solve for jitter, producing a globally consistent reconstruction rather than a per-pair patchwork.

## Audience

You've run [S1](s1-cameras.md) on multiple overlapping pairs and want to combine them into one solution. You're comfortable with ASP `jitter_solve` and have read the relevant ASP documentation.

## TODO — sections to write

- [ ] When to run S2 (overlap requirements, strip selection)
- [ ] The 17 raw-space phases at a glance
- [ ] Raw tile matching via `scripts/lib/raw_tile_match.py`
- [ ] NPZ → GCP conversion via `scripts/lib/raw_npz_to_gcp.py` and `batch_npz_to_gcp.py`
- [ ] GCP concatenation and filtering: `scripts/lib/concat_gcps.py`, `filter_gcps_by_camera.py`
- [ ] `jitter_solve` invocation and key flags
- [ ] Outputs: refined cameras, final point cloud
- [ ] Validation: pixel-residual diagnostics, before/after DEM comparisons
- [ ] Common pitfalls
