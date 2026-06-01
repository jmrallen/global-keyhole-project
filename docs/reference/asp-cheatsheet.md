# ASP cheatsheet

Quick reference for the NASA Ames Stereo Pipeline tools used in this pipeline.

## Audience

You know the pipeline calls `bundle_adjust`, `stereo`, `parallel_stereo`, `jitter_solve`, `dem2gcp`, `cam_gen`, `mapproject`, `point2dem`, `pc_align`, `dem_mosaic`, `hillshade`, `image_align` — and you want a one-stop reminder of which flags matter for KH-9 specifically.

## TODO — sections to write

- [ ] `cam_gen` — KH-9 OPTICAL_BAR setup, corner seeding
- [ ] `bundle_adjust` — KH-9 flags, GCP weighting, IP filtering
- [ ] `parallel_stereo` — stereo session selection, correlator settings
- [ ] `dem2gcp` — input requirements, output format
- [ ] `jitter_solve` — multi-strip setup
- [ ] `mapproject` — projection choices for KH-9 raw vs camera-corrected
- [ ] `pc_align` — when to use it (NOT in the canonical KH-9 path)
- [ ] Common flag-set patterns used across S0/S1/S2
