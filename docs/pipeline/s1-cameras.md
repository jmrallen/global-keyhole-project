# S1 — Per-pair camera optimization

Driver: `scripts/S1.sh`. Purpose: refine each stereo pair's KH-9 camera model from initial corner-seeded `cam_gen` output through hillshade-matched DEM ground control, producing pair-level CSM cameras suitable for stereo and (eventually) multi-strip jitter solve.

This stage implements the canonical KH-9 ASP recipe: hillshade + correlator + `dem2gcp`. Do **not** silently substitute `pc_align` — that's the KH-4B path (ASP §8.28.9), not the KH-9 path (ASP §8.29.9).

## Audience

You've run [S0](s0-preprocessing.md), have a region manifest, and want to know what S1's 11 phases actually do and how to debug a pair that won't converge.

## TODO — sections to write

- [ ] The 11 phases at a glance (table: phase → tool → output)
- [ ] Phase 1–3: crop, subres, `cam_gen` corner seeding via `scripts/lib/cam_gen_corners.py`
- [ ] Phase 4–5: mapproject + initial bundle_adjust
- [ ] Phase 6–8: stereo, hillshade matching, correlator
- [ ] Phase 9: `dem2gcp` — generating refined GCPs from the DEM
- [ ] Phase 10–11: final bundle_adjust, output
- [ ] Camera model considerations: `scripts/lib/scale_opticalbar.py`, `scale_linescan.py`
- [ ] QC: `scripts/lib/qc_csm_cameras.py`
- [ ] Common pitfalls (see [troubleshooting](../reference/troubleshooting.md) — esp. `scan_dir = right`)
- [ ] Next: [S2](s2-jitter-solve.md)
