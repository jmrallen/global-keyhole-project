# Troubleshooting

Common pitfalls and their fixes.

## Audience

You're running the pipeline and something has gone wrong. Start here before opening an issue.

## TODO — sections to write

### KH-9 specific

- [ ] **`scan_dir` must be `right`** — the velocity-vector OPTICAL_BAR model hard-errors on `scan_dir = left`. Do not propose KH-4B-style asymmetric `scan_dir`. Troubleshoot mirror-flipped DEMs via `mean_surface_elevation` or `forward_tilt` symmetry breaking instead.
- [ ] **Use the canonical S1 path** — hillshade + correlator + `dem2gcp` is the documented KH-9 recipe (ASP §8.29.9). Do not silently swap to `pc_align` (that's §8.28.9 / KH-4B).

### ASP gotchas

- [ ] **Flags don't cross tools** — `--min-matches` is bundle_adjust-only, `--max-pairwise-matches` is bundle_adjust/jitter_solve-only, etc. Always check `<tool> --help` before adding a flag copied from a different ASP tool's docs.
- [ ] Match file naming conventions
- [ ] When stereo correlation produces empty point clouds
- [ ] When `dem2gcp` produces few GCPs

### Environment

- [ ] `csmapi` is bundled with ASP, not pip-installable
- [ ] Config path resolution issues (`config/config.yaml` is gitignored — must exist locally)
