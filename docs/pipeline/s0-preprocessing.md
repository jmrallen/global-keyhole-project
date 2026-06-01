# S0 — Pre-processing

Driver: `scripts/S0.sh`. Purpose: get a target region ready for stereo processing by resolving frame metadata, building basemap/DEM tiles, and detecting per-frame crop windows.

## Audience

You've picked a region of interest and identified candidate KH-9 frames (e.g., via the [Image Archive](../archive/index.md)). You want to set up the working directory before running S1.

## TODO — sections to write

- [ ] What S0 produces (per-frame manifest, mosaic tiles, crop bboxes)
- [ ] Walkthrough: metadata resolution via `scripts/lib/metadata.py`
- [ ] Walkthrough: DEM tile assembly via `scripts/lib/dem_tiles.py`
- [ ] Walkthrough: Planet basemap mosaic via `scripts/lib/planet_tiles.py`
- [ ] Walkthrough: crop-window auto-detection via `scripts/lib/crop_detect.py`
- [ ] Walkthrough: manifest generation via `scripts/lib/manifest.py`
- [ ] Outputs: directory layout, what each artifact is
- [ ] Common pitfalls (frame downloads from USGS, tile cache locations)
- [ ] Next: [S1](s1-cameras.md)
