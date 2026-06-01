# `scripts/lib/` reference

Quick-reference catalog of the Python helpers in `scripts/lib/`. Each helper is a single-purpose module called from one of the shell drivers.

## TODO — sections to write

For each module, document: what it does, key CLI flags, inputs, outputs, which stage uses it.

### Metadata & manifest
- [ ] `metadata.py` — frame catalog lookup
- [ ] `manifest.py` — per-region manifest generation
- [ ] `resolved.py` — resolved-path tracking
- [ ] `config.py` — config.yaml loader

### Footprints & tiles
- [ ] `footprint.py` — per-frame footprint geometry
- [ ] `crop_detect.py` — auto crop window detection
- [ ] `dem_tiles.py` — DEM mosaic tile assembly
- [ ] `planet_tiles.py` — Planet basemap tile assembly

### Cameras (S1)
- [ ] `cam_gen_corners.py` — corner-seeded `cam_gen` input
- [ ] `scale_opticalbar.py` — OPTICAL_BAR camera scaling (KH-9 panoramic)
- [ ] `scale_linescan.py` — linescan camera scaling
- [ ] `qc_csm_cameras.py` — CSM camera QC

### Jitter solve (S2)
- [ ] `raw_tile_match.py` — raw-image-space tile matching
- [ ] `raw_npz_to_gcp.py` — convert matches to GCPs
- [ ] `batch_npz_to_gcp.py` — batch wrapper
- [ ] `concat_gcps.py` — combine GCPs across pairs
- [ ] `filter_gcps_by_camera.py` — per-camera GCP filtering
