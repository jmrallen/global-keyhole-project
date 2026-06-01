# Global Keyhole Pipeline

Pipeline for orthorectifying declassified 1970s KH-9 HEXAGON spy satellite imagery using NASA Ames Stereo Pipeline. Uses automatic match points with reference topography and imagery to correct systematic camera distortions, enabling the creation of orthophotos with sub-pixel geometric accuracy.

📖 **Documentation:** <https://jmrallen.github.io/global-keyhole-pipeline/>
🗺️ **Interactive image archive:** <https://jmrallen.github.io/global-keyhole-pipeline/archive/>

---

## What this pipeline does

KH-9 HEXAGON frames were acquired by a panoramic optical-bar camera between 1971 and 1986 and declassified in phases starting in 2002. Each frame has only approximate ephemeris and pointing metadata, so a raw `cam_gen`-seeded camera model is geometrically off by hundreds of meters or more. This pipeline refines those camera models against:

1. **Reference topography** — modern DEMs (Copernicus GLO-30, ArcticDEM, etc.) used as ground control via hillshade matching and `dem2gcp` GCP generation.
2. **Reference imagery** — modern orthorectified basemaps (Planet, Sentinel-2) for additional tie-point constraints.

The output is a refined camera model per frame and per strip, suitable for producing orthophotos and stereo-derived DEMs with sub-pixel geometric accuracy.

## Pipeline structure

The pipeline runs in three sequential stages, each a shell driver in `scripts/` backed by Python helpers in `scripts/lib/`. Stages are independent per region and can be re-run incrementally.

### S0 — Pre-processing
**Driver:** `scripts/S0.sh`

Resolves frame metadata from the declassified catalog (`assets/declass3_metadata.parquet`), assembles DEM and modern-imagery mosaics over the target region, and auto-detects per-frame crop windows. Output is a region manifest consumed by S1.

Key helpers: `lib/metadata.py`, `lib/dem_tiles.py`, `lib/planet_tiles.py`, `lib/crop_detect.py`, `lib/manifest.py`, `lib/footprint.py`.

### S1 — Per-pair camera optimization
**Driver:** `scripts/S1.sh`

An 11-phase cascade run per stereo pair that refines each frame's KH-9 OPTICAL_BAR camera model from corner-seeded `cam_gen` through hillshade-matched ground control:

1. Crop + sub-resolution preview
2. Corner-seeded `cam_gen`
3. Mapproject against the reference DEM
4. Initial `bundle_adjust`
5. Stereo correlation
6. Hillshade rendering of the produced DEM
7. Correlator matching to the reference DEM hillshade
8. `dem2gcp` — convert matches to ground control points
9. Filter GCPs by camera
10. Refined `bundle_adjust` with GCP constraints
11. Output of refined CSM cameras + QC

This is the canonical KH-9 recipe per ASP §8.29.9. The output is per-pair CSM cameras ready for S2.

Key helpers: `lib/cam_gen_corners.py`, `lib/scale_opticalbar.py`, `lib/qc_csm_cameras.py`, `lib/filter_gcps_by_camera.py`.

### S2 — Multi-strip jitter solve
**Driver:** `scripts/S2.sh`

A 17-phase raw-image-space cascade that takes S1-optimized pairs from overlapping strips and jointly solves for jitter, producing a globally consistent reconstruction over an entire region rather than a per-pair patchwork. Built around ASP's `jitter_solve` with custom raw-space tile matching to seed the cross-strip tie points.

Key helpers: `lib/raw_tile_match.py`, `lib/raw_npz_to_gcp.py`, `lib/batch_npz_to_gcp.py`, `lib/concat_gcps.py`.

## Repository layout

```
scripts/                S0.sh, S1.sh, S2.sh + Python helpers in lib/
config/                 Per-machine paths (config.yaml — gitignored)
cameras/                Camera model templates per KH series
assets/                 declass3_metadata.parquet — master frame catalog (586k rows)
docs/                   Public-facing documentation site (MkDocs Material)
.github/workflows/      CI for building and deploying the docs site
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/jmrallen/global-keyhole-pipeline.git
cd global-keyhole-pipeline

# 2. Install Python deps into your ASP conda env (or any 3.10+ env)
conda install -n asp_py -c conda-forge \
    shapely pandas pyarrow pyyaml rasterio opencv numpy

# 3. Configure per-machine paths
cp config/config.example.yaml config/config.yaml
# edit working_dir, archive_dir, tile dirs

# 4. Pick a region from the interactive archive and run S0
bash scripts/S0.sh <region-name>
```

See the [Getting Started guide](https://jmrallen.github.io/global-keyhole-pipeline/getting-started/) for full setup details.

## Data sources

- **KH-9 imagery:** USGS EarthExplorer (Declassified Satellite Imagery — DSI). The interactive [Image Archive](https://jmrallen.github.io/global-keyhole-pipeline/archive/) is built from the USGS declass3 metadata catalog.
- **Reference DEMs:** Copernicus GLO-30, ArcticDEM, REMA (per-region).
- **Reference imagery:** Planet basemaps, Sentinel-2 (per-region).

## Documentation

The full user guide — tutorials for each stage, library reference, troubleshooting, and the interactive image archive — lives at <https://jmrallen.github.io/global-keyhole-pipeline/>.

## License

See [LICENSE](LICENSE).
