# Global Keyhole Pipeline

Pipeline for orthorectifying declassified 1970s KH-9 HEXAGON spy satellite imagery using NASA Ames Stereo Pipeline. Uses automatic match points with reference topography and imagery to correct systematic camera distortions, enabling the creation of orthophotos with sub-pixel geometric accuracy.

The pipeline runs on top of [NASA Ames Stereo Pipeline (ASP)](https://stereopipeline.readthedocs.io/) and is organized as three stages that can be run independently per region:

| Stage | What it does |
|-------|--------------|
| **[S0](pipeline/s0-preprocessing.md)** | Pre-processing: resolve frame metadata, build DEM and modern-imagery mosaics for the target region, auto-detect crop windows. |
| **[S1](pipeline/s1-cameras.md)** | Per-pair camera optimization: an 11-phase cascade that refines each stereo pair's camera model from initial corner GCPs through hillshade-matched DEM ground control. |
| **[S2](pipeline/s2-jitter-solve.md)** | Multi-strip jitter solve: a 17-phase raw-image-space cascade that jointly solves jitter across overlapping strips for a globally consistent reconstruction. |

## Where to start

- **New to the pipeline?** → [Getting started](getting-started.md)
- **Setting up a new region?** → [S0 — Pre-processing](pipeline/s0-preprocessing.md)
- **Looking for a specific KH frame?** → [Image Archive](archive/index.md)
- **Pipeline failing in a confusing way?** → [Troubleshooting](reference/troubleshooting.md)

## What's in this site

This is the user-facing documentation. The pipeline source code (shell drivers in `scripts/`, Python helpers in `scripts/lib/`) lives in the same [GitHub repository](https://github.com/jmrallen/global-keyhole-pipeline).

The [Image Archive](archive/index.md) page renders all 586,076 declassified KH-9 frames as an interactive globe — filter by mission, date, or camera; click a footprint for metadata and a link to USGS EarthExplorer.
