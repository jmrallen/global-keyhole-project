# Getting started

This page walks through cloning the repo, installing dependencies, and running the pipeline on a small example region.

## Audience

You're comfortable on a Linux shell, have used `conda` before, and have heard of NASA Ames Stereo Pipeline. You don't need to be a photogrammetry expert — the pipeline's job is to encapsulate that.

## TODO — sections to write

- [ ] Prerequisites (ASP install, conda env, USGS EarthExplorer account for frame downloads)
- [ ] Clone the repo and inspect the layout (`scripts/`, `config/`, `cameras/`, `assets/`)
- [ ] Configure `config/config.yaml` (working_dir, archive_dir, tile dirs — see `config/config.example.yaml`)
- [ ] Pick a region: walk through identifying KH-9 frames from the [Image Archive](archive/index.md)
- [ ] Run S0 end-to-end on a small example region
- [ ] Where outputs land; how to inspect intermediate products
- [ ] Next: link to [S1](pipeline/s1-cameras.md) for camera optimization
