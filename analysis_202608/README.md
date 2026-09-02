# First-Batch Focused Metric Scripts

This package contains only calculation scripts for the first-batch Cas12a MD dataset. Plotting scripts and unrelated analyses such as SASA/RMSF, entry-grid volume, networks, clustering, and keyframe rendering were intentionally removed.

## Scripts

- `analysis_scripts/trajectory_io_first_batch.py`: shared solute-trajectory loader.
- `analysis_scripts/calculate_ruvc_pocket_backbone_rmsd.py`: RuvC catalytic-pocket RMSD and whole-protein backbone RMSD.

## Usage

Run from a checkout that contains the first-batch data directories, or set `CAS12A_MD_ROOT` to the project root.

```bash
python analysis_scripts/calculate_ruvc_pocket_backbone_rmsd.py
```

The scripts write CSV outputs under `outputs/` inside this package.
