## Scripts

- `analysis_scripts/trajectory_io_first_batch.py`: shared solute-trajectory loader.
- `analysis_scripts/calculate_crrna_pairing_metrics.py`: full-crRNA base-centroid distances, N/O contacts, paired-state occupancy, and per-ns pairing ratios.
- `analysis_scripts/calculate_ruvc_com_pocket_metrics.py`: RuvC-I/II/III center-of-geometry distances and catalytic-pocket RMSD.

## Usage

Run from a checkout that contains the first-batch data directories, or set `CAS12A_MD_ROOT` to the project root.

```bash
python analysis_scripts/calculate_crrna_pairing_metrics.py
python analysis_scripts/calculate_ruvc_com_pocket_metrics.py
```

The scripts write CSV outputs under `outputs/` inside this package.

