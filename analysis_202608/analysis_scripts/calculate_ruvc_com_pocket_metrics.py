#!/usr/bin/env python3
"""Calculate RuvC I/II/III COM distances and catalytic pocket RMSD."""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from trajectory_io_first_batch import SYSTEMS, load_system, residue_range_heavy_atoms, residue_heavy_atoms


OUT = Path(__file__).resolve().parents[1] / "outputs" / "ruvc_com_pocket_metrics"
RUVC_DOMAINS = {
    "RuvC-I": (809, 872),
    "RuvC-II": (891, 997),
    "RuvC-III": (1180, 1226),
}
CATALYTIC_RESIDUES = (832, 925, 1180)


def heavy_atom_com(xyz_A: np.ndarray, atom_ids: np.ndarray) -> np.ndarray:
    return xyz_A[:, atom_ids].mean(axis=1)


def rmsd_to_first_frame(trajectory, atom_ids: np.ndarray) -> np.ndarray:
    aligned = trajectory[:]
    aligned.superpose(aligned, 0, atom_indices=atom_ids)
    return md.rmsd(aligned, aligned, 0, atom_indices=atom_ids) * 10.0


def calculate_system(system: str) -> pd.DataFrame:
    trajectory, topology, time_ns = load_system(system, start_ns=20.0, end_ns=100.0, stride_ns=1.0)
    xyz_A = trajectory.xyz * 10.0
    domain_atoms = {
        name: residue_range_heavy_atoms(topology, "A", lo, hi)
        for name, (lo, hi) in RUVC_DOMAINS.items()
    }
    catalytic_atoms = np.concatenate([residue_heavy_atoms(topology, "A", res) for res in CATALYTIC_RESIDUES])
    mg_atoms = np.array([atom.index for atom in topology.atoms if atom.element and atom.element.symbol == "Mg"], dtype=int)
    pocket_atoms = np.concatenate([catalytic_atoms, mg_atoms])

    centers = {name: heavy_atom_com(xyz_A, atom_ids) for name, atom_ids in domain_atoms.items()}
    pocket_rmsd_A = rmsd_to_first_frame(trajectory, pocket_atoms)

    rows = []
    for i, time in enumerate(time_ns):
        rows.append(
            {
                "system": system,
                "time_ns": float(time),
                "RuvCI_RuvCII_COM_A": float(np.linalg.norm(centers["RuvC-I"][i] - centers["RuvC-II"][i])),
                "RuvCI_RuvCIII_COM_A": float(np.linalg.norm(centers["RuvC-I"][i] - centers["RuvC-III"][i])),
                "RuvCII_RuvCIII_COM_A": float(np.linalg.norm(centers["RuvC-II"][i] - centers["RuvC-III"][i])),
                "catalytic_pocket_rmsd_A": float(pocket_rmsd_A[i]),
                "catalytic_pocket_atom_count": int(len(pocket_atoms)),
                "mg_atom_count": int(len(mg_atoms)),
            }
        )
    return pd.DataFrame(rows)


def summarize(timeseries: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "RuvCI_RuvCII_COM_A",
        "RuvCI_RuvCIII_COM_A",
        "RuvCII_RuvCIII_COM_A",
        "catalytic_pocket_rmsd_A",
    ]
    return (
        timeseries.groupby("system")[metrics]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    timeseries = pd.concat([calculate_system(system) for system in SYSTEMS], ignore_index=True)
    timeseries.to_csv(OUT / "ruvc_com_pocket_timeseries_20_100ns.csv", index=False)
    summarize(timeseries).to_csv(OUT / "ruvc_com_pocket_summary.csv", index=False)
    print(OUT)


if __name__ == "__main__":
    main()

