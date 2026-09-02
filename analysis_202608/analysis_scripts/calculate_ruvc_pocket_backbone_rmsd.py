#!/usr/bin/env python3
"""Calculate catalytic-pocket and whole-protein backbone RMSD."""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from trajectory_io_first_batch import SYSTEMS, load_system, residue_heavy_atoms


OUT = Path(__file__).resolve().parents[1] / "outputs" / "ruvc_pocket_backbone_rmsd"
CATALYTIC_RESIDUES = (832, 925, 1180)
PROTEIN_BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}


def protein_backbone_atoms(topology) -> np.ndarray:
    return np.array(
        [
            atom.index
            for atom in topology.atoms
            if atom.residue.is_protein and atom.name in PROTEIN_BACKBONE_ATOM_NAMES
        ],
        dtype=int,
    )


def require_atoms(system: str, label: str, atom_ids: np.ndarray) -> np.ndarray:
    if len(atom_ids) == 0:
        raise ValueError(f"{system}: no atoms found for {label}")
    return atom_ids


def rmsd_to_first_frame(trajectory, atom_ids: np.ndarray) -> np.ndarray:
    aligned = trajectory[:]
    aligned.superpose(aligned, 0, atom_indices=atom_ids)
    return md.rmsd(aligned, aligned, 0, atom_indices=atom_ids) * 10.0


def calculate_system(system: str) -> pd.DataFrame:
    trajectory, topology, time_ns = load_system(system, start_ns=20.0, end_ns=100.0, stride_ns=1.0)
    catalytic_atoms = require_atoms(
        system,
        "RuvC catalytic residues",
        np.concatenate([residue_heavy_atoms(topology, "A", res) for res in CATALYTIC_RESIDUES]),
    )
    mg_atoms = np.array([atom.index for atom in topology.atoms if atom.element and atom.element.symbol == "Mg"], dtype=int)
    pocket_atoms = require_atoms(system, "RuvC catalytic pocket", np.concatenate([catalytic_atoms, mg_atoms]))
    protein_backbone = require_atoms(system, "protein backbone", protein_backbone_atoms(topology))

    pocket_rmsd_A = rmsd_to_first_frame(trajectory, pocket_atoms)
    protein_backbone_rmsd_A = rmsd_to_first_frame(trajectory, protein_backbone)

    rows = []
    for i, time in enumerate(time_ns):
        rows.append(
            {
                "system": system,
                "time_ns": float(time),
                "catalytic_pocket_rmsd_A": float(pocket_rmsd_A[i]),
                "protein_backbone_rmsd_A": float(protein_backbone_rmsd_A[i]),
                "catalytic_pocket_atom_count": int(len(pocket_atoms)),
                "mg_atom_count": int(len(mg_atoms)),
                "protein_backbone_atom_count": int(len(protein_backbone)),
            }
        )
    return pd.DataFrame(rows)


def summarize(timeseries: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "catalytic_pocket_rmsd_A",
        "protein_backbone_rmsd_A",
    ]
    return (
        timeseries.groupby("system")[metrics]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    timeseries = pd.concat([calculate_system(system) for system in SYSTEMS], ignore_index=True)
    timeseries.to_csv(OUT / "ruvc_pocket_backbone_rmsd_timeseries_20_100ns.csv", index=False)
    summarize(timeseries).to_csv(OUT / "ruvc_pocket_backbone_rmsd_summary.csv", index=False)
    print(OUT)


if __name__ == "__main__":
    main()
