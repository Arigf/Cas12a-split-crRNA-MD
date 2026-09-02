#!/usr/bin/env python3
"""Shared trajectory loading utilities for the first-batch four-system dataset."""

from __future__ import annotations

import os
from pathlib import Path

import mdtraj as md
import numpy as np


SYSTEMS = ("Match-Full", "Match-Split", "MM-Full", "MM-Split")


def project_root() -> Path:
    env = os.environ.get("CAS12A_MD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


ROOT = project_root()
PROD = ROOT / "production_md_2p5mM_MgCl2_37p5mM_KCl"
EQ = ROOT / "equilibration_smoke_2p5mM_MgCl2_37p5mM_KCl"


def repair_nucleic_bonds(topology) -> None:
    for chain in topology.chains:
        residues = list(chain.residues)
        if residues and residues[0].is_nucleic:
            for left, right in zip(residues[:-1], residues[1:]):
                o3 = next((atom for atom in left.atoms if atom.name in {"O3'", "O3*"}), None)
                p = next((atom for atom in right.atoms if atom.name == "P"), None)
                if o3 is not None and p is not None:
                    topology.add_bond(o3, p)


def production_prefixes(system: str) -> list[Path]:
    base = PROD / system
    if system == "MM-Split":
        return [base / f"prod_{i:04d}" for i in range(1, 6)] + [base / "prod_0006_95ns"]
    return [base / "prod_0001_100ns"]


def load_system(system: str, start_ns: float = 20.0, end_ns: float = 100.0, stride_ns: float = 1.0):
    """Load a solute-only trajectory and return common-window frames.

    Returns `(trajectory, topology, time_ns)`. The default window is 20-100 ns
    sampled every 1 ns, matching the published first-batch analyses.
    """
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system: {system}")
    prefixes = production_prefixes(system)
    atom_indices = np.loadtxt(str(prefixes[0]) + "_solute_atom_indices.txt", dtype=int)
    full = md.load_pdb(str(EQ / system / "06_unrestrained_stability_npt.pdb"))
    topology = full.atom_slice(atom_indices).topology
    del full
    repair_nucleic_bonds(topology)

    chunks = [md.load(str(prefix) + "_solute.dcd", top=topology) for prefix in prefixes]
    if system == "MM-Split":
        chunks = [chunk[1::2] for chunk in chunks[:5]] + [chunks[5]]
    trajectory = md.join(chunks, check_topology=True, discard_overlapping_frames=False)
    if trajectory.n_frames != 1000:
        raise ValueError(f"{system}: expected 1000 common-cadence frames, got {trajectory.n_frames}")

    molecules = list(topology.find_molecules())
    anchor = max(molecules, key=len)
    trajectory = trajectory.image_molecules(
        anchor_molecules=[anchor],
        other_molecules=[mol for mol in molecules if mol is not anchor],
        make_whole=True,
    )

    all_time = np.arange(trajectory.n_frames, dtype=float) * 0.1
    step = max(1, round(stride_ns / 0.1))
    mask = np.where((all_time >= start_ns) & (all_time < end_ns))[0][::step]
    return trajectory[mask], topology, all_time[mask]


def atom_indices(topology, predicate) -> np.ndarray:
    return np.array([atom.index for atom in topology.atoms if predicate(atom)], dtype=int)


def residue_heavy_atoms(topology, chain_id: str, resseq: int) -> np.ndarray:
    return atom_indices(
        topology,
        lambda atom: (
            atom.residue.chain.chain_id == chain_id
            and atom.residue.resSeq == resseq
            and atom.element is not None
            and atom.element.symbol != "H"
        ),
    )

