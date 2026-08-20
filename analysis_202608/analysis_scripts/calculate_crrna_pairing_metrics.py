#!/usr/bin/env python3
"""Calculate full-crRNA base-centroid distances and pairing indices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_io_first_batch import SYSTEMS, base_heavy_atoms, load_system


OUT = Path(__file__).resolve().parents[1] / "outputs" / "crrna_pairing_metrics"
PAIR_DISTANCE_CUTOFF_A = 6.5
NO_CONTACT_CUTOFF_A = 3.5


def guide_location(system: str, guide_eq: int) -> tuple[str, int]:
    if "Split" in system and guide_eq >= 35:
        return "D", guide_eq - 34
    return "C", guide_eq


def calculate_system(system: str) -> pd.DataFrame:
    trajectory, topology, time_ns = load_system(system, start_ns=20.0, end_ns=100.0, stride_ns=1.0)
    xyz_A = trajectory.xyz * 10.0
    atoms = list(topology.atoms)
    rows = []

    for guide_eq in range(1, 45):
        guide_chain, guide_residue = guide_location(system, guide_eq)
        target_residue = 64 - guide_eq
        target_present = 1 <= target_residue <= 50
        guide_atoms = base_heavy_atoms(topology, guide_chain, guide_residue)
        target_atoms = base_heavy_atoms(topology, "B", target_residue) if target_present else np.array([], dtype=int)
        can_pair = target_present and len(guide_atoms) > 0 and len(target_atoms) > 0

        if can_pair:
            guide_centroid = xyz_A[:, guide_atoms].mean(axis=1)
            target_centroid = xyz_A[:, target_atoms].mean(axis=1)
            basecentroid_A = np.linalg.norm(guide_centroid - target_centroid, axis=1)
            base_dist = np.linalg.norm(
                xyz_A[:, guide_atoms, None, :] - xyz_A[:, None, target_atoms, :],
                axis=3,
            )
            min_base_A = base_dist.min(axis=(1, 2))
            base_contacts_3p5A = (base_dist < NO_CONTACT_CUTOFF_A).sum(axis=(1, 2))
            guide_no = np.array([idx for idx in guide_atoms if atoms[idx].element.symbol in {"N", "O"}], dtype=int)
            target_no = np.array([idx for idx in target_atoms if atoms[idx].element.symbol in {"N", "O"}], dtype=int)
            no_dist = np.linalg.norm(
                xyz_A[:, guide_no, None, :] - xyz_A[:, None, target_no, :],
                axis=3,
            )
            no_contacts_3p5A = (no_dist < NO_CONTACT_CUTOFF_A).sum(axis=(1, 2))
            paired = (basecentroid_A < PAIR_DISTANCE_CUTOFF_A) & (no_contacts_3p5A >= 1)
            paired_binary = paired.astype(float)
        else:
            n = len(time_ns)
            basecentroid_A = np.full(n, np.nan)
            min_base_A = np.full(n, np.nan)
            base_contacts_3p5A = np.full(n, np.nan)
            no_contacts_3p5A = np.full(n, np.nan)
            paired = np.full(n, np.nan)
            paired_binary = np.full(n, np.nan)

        for i, time in enumerate(time_ns):
            rows.append(
                {
                    "system": system,
                    "time_ns": float(time),
                    "crRNA_chain": guide_chain,
                    "crRNA_residue": guide_residue,
                    "guide_equivalent_position": guide_eq,
                    "target_DNA_residue_B": target_residue if target_present else np.nan,
                    "target_present": bool(target_present),
                    "pairing_calculated": bool(can_pair),
                    "basecentroid_A": basecentroid_A[i],
                    "min_base_A": min_base_A[i],
                    "base_contacts_3p5A": base_contacts_3p5A[i],
                    "NO_contacts_3p5A": no_contacts_3p5A[i],
                    "paired_state": bool(paired[i]) if can_pair else np.nan,
                    "paired_binary": paired_binary[i],
                }
            )
    return pd.DataFrame(rows)


def summarize(timeseries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_position = (
        timeseries.groupby(
            [
                "system",
                "crRNA_chain",
                "crRNA_residue",
                "guide_equivalent_position",
                "target_DNA_residue_B",
                "target_present",
                "pairing_calculated",
            ],
            dropna=False,
        )
        .agg(
            frames=("time_ns", "size"),
            paired_occupancy=("paired_binary", "mean"),
            basecentroid_mean_A=("basecentroid_A", "mean"),
            basecentroid_sd_A=("basecentroid_A", "std"),
            basecentroid_p50_A=("basecentroid_A", "median"),
            NO_contacts_mean=("NO_contacts_3p5A", "mean"),
        )
        .reset_index()
    )

    rows = []
    for (system, time), group in timeseries.groupby(["system", "time_ns"], sort=True):
        calculated = group[group["pairing_calculated"]].copy()
        rows.append(
            {
                "system": system,
                "time_ns": time,
                "n_full_crRNA_positions": len(group),
                "n_target_mapped_positions": len(calculated),
                "paired_positions_target_mapped": int(calculated["paired_binary"].sum()),
                "pairing_ratio_target_mapped_eq14_eq44": float(calculated["paired_binary"].mean()),
                "pairing_ratio_full_crRNA_denominator44": float(group["paired_binary"].fillna(0).sum() / len(group)),
                "mean_basecentroid_A_target_mapped": float(calculated["basecentroid_A"].mean()),
            }
        )
    return by_position, pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    timeseries = pd.concat([calculate_system(system) for system in SYSTEMS], ignore_index=True)
    by_position, per_ns = summarize(timeseries)
    timeseries.to_csv(OUT / "crrna_pairing_timeseries_eq1_eq44.csv", index=False)
    by_position.to_csv(OUT / "crrna_pairing_summary_by_position.csv", index=False)
    per_ns.to_csv(OUT / "crrna_pairing_ratio_per_ns.csv", index=False)
    print(OUT)


if __name__ == "__main__":
    main()

