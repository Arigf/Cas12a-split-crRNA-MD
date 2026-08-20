from __future__ import annotations

import json
from pathlib import Path

import mdtraj as md
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "Match-Full"
PROD = ROOT / "production_md_2p5mM_MgCl2_37p5mM_KCl" / SYSTEM
TOP_PDB = ROOT / "equilibration_smoke_2p5mM_MgCl2_37p5mM_KCl" / SYSTEM / "06_unrestrained_stability_npt.pdb"
PREFIX = PROD / "prod_0001_100ns"
OUT = PROD / "analysis"


def stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    atom_indices = np.loadtxt(f"{PREFIX}_solute_atom_indices.txt", dtype=int)
    full = md.load_pdb(str(TOP_PDB))
    solute_top = full.atom_slice(atom_indices).topology
    # The PDB omits inter-residue nucleic-acid bonds. Restore O3'-P links so
    # periodic imaging treats each DNA/RNA strand as one molecule.
    for chain in solute_top.chains:
        residues = list(chain.residues)
        if not residues or not residues[0].is_nucleic:
            continue
        for left, right in zip(residues[:-1], residues[1:]):
            o3 = next((a for a in left.atoms if a.name in {"O3'", "O3*"}), None)
            p = next((a for a in right.atoms if a.name == "P"), None)
            if o3 is not None and p is not None:
                solute_top.add_bond(o3, p)
    traj = md.load(f"{PREFIX}_solute.dcd", top=solute_top)
    molecules = list(solute_top.find_molecules())
    protein_molecule = max(molecules, key=len)
    traj = traj.image_molecules(
        anchor_molecules=[protein_molecule],
        other_molecules=[m for m in molecules if m is not protein_molecule],
        make_whole=True,
    )
    time_ns = np.arange(traj.n_frames, dtype=float) * 0.1

    ca = traj.topology.select("protein and name CA")
    protein_heavy = traj.topology.select("protein and not element H")
    nucleic_heavy = traj.topology.select("nucleic and not element H")
    dna_heavy = traj.topology.select("chainid 1 and not element H")
    rna_heavy = traj.topology.select("chainid 2 and not element H")

    aligned = traj[:]
    aligned.superpose(aligned, 0, atom_indices=ca)
    protein_rmsd_nm = md.rmsd(aligned, aligned, 0, atom_indices=ca, precentered=False)
    nucleic_in_protein_frame_nm = md.rmsd(aligned, aligned, 0, atom_indices=nucleic_heavy, precentered=False)
    nucleic_aligned = traj[:]
    nucleic_aligned.superpose(nucleic_aligned, 0, atom_indices=nucleic_heavy)
    nucleic_internal_rmsd_nm = md.rmsd(
        nucleic_aligned, nucleic_aligned, 0, atom_indices=nucleic_heavy, precentered=False
    )
    ca_rmsf_nm = md.rmsf(aligned, aligned[0], atom_indices=ca, precentered=False)
    dna_aligned = traj[:]
    dna_aligned.superpose(dna_aligned, 0, atom_indices=dna_heavy)
    dna_rmsd_nm = md.rmsd(dna_aligned, dna_aligned, 0, atom_indices=dna_heavy)
    rna_aligned = traj[:]
    rna_aligned.superpose(rna_aligned, 0, atom_indices=rna_heavy)
    rna_rmsd_nm = md.rmsd(rna_aligned, rna_aligned, 0, atom_indices=rna_heavy)
    rg_protein_nm = md.compute_rg(traj.atom_slice(protein_heavy))
    rg_nucleic_nm = md.compute_rg(traj.atom_slice(nucleic_heavy))

    mg = np.array([a.index for a in traj.topology.atoms if a.element and a.element.symbol == "Mg"])
    ligand_atoms = np.array(
        [
            a.index
            for a in traj.topology.atoms
            if a.index not in set(mg) and a.element and a.element.symbol != "H"
        ]
    )
    coordination = []
    ligand_rows = []
    for mg_idx in mg:
        pairs = np.column_stack((np.full(ligand_atoms.size, mg_idx), ligand_atoms))
        dist = md.compute_distances(traj, pairs, periodic=True)
        counts = np.sum(dist < 0.26, axis=1)
        occupancy = np.mean(dist < 0.26, axis=0)
        top_ligands = np.argsort(occupancy)[::-1][:12]
        label = str(traj.topology.atom(int(mg_idx)))
        coordination.append({"magnesium": label, "coordination_count": stats(counts)})
        for j in top_ligands:
            if occupancy[j] < 0.01:
                continue
            atom = traj.topology.atom(int(ligand_atoms[j]))
            row = {
                "magnesium": label,
                "ligand": str(atom),
                "chain": atom.residue.chain.chain_id,
                "resSeq": atom.residue.resSeq,
                "occupancy_lt_2p6A": float(occupancy[j]),
                "distance_A_mean": float(np.mean(dist[:, j]) * 10),
                "distance_A_p95": float(np.percentile(dist[:, j], 95) * 10),
            }
            ligand_rows.append(row)

    mg_mg_A = md.compute_distances(traj, [[int(mg[0]), int(mg[1])]], periodic=True)[:, 0] * 10
    def atom_index(chain: str, resseq: int, name: str) -> int:
        return next(a.index for a in traj.topology.atoms
                    if a.residue.chain.chain_id == chain and a.residue.resSeq == resseq and a.name == name)
    catalytic_pairs = {
        "Mg1-D832_OD1": (mg[0], atom_index("A", 832, "OD1")),
        "Mg1-E925_OE1": (mg[0], atom_index("A", 925, "OE1")),
        "Mg1-E925_OE2": (mg[0], atom_index("A", 925, "OE2")),
        "Mg1-cleavage_OP1": (mg[0], atom_index("B", 18, "OP1")),
        "Mg2-D832_OD2": (mg[1], atom_index("A", 832, "OD2")),
        "Mg2-D1180_OD1": (mg[1], atom_index("A", 1180, "OD1")),
        "Mg2-cleavage_OP2": (mg[1], atom_index("B", 18, "OP2")),
    }
    catalytic_dist_A = {
        name: md.compute_distances(traj, [pair], periodic=True)[:, 0] * 10
        for name, pair in catalytic_pairs.items()
    }
    productive = (mg_mg_A >= 3.0) & (mg_mg_A <= 5.0)
    for values in catalytic_dist_A.values():
        productive &= values < 2.6

    # SASA is calculated with the entire solute present so occlusion by the
    # surrounding macromolecules is retained.
    sasa_nm2 = md.shrake_rupley(traj, mode="atom", n_sphere_points=320)
    catalytic_residue_sasa = {}
    for resseq in (832, 925, 1180):
        sel = traj.topology.select(f"chainid 0 and residue {resseq - 1}")
        catalytic_residue_sasa[f"A:{resseq}"] = np.sum(sasa_nm2[:, sel], axis=1) * 100
    cleavage_sel = np.array([atom_index("B", 18, n) for n in ("P", "OP1", "OP2")])
    cleavage_sasa_A2 = np.sum(sasa_nm2[:, cleavage_sel], axis=1) * 100

    state = np.loadtxt(f"{PREFIX}.tsv", skiprows=1, usecols=range(9))
    # Columns: step, time, potential, kinetic, total, temperature, volume, density, speed, ETA.
    temperature = state[:, 5]
    density = state[:, 7]
    potential = state[:, 2]
    total_energy = state[:, 4]
    x_ns = (state[:, 1] - state[0, 1]) / 1000.0
    potential_slope = float(np.polyfit(x_ns, potential, 1)[0])

    result = {
        "system": SYSTEM,
        "frames": traj.n_frames,
        "sampling_interval_ns": 0.1,
        "trajectory_span_ns": [float(time_ns[0]), float(time_ns[-1])],
        "rmsd_A": {
            "protein_CA_vs_first": stats(protein_rmsd_nm * 10),
            "nucleic_heavy_internal_vs_first": stats(nucleic_internal_rmsd_nm * 10),
            "nucleic_heavy_in_protein_frame_vs_first": stats(nucleic_in_protein_frame_nm * 10),
            "DNA_heavy_internal_vs_first": stats(dna_rmsd_nm * 10),
            "RNA_heavy_internal_vs_first": stats(rna_rmsd_nm * 10),
            "protein_CA_last_20ns": stats(protein_rmsd_nm[-200:] * 10),
        },
        "protein_CA_rmsf_A": stats(ca_rmsf_nm * 10),
        "radius_of_gyration_A": {
            "protein_heavy": stats(rg_protein_nm * 10),
            "nucleic_heavy": stats(rg_nucleic_nm * 10),
        },
        "thermodynamics": {
            "temperature_K": stats(temperature),
            "density_g_ml": stats(density),
            "potential_energy_kj_mol": stats(potential),
            "total_energy_kj_mol": stats(total_energy),
            "potential_energy_linear_slope_kj_mol_per_ns": potential_slope,
            "last_20_percent_temperature_K": stats(temperature[-len(temperature)//5:]),
            "last_20_percent_density_g_ml": stats(density[-len(density)//5:]),
        },
        "magnesium": coordination,
        "magnesium_ligands": ligand_rows,
        "catalytic_geometry": {
            "Mg_Mg_distance_A": stats(mg_mg_A),
            "distances_A": {name: stats(values) for name, values in catalytic_dist_A.items()},
            "productive_geometry_definition": "Mg-Mg 3.0-5.0 A and all listed metal-ligand distances <2.6 A",
            "productive_geometry_fraction": float(np.mean(productive)),
        },
        "SASA_A2": {
            "catalytic_residues": {name: stats(values) for name, values in catalytic_residue_sasa.items()},
            "cleavage_phosphate_B18_P_OP1_OP2": stats(cleavage_sasa_A2),
            "cleavage_phosphate_last_20ns": stats(cleavage_sasa_A2[-200:]),
        },
    }
    (OUT / "match_full_100ns_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savetxt(
        OUT / "match_full_100ns_timeseries.csv",
        np.column_stack((time_ns, protein_rmsd_nm * 10, nucleic_internal_rmsd_nm * 10,
                         nucleic_in_protein_frame_nm * 10, rg_protein_nm * 10, rg_nucleic_nm * 10)),
        delimiter=",",
        header="time_ns,protein_ca_rmsd_A,nucleic_internal_rmsd_A,nucleic_in_protein_frame_rmsd_A,protein_rg_A,nucleic_rg_A",
        comments="",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
