#!/usr/bin/env python3
"""Extract time-matched eq39-B25 keyframes for Split-PM and FL-MM.

Project naming:
- Split-PM maps to Match-Split
- FL-MM maps to MM-Full
"""

from pathlib import Path
import sys

import mdtraj as md
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_ruvc_entry_grid as ag
import analyze_precatalytic_allostery as pa

OUT = ROOT / "analysis_four_systems_20_100ns/eq39_keyframes_split_pm_fl_mm"
FEATURES = ROOT / "analysis_four_systems_20_100ns/precatalytic_allostery/feature_timeseries.csv"
TIMES = [26.0, 27.0, 65.0, 97.0]
ALIASES = {
    "Match-Split": "Split-PM",
    "MM-Full": "FL-MM",
}
LABELS = {
    26.0: "time_matched_26ns",
    27.0: "time_matched_27ns",
    65.0: "time_matched_65ns",
    97.0: "time_matched_97ns",
}


def local_indices(top, system):
    if "Split" in system:
        guide_chain = "D"
        guide_res = range(3, 7)  # equivalent guide positions 37-40
    else:
        guide_chain = "C"
        guide_res = range(37, 41)
    return np.array(
        [
            a.index
            for a in top.atoms
            if (a.residue.chain.chain_id == "B" and 24 <= a.residue.resSeq <= 27)
            or (a.residue.chain.chain_id == guide_chain and a.residue.resSeq in guide_res)
        ],
        dtype=int,
    )


def write_readme(path):
    readme = """# Split-PM and FL-MM eq39-B25 keyframes

This directory contains time-matched keyframes from the first four-system MD
round without the added common cleavage strand.

System name mapping:

- `Split-PM` = project system `Match-Split`
- `FL-MM` = project system `MM-Full`

Frames were extracted at 26, 27, 65, and 97 ns from the processed 20-100 ns
solute trajectories. Structures were imaged and aligned to the common
Match-Full protein C-alpha reference using the same helper functions as the
original `eq39_keyframes_comparison` extraction.

Directory contents:

- `full_complex/`: complete solute-complex PDBs for each system and time.
- `local/`: local B24-B27 target-DNA plus equivalent guide positions 37-40.
  For Split-PM these guide residues are chain D residues 3-6; for FL-MM these
  guide residues are chain C residues 37-40.
- `*_four_keyframes_full_aligned.pdb`: four full-complex frames in one
  multi-model PDB, chronological order.
- `*_four_keyframes_local.pdb`: four local frames in one multi-model PDB,
  chronological order.
- `frame_selection_and_metrics.csv`: extracted file paths and eq39-B25 metrics.
- `compare_split_pm_fl_mm_eq39_B25.pml`: PyMOL loader for the local structures.

Pairing proxy: `eq39_B25_basecentroid_A < 6.5` and
`eq39_B25_NO_contacts_lt3p5A >= 1`.
"""
    path.write_text(readme)


def write_pml(path):
    lines = [
        "reinitialize",
        "set dash_width, 2.0",
        "set dash_gap, 0.25",
        "set stick_radius, 0.16",
        "bg_color white",
        "set ray_opaque_background, off",
    ]
    for system, alias in ALIASES.items():
        prefix = alias.replace("-", "_")
        for t in TIMES:
            label = LABELS[t]
            lines.append(f"load local/{alias}_{system}_{int(t)}ns_{label}_local.pdb, {prefix}_{int(t)}")
    lines += [
        "hide everything",
        "show sticks, all",
        "color gray70, chain B",
        "color marine, Split_PM_* and chain D",
        "color orange, FL_MM_* and chain C",
        "color tv_red, chain B and resi 25",
        "color violet, (Split_PM_* and chain D and resi 5) or (FL_MM_* and chain C and resi 39)",
        "group Split_PM, Split_PM_26 Split_PM_27 Split_PM_65 Split_PM_97",
        "group FL_MM, FL_MM_26 FL_MM_27 FL_MM_65 FL_MM_97",
        "disable all",
        "enable Split_PM_26",
        "enable FL_MM_26",
        "orient",
        "zoom all, 4",
        "set orthoscopic, on",
        "# Structures are aligned to the common Match-Full protein reference frame.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_complex").mkdir(exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    feat = pd.read_csv(FEATURES)
    ref = pa.global_reference()
    rows = []
    for system, alias in ALIASES.items():
        tr, top, time = ag.load_system(system)
        xyz = pa.align_global(tr, top, ref)
        local = local_indices(top, system)
        selected = []
        selected_local = []
        for t in TIMES:
            matches = np.where(np.isclose(time, t))[0]
            if len(matches) != 1:
                raise ValueError(f"{system}: expected exactly one frame at {t} ns, found {len(matches)}")
            fi = int(matches[0])
            label = LABELS[t]
            one = md.Trajectory(xyz[fi : fi + 1] / 10.0, top)
            loc = one.atom_slice(local)
            selected.append(one)
            selected_local.append(loc)
            stem = f"{alias}_{system}_{int(t)}ns_{label}"
            full_rel = f"full_complex/{stem}_full_aligned.pdb"
            local_rel = f"local/{stem}_local.pdb"
            one.save_pdb(str(OUT / full_rel))
            loc.save_pdb(str(OUT / local_rel))
            r = feat[(feat.system == system) & np.isclose(feat.time_ns, t)].iloc[0]
            rows.append(
                {
                    "alias": alias,
                    "system": system,
                    "time_ns": t,
                    "role": label,
                    "eq39_B25_basecentroid_A": r.hybrid_eq39_basecentroid_A,
                    "eq39_B25_min_base_A": r.hybrid_eq39_min_base_A,
                    "eq39_B25_NO_contacts_lt3p5A": int(r.hybrid_eq39_NO_contacts_3p5A),
                    "paired_state": bool(
                        (r.hybrid_eq39_basecentroid_A < 6.5)
                        and (r.hybrid_eq39_NO_contacts_3p5A >= 1)
                    ),
                    "full_pdb": full_rel,
                    "local_pdb": local_rel,
                }
            )
        md.join(selected, check_topology=True, discard_overlapping_frames=False).save_pdb(
            str(OUT / f"{alias}_{system}_four_keyframes_full_aligned.pdb")
        )
        md.join(selected_local, check_topology=True, discard_overlapping_frames=False).save_pdb(
            str(OUT / f"{alias}_{system}_four_keyframes_local.pdb")
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "frame_selection_and_metrics.csv", index=False)
    write_pml(OUT / "compare_split_pm_fl_mm_eq39_B25.pml")
    write_readme(OUT / "README.md")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
