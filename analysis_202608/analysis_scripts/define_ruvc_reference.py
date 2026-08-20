#!/usr/bin/env python3
"""Define reproducible LbCas12a RuvC lid/BH selections and ssDNA entry paths.

Experimental AsCas12a structures 8SFO (NTS in RuvC) and 8SFP (TS in RuvC)
are mapped into the local LbCas12a frame with the three catalytic residues and
their complete common heavy-atom sets.  Path pseudo-atoms are sampled at 1 A.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_four_systems_20_100ns/ruvc_reference_definition"
REFS = OUT / "reference_sources"
LOCAL = ROOT / "equilibration_smoke_2p5mM_MgCl2_37p5mM_KCl/Match-Full/06_unrestrained_stability_npt.pdb"

CAT_MAP = {908: 832, 993: 925, 1263: 1180}
BH_REF = range(939, 958)       # 8SFO HELIX record; maps to Lb 871-889
LID_REF = range(995, 1010)     # catalytic-E-adjacent lid core + one-residue margins


def atoms(path: Path):
    result = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        try:
            result.append({
                "record": line[:6].strip(), "name": line[12:16].strip(),
                "resname": line[17:20].strip(), "chain": line[21],
                "resid": int(line[22:26]),
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                "element": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
            })
        except ValueError:
            continue
    return result


def kabsch(source, target):
    sc, tc = source.mean(0), target.mean(0)
    u, _s, vt = np.linalg.svd((source - sc).T @ (target - tc))
    corr = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ corr @ vt
    trans = tc - sc @ rot
    fitted = source @ rot + trans
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return rot, trans, rmsd


def catalytic_fit(ref, local):
    ra = {(a["resid"], a["name"]): a["xyz"] for a in ref if a["chain"] == "A"}
    la = {(a["resid"], a["name"]): a["xyz"] for a in local if a["chain"] == "A"}
    src, dst, labels = [], [], []
    for rr, lr in CAT_MAP.items():
        for name in ("N", "CA", "C", "O", "CB", "CG", "CD", "OD1", "OD2", "OE1", "OE2"):
            if (rr, name) in ra and (lr, name) in la:
                src.append(ra[rr, name]); dst.append(la[lr, name]); labels.append(f"A:{rr}:{name}->A:{lr}:{name}")
    rot, trans, rmsd = kabsch(np.array(src), np.array(dst))
    return rot, trans, rmsd, labels


def global_align_map(ref_atoms, local_atoms, ref_lo=880, ref_hi=1025, local_lo=800, local_hi=950):
    aa3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    def residues(data, lo, hi):
        return [(a["resid"], aa3[a["resname"]]) for a in data if a["chain"] == "A" and a["name"] == "CA" and lo <= a["resid"] <= hi and a["resname"] in aa3]
    r, q = residues(ref_atoms, ref_lo, ref_hi), residues(local_atoms, local_lo, local_hi)
    a, b = "".join(x[1] for x in r), "".join(x[1] for x in q)
    n, m = len(a), len(b)
    score = np.empty((n+1, m+1), dtype=np.int16); trace = np.zeros((n+1, m+1), dtype=np.int8)
    score[:, 0] = np.arange(n+1) * -2; score[0, :] = np.arange(m+1) * -2
    trace[1:, 0] = 1; trace[0, 1:] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            v = (score[i-1,j-1] + (2 if a[i-1] == b[j-1] else -1), score[i-1,j]-2, score[i,j-1]-2)
            trace[i,j] = int(np.argmax(v)); score[i,j] = max(v)
    mapping = {}; i, j = n, m
    while i or j:
        t = trace[i,j]
        if i and j and t == 0:
            mapping[r[i-1][0]] = q[j-1][0]; i -= 1; j -= 1
        elif i and (j == 0 or t == 1): i -= 1
        else: j -= 1
    return mapping, int(score[n,m])


def nucleotide_anchors(data, chain, residues):
    by_res = {}
    for a in data:
        if a["chain"] == chain and a["resid"] in residues:
            by_res.setdefault(a["resid"], {})[a["name"]] = a["xyz"]
    points, labels = [], []
    for resid in residues:
        if resid not in by_res: continue
        atom = "P" if "P" in by_res[resid] else "C4'"
        if atom in by_res[resid]: points.append(by_res[resid][atom]); labels.append(f"{chain}:{resid}:{atom}")
    return np.array(points), labels


def resample(points, spacing=1.0):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    wanted = np.arange(0, s[-1] + spacing * 0.5, spacing)
    return np.column_stack([np.interp(wanted, s, points[:, k]) for k in range(3)])


def write_path_pdb(path, paths):
    lines = ["REMARK RuvC ssDNA reference paths mapped to local LbCas12a catalytic frame",
             "REMARK chain N = 8SFO nontarget strand (primary); chain T = 8SFP target strand (sensitivity)"]
    serial = 1
    for chain, coords in paths.items():
        for i, xyz in enumerate(coords, 1):
            lines.append(f"HETATM{serial:5d}  C   PTH {chain}{i:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C")
            serial += 1
    lines += ["END", ""]
    path.write_text("\n".join(lines))


def residue_shell(local, path_points, cutoff=6.0):
    shell = set()
    for a in local:
        if a["chain"] != "A" or a["element"] == "H": continue
        if np.min(np.linalg.norm(path_points - a["xyz"], axis=1)) <= cutoff:
            shell.add(a["resid"])
    return sorted(shell)


def compact_ranges(values):
    if not values: return ""
    runs=[]; start=prev=values[0]
    for x in values[1:]:
        if x == prev + 1: prev=x; continue
        runs.append(str(start) if start == prev else f"{start}-{prev}"); start=prev=x
    runs.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(runs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local = atoms(LOCAL); sfo = atoms(REFS / "8SFO.pdb"); sfp = atoms(REFS / "8SFP.pdb")
    map_res, aln_score = global_align_map(sfo, local)
    rot_o, tr_o, rms_o, labels_o = catalytic_fit(sfo, local)
    rot_p, tr_p, rms_p, labels_p = catalytic_fit(sfp, local)

    # Primary path is the experimentally observed NTS traversal through Mg (D:22-36).
    nts_raw, nts_labels = nucleotide_anchors(sfo, "D", range(22, 37))
    # Orthogonal sensitivity path is the target-strand segment entering RuvC (C:3-10).
    ts_raw, ts_labels = nucleotide_anchors(sfp, "C", range(3, 11))
    nts = resample(nts_raw @ rot_o + tr_o); ts = resample(ts_raw @ rot_p + tr_p)
    write_path_pdb(OUT / "ruvc_ssdna_reference_paths.pdb", {"N": nts, "T": ts})
    np.savetxt(OUT / "ruvc_primary_nts_path.csv", nts, delimiter=",", header="x_A,y_A,z_A", comments="")
    np.savetxt(OUT / "ruvc_sensitivity_ts_path.csv", ts, delimiter=",", header="x_A,y_A,z_A", comments="")

    bh_core = sorted({map_res[x] for x in BH_REF if x in map_res})
    lid_core = sorted({map_res[x] for x in LID_REF if x in map_res})
    # Analysis ranges add one residue at each end to absorb ortholog/helix-boundary uncertainty.
    bh_analysis = list(range(min(bh_core)-1, max(bh_core)+2))
    lid_analysis = list(range(min(lid_core)-1, max(lid_core)+2))
    shell = residue_shell(local, nts, 6.0)

    with (OUT / "residue_mapping.csv").open("w", newline="") as fh:
        w=csv.writer(fh); w.writerow(["feature","8SFO_residue","LbCas12a_residue"])
        for feature, rr in [("bridge_helix", x) for x in BH_REF] + [("ruvc_lid", x) for x in LID_REF]:
            if rr in map_res: w.writerow([feature, rr, map_res[rr]])

    result = {
        "local_reference": str(LOCAL.relative_to(ROOT)),
        "fit": {"8SFO_to_local_rmsd_A": rms_o, "8SFP_to_local_rmsd_A": rms_p,
                "atom_pairs_8SFO": len(labels_o), "atom_pairs_8SFP": len(labels_p),
                "catalytic_mapping": {str(k): v for k,v in CAT_MAP.items()}},
        "definitions": {
            "bridge_helix_core": bh_core, "bridge_helix_analysis": bh_analysis,
            "ruvc_lid_core": lid_core, "ruvc_lid_analysis": lid_analysis,
            "ruvc_entry_shell_6A": shell,
        },
        "paths": {
            "primary": {"source":"8SFO chain D residues 22-36 (NTS in RuvC)", "anchors":nts_labels, "sample_spacing_A":1.0, "points":len(nts)},
            "sensitivity": {"source":"8SFP chain C residues 3-10 (TS in RuvC)", "anchors":ts_labels, "sample_spacing_A":1.0, "points":len(ts)},
        },
        "alignment_score_local_window": aln_score,
        "cpptraj_selections": {
            "bridge_helix": f":{compact_ranges(bh_analysis)}",
            "ruvc_lid": f":{compact_ranges(lid_analysis)}",
            "ruvc_entry_reference": f":{compact_ranges(shell)}",
            "ruvc_catalytic_residues": ":832,925,1180",
        },
    }
    (OUT / "ruvc_reference_definition.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
