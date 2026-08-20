#!/usr/bin/env python3
"""Four-system unified-grid RuvC entry volume and bottleneck analysis."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_propagation
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "production_md_2p5mM_MgCl2_37p5mM_KCl"
EQ = ROOT / "equilibration_smoke_2p5mM_MgCl2_37p5mM_KCl"
REFDIR = ROOT / "analysis_four_systems_20_100ns/ruvc_reference_definition"
OUT = ROOT / "analysis_four_systems_20_100ns/ruvc_entry_grid_analysis"
SYSTEMS = ["Match-Full", "Match-Split", "MM-Full", "MM-Split"]
RUVC_RANGES = [(809, 872), (891, 997), (1180, 1226)]
PROBES = (1.4, 2.5)
SPACING = 1.0
TUBE_RADIUS = 6.0
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
       "P": 1.80, "S": 1.80, "CL": 1.75, "MG": 1.73}


def repair(top):
    for chain in top.chains:
        rs = list(chain.residues)
        if rs and rs[0].is_nucleic:
            for left, right in zip(rs[:-1], rs[1:]):
                a = next((x for x in left.atoms if x.name in {"O3'", "O3*"}), None)
                b = next((x for x in right.atoms if x.name == "P"), None)
                if a and b:
                    top.add_bond(a, b)


def load_system(system):
    base = PROD / system
    if system == "MM-Split":
        prefixes = [base / f"prod_{i:04d}" for i in range(1, 6)] + [base / "prod_0006_95ns"]
    else:
        prefixes = [base / "prod_0001_100ns"]
    idx = np.loadtxt(str(prefixes[0]) + "_solute_atom_indices.txt", dtype=int)
    full = md.load_pdb(str(EQ / system / "06_unrestrained_stability_npt.pdb"))
    top = full.atom_slice(idx).topology
    del full
    repair(top)
    chunks = [md.load(str(p) + "_solute.dcd", top=top) for p in prefixes]
    if system == "MM-Split":
        chunks = [x[1::2] for x in chunks[:5]] + [chunks[5]]
    tr = md.join(chunks, check_topology=True, discard_overlapping_frames=False)
    assert tr.n_frames == 1000
    molecules = list(top.find_molecules())
    anchor = max(molecules, key=len)
    tr = tr.image_molecules(anchor_molecules=[anchor],
                            other_molecules=[m for m in molecules if m is not anchor],
                            make_whole=True)
    # 20-100 ns, sampled every 1 ns from the common 0.1-ns trajectory cadence.
    return tr[200::10], top, np.arange(20.0, 100.0, 1.0)


def atom_map(top, predicate):
    return np.array([a.index for a in top.atoms if predicate(a)], dtype=int)


def is_ruvc_ca(a):
    return (a.residue.chain.chain_id == "A" and a.name == "CA" and
            any(lo <= a.residue.resSeq <= hi for lo, hi in RUVC_RANGES))


def reference_ca():
    t = md.load_pdb(str(EQ / "Match-Full" / "06_unrestrained_stability_npt.pdb"))
    d = {(a.residue.resSeq, a.name): t.xyz[0, a.index] * 10.0 for a in t.topology.atoms
         if is_ruvc_ca(a)}
    return d


def kabsch(source, target):
    sc, tc = source.mean(0), target.mean(0)
    u, _s, vt = np.linalg.svd((source - sc).T @ (target - tc))
    corr = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ corr @ vt
    return rot, tc - sc @ rot


def align_frames(tr, top, ref_ca):
    ids, target = [], []
    for a in top.atoms:
        key = (a.residue.resSeq, a.name)
        if is_ruvc_ca(a) and key in ref_ca:
            ids.append(a.index); target.append(ref_ca[key])
    ids = np.asarray(ids, int); target = np.asarray(target)
    xyz = tr.xyz * 10.0
    rmsd = []
    for f in range(tr.n_frames):
        rot, trans = kabsch(xyz[f, ids], target)
        xyz[f] = xyz[f] @ rot + trans
        rmsd.append(np.sqrt(np.mean(np.sum((xyz[f, ids] - target) ** 2, axis=1))))
    return xyz, np.asarray(rmsd)


def grid_definition(paths):
    allp = np.vstack(list(paths.values()))
    origin = np.floor(allp.min(0) - 7.0)
    upper = np.ceil(allp.max(0) + 7.0)
    shape = tuple((np.rint((upper - origin) / SPACING).astype(int) + 1).tolist())
    axes = [origin[i] + np.arange(shape[i]) * SPACING for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([x.ravel() for x in mesh])
    tube = {}
    for name, path in paths.items():
        tube[name] = (cKDTree(path).query(points, k=1)[0] <= TUBE_RADIUS).reshape(shape)
    return origin, shape, points, tube


def element(a):
    if a.element is not None:
        return a.element.symbol.upper()
    return a.name.strip("0123456789").upper()[:1]


def wall_indices(top, mode):
    keep = []
    for a in top.atoms:
        ch = a.residue.chain.chain_id
        el = element(a)
        if el == "H" or el == "MG":
            continue
        if mode == "protein" and ch != "A":
            continue
        if mode == "complex" and ch not in {"A", "B", "C", "D"}:
            continue
        keep.append(a.index)
    return np.asarray(keep, int)


def rasterize(coords, radii, origin, shape, probe):
    occ = np.zeros(shape, dtype=bool)
    maxrad = max(VDW.values()) + probe
    upper = origin + (np.asarray(shape) - 1) * SPACING
    near = np.all((coords >= origin - maxrad) & (coords <= upper + maxrad), axis=1)
    for xyz, rad in zip(coords[near], radii[near] + probe):
        lo = np.maximum(0, np.floor((xyz - rad - origin) / SPACING).astype(int))
        hi = np.minimum(np.asarray(shape) - 1, np.ceil((xyz + rad - origin) / SPACING).astype(int))
        xs = origin[0] + np.arange(lo[0], hi[0]+1) * SPACING
        ys = origin[1] + np.arange(lo[1], hi[1]+1) * SPACING
        zs = origin[2] + np.arange(lo[2], hi[2]+1) * SPACING
        inside = ((xs[:,None,None]-xyz[0])**2 + (ys[None,:,None]-xyz[1])**2 +
                  (zs[None,None,:]-xyz[2])**2) <= rad**2
        occ[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] |= inside
    return occ


def exterior_free(occ):
    free = ~occ
    seed = np.zeros_like(free)
    seed[0,:,:] = free[0,:,:]; seed[-1,:,:] = free[-1,:,:]
    seed[:,0,:] |= free[:,0,:]; seed[:,-1,:] |= free[:,-1,:]
    seed[:,:,0] |= free[:,:,0]; seed[:,:,-1] |= free[:,:,-1]
    return binary_propagation(seed, mask=free)


def path_clearance(path, coords, radii):
    tree = cKDTree(coords)
    k = min(24, len(coords))
    d, ix = tree.query(path, k=k)
    if k == 1:
        d, ix = d[:,None], ix[:,None]
    surf = d - radii[ix]
    owner = np.argmin(surf, axis=1)
    return surf[np.arange(len(path)), owner], ix[np.arange(len(path)), owner]


def selection_rmsd(xyz, top, lo, hi, core=False):
    if core:
        lo, hi = lo+1, hi-1
    ids = atom_map(top, lambda a: a.residue.chain.chain_id == "A" and lo <= a.residue.resSeq <= hi and a.name == "CA")
    ref = xyz[0, ids]
    return np.sqrt(np.mean(np.sum((xyz[:,ids] - ref[None,:,:])**2, axis=2), axis=1))


def summarize(x):
    x = np.asarray(x, float)
    # Eight 10-ns block means provide a conservative descriptive interval.
    blocks = np.asarray([b.mean() for b in np.array_split(x, 8)])
    return {"mean": float(x.mean()), "sd": float(x.std()), "median": float(np.median(x)),
            "p05": float(np.percentile(x,5)), "p95": float(np.percentile(x,95)),
            "block_mean_95pct_interval": [float(np.percentile(blocks,2.5)), float(np.percentile(blocks,97.5))]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "NTS_primary": np.loadtxt(REFDIR / "ruvc_primary_nts_path.csv", delimiter=",", skiprows=1),
        "TS_sensitivity": np.loadtxt(REFDIR / "ruvc_sensitivity_ts_path.csv", delimiter=",", skiprows=1),
    }
    origin, shape, grid_points, tube_masks = grid_definition(paths)
    ref = reference_ca()
    rows = []
    alignment = {}
    for system in SYSTEMS:
        tr, top, times = load_system(system)
        xyz, fit_rmsd = align_frames(tr, top, ref)
        alignment[system] = summarize(fit_rmsd)
        walls = {m: wall_indices(top, m) for m in ("protein", "complex")}
        radii = {m: np.asarray([VDW.get(element(list(top.atoms)[i]), 1.70) for i in ids]) for m,ids in walls.items()}
        lid_buffer = selection_rmsd(xyz, top, 926, 941)
        lid_core = selection_rmsd(xyz, top, 926, 941, core=True)
        bh_buffer = selection_rmsd(xyz, top, 870, 890)
        bh_core = selection_rmsd(xyz, top, 870, 890, core=True)
        for f, time in enumerate(times):
            base = {"system": system, "time_ns": time, "alignment_rmsd_A": fit_rmsd[f],
                    "lid_buffer_rmsd_A": lid_buffer[f], "lid_core_rmsd_A": lid_core[f],
                    "bh_buffer_rmsd_A": bh_buffer[f], "bh_core_rmsd_A": bh_core[f]}
            for mode in ("protein", "complex"):
                c = xyz[f, walls[mode]]; r = radii[mode]
                for pname, path in paths.items():
                    clear, nearest = path_clearance(path, c, r)
                    active = int(np.argmin(np.linalg.norm(path - np.mean(path,axis=0), axis=1)))
                    # D31 is point 57 on the resampled NTS curve; for TS use the closest approach
                    # to the local catalytic-residue centroid calculated from the fixed reference.
                    if pname == "NTS_primary": active = min(57, len(path)-1)
                    base[f"{mode}_{pname}_min_clearance_A"] = float(clear.min())
                    base[f"{mode}_{pname}_arm1_min_clearance_A"] = float(clear[:active+1].min())
                    base[f"{mode}_{pname}_arm2_min_clearance_A"] = float(clear[active:].min())
                    near_atom = list(top.atoms)[walls[mode][nearest[np.argmin(clear)]]]
                    base[f"{mode}_{pname}_bottleneck_owner"] = f"{near_atom.residue.chain.chain_id}:{near_atom.residue.resSeq}:{near_atom.name}"
                for probe in PROBES:
                    occ = rasterize(c, r, origin, shape, probe)
                    accessible = exterior_free(occ)
                    for pname in paths:
                        base[f"{mode}_{pname}_accessible_volume_probe{probe:.1f}_A3"] = float(np.count_nonzero(accessible & tube_masks[pname]) * SPACING**3)
            rows.append(base)
        del tr, xyz
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "timeseries_20_100ns_1ns.csv", index=False)
    numeric = [c for c in df.columns if c not in {"system","time_ns"} and not c.endswith("owner")]
    summary = {s: {c: summarize(df.loc[df.system == s, c]) for c in numeric} for s in SYSTEMS}
    owners = {s: {c: df.loc[df.system == s, c].value_counts().head(10).to_dict()
                  for c in df.columns if c.endswith("owner")} for s in SYSTEMS}
    output = {"method": {
        "window_ns":"20-100", "sampling_ns":1.0, "frames_per_system":80,
        "alignment":"local RuvC CA: A:809-872,891-997,1180-1226",
        "grid_origin_A":origin.tolist(), "grid_shape":shape, "grid_spacing_A":SPACING,
        "tube_radius_A":TUBE_RADIUS, "probe_radii_A":list(PROBES),
        "wall_modes":{"protein":"chain A heavy atoms", "complex":"protein + resident DNA/RNA heavy atoms; Mg excluded"},
        "volume":"exterior-connected free voxels inside fixed 6-A path tube",
        "bottleneck":"minimum path-center to receptor van-der-Waals surface clearance",
        "uncertainty":"descriptive 95% interval across eight contiguous 10-ns block means; one trajectory per condition"
    }, "alignment_fit_rmsd_A":alignment, "summary":summary, "bottleneck_owner_counts":owners}
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")

    colors={"Match-Full":"#1f77b4","Match-Split":"#2ca02c","MM-Full":"#ff7f0e","MM-Split":"#d62728"}
    fig, ax = plt.subplots(2,2,figsize=(13,9))
    metrics=[("protein_NTS_primary_min_clearance_A","NTS minimum clearance (A)"),
             ("protein_NTS_primary_accessible_volume_probe1.4_A3","NTS accessible volume, 1.4-A probe (A3)"),
             ("protein_TS_sensitivity_min_clearance_A","TS-path minimum clearance (A)"),
             ("protein_TS_sensitivity_accessible_volume_probe1.4_A3","TS-path accessible volume, 1.4-A probe (A3)")]
    for a,(metric,label) in zip(ax.flat,metrics):
        vals=[df.loc[df.system==s,metric].values for s in SYSTEMS]
        parts=a.violinplot(vals,showmedians=True,showextrema=False)
        for body,s in zip(parts['bodies'],SYSTEMS): body.set_facecolor(colors[s]);body.set_alpha(.65)
        a.set_xticks(range(1,5),SYSTEMS,rotation=20);a.set_ylabel(label);a.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'four_system_entry_volume_bottleneck.png',dpi=200);plt.close(fig)

    corrrows=[]
    for s in SYSTEMS:
        sub=df[df.system==s]
        for geom in ["lid_buffer_rmsd_A","lid_core_rmsd_A","bh_buffer_rmsd_A","bh_core_rmsd_A"]:
            for gate in ["protein_NTS_primary_min_clearance_A","protein_NTS_primary_accessible_volume_probe1.4_A3"]:
                corrrows.append({"system":s,"geometry":geom,"gate_metric":gate,"pearson_r":float(sub[geom].corr(sub[gate]))})
    pd.DataFrame(corrrows).to_csv(OUT/'lid_bh_gate_correlations.csv',index=False)
    print(json.dumps({"output":str(OUT),"rows":len(df),"grid_shape":shape},indent=2))


if __name__ == "__main__":
    main()
