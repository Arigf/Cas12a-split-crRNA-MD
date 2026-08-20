#!/usr/bin/env python3
"""Export full-crRNA target pairing data for selected first-round systems."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_ruvc_entry_grid as ag
import analyze_precatalytic_allostery as pa

OUT = ROOT / "analysis_four_systems_20_100ns/full_crrna_pairing_FLPM_FLMM17_SplitMM17"
SYSTEMS = {
    "FL-PM": "Match-Full",
    "FL-MM17": "MM-Full",
    "Split-MM17": "MM-Split",
}


def guide_location(system: str, eq: int) -> tuple[str, int, str]:
    if "Split" in system and eq >= 35:
        return "D", eq - 34, "back_D"
    return "C", eq, "front_C" if eq <= 34 else "full_C_tail"


def calc_system(alias: str, system: str) -> pd.DataFrame:
    tr, top, time = ag.load_system(system)
    x = tr.xyz * 10.0
    atoms = list(top.atoms)
    rows = []
    for eq in range(1, 45):
        chain, resseq, segment = guide_location(system, eq)
        target = 64 - eq
        target_present = 1 <= target <= 50
        ga = pa.base_atoms(top, chain, resseq)
        da = pa.base_atoms(top, "B", target) if target_present else np.array([], dtype=int)
        can_pair = target_present and len(ga) > 0 and len(da) > 0
        if can_pair:
            gc = x[:, ga].mean(axis=1)
            dc = x[:, da].mean(axis=1)
            basecentroid = np.linalg.norm(gc - dc, axis=1)
            bd = np.linalg.norm(x[:, ga, None, :] - x[:, None, da, :], axis=3)
            min_base = bd.min(axis=(1, 2))
            base_contacts = (bd < 3.5).sum(axis=(1, 2))
            gp = np.array([i for i in ga if atoms[i].element.symbol in {"N", "O"}], dtype=int)
            dp = np.array([i for i in da if atoms[i].element.symbol in {"N", "O"}], dtype=int)
            hd = np.linalg.norm(x[:, gp, None, :] - x[:, None, dp, :], axis=3)
            no_contacts = (hd < 3.5).sum(axis=(1, 2))
            paired = (basecentroid < 6.5) & (no_contacts >= 1)
            paired_binary = paired.astype(float)
        else:
            n = len(time)
            basecentroid = np.full(n, np.nan)
            min_base = np.full(n, np.nan)
            base_contacts = np.full(n, np.nan)
            no_contacts = np.full(n, np.nan)
            paired = np.full(n, np.nan)
            paired_binary = np.full(n, np.nan)

        for i, t in enumerate(time):
            rows.append(
                {
                    "alias": alias,
                    "system": system,
                    "time_ns": float(t),
                    "block_start_ns": int((t // 10) * 10),
                    "block_end_ns": int((t // 10) * 10 + 10),
                    "crRNA_chain": chain,
                    "crRNA_residue": resseq,
                    "crRNA_segment": segment,
                    "guide_equivalent_position": eq,
                    "target_DNA_residue_B": target if target_present else np.nan,
                    "target_present": bool(target_present),
                    "pairing_calculated": bool(can_pair),
                    "basecentroid_A": basecentroid[i],
                    "min_base_A": min_base[i],
                    "base_contacts_3p5A": base_contacts[i],
                    "NO_contacts_3p5A": no_contacts[i],
                    "paired_state": bool(paired[i]) if can_pair else np.nan,
                    "paired_binary": paired_binary[i],
                }
            )
    return pd.DataFrame(rows)


def summaries(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "alias",
        "system",
        "crRNA_chain",
        "crRNA_residue",
        "crRNA_segment",
        "guide_equivalent_position",
        "target_DNA_residue_B",
        "target_present",
        "pairing_calculated",
    ]
    by_position = (
        d.groupby(keys, dropna=False)
        .agg(
            frames=("time_ns", "size"),
            paired_occupancy=("paired_binary", "mean"),
            basecentroid_mean_A=("basecentroid_A", "mean"),
            basecentroid_sd_A=("basecentroid_A", "std"),
            basecentroid_p05_A=("basecentroid_A", lambda z: z.quantile(0.05)),
            basecentroid_p50_A=("basecentroid_A", "median"),
            basecentroid_p95_A=("basecentroid_A", lambda z: z.quantile(0.95)),
            min_base_mean_A=("min_base_A", "mean"),
            NO_contacts_mean=("NO_contacts_3p5A", "mean"),
        )
        .reset_index()
    )

    calc = d[d["pairing_calculated"]].copy()
    per_ns = []
    for (alias, system, t), g in calc.groupby(["alias", "system", "time_ns"]):
        front = g[g["guide_equivalent_position"] <= 34]
        back = g[g["guide_equivalent_position"] >= 35]
        per_ns.append(
            {
                "alias": alias,
                "system": system,
                "time_ns": t,
                "n_positions_calculated": len(g),
                "paired_positions": int(g["paired_binary"].sum()),
                "pairing_ratio_all_calculated": float(g["paired_binary"].mean()),
                "mean_basecentroid_A_all_calculated": float(g["basecentroid_A"].mean()),
                "front_eq14_eq34_n": len(front),
                "front_eq14_eq34_pairing_ratio": float(front["paired_binary"].mean()) if len(front) else np.nan,
                "front_eq14_eq34_mean_basecentroid_A": float(front["basecentroid_A"].mean()) if len(front) else np.nan,
                "back_eq35_eq44_n": len(back),
                "back_eq35_eq44_pairing_ratio": float(back["paired_binary"].mean()) if len(back) else np.nan,
                "back_eq35_eq44_mean_basecentroid_A": float(back["basecentroid_A"].mean()) if len(back) else np.nan,
            }
        )
    per_ns = pd.DataFrame(per_ns)
    return by_position, per_ns


def plot_outputs(d: pd.DataFrame, by_position: pd.DataFrame, per_ns: pd.DataFrame) -> None:
    colors = {"FL-PM": "#1f77b4", "FL-MM17": "#ff7f0e", "Split-MM17": "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for alias in SYSTEMS:
        x = per_ns[per_ns["alias"] == alias]
        ax.plot(x["time_ns"], x["pairing_ratio_all_calculated"], label=f"{alias} all eq14-eq44", color=colors[alias], lw=2)
        ax.plot(x["time_ns"], x["back_eq35_eq44_pairing_ratio"], color=colors[alias], lw=1, ls="--", alpha=0.8, label=f"{alias} eq35-eq44")
    ax.set(xlabel="Time (ns)", ylabel="Pairing ratio", ylim=(-0.04, 1.04), title="Full crRNA-target pairing ratio per ns")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "FLPM_FLMM17_SplitMM17_pairing_ratio_per_ns.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, alias in zip(axes, SYSTEMS):
        sub = d[(d["alias"] == alias) & d["pairing_calculated"]]
        mat = sub.pivot(index="guide_equivalent_position", columns="time_ns", values="basecentroid_A").sort_index()
        im = ax.imshow(mat.values, aspect="auto", origin="lower", cmap="viridis", vmin=4.5, vmax=16.0)
        ax.set_title(alias)
        ax.set_xlabel("Frame index (20-99 ns)")
        ax.set_yticks(range(0, len(mat.index), 5), mat.index[::5])
        ax.axhline(34 - mat.index.min() + 0.5, color="white", lw=1, ls="--")
    axes[0].set_ylabel("Guide equivalent position")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Base-centroid distance (A)")
    fig.suptitle("crRNA-target base-centroid distance heatmaps")
    fig.savefig(OUT / "FLPM_FLMM17_SplitMM17_basecentroid_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    calc = by_position[by_position["pairing_calculated"]].copy()
    for alias in SYSTEMS:
        x = calc[calc["alias"] == alias]
        ax.plot(x["guide_equivalent_position"], x["paired_occupancy"], marker="o", label=alias, color=colors[alias])
    ax.axvline(34.5, color="k", lw=1, ls="--", alpha=0.6)
    ax.set(xlabel="Guide equivalent position", ylabel="Paired occupancy", ylim=(-0.04, 1.04), title="Per-position paired occupancy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "FLPM_FLMM17_SplitMM17_per_position_paired_occupancy.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [calc_system(alias, system) for alias, system in SYSTEMS.items()]
    d = pd.concat(frames, ignore_index=True)
    by_position, per_ns = summaries(d)

    block_keys = [
        "alias",
        "system",
        "crRNA_chain",
        "crRNA_residue",
        "crRNA_segment",
        "guide_equivalent_position",
        "target_DNA_residue_B",
        "target_present",
        "pairing_calculated",
        "block_start_ns",
        "block_end_ns",
    ]
    blocks = (
        d.groupby(block_keys, dropna=False)
        .agg(
            frames_10ns_block=("time_ns", "size"),
            paired_occupancy_10ns_block=("paired_binary", "mean"),
            basecentroid_mean_A_10ns_block=("basecentroid_A", "mean"),
            NO_contacts_mean_10ns_block=("NO_contacts_3p5A", "mean"),
        )
        .reset_index()
    )

    merge_keys = block_keys[:-2]
    all_in_one = d.merge(by_position, on=merge_keys, how="left").merge(blocks, on=block_keys, how="left")

    d.to_csv(OUT / "FLPM_FLMM17_SplitMM17_full_crRNA_pairing_timeseries_eq1_eq44.csv", index=False)
    by_position.to_csv(OUT / "FLPM_FLMM17_SplitMM17_full_crRNA_pairing_summary_by_position.csv", index=False)
    per_ns.to_csv(OUT / "FLPM_FLMM17_SplitMM17_pairing_ratio_per_ns.csv", index=False)
    blocks.to_csv(OUT / "FLPM_FLMM17_SplitMM17_full_crRNA_pairing_10ns_blocks.csv", index=False)
    all_in_one.to_csv(OUT / "FLPM_FLMM17_SplitMM17_full_crRNA_pairing_ALL_IN_ONE.csv", index=False)

    readme = """# Full crRNA pairing data for FL-PM, FL-MM17, and Split-MM17

System mapping:

- FL-PM = Match-Full
- FL-MM17 = MM-Full
- Split-MM17 = MM-Split

The exported crRNA positions are guide-equivalent positions eq1-eq44. Target-hybrid
distances are calculated with the project mapping:

    target DNA chain B residue = 64 - guide equivalent position

Under this mapping eq14-eq44 map to B50-B20. Positions eq1-eq13 have no target
residue in chain B and are retained with target_present=False and blank distance
metrics.

Pairing proxy:

    paired_state = basecentroid_A < 6.5 and NO_contacts_3p5A >= 1

basecentroid_A is the base-heavy-atom centroid distance. NO_contacts_3p5A is an
N/O atom contact count within 3.5 A and is not a strict angle-based hydrogen-bond
definition.
"""
    (OUT / "README.md").write_text(readme)
    plot_outputs(d, by_position, per_ns)
    print(OUT)
    print(per_ns.groupby("alias")["pairing_ratio_all_calculated"].describe().to_string())
    print(by_position[by_position["guide_equivalent_position"].isin([33, 34, 35, 39, 44])][
        ["alias", "system", "guide_equivalent_position", "paired_occupancy", "basecentroid_mean_A", "NO_contacts_mean"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
