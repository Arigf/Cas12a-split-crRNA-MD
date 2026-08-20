#!/usr/bin/env python3
"""Plot RNA-DNA hybrid base-pair distances for positions eq35-eq44."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis_four_systems_20_100ns"
OUTDIR = ANALYSIS / "basepair_distance_plots"
SYSTEMS = ["Match-Full", "MM-Full", "Match-Split", "MM-Split"]
POSITIONS = list(range(35, 45))
COLORS = {
    "Match-Full": "#2b6cb0",
    "MM-Full": "#c53030",
    "Match-Split": "#2f855a",
    "MM-Split": "#805ad5",
}


def save_all_formats(fig: plt.Figure, stem: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def rolling(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def load_long_table() -> pd.DataFrame:
    src = pd.read_csv(ANALYSIS / "precatalytic_allostery" / "feature_timeseries.csv")
    rows = []
    for pos in POSITIONS:
        for _, row in src.iterrows():
            rows.append(
                {
                    "system": row["system"],
                    "time_ns": row["time_ns"],
                    "position": pos,
                    "basecentroid_A": row[f"hybrid_eq{pos}_basecentroid_A"],
                    "min_base_A": row[f"hybrid_eq{pos}_min_base_A"],
                    "base_contacts_3p5A": row[f"hybrid_eq{pos}_base_contacts_3p5A"],
                    "NO_contacts_3p5A": row[f"hybrid_eq{pos}_NO_contacts_3p5A"],
                }
            )
    long = pd.DataFrame(rows)
    long["system"] = pd.Categorical(long["system"], categories=SYSTEMS, ordered=True)
    long["paired_state"] = (long["basecentroid_A"] < 6.5) & (long["NO_contacts_3p5A"] >= 1)
    return long.sort_values(["system", "position", "time_ns"]).reset_index(drop=True)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long.groupby(["system", "position"], observed=False)
        .agg(
            basecentroid_mean_A=("basecentroid_A", "mean"),
            basecentroid_sd_A=("basecentroid_A", "std"),
            basecentroid_p05_A=("basecentroid_A", lambda x: np.percentile(x, 5)),
            basecentroid_p50_A=("basecentroid_A", "median"),
            basecentroid_p95_A=("basecentroid_A", lambda x: np.percentile(x, 95)),
            min_base_mean_A=("min_base_A", "mean"),
            min_base_p05_A=("min_base_A", lambda x: np.percentile(x, 5)),
            min_base_p95_A=("min_base_A", lambda x: np.percentile(x, 95)),
            NO_contacts_mean=("NO_contacts_3p5A", "mean"),
            paired_occupancy=("paired_state", "mean"),
        )
        .reset_index()
    )
    return summary


def plot_eq38_eq39_timeseries(long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True)
    for ax, pos, title in zip(
        axes,
        [38, 39],
        ["eq38-B26 / M17 site", "eq39-B25 / M17-adjacent site"],
    ):
        for system in SYSTEMS:
            sub = long[(long["system"] == system) & (long["position"] == pos)]
            ax.plot(sub["time_ns"], sub["basecentroid_A"], color=COLORS[system], alpha=0.2, linewidth=0.8)
            ax.plot(
                sub["time_ns"],
                rolling(sub["basecentroid_A"]),
                color=COLORS[system],
                linewidth=2.0,
                label=system,
            )
        ax.axhline(6.5, color="#4a5568", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("Base-centroid distance (A)")
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[-1].set_xlabel("Time (ns)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("RNA-DNA base-pair distance at M17 and adjacent site", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    save_all_formats(fig, "Figure_eq38_eq39_basecentroid_timeseries")
    plt.close(fig)


def plot_position_profiles(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for system in SYSTEMS:
        sub = summary[summary["system"] == system].sort_values("position")
        ax.plot(
            sub["position"],
            sub["basecentroid_mean_A"],
            marker="o",
            color=COLORS[system],
            linewidth=2,
            label=system,
        )
        ax.fill_between(
            sub["position"].to_numpy(dtype=float),
            sub["basecentroid_p05_A"].to_numpy(dtype=float),
            sub["basecentroid_p95_A"].to_numpy(dtype=float),
            color=COLORS[system],
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(6.5, color="#4a5568", linestyle=":", linewidth=1.0)
    ax.axvline(38, color="#718096", linestyle="--", linewidth=0.8)
    ax.axvline(39, color="#718096", linestyle="--", linewidth=0.8)
    ax.text(38, ax.get_ylim()[1], "M17", va="top", ha="center", fontsize=8)
    ax.text(39, ax.get_ylim()[1], "adjacent", va="top", ha="center", fontsize=8)
    ax.set_xlabel("Equivalent guide position")
    ax.set_ylabel("Base-centroid distance (A)")
    ax.set_title("Position-resolved RNA-DNA base-pair distances")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_all_formats(fig, "Figure_basepair_distance_position_profile")
    plt.close(fig)


def plot_heatmaps(summary: pd.DataFrame) -> None:
    pivot_dist = summary.pivot(index="system", columns="position", values="basecentroid_mean_A").loc[SYSTEMS]
    pivot_occ = summary.pivot(index="system", columns="position", values="paired_occupancy").loc[SYSTEMS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    im0 = axes[0].imshow(pivot_dist.to_numpy(), cmap="magma_r", aspect="auto", vmin=5.0, vmax=10.0)
    axes[0].set_title("Mean base-centroid distance (A)")
    im1 = axes[1].imshow(pivot_occ.to_numpy(), cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_title("Paired-state occupancy")

    for ax, pivot in zip(axes, [pivot_dist, pivot_occ]):
        ax.set_xticks(np.arange(len(POSITIONS)))
        ax.set_xticklabels(POSITIONS)
        ax.set_yticks(np.arange(len(SYSTEMS)))
        ax.set_yticklabels(SYSTEMS)
        ax.set_xlabel("Equivalent guide position")
        for i in range(len(SYSTEMS)):
            for j in range(len(POSITIONS)):
                value = pivot.iloc[i, j]
                text = f"{value:.1f}" if ax is axes[0] else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white")

    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.suptitle("RNA-DNA hybrid pairing geometry across positions eq35-eq44", y=1.02, fontsize=13)
    fig.tight_layout()
    save_all_formats(fig, "Figure_basepair_distance_heatmaps")
    plt.close(fig)


def plot_all_position_timeseries(long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 13), sharex=True)
    axes = axes.ravel()
    for ax, pos in zip(axes, POSITIONS):
        for system in SYSTEMS:
            sub = long[(long["system"] == system) & (long["position"] == pos)]
            ax.plot(
                sub["time_ns"],
                rolling(sub["basecentroid_A"]),
                color=COLORS[system],
                linewidth=1.4,
                label=system,
            )
        ax.axhline(6.5, color="#4a5568", linestyle=":", linewidth=0.8)
        ax.set_title(f"eq{pos}")
        ax.set_ylabel("Distance (A)")
        ax.grid(True, alpha=0.22, linewidth=0.5)
    axes[-2].set_xlabel("Time (ns)")
    axes[-1].set_xlabel("Time (ns)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Base-centroid distance time series for RNA-DNA pairs eq35-eq44", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    save_all_formats(fig, "Figure_all_basepair_distance_timeseries")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    eq39 = summary[summary["position"] == 39].copy()
    occ = {row["system"]: row["paired_occupancy"] for _, row in eq39.iterrows()}
    dist = {row["system"]: row["basecentroid_mean_A"] for _, row in eq39.iterrows()}
    report = f"""# RNA-DNA base-pair distance plots

Source: `analysis_four_systems_20_100ns/precatalytic_allostery/feature_timeseries.csv`.

The plotted distance is the RNA-DNA base-centroid distance for equivalent guide positions eq35-eq44. The dotted 6.5 A line marks the distance component of the paired-state criterion; paired state also requires at least one inter-base N/O atom pair within 3.5 A.

## Key eq39-B25 result

| System | Mean eq39-B25 distance (A) | Paired occupancy |
|---|---:|---:|
| Match-Full | {dist['Match-Full']:.2f} | {occ['Match-Full']:.2%} |
| MM-Full | {dist['MM-Full']:.2f} | {occ['MM-Full']:.2%} |
| Match-Split | {dist['Match-Split']:.2f} | {occ['Match-Split']:.2%} |
| MM-Split | {dist['MM-Split']:.2f} | {occ['MM-Split']:.2%} |

MM-Split selectively loses the M17-adjacent eq39-B25 pair, while the M17 site itself is not uniquely disrupted compared with MM-Full.

## Output files

- `Figure_eq38_eq39_basecentroid_timeseries.*`
- `Figure_basepair_distance_position_profile.*`
- `Figure_basepair_distance_heatmaps.*`
- `Figure_all_basepair_distance_timeseries.*`
- `basepair_distances_long.csv`
- `basepair_distance_summary.csv`
"""
    (OUTDIR / "analysis_report.md").write_text(report)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    long = load_long_table()
    summary = summarize(long)
    long.to_csv(OUTDIR / "basepair_distances_long.csv", index=False)
    summary.to_csv(OUTDIR / "basepair_distance_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    plot_eq38_eq39_timeseries(long)
    plot_position_profiles(summary)
    plot_heatmaps(summary)
    plot_all_position_timeseries(long)
    write_report(summary)


if __name__ == "__main__":
    main()
