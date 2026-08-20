#!/usr/bin/env python3
"""Plot focused RuvC reorganization metrics from trajectory-derived tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis_four_systems_20_100ns"
OUTDIR = ANALYSIS / "ruvc_reorganization"

SYSTEMS = ["Match-Full", "MM-Full", "Match-Split", "MM-Split"]
COLORS = {
    "Match-Full": "#2b6cb0",
    "MM-Full": "#c53030",
    "Match-Split": "#2f855a",
    "MM-Split": "#805ad5",
}


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature = pd.read_csv(ANALYSIS / "precatalytic_allostery" / "feature_timeseries.csv")
    grid = pd.read_csv(ANALYSIS / "ruvc_entry_grid_analysis" / "timeseries_20_100ns_1ns.csv")

    pocket_parts = []
    for system in SYSTEMS:
        path = (
            ROOT
            / "production_md_2p5mM_MgCl2_37p5mM_KCl"
            / system
            / "analysis_brief"
            / "timeseries_20_100ns.csv"
        )
        df = pd.read_csv(path)
        df.insert(0, "system", system)
        pocket_parts.append(df[["system", "time_ns", "pocket_rmsd_A", "RuvC-overall_rmsd_A"]])
    pocket = pd.concat(pocket_parts, ignore_index=True)
    return feature, grid, pocket


def build_focused_table(feature: pd.DataFrame, grid: pd.DataFrame, pocket: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "system",
        "time_ns",
        "hybrid_eq39_basecentroid_A",
        "hybrid_eq39_NO_contacts_3p5A",
        "RuvCI_RuvCIII_COM_A",
        "RuvCII_RuvCIII_COM_A",
        "lid_RuvCIII_COM_A",
        "A999_A1084_COM_A",
        "A999_A1138_COM_A",
        "A1084_A1138_COM_A",
        "A1084_A1142_COM_A",
        "A1138_A1210_COM_A",
        "A1142_A1210_COM_A",
        "Mg_Mg_A",
    ]
    grid_cols = [
        "system",
        "time_ns",
        "protein_NTS_primary_min_clearance_A",
        "protein_NTS_primary_accessible_volume_probe1.4_A3",
        "protein_NTS_primary_bottleneck_owner",
        "protein_TS_sensitivity_min_clearance_A",
        "protein_TS_sensitivity_accessible_volume_probe1.4_A3",
    ]
    merged = feature[feature_cols].merge(grid[grid_cols], on=["system", "time_ns"], how="inner")
    merged = merged.merge(pocket, on=["system", "time_ns"], how="inner")
    merged["system"] = pd.Categorical(merged["system"], categories=SYSTEMS, ordered=True)
    return merged.sort_values(["system", "time_ns"]).reset_index(drop=True)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "RuvCI_RuvCIII_COM_A",
        "RuvCII_RuvCIII_COM_A",
        "lid_RuvCIII_COM_A",
        "pocket_rmsd_A",
        "RuvC-overall_rmsd_A",
        "protein_NTS_primary_min_clearance_A",
        "protein_NTS_primary_accessible_volume_probe1.4_A3",
        "A999_A1084_COM_A",
        "A1084_A1142_COM_A",
        "A1138_A1210_COM_A",
        "hybrid_eq39_basecentroid_A",
        "hybrid_eq39_NO_contacts_3p5A",
        "Mg_Mg_A",
    ]
    rows = []
    for metric in metrics:
        for system in SYSTEMS:
            vals = df.loc[df["system"] == system, metric].dropna().to_numpy()
            rows.append(
                {
                    "metric": metric,
                    "system": system,
                    "mean": vals.mean(),
                    "sd": vals.std(ddof=1),
                    "p05": np.percentile(vals, 5),
                    "p50": np.percentile(vals, 50),
                    "p95": np.percentile(vals, 95),
                }
            )
        active_vals = df.loc[df["system"].isin(["Match-Full", "MM-Full", "Match-Split"]), metric]
        mm_split_vals = df.loc[df["system"] == "MM-Split", metric]
        rows.append(
            {
                "metric": metric,
                "system": "MM-Split_minus_active_pool",
                "mean": mm_split_vals.mean() - active_vals.mean(),
                "sd": np.nan,
                "p05": np.nan,
                "p50": np.nan,
                "p95": np.nan,
            }
        )
    return pd.DataFrame(rows)


def rolling(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def save_all_formats(fig: plt.Figure, stem: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def plot_reorganization_timeseries(df: pd.DataFrame) -> None:
    panels = [
        ("RuvCI_RuvCIII_COM_A", "RuvC-I to RuvC-III COM (A)"),
        ("RuvCII_RuvCIII_COM_A", "RuvC-II to RuvC-III COM (A)"),
        ("lid_RuvCIII_COM_A", "Lid to RuvC-III COM (A)"),
        ("pocket_rmsd_A", "RuvC pocket RMSD (A)"),
        ("protein_NTS_primary_min_clearance_A", "NTS path min clearance (A)"),
        ("protein_NTS_primary_accessible_volume_probe1.4_A3", "NTS accessible volume, 1.4 A probe (A3)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for ax, (metric, ylabel) in zip(axes, panels):
        for system in SYSTEMS:
            sub = df[df["system"] == system]
            ax.plot(
                sub["time_ns"],
                sub[metric],
                color=COLORS[system],
                alpha=0.22,
                linewidth=0.8,
            )
            ax.plot(
                sub["time_ns"],
                rolling(sub[metric]),
                color=COLORS[system],
                linewidth=1.8,
                label=system,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        if "clearance" in metric:
            ax.axhline(0, color="#4a5568", linewidth=0.8, linestyle=":")

    axes[-2].set_xlabel("Time (ns)")
    axes[-1].set_xlabel("Time (ns)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("RuvC geometry and entry-path reorganization", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    save_all_formats(fig, "Figure_RuvC_reorganization_timeseries")
    plt.close(fig)


def plot_block_distributions(df: pd.DataFrame) -> None:
    metrics = [
        ("RuvCI_RuvCIII_COM_A", "RuvC-I to III"),
        ("RuvCII_RuvCIII_COM_A", "RuvC-II to III"),
        ("lid_RuvCIII_COM_A", "Lid to III"),
        ("pocket_rmsd_A", "Pocket RMSD"),
    ]
    blocks = df.copy()
    blocks["block"] = ((blocks["time_ns"] - 20) // 10).astype(int)
    block_means = (
        blocks.groupby(["system", "block"], observed=False)[[m for m, _ in metrics]]
        .mean()
        .reset_index()
    )
    block_means.to_csv(OUTDIR / "ruvc_reorganization_10ns_block_means.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharex=False)
    x = np.arange(len(SYSTEMS))
    for ax, (metric, title) in zip(axes, metrics):
        for i, system in enumerate(SYSTEMS):
            vals = block_means.loc[block_means["system"] == system, metric].dropna().to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) else []
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                color=COLORS[system],
                s=22,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.hlines(vals.mean(), i - 0.28, i + 0.28, colors=COLORS[system], linewidth=2.0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(SYSTEMS, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("10 ns block mean (A)")
    fig.suptitle("RuvC reorganization block means", y=1.02, fontsize=14)
    fig.tight_layout()
    save_all_formats(fig, "Figure_RuvC_reorganization_block_means")
    plt.close(fig)


def plot_bottleneck_owners(df: pd.DataFrame) -> None:
    owner = df[["system", "protein_NTS_primary_bottleneck_owner"]].copy()
    owner["owner_residue"] = owner["protein_NTS_primary_bottleneck_owner"].str.extract(r"^(A:\d+)")
    freq = (
        owner.groupby(["system", "owner_residue"], observed=False)
        .size()
        .reset_index(name="frames")
    )
    freq["fraction"] = freq["frames"] / freq.groupby("system", observed=False)["frames"].transform("sum")
    freq.to_csv(OUTDIR / "ruvc_nts_bottleneck_owner_frequencies.csv", index=False)

    key_residues = ["A:999", "A:1084", "A:1138", "A:1142", "A:1210"]
    rows = []
    for system in SYSTEMS:
        sub = freq[freq["system"] == system]
        used = 0.0
        row = {"system": system}
        for residue in key_residues:
            val = sub.loc[sub["owner_residue"] == residue, "fraction"].sum()
            row[residue] = val
            used += val
        row["Other"] = max(0.0, 1.0 - used)
        rows.append(row)
    stacked = pd.DataFrame(rows).set_index("system")
    stacked.to_csv(OUTDIR / "ruvc_nts_bottleneck_owner_stacked.csv")

    colors = ["#805ad5", "#dd6b20", "#2b6cb0", "#319795", "#d69e2e", "#a0aec0"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(stacked))
    x = np.arange(len(stacked))
    for color, col in zip(colors, stacked.columns):
        vals = stacked[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=col, color=color, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(stacked.index, rotation=30, ha="right")
    ax.set_ylabel("Fraction of frames")
    ax.set_ylim(0, 1.0)
    ax.set_title("NTS path bottleneck-owner residue frequencies")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_all_formats(fig, "Figure_RuvC_NTS_bottleneck_owner_frequencies")
    plt.close(fig)


def plot_entry_residue_network(df: pd.DataFrame) -> None:
    metrics = [
        "A999_A1084_COM_A",
        "A999_A1138_COM_A",
        "A1084_A1138_COM_A",
        "A1084_A1142_COM_A",
        "A1138_A1210_COM_A",
        "A1142_A1210_COM_A",
    ]
    mean_table = df.groupby("system", observed=False)[metrics].mean().loc[SYSTEMS].T
    active_mean = mean_table[["Match-Full", "MM-Full", "Match-Split"]].mean(axis=1)
    delta = mean_table.sub(active_mean, axis=0)
    delta.to_csv(OUTDIR / "ruvc_entry_residue_distance_delta_vs_active_pool.csv")

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    vmax = np.nanmax(np.abs(delta.to_numpy()))
    image = ax.imshow(delta.to_numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(SYSTEMS)))
    ax.set_xticklabels(SYSTEMS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([m.replace("_COM_A", "").replace("_", " ") for m in metrics])
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{delta.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Entry-wall residue COM-distance shift vs active-pool mean")
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Delta distance (A)")
    fig.tight_layout()
    save_all_formats(fig, "Figure_RuvC_entry_residue_network_delta")
    plt.close(fig)


def plot_eq39_ruvc_coupling(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    y_metrics = [
        ("RuvCI_RuvCIII_COM_A", "RuvC-I to III (A)"),
        ("pocket_rmsd_A", "Pocket RMSD (A)"),
        ("protein_NTS_primary_min_clearance_A", "NTS min clearance (A)"),
    ]
    for ax, (metric, ylabel) in zip(axes, y_metrics):
        for system in SYSTEMS:
            sub = df[df["system"] == system]
            ax.scatter(
                sub["hybrid_eq39_basecentroid_A"],
                sub[metric],
                color=COLORS[system],
                s=18,
                alpha=0.62,
                label=system,
            )
        ax.set_xlabel("guide39-B25 base-centroid distance (A)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Exploratory coupling between eq39 dissociation and RuvC metrics", y=1.02, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_all_formats(fig, "Figure_eq39_RuvC_coupling_scatter")
    plt.close(fig)

    corr_rows = []
    for system in SYSTEMS:
        sub = df[df["system"] == system]
        for metric, _ in y_metrics:
            corr_rows.append(
                {
                    "system": system,
                    "x_metric": "hybrid_eq39_basecentroid_A",
                    "y_metric": metric,
                    "pearson_r": sub["hybrid_eq39_basecentroid_A"].corr(sub[metric]),
                }
            )
    pd.DataFrame(corr_rows).to_csv(OUTDIR / "eq39_ruvc_metric_correlations.csv", index=False)


def write_report(summary: pd.DataFrame) -> None:
    def mean(metric: str, system: str) -> float:
        return float(summary[(summary["metric"] == metric) & (summary["system"] == system)]["mean"].iloc[0])

    report = f"""# RuvC reorganization focused figure set

Input tables were derived from the production trajectories:

- `precatalytic_allostery/feature_timeseries.csv`
- `ruvc_entry_grid_analysis/timeseries_20_100ns_1ns.csv`
- per-system `production_md_*/<System>/analysis_brief/timeseries_20_100ns.csv`

## Main observations

- MM-Split has the largest RuvC-I to RuvC-III separation: {mean('RuvCI_RuvCIII_COM_A', 'MM-Split'):.2f} A.
- MM-Split has the largest RuvC-II to RuvC-III separation: {mean('RuvCII_RuvCIII_COM_A', 'MM-Split'):.2f} A.
- MM-Split has the largest lid to RuvC-III separation: {mean('lid_RuvCIII_COM_A', 'MM-Split'):.2f} A.
- MM-Split has the largest RuvC pocket RMSD: {mean('pocket_rmsd_A', 'MM-Split'):.2f} A.
- The NTS bottleneck-owner pattern is redistributed in MM-Split across A:1142, A:1084, A:999, and A:1210, rather than being dominated by A:1084 alone.

These plots support a conservative statement: MM-Split shows RuvC pocket and entry-wall reorganization accompanying guide39-B25 dissociation. They do not prove that the dissociation directly causes the RuvC shift, and they do not reduce the mechanism to lid opening or closing.

## Output files

- `Figure_RuvC_reorganization_timeseries.*`
- `Figure_RuvC_reorganization_block_means.*`
- `Figure_RuvC_NTS_bottleneck_owner_frequencies.*`
- `Figure_RuvC_entry_residue_network_delta.*`
- `Figure_eq39_RuvC_coupling_scatter.*`
- `ruvc_reorganization_focused_timeseries.csv`
- `ruvc_reorganization_metric_summary.csv`
- `ruvc_reorganization_10ns_block_means.csv`
- `ruvc_nts_bottleneck_owner_frequencies.csv`
- `ruvc_entry_residue_distance_delta_vs_active_pool.csv`
- `eq39_ruvc_metric_correlations.csv`
"""
    (OUTDIR / "analysis_report.md").write_text(report)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    feature, grid, pocket = load_tables()
    focused = build_focused_table(feature, grid, pocket)
    focused.to_csv(OUTDIR / "ruvc_reorganization_focused_timeseries.csv", index=False)
    summary = summarize_metrics(focused)
    summary.to_csv(OUTDIR / "ruvc_reorganization_metric_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    plot_reorganization_timeseries(focused)
    plot_block_distributions(focused)
    plot_bottleneck_owners(focused)
    plot_entry_residue_network(focused)
    plot_eq39_ruvc_coupling(focused)
    write_report(summary)


if __name__ == "__main__":
    main()
