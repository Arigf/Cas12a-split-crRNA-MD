from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "production_md_2p5mM_MgCl2_37p5mM_KCl"
OUT = ROOT / "analysis_four_systems_20_100ns"
SYSTEMS = ["Match-Full", "MM-Full", "Match-Split", "MM-Split"]


def mean(summary, *path):
    value = summary
    for key in path:
        value = value[key]
    return value["mean"] if isinstance(value, dict) and "mean" in value else value


def interaction(values):
    return values["MM-Split"] - values["Match-Split"] - values["MM-Full"] + values["Match-Full"]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    summaries = {s: json.loads((PROD / s / "analysis_brief" / "summary.json").read_text()) for s in SYSTEMS}
    series = {s: pd.read_csv(PROD / s / "analysis_brief" / "timeseries_20_100ns.csv") for s in SYSTEMS}
    definitions = {
        "protein_rmsd_A": lambda s, d: mean(s, "global", "protein_rmsd_A"),
        "dna_rmsd_A": lambda s, d: mean(s, "global", "dna_rmsd_A"),
        "rna_rmsd_A": lambda s, d: mean(s, "global", "rna_rmsd_A"),
        "M17_min_base_distance_A": lambda s, d: mean(s, "M17", "min_base_distance_A"),
        "M17_polar_contacts": lambda s, d: mean(s, "M17", "polar_contact_count"),
        "M17_local_rmsd_A": lambda s, d: mean(s, "M17", "local_rmsd_A"),
        "RuvCII_global_rmsd_A": lambda s, d: mean(s, "RuvC_domains", "RuvC-II", "global_frame_rmsd_A"),
        "RuvCII_internal_rmsd_A": lambda s, d: mean(s, "RuvC_domains", "RuvC-II", "internal_rmsd_A"),
        "RuvCII_crRNA_contacts": lambda s, d: float(d["RuvCII_crRNA_contacts_lt4p5A"].mean()),
        "RuvCII_to_Mg_centroid_A": lambda s, d: float(d["RuvCII_to_Mg_centroid_A"].mean()),
        "Mg_Mg_A": lambda s, d: mean(s, "catalytic", "Mg_Mg_A"),
        "pocket_rmsd_A": lambda s, d: mean(s, "catalytic", "pocket_rmsd_A"),
        "productive_fraction": lambda s, d: s["catalytic"]["productive_fraction"],
        "temperature_K": lambda s, d: mean(s, "thermodynamics", "temperature_K"),
        "density_g_ml": lambda s, d: mean(s, "thermodynamics", "density_g_ml"),
    }
    rows = []
    metric_values = {}
    for metric, getter in definitions.items():
        values = {system: float(getter(summaries[system], series[system])) for system in SYSTEMS}
        metric_values[metric] = values
        rows.append({"metric": metric, **values, "interaction_effect": interaction(values)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "four_system_summary_metrics.csv", index=False)

    split_rows = []
    for metric in ("split_break_O3_O5_A", "split_local_contacts_lt4p5A", "split_back_fragment_positional_rmsd_A"):
        split_rows.append({
            "metric": metric,
            "Match-Split": summaries["Match-Split"]["split"][metric]["mean"],
            "MM-Split": summaries["MM-Split"]["split"][metric]["mean"],
            "MM_minus_Match": summaries["MM-Split"]["split"][metric]["mean"] - summaries["Match-Split"]["split"][metric]["mean"],
        })
    pd.DataFrame(split_rows).to_csv(OUT / "split_specific_metrics.csv", index=False)

    plot_metrics = ["protein_rmsd_A", "rna_rmsd_A", "M17_min_base_distance_A",
                    "M17_polar_contacts_lt3p5A", "RuvC-II_rmsd_A", "Mg_Mg_A"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, plot_metrics):
        ax.boxplot([series[s][metric].values for s in SYSTEMS], tick_labels=SYSTEMS, showfliers=False)
        ax.set_title(metric); ax.tick_params(axis="x", rotation=35); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "four_system_metric_distributions.png", dpi=200); plt.close(fig)

    time = series["MM-Split"]["time_ns"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for system in SYSTEMS:
        axes[0].plot(time, series[system]["M17_min_base_distance_A"], lw=.7, label=system)
        axes[1].plot(time, series[system]["RuvC-II_rmsd_A"], lw=.7, label=system)
        axes[2].plot(time, series[system]["Mg_Mg_A"], lw=.7, label=system)
    for ax, label in zip(axes, ["M17 minimum base distance (Å)", "RuvC-II RMSD (Å)", "Mg-Mg distance (Å)"]):
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=8); axes[-1].set_xlabel("Production time (ns)")
    fig.tight_layout(); fig.savefig(OUT / "four_system_key_timeseries.png", dpi=200); plt.close(fig)

    v = metric_values
    report = f"""# Four-system Cas12a M17 trajectory comparison

Analyzed window: 20–100 ns; sampling interval: 100 ps; one trajectory per condition.

## Main numerical findings

- M17 minimum base distance (Å): Match-Full {v['M17_min_base_distance_A']['Match-Full']:.3f}, MM-Full {v['M17_min_base_distance_A']['MM-Full']:.3f}, Match-Split {v['M17_min_base_distance_A']['Match-Split']:.3f}, MM-Split {v['M17_min_base_distance_A']['MM-Split']:.3f}. The mismatch-by-split interaction is {interaction(v['M17_min_base_distance_A']):+.3f} Å.
- M17 local RMSD interaction: {interaction(v['M17_local_rmsd_A']):+.3f} Å; RNA RMSD interaction: {interaction(v['rna_rmsd_A']):+.3f} Å.
- M17 polar-contact interaction: {interaction(v['M17_polar_contacts']):+.3f} contacts. The mismatch reduces the simple polar-contact count more in Full than in Split, so this metric alone does not support split-specific amplification.
- Split back-fragment positional RMSD rises from {summaries['Match-Split']['split']['split_back_fragment_positional_rmsd_A']['mean']:.3f} to {summaries['MM-Split']['split']['split_back_fragment_positional_rmsd_A']['mean']:.3f} Å.
- Split-break O3′–O5′ distance falls from {summaries['Match-Split']['split']['split_break_O3_O5_A']['mean']:.3f} to {summaries['MM-Split']['split']['split_break_O3_O5_A']['mean']:.3f} Å, while local split-fragment contacts rise from {summaries['Match-Split']['split']['split_local_contacts_lt4p5A']['mean']:.2f} to {summaries['MM-Split']['split']['split_local_contacts_lt4p5A']['mean']:.2f}.
- RuvC-II RMSD interaction: {interaction(v['RuvCII_global_rmsd_A']):+.3f} Å. MM-Split does not show a larger RuvC-II displacement than Match-Split in this trajectory.
- Catalytic-pocket RMSD rises from {v['pocket_rmsd_A']['Match-Split']:.3f} Å in Match-Split to {v['pocket_rmsd_A']['MM-Split']:.3f} Å in MM-Split (interaction {interaction(v['pocket_rmsd_A']):+.3f} Å), indicating local pocket reorganization despite retained metal geometry.
- RuvC-II–crRNA contacts change only from {v['RuvCII_crRNA_contacts']['Match-Split']:.2f} to {v['RuvCII_crRNA_contacts']['MM-Split']:.2f}; this trajectory does not show strong new RuvC-II attachment.
- Catalytic-like Mg geometry fractions are {', '.join(f'{s} {100*v["productive_fraction"][s]:.2f}%' for s in SYSTEMS)}.

## Interpretation

The results support a modest split-specific increase in M17 separation/local disorder and a strong increase in positional mobility of the split back fragment. The shorter break distance and greater front/back contact count indicate compaction or renewed RNA–RNA contact near the split, consistent with the previously observed lower split-region SASA. They do not by themselves demonstrate attachment to Cas12a or obstruction of the RuvC entry path.

RuvC-II motion is not enhanced in MM-Split, and catalytic Mg geometry remains highly occupied in every condition. The larger local pocket RMSD suggests a nearby structural response, but the nearly unchanged RuvC-II–crRNA contact count does not establish direct RNA occlusion. Thus this 100 ns comparison does not support loss of catalytic Mg organization as the explanation for reduced activity. A steric-entry mechanism remains possible but requires configured RuvC-lid/bridge-helix selections and a common reporter-path pocket grid, which are placeholders in the supplied YAML and cannot be inferred uniquely from these trajectories.

## Statistical limitation

There is only one production trajectory per condition. Frames are autocorrelated and are not independent biological/statistical replicates; interaction values are descriptive effect sizes, not inferential tests. Additional replicas are required for confidence intervals and significance claims.
"""
    (OUT / "comparison_report.md").write_text(report)
    (OUT / "comparison_summary.json").write_text(json.dumps({"metrics": metric_values, "split": split_rows}, indent=2) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
