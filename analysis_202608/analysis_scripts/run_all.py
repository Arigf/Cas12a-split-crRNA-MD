from __future__ import annotations

import argparse
from pathlib import Path


STEP_SCRIPTS = {
    "validate": "validate_topology.py",
    "preprocess": "preprocess.py",
    "sasa": "analyze_sasa.py",
    "basepair": "analyze_basepair.py",
    "stacking": "analyze_stacking.py",
    "fraying": "analyze_fraying.py",
    "bsa": "analyze_bsa.py",
    "contacts": "analyze_contacts.py",
    "ruvc": "analyze_ruvc_distances.py",
    "pocket": "analyze_pocket.py",
    "channel": "analyze_channel.py",
    "density": "analyze_density.py",
    "framewise": "build_framewise_table.py",
    "correlations": "analyze_correlations.py",
    "fes": "analyze_fes.py",
    "time_lag": "analyze_time_lag.py",
    "interaction": "analyze_interaction_effects.py",
    "structures": "extract_representative_structures.py",
    "figures": "make_figures.py",
}


DEFAULT_PRIORITY1 = [
    "validate",
    "preprocess",
    "sasa",
    "basepair",
    "stacking",
    "fraying",
    "bsa",
    "contacts",
    "ruvc",
    "interaction",
    "figures",
]


def parse_steps(raw_steps: str) -> list[str]:
    if raw_steps == "priority1":
        return DEFAULT_PRIORITY1
    if raw_steps == "all":
        return list(STEP_SCRIPTS)
    return [step.strip() for step in raw_steps.split(",") if step.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run planned Cas12a M17 MD analysis steps.")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--steps", default="priority1")
    parser.add_argument("--system", help="Optional system name to pass to module scripts.")
    parser.add_argument("--replicate", help="Optional replicate identifier to pass to module scripts.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file does not exist: {config_path}")

    steps = parse_steps(args.steps)
    unknown = [step for step in steps if step not in STEP_SCRIPTS]
    if unknown:
        parser.error(f"Unknown step(s): {', '.join(unknown)}")

    print(f"Config: {config_path}")
    print("Planned steps:")
    for step in steps:
        script = STEP_SCRIPTS[step]
        command = ["python", f"scripts/{script}", "--config", str(config_path)]
        if args.system:
            command.extend(["--system", args.system])
        if args.replicate:
            command.extend(["--replicate", args.replicate])
        print("  " + " ".join(command))

    if args.dry_run:
        print("Dry run completed.")
        return 0

    print("Scaffold only; module implementations are pending.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

