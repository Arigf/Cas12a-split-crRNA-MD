from __future__ import annotations

import argparse
from pathlib import Path


def build_parser(module_name: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="config/analysis_config.yaml",
        help="Path to the YAML analysis configuration.",
    )
    parser.add_argument("--system", help="Optional system name such as MM-Split.")
    parser.add_argument("--replicate", help="Optional replicate identifier such as rep1.")
    parser.add_argument("--output", help="Optional output directory for this module.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate command-line arguments without running analysis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite this module's generated outputs when implemented.",
    )
    parser.set_defaults(module_name=module_name)
    return parser


def main(module_name: str, description: str) -> int:
    parser = build_parser(module_name, description)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file does not exist: {config_path}")

    output_text = args.output or "(module default output path)"
    print(f"[{module_name}] config: {config_path}")
    print(f"[{module_name}] system: {args.system or 'all systems'}")
    print(f"[{module_name}] replicate: {args.replicate or 'all replicates'}")
    print(f"[{module_name}] output: {output_text}")

    if args.dry_run:
        print(f"[{module_name}] dry run completed.")
        return 0

    print(f"[{module_name}] scaffold only; implementation is pending.")
    return 2

