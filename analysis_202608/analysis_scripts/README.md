# Script Map

The script names follow the analysis plan. They currently provide stable command-line entry points and will be filled module by module.

Core workflow:

- `validate_topology.py`
- `preprocess.py`
- `analyze_sasa.py`
- `analyze_basepair.py`
- `analyze_stacking.py`
- `analyze_fraying.py`
- `analyze_bsa.py`
- `analyze_contacts.py`
- `analyze_ruvc_distances.py`
- `analyze_pocket.py`
- `analyze_channel.py`
- `analyze_density.py`
- `build_framewise_table.py`
- `analyze_correlations.py`
- `analyze_fes.py`
- `analyze_time_lag.py`
- `analyze_interaction_effects.py`
- `extract_representative_structures.py`
- `make_figures.py`
- `run_all.py`

All scripts should accept `--config`, and system-specific scripts should accept `--system`, `--replicate`, and `--output`.

