from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _module_stub import main


if __name__ == "__main__":
    raise SystemExit(main("fraying", "Analyze M15-M20 distal pairing and fraying metrics."))

