from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _module_stub import main


if __name__ == "__main__":
    raise SystemExit(main("basepair", "Analyze M17 pairing, hydrogen bonds, and base-opening metrics."))

