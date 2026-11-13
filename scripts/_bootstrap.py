"""Let scripts run from a fresh clone without `pip install -e .` (adds ./src to sys.path)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
