"""S-BACKLOG smoke: convert ONE real loop14 branch tape with the fixed converter.

Run on cpuP/cpuQ beside the tapes.  Verifies the exact entry_backlog clocks
reconcile to the tape's authoritative pair-summary ledger (fail closed)."""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wms.data.c4_episodes import convert_episode  # noqa: E402

tape = Path(sys.argv[1])
out = Path(sys.argv[2])
try:
    entry = convert_episode(tape, out)
    keep = ("episode", "dir", "pair_key", "queue_arrays", "branch")
    print(json.dumps({k: entry.get(k) for k in keep if k in entry}, indent=1)[:1800])
except Exception:
    traceback.print_exc()
    sys.exit(1)
