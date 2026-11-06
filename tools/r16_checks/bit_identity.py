"""R16: flag-off bit identity against the c4/v10r14b tip (7adf5663; 13E + R14 D1/D3 + driver_b).

Reuses the R14 harness (``tools/r14_checks/bit_identity.py``): the base package is exported from the
local Git object database at the R14B tip and every tensor (state dict, forward, step, 6-step rollout;
the four R13E identity cases plus the two 13E-like lane_learnable first-passage cases with ramp cars,
CAVs and an absent slot) of the current tree with ``model.ramp_speed_adapt_learnable`` at its default
(off) is compared byte for byte.  ``explicit_off`` additionally passes the R14 physics defaults.

  python tools/r16_checks/bit_identity.py [--out-dir runs/r16_bit_identity]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

R14B_TIP = "7adf5663"
os.environ["R14_BIT_BASE"] = os.environ.get("R16_BIT_BASE", R14B_TIP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r14_checks"))
import bit_identity as B  # noqa: E402


def run_check(out_dir: str = "runs/r16_bit_identity") -> dict:
    report = B.run_check(out_dir)
    report["study"] = "S-RAMPSPEED R16 flag-off identity vs c4/v10r14b tip"
    return report


if __name__ == "__main__":
    if "--snapshot" in sys.argv:
        raise SystemExit(B.main())
    out = sys.argv[sys.argv.index("--out-dir") + 1] if "--out-dir" in sys.argv else "runs/r16_bit_identity"
    print(json.dumps(run_check(out), indent=2))
