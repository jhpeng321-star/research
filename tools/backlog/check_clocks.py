"""S-BACKLOG evidence: every re-exported loop pair carries exact entry_backlog clocks that
reconcile (long_credit rule: np.isclose rtol 1e-6 atol 1e-3) to the authoritative returns.

This is the exact precondition ``wms.control.long_credit.fit_from_loop`` enforces (it raised
``MissingCostClocks`` on the original datasets).  No WM, no policy, no rollout.

Usage: python check_clocks.py --home ~/backlog [--datasets pairs_c00.a1.backlog ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(Path.home() / "backlog"))
    a = ap.parse_args()
    os.environ["BACKLOG_HOME"] = a.home
    os.environ.setdefault("C4_ALLOWED_ROOTS", a.home)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cqr_rows as C
    from wms.control.real_returns14 import NPZRealReturnProvider
    report = {}
    for cyc in C.LOOP:
        data = Path(a.home) / f"reexport/pairs_{cyc}.a1.backlog"
        pairing = Path(a.home) / f"inputs/returns/pairing_{cyc}.a1.json"
        proof = json.loads(pairing.read_text())
        p = NPZRealReturnProvider(str(Path(a.home) / f"inputs/returns/returns_{cyc}.a1.npz"))
        b = NPZRealReturnProvider(str(Path(a.home) / f"inputs/returns/native_returns_{cyc}.a1.npz"))
        pairs, _ = C.make_source(str(data))
        loader = pairs.dataset._loader
        by_key = {g["pair_key"]: g for g in pairs.groups}
        ok, waiting = 0, []
        for i, item in enumerate(proof["pairs"]):
            group = by_key[item["pair_key"]]
            native_row = group["rows"][0]
            policy_row = next(r for r in group["rows"][1:] if loader.entries[r]["branch"]["kind"] == "POLICY")
            cp = C.loop_curves(pairs, policy_row, float(p._returns[i]))      # raises on any mismatch
            cn = C.loop_curves(pairs, native_row, float(b._returns[i]))
            ok += 1
            waiting.append(float(cn.sum()))
        report[cyc] = {"pairs": len(proof["pairs"]), "reconciled": ok,
                       "native_J_mean": float(np.mean(waiting))}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
