"""Extract OLDHIST recovered per-driver IDM parameters for the worst states.

Runs the model once per selected state, capturing the rollout's
``driver_parameters`` [1,K,9] (oldhist runs in ``driver_params`` mode).  Only
the encoder outputs are kept; nothing full-horizon is stored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WS = Path(r"E:/rd-wt/ws-diag120")
sys.path.insert(0, str(WS / "output/wd01/wm_n1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wms_eval.c1v2 import iter_windows  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointModel  # noqa: E402
from wms.driver_encoder import PARAMETER_NAMES  # noqa: E402

CK = WS / "_out/diag120/ckpt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", required=True, help="file with one world id per line")
    ap.add_argument("--arm", default="pi")
    args = ap.parse_args()

    worlds = set(Path(args.worlds).read_text(encoding="utf-8").split())
    physics = json.loads((CK / "oldhist_physics.json").read_text())
    model = WmsCheckpointModel(str(CK / "oldhist_wm_s.pt"), physics=physics)
    model._load()
    captured: dict = {}
    orig = model.model.rollout

    def wrap(*a, **k):
        out = orig(*a, **k)
        captured["dp"] = out.get("driver_parameters")
        return out
    model.model.rollout = wrap

    results = {}
    for record in iter_windows(args.states):
        world = str(record.get("world"))
        if world not in worlds:
            continue
        init_ids = sorted(str(r["id"]) for r in record["initial_state"]["vehicles"]
                          if r.get("present", True) and int(r.get("lane", -1)) >= 0)
        # only the t0 encoder output is needed; a 2 s rollout captures it cheaply.
        model.predict_states(record, {"arm": args.arm}, 2.0, int(record.get("seed", 0)))
        dp = captured.get("dp")
        if dp is None:
            results[world] = {"error": "no driver_parameters in rollout"}
            continue
        vals = dp[0].numpy()
        results[world] = {
            "ids": init_ids,
            "params": {name: [float(x) for x in vals[:, i]]
                       for i, name in enumerate(PARAMETER_NAMES)},
        }
        print("params", world, "n", len(init_ids), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
