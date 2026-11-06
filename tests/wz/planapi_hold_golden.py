"""WM-A: golden of ``wms.planapi.evaluate`` for 'hold'-only action sets (captured BEFORE the 'prog'
role was added, c4/wz-wm-port 2fb9d454).  ``test_planapi_hold_golden.py`` requires every J draw
(float.hex) and every predicted speed / lane trace (sha256) to stay bit-identical.

    python tests/wz/planapi_hold_golden.py --capture
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

GOLDEN = HERE / "data" / "planapi_hold_golden.json"


def moment() -> dict:
    cavs = [{"id": f"c{i}", "edge": "up", "lane": i % 3, "x": 2300.0 + 45.0 * i, "v": 24.0 + 0.5 * i}
            for i in range(5)]
    veh = [{"id": f"h{i}", "edge": "up", "lane": i % 3, "x": 2150.0 + 33.0 * i, "v": 20.0 + 0.3 * i}
           for i in range(14)]
    veh += [{"id": "j1", "edge": ":n1_0", "lane": 1, "x": 2700.0, "v": 26.0},
            {"id": "w1", "edge": "wz", "lane": 0, "pos": 40.0, "v": 28.0}]
    return {"t0": 600.0, "map": "wz3to2", "cavs": cavs, "vehicles": veh}


def actions() -> list:
    def hold(**kw):
        p = {"v_w": 18.0, "hold_s": 4.0, "dmax": 1.5, "amax": 1.0, "release_x": 2696.0}
        p.update(kw)
        return p
    return [
        {"id": "base", "kind": "base", "members": []},
        {"id": "solo_c0", "kind": "solo", "members": [{"id": "c0", "role": "hold", "params": hold()}]},
        {"id": "solo_c2_late", "kind": "solo",
         "members": [{"id": "c2", "role": "hold", "params": hold(start_s=2.0, v_w=15.0)}]},
        {"id": "solo_c4_release", "kind": "solo",
         "members": [{"id": "c4", "role": "hold", "params": hold(release_x=2490.0)}]},
        {"id": "wall", "kind": "group", "members": [
            {"id": "c0", "role": "hold", "params": hold()},
            {"id": "c1", "role": "hold", "params": hold(anchor="c0", anchor_release_x=2380.0)},
            {"id": "c2", "role": "hold", "params": hold(dmax=1.0)}]},
    ]


def compute() -> dict:
    import torch
    from wms import planapi as A
    from wms.core import road as R

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    R._reset_for_tests()
    road = R.road_from_descriptor(R.stub_descriptor("wz3to2"))
    out = {}
    with R.road_scope(road):
        h = A.load(None, max_rows=4)
        st = A.observe(h, moment())
        trace = {}
        scores = A.evaluate(h, st, actions(), K=2, H=6.0, seed=11, trace=trace)
    out["state"] = hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest()
    for aid, draws in scores.items():
        out[f"J.{aid}"] = [float.hex(float(x)) for x in draws]
    for name in ("speeds", "lanes"):
        arr = torch.cat(trace[name], 0).contiguous().numpy()
        out[f"trace.{name}"] = hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()
    out["trace.rows"] = [list(r) for r in trace["rows"]]
    R._reset_for_tests()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args(argv)
    result = compute()
    if args.capture:
        GOLDEN.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {GOLDEN}")
    print(json.dumps({k: v for k, v in result.items() if k.startswith("J.")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
