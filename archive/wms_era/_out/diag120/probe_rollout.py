"""Time one WM-S rollout and inspect per-vehicle / per-section outputs."""
from __future__ import annotations

import sys
import time
from pathlib import Path

WS = Path(r"E:/rd-wt/ws-diag120")
sys.path.insert(0, str(WS / "output/wd01/wm_n1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wms_eval.c1v2 import iter_windows  # noqa: E402
from wms_eval.models import load_model  # noqa: E402
from wms_eval.data import initial_frame, recorded_for  # noqa: E402

CK = WS / "_out/diag120/ckpt"


def main() -> int:
    spec = sys.argv[1] if len(sys.argv) > 1 else "struct"
    state_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(r"E:/rd-wt/ws-evalshift/_out/c1_states_v2shift/u_new.json"))
    record = next(iter_windows(state_path, limit=1))
    ck = str(CK / f"{spec}_wm_s.pt")
    physics_path = CK / f"{spec}_physics.json"
    import json
    physics = json.loads(physics_path.read_text()) if physics_path.is_file() else None
    t = time.time()
    predictor, flags = load_model("wms", checkpoint=ck, physics=physics)
    print("load", f"{time.time()-t:.2f}s")
    arm = "pi"
    t = time.time()
    frames = predictor(record, {"arm": arm}, 120.0, int(record.get("seed", 0)))
    dt = time.time() - t
    print("rollout", f"{dt:.2f}s", "frames", len(frames),
          "nv_last", len(frames[-1]["vehicles"]))
    # quick error check
    rec = recorded_for(record, arm)
    from wms_eval.metrics import car_metrics
    cm = car_metrics(frames, rec, t0_s=float(record.get("t0_s", 0.0)),
                     horizons=(10.0, 30.0, 60.0, 120.0))
    print("hdv ds60", cm["60.0"]["hdv"]["abs_delta_s_m"]["median"],
          "ds120", cm["120.0"]["hdv"]["abs_delta_s_m"]["median"])
    init = initial_frame(record)
    print("n_init", len(init["vehicles"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
