"""Probe: does enabling the run3 FeasibilityProjection reproduce stored run3 scores?"""
from __future__ import annotations
import json
import sys
from pathlib import Path

WS = Path(r"E:/rd-wt/ws-diag120")
sys.path.insert(0, str(WS / "output/wd01/wm_n1"))
from wms_eval.c1v2 import iter_windows  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointModel  # noqa: E402
from wms.stability import FeasibilityProjection, ProjectionConfig  # noqa: E402
from wms_eval.metrics import car_metrics  # noqa: E402
from wms_eval.data import recorded_for  # noqa: E402

CK = WS / "_out/diag120/ckpt"
physics = json.loads((CK / "run3_physics.json").read_text())
cfg = physics
proj = FeasibilityProjection(ProjectionConfig(min_gap_m=2.0, v_max_mps=31.29))


def main():
    m = WmsCheckpointModel(str(CK / "run3_wm_s.pt"), physics=physics)
    m._load()
    orig = m.model.rollout

    def wrapped(*a, **k):
        k.setdefault("projection", proj)
        return orig(*a, **k)
    m.model.rollout = wrapped
    path = Path(r"E:/rd-wt/ws-evalshift/_out/c1_states_v2shift/u_same.json")
    stored = {s["world"]: s for s in json.loads(
        Path(r"E:/rd-wt/ws-eval2/_out/run3_c1_entries_off/states__run3__u_same.json").read_text())["states"]}
    for record in iter_windows(path, limit=3):
        w = record["world"]
        frames = m.predict_states(record, {"arm": "pi"}, 120.0, int(record.get("seed", 0)))
        cm = car_metrics(frames, recorded_for(record, "pi"), t0_s=float(record["t0_s"]),
                         horizons=(60.0, 120.0))
        c = stored[w]["arms"]["pi"]["car"]
        print(w, "mine %.2f/%s stored %.2f/%s" % (
            cm["60.0"]["hdv"]["abs_delta_s_m"]["median"],
            cm["120.0"]["hdv"]["abs_delta_s_m"]["median"],
            c["60.0"]["hdv"]["ds_med"], c["120.0"]["hdv"]["ds_med"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
