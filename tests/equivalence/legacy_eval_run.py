"""Run the LEGACY scorer (wms_eval) on a state subset and dump results as JSON.

Executed in a subprocess with only ``output/wd01/wm_n1`` on ``sys.path`` (the legacy
``wms`` package name collides with the clean one).  Usage:
``legacy_eval_run.py STATES CKPT RECEIPT MODE OUT`` with MODE ``first3`` or ``per_world20``.
"""
import json
import sys
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[2] / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(WM_N1))

from wms_eval.c1v2 import _car_compact, iter_windows  # noqa: E402
from wms_eval.data import recorded_for  # noqa: E402
from wms_eval.metrics import car_metrics  # noqa: E402
from wms_eval.models import flags_for  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointModel  # noqa: E402

HORIZONS = (10.0, 30.0, 60.0)


def select(path, mode):
    metas = sorted((str(r["world"]), float(r["t0_s"]), str(r["key"])) for r in iter_windows(path))
    if mode == "first3":
        chosen = set(metas[:3])
    else:
        best = {}
        for w, t, k in metas:
            if w not in best or (t, k) < best[w][1:]:
                best[w] = (w, t, k)
        chosen = {best[w] for w in sorted(best)[:20]}
    recs = [r for r in iter_windows(path) if (str(r["world"]), float(r["t0_s"]), str(r["key"])) in chosen]
    return sorted(recs, key=lambda r: (str(r["world"]), float(r["t0_s"]), str(r["key"])))


def main(states, ckpt, receipt, mode, out):
    physics = json.loads(Path(receipt).read_text(encoding="utf-8"))["physics"]
    model = WmsCheckpointModel(ckpt, flags_for("wms"), physics=physics, model_kind="structured")
    rows = []
    for rec in select(states, mode):
        t0 = float(rec["t0_s"])
        arms = {}
        for arm in sorted(rec["recorded"]):
            pred = model.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm}, 60.0, 0)
            car = _car_compact(car_metrics(pred, recorded_for(rec, arm), t0_s=t0, horizons=HORIZONS))
            last = {v["id"]: [v["s"], v["v"], v["lane"]] for v in pred[-1]["vehicles"]}
            arms[arm] = {"car": car, "last_frame": last}
        rows.append({"world": rec["world"], "t0_s": t0, "key": rec["key"], "arms": arms})
    Path(out).write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:6])
