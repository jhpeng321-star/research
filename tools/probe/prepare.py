"""Prepare S-PROBE config; --enable is for main at the cycle-2 -> 3 boundary.

This command never starts a driver, simulator or trainer. No registration budget
is increased. The same command and constants apply to the E5 coop-off arm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from traffic_shepherd.c4.probe import SCHEMA, world_options  # noqa: E402


def prepare(arm, enable=False):
    path = arm / "CONFIG.json"
    cfg = json.loads(path.read_text())
    state = json.loads((Path(cfg["loop_root"]) / "LOOP14_STATE.json").read_text())
    cqr_path = arm / "probe_cqr.json"
    cqr = json.loads(cqr_path.read_text())
    if (cqr.get("schema") != SCHEMA or not cqr.get("ready")
            or cqr["wm_sha256"] != state["accepted_wm"]["sha256"]):
        raise ValueError("requires finite, accepted-WM-bound real-pair CQR before enabling")
    if not cfg.get("fastloop", {}).get("two_box"):
        raise ValueError("probe requires the existing two-box collection dispatcher")
    c = cfg["collect"]
    if not all(c.get(k, {}).get("enabled") for k in ("long_group", "long_single_hold")):
        raise ValueError("both original long slots must already be budgeted")
    if 600 * c["worlds_per_cycle"] > 8000:
        raise ValueError("original long reservation exceeds 8,000 sim-s")
    c["probe"] = {"enabled": True, "from_cycle": 3, "calibration": str(cqr_path.resolve())}
    world_options(c, index=4, cycle=3)
    if enable:
        if not state.get("cycles") or max(row["cycle"] for row in state["cycles"]) != 2:
            raise ValueError("main may enable only before cycle 3 is created")
        target = path
    else:
        target = arm / "CONFIG.probe.json"
    tmp = target.with_suffix(".probe.tmp")
    with tmp.open("x") as stream:
        stream.write(json.dumps(cfg, indent=1) + "\n")
    tmp.replace(target)
    return {"config": str(target), "probe": c["probe"], "extra_sim_s": 0,
            "cqr_sha256": hashlib.sha256(cqr_path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.arm, args.enable), indent=2))
