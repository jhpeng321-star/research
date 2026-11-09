"""DIST-RL STUDENT_RL helpers (DISTILL, 2026-11-09).  Zero SUMO.

    python tools/nm_mappo/distill_rl.py render  --seeds LO-HI --out RUN/c4_distill_rl.toml
    python tools/nm_mappo/distill_rl.py student --ckpt STUDENT.pt [--sha256 X]
    python tools/nm_mappo/distill_rl.py quote   [--json]

* ``render``  writes the launch config = configs/c4_distill_rl_d2800p20.toml with the
  ``@DISTILL_RL_SEEDS@`` placeholder replaced by the DIST-REG fresh TRAIN block (validated by the map
  module: never 989xxx / 993xxx / 991xxx / 962xxx / 964000-964199 / 977xxx / registered panels).  The
  block must ALSO be passed to ``launch.py plan --extra-train-ranges`` (PLAN env opens the seed guard).
* ``student`` checks the STUDENT is an EXPORTED lane-on MAPPO_LANE actor that loads STRICTLY into the
  lane-on MAPPO networks (policy incl. lane head + group encoder), accel bound 2.6, no critic (a trainer
  checkpoint would load a critic; MAPPO_CT started with a fresh critic, so it is refused).
* ``quote``   sim-s and wall estimate (MAPPO_CT measured on cpuZ, lane-on factor 1.05-1.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _p in (CODE / "src", CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TEMPLATE = CODE / "configs" / "c4_distill_rl_d2800p20.toml"
PLACEHOLDER = "@DISTILL_RL_SEEDS@"
ACCEL_MAX = 2.6

# MAPPO_CT measured on cpuZ (S-MAPPOCT production log, 12 iterations x 8 episodes, 8 workers, speed-only):
CT_ITER_WALL_S = (96.75, 123.36)          # min / max iteration wall
CT_ITER_WALL_MEAN_S = 104.4               # 1252.3 s / 12
CT_COLLECT_S, CT_UPDATE_S = 44.3, 59.1    # means
LANE_FACTOR = (1.05, 1.5)                 # lane-on (LCEXEC gate + writer, lane head), as S-MAPPOWZL quote
EPISODE_SIM_S, EPISODES, ITERATIONS = 900.0, 8, 12
SMOKE_SIM_S = 2 * (300.0 + 30.0)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render(seeds: str, out: Path, template: Path = TEMPLATE) -> dict:
    from tools.coopmwm.map_scenarios import parse_extra_train_ranges
    blocks = parse_extra_train_ranges(seeds)
    if len(blocks) != 1:
        raise SystemExit("exactly one 'lo-hi' block")
    lo, hi = blocks[0]
    text = template.read_text(encoding="utf-8").replace("\r\n", "\n")
    if text.count(PLACEHOLDER) != 2:          # header mention + the [rollout] value
        raise SystemExit("template placeholder count changed")
    text = text.replace(f'seeds = "{PLACEHOLDER}"', f'seeds = "{lo}-{hi}"', 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return {"config": str(out), "sha256_lf": hashlib.sha256(text.encode()).hexdigest(),
            "seeds": f"{lo}-{hi}", "template_sha256_lf": hashlib.sha256(
                template.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def check_student(path, sha256: str | None = None) -> dict:
    import torch
    from wms.control.mappo.agent import build_networks
    path = Path(path)
    digest = sha256_file(path)
    if sha256 and digest != sha256:
        raise SystemExit(f"{path}: sha256 {digest} != pinned {sha256}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "policy" not in state or "group_encoder" not in state:
        raise SystemExit(f"{path}: not an exported MAPPO actor (policy + group_encoder)")
    if "critic" in state:
        raise SystemExit(f"{path}: carries a critic (trainer checkpoint); export the actor only "
                         "(MAPPO_CT started with a fresh critic)")
    if state.get("lane_control", "learned") != "learned" or not state.get("lane_head", True):
        raise SystemExit(f"{path}: not a lane-on actor")
    keys = sorted(k for k in state["policy"] if k.startswith("lane_head."))
    if not keys:
        raise SystemExit(f"{path}: no lane_head.* weights (would need a fresh head: refused)")
    policy, encoder, _ = build_networks(accel_max=ACCEL_MAX, lane_control="learned")
    policy.load_state_dict(state["policy"], strict=True)
    encoder.load_state_dict(state["group_encoder"], strict=True)
    if abs(float(policy.accel_max) - ACCEL_MAX) > 1e-9:
        raise SystemExit(f"{path}: stored accel_max {float(policy.accel_max)} != {ACCEL_MAX}")
    return {"path": str(path), "sha256": digest, "format": state.get("format"),
            "trainer": state.get("trainer"), "lane_head_keys": len(keys),
            "accel_bounds": [float(policy.accel_min), float(policy.accel_max)],
            "lane_rules": state.get("lane_rules"), "critic": "fresh (exported actor)", "ok": True}


def quote() -> dict:
    planned = ITERATIONS * EPISODES * EPISODE_SIM_S
    lo = CT_ITER_WALL_MEAN_S * LANE_FACTOR[0]
    hi = CT_ITER_WALL_S[1] * LANE_FACTOR[1]
    return {"per_update_sim_s": EPISODES * EPISODE_SIM_S, "updates": ITERATIONS,
            "planned_sim_s": planned, "smoke_sim_s": SMOKE_SIM_S, "planned_total_sim_s": planned + SMOKE_SIM_S,
            "retry_reserve_sim_s": planned, "hard_cap_sim_s": 2 * planned + 2 * SMOKE_SIM_S,
            "per_update_wall_s": [round(lo), round(hi)],
            "production_wall_min": [round(ITERATIONS * lo / 60), round(ITERATIONS * hi / 60)],
            "basis": f"MAPPO_CT cpuZ 8 workers: mean {CT_ITER_WALL_MEAN_S} s/update (collect {CT_COLLECT_S}, "
                     f"update {CT_UPDATE_S}), max {CT_ITER_WALL_S[1]}; lane-on factor {LANE_FACTOR}",
            "host": "cpuZ (one trainer, 8 spawn workers = one wave of 8 episodes); cpuW not needed "
                    "(a MAPPO run is single-box); keep it as the fallback host",
            "waves": math.ceil(EPISODES / 8)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render")
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("student")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sha256", default=None)
    sub.add_parser("quote")
    a = ap.parse_args(argv)
    if a.cmd == "render":
        out = render(a.seeds, Path(a.out))
    elif a.cmd == "student":
        out = check_student(a.ckpt, a.sha256)
    else:
        out = quote()
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
