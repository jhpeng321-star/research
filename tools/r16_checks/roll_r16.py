"""S-RAMPSPEED (R16) offline inference (CPU only, no training): the 13E stage-B FINAL (b2) with the
R14 switches, R16 ramp speed adaptation OFF vs ON (TRAIN-fitted initialization, no training).

Arms (native switches of the code under test, never emulated):
  R14   emergency_braking + secure_gap_veto + lane_change_secure_gap + emergency_onset=driver_b
        + learned yield (13E pass rule)                              = the R14 launch physics
  R16   R14 + model.ramp_speed_adapt (wms.model.ramp_speed_adapt, init = ramp_speed_adapt_init.json)
Harness = roll_r14.py (panel: R13E TRAIN-role 48-state panel re-extended to H s, entries v2 + queue
state, CRN seed 130013 + state index identical across arms).  Outputs: COMBO300-compatible compact
1 s npz files (p<idx>_ref_<arm>_k0.npz, p<idx>_ref_sumo.npz).  Own PIDs only, nice 19.

  python roll_r16.py --repo REPO --out OUT --ckpt CK --horizon 120 --n 48 --arms R14,R16 --workers 4
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r14_checks"))
import roll_r14 as RR  # noqa: E402  (imports sanity_roll: CPU/thread environment)

SR = RR.SR
#: The 13E stage-B FINAL (resumed b2), gpuR runs/c4_wm_v10_round13e_b2/wm_s.pt (read only).
CKPT_SHA = os.environ.get("R16_CKPT_SHA", "")
R14_PHYS = {"emergency_braking": True, "secure_gap_veto": True, "lane_change_secure_gap": True,
            "emergency_onset": "driver_b"}
SR.ARMS.update({"R14": dict(phys=dict(R14_PHYS), yld=True, ramp=False),
                "R16": dict(phys=dict(R14_PHYS), yld=True, ramp=True)})
_base_set_arm = SR.set_arm


def set_arm(arm: str) -> None:
    _base_set_arm(arm)
    m = SR.G["m"]
    if SR.ARMS[arm].get("ramp"):
        m.enable_ramp_speed_adapt_learnable()
        for q in m.ramp_speed_adapt.parameters():
            q.requires_grad_(False)
    elif hasattr(m, "ramp_speed_adapt_learnable"):
        m.ramp_speed_adapt_learnable = False


def arm_supported(arm: str):
    why = _base_supported(arm)
    if why is None and SR.ARMS[arm].get("ramp") and not hasattr(SR.G["m"], "enable_ramp_speed_adapt_learnable"):
        return "model lacks enable_ramp_speed_adapt_learnable"
    return why


_base_supported = SR.arm_supported
SR.set_arm = set_arm
SR.arm_supported = arm_supported


def main() -> int:
    argv = sys.argv
    ck = argv[argv.index("--ckpt") + 1]
    if not CKPT_SHA or SR.sha256(ck) != CKPT_SHA:
        raise SystemExit("checkpoint is not the registered 13E stage-B FINAL (set R16_CKPT_SHA)")
    if "--workers" in argv and not 1 <= int(argv[argv.index("--workers") + 1]) <= 4:
        raise SystemExit("workers must be 1..4 on gpuR")
    return _main_r16()


def _main_r16() -> int:
    import argparse
    import json
    import time
    from multiprocessing import get_context

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", default=str(SR.STATES))
    ap.add_argument("--receipt", default=str(SR.RECEIPT))
    ap.add_argument("--data-root", default=str(SR.L5 / "data"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--arms", default="R14,R16")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.mode, a.k = "panel", 0
    if os.name == "posix":
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    arms = a.arms.split(",")
    if any(x not in ("R14", "R16") for x in arms):
        raise SystemExit(f"unknown arm in {arms}")
    repo = Path(SR.guard(a.repo)).resolve()
    for p in (repo, repo / "src", repo / "tools"):
        sys.path.insert(0, str(p))
    out = Path(SR.guard(a.out))
    out.mkdir(parents=True, exist_ok=True)
    records, info, tags = RR.load_panel(a)
    RR.G["records"] = records
    RR.write_sumo("panel", records, out, a.horizon)
    meta = {"argv": sys.argv, "pid": os.getpid(), "repo": str(repo), "ckpt_sha256": CKPT_SHA,
            "mode": "panel", "horizon_s": a.horizon, "arms": arms, "records": info,
            "started": time.strftime("%F %T")}
    (out / f"run_meta_{os.getpid()}.json").write_text(json.dumps(meta, indent=1))
    todo = [("panel", i, "ref", arm, str(out), a.horizon, 0) for i in records for arm in arms]
    print(f"todo {len(todo)} workers {a.workers}", flush=True)
    t0 = time.time()
    pool = get_context("fork").Pool(a.workers, initializer=RR.init_worker, initargs=(a.ckpt, str(repo)),
                                    maxtasksperchild=4)
    for idx, tag, arm, status, errs, sec in pool.imap_unordered(RR.process, todo, chunksize=1):
        print(f"[{time.time() - t0:6.0f}s] p{idx:02d} {tag} {arm} {status} {sec:.0f}s", flush=True)
        if errs:
            print(errs[0], flush=True)
    (out / f"DONE_{os.getpid()}").write_text(time.strftime("%F %T"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
