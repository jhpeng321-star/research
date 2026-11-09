"""Prove a DISTILL student checkpoint loads like the MAPPO_LANE comparator and acts (zero SUMO).

Checks, in one process with the wz3to2 N1 binding installed (as the MAPPO_LANE training/eval path):
1. export metadata as ``ops/nm_finaleval.export_meta`` / ``check_export`` read it (weights_only load):
   ``format`` c4_dreamer/3, ``trainer == "mappo"``, lane head present;
2. ``tools.nm_mappo.sanity_eval.load_actor(path, [lanes] of the MAPPO_LANE config, lcexec)`` returns the
   lane-capable ``LaneActor`` with ``lane_control == "learned"`` (the exact MAPPO_LANE loader);
3. ``wms.control.policy.load_policy_fn`` also loads it (deterministic PolicyExecutor);
4. a dummy ``[1,K]`` context (K = map pool) -> finite commands on every CAV, NaN elsewhere, integer lane
   targets; two fresh actors give identical outputs (deterministic).

``python -m tools.distill.check_ckpt CKPT [--lanes-config TOML]``; prints a JSON summary, exit 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

LCEXEC = "tools.nm_mappo.lane_actions:lcexec"


def dummy_state(k: int, *, n: int = 24, seed: int = 0):
    """[1,K] env-style state: n present vehicles on 3 lanes upstream of the drop, every third a CAV."""
    g = torch.Generator().manual_seed(seed)
    present = torch.zeros(1, k, dtype=torch.bool)
    slots = torch.randperm(k, generator=g)[:n].sort().values
    present[0, slots] = True
    s = torch.zeros(1, k)
    s[0, slots] = torch.linspace(-600.0, 50.0, n)
    v = torch.zeros(1, k)
    v[0, slots] = 10.0 + 10.0 * torch.rand(n, generator=g)
    lane = torch.full((1, k), -1, dtype=torch.long)
    lane[0, slots] = torch.arange(n) % 3
    is_cav = torch.zeros(1, k, dtype=torch.bool)
    is_cav[0, slots[::3]] = True
    return {"present": present, "s_m": s, "v_mps": v, "a_mps2": torch.zeros(1, k), "lane": lane,
            "is_cav": is_cav, "length_m": torch.where(present, torch.full((1, k), 5.0), torch.zeros(1, k)),
            "junction": torch.zeros(1, k, dtype=torch.bool)}


def check(path, lanes_config="configs/c4_mappo_lane_wz3to2.toml", map_name="wz3to2") -> dict:
    from wms.control.mappo import agent  # noqa: F401  (imported before the install, as in train_bc)
    from tools.nm_mappo import n1_adapter
    from tools.nm_mappo.sanity_eval import load_actor, load_cfg
    desc, _ = n1_adapter.install(map_name)
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy_keys = payload.get("policy") or {}
    meta = dict(format=payload.get("format"), trainer=payload.get("trainer"), iteration=payload.get("iteration"),
                lane_control=payload.get("lane_control"),
                lane_head=payload.get("lane_head") is True and any(str(k).startswith("lane_head.") for k in policy_keys),
                accel_bounds=[payload.get("accel_min_mps2"), payload.get("accel_max_mps2")],
                student=payload.get("student"))
    assert meta["format"] == "c4_dreamer/3", meta
    assert meta["trainer"] == "mappo", "nm_finaleval.check_export requires trainer == 'mappo'"
    assert meta["lane_head"], "MAPPO_LANE must carry its lane head"
    lanes = (load_cfg(ROOT / lanes_config if not Path(lanes_config).is_absolute() else lanes_config) or {}).get("lanes")
    outs = []
    for _ in range(2):
        exe, lane_control = load_actor(path, lanes, LCEXEC)
        assert lane_control == "learned" and type(exe).__name__ == "LaneActor", (lane_control, type(exe))
        exe.reset()
        state = dummy_state(int(desc.k_slots))
        steps = []
        for t in range(3):
            cmd, lanes_out = exe(state)
            steps.append((cmd.detach().cpu().numpy().copy(), lanes_out.detach().cpu().numpy().copy()))
            state = {**state, "s_m": state["s_m"] + state["v_mps"]}
        outs.append(steps)
    cav = dummy_state(int(desc.k_slots))["is_cav"][0].numpy()
    for cmd, lanes_out in outs[0]:
        assert np.isfinite(cmd[0, cav]).all(), "non-finite CAV command"
        assert np.isnan(cmd[0, ~cav]).all(), "command on a non-CAV slot"
        assert lanes_out.dtype.kind == "i", lanes_out.dtype
    same = all(np.array_equal(a[0], b[0], equal_nan=True) and np.array_equal(a[1], b[1])
               for a, b in zip(outs[0], outs[1]))
    assert same, "two fresh actors disagree (non-deterministic)"
    from wms.control.policy import load_policy_fn
    exe2 = load_policy_fn(str(path))
    a2, l2 = exe2(dummy_state(int(desc.k_slots)), with_lanes=True)
    assert np.isfinite(a2[0, cav]).all()
    cmd0, lanes0 = outs[0][0]
    return dict(ok=True, path=str(path), meta=meta, k_slots=int(desc.k_slots), n_cav=int(cav.sum()),
                laneactor_cmd_first_cavs=[round(float(x), 4) for x in cmd0[0, cav][:6]],
                laneactor_lane_targets_on_cavs=lanes0[0, cav].tolist()[:8], deterministic=same,
                load_policy_fn_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--lanes-config", default="configs/c4_mappo_lane_wz3to2.toml")
    a = p.parse_args(argv)
    try:
        print(json.dumps(check(a.checkpoint, a.lanes_config), indent=1))
    except AssertionError as exc:
        print(json.dumps(dict(ok=False, error=str(exc))))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
