"""One-world CPU orchestrator mechanics: synthetic recorder endpoint, real conversion/merge.

No SUMO executable, GPU, learned checkpoint, or live arm is used. The only mocked
endpoint is the collector process; all orchestration and data/return exports run normally.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/c4")]
import loop14 as loop  # noqa: E402
from tools.fastloop.world import build  # noqa: E402


def helper(name):
    spec = importlib.util.spec_from_file_location("fastloop_dry_helper", ROOT / "tests/unit" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_collect(argv):
    import numpy as np
    out = Path(argv[argv.index("--out") + 1])
    seed = int(argv[argv.index("--seeds") + 1])
    for kind in ("POLICY", "B"):
        recorder = helper("test_c4_loop.py")
        recorder.TOTAL = 310.
        ep = recorder.write_episode(out / "branches" / kind, seed=seed, policy=kind)
        with np.load(ep / "episode.npz", allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        meta = json.loads(str(arrays["meta_json"]))
        meta["branch"] = {"kind": kind, "pair_key": f"w{seed}@t10"}
        arrays["meta_json"] = np.asarray(json.dumps(meta))
        np.savez_compressed(ep / "episode.npz", **arrays)
        manifest = loop.read_json(ep / "manifest.json")
        manifest["branch"] = meta["branch"]
        loop.write_atomic(ep / "manifest.json", manifest)
        loop.write_atomic(ep / "summary.json", {"schema": "c4_loop14_pair_summary/1", "seed": seed,
            "warmup_s": 10., "horizon_s": 300., "collisions": 0, "teleports": 0,
            "dropped_vehicle_steps": 0, "requested_population": 6, "start_entry_backlog": [0, 0],
            "ledger": {"speed_deficit_vehicle_seconds": 90., "source_wait_vehicle_seconds": 10.}})
    loop.write_atomic(out / "COLLECT_SUMMARY.json", {"episodes_complete": 1, "episodes_failed": 0,
        "branches": 2, "branch_collisions": 0, "branch_teleports": 0, "pair_dJ": {f"w{seed}@t10": 0.}})
    return 0


def run(out, worlds=1):
    out.mkdir(parents=True, exist_ok=False)
    helpers = helper("test_loop14.py")
    path = helpers._config(out, helpers._registration(out))
    cfg = loop.read_json(path)
    cfg["collect"].update(worlds_per_cycle=worlds, warmup=10, branch_first=10, branch_horizon=300)
    cfg["hosts"].update(cpuP=dict(cfg["hosts"]["cpu"]), cpuQ=dict(cfg["hosts"]["cpu"]))
    cfg["fastloop"] = {"two_box": True, "spot_resume": True, "boxes": ["cpuP", "cpuQ"], "workers_per_box": 1}
    policy = out / "policy.pt"
    policy.write_bytes(b"mechanics-only; never loaded")
    loop.init(cfg, SimpleNamespace(adopt_policy=str(policy)))
    real_run = loop.run_cmd

    def endpoint(h, argv, **kwargs):
        if "traffic_shepherd.c4.collect" in argv:
            return synthetic_collect(argv)
        # Call the ordinary CPU converter in-process to keep all fixture IO contained.
        if str(argv[1]).endswith("tools/fastloop/world.py"):
            build(Path(argv[argv.index("--work") + 1]), 0, 0.)
            return 0
        return real_run(h, argv, **kwargs)

    loop.run_cmd = endpoint
    try:
        assert loop.run_phase(cfg, "collect") == 0
        assert loop.run_phase(cfg, "convert") == 0
        assert loop.run_phase(cfg, "wm_ft", dry_run=True) == 0
    finally:
        loop.run_cmd = real_run
    state = loop.load_state(cfg)
    converted = state["cycles"][0]["phases"]["convert"]["receipt"]
    assert converted["episodes"] == 2 * worlds and converted["targets"] == worlds
    report = {"status": "PASS", "worlds": worlds, "actual_sumo_sim_s": 0, "gpu_used": False,
              "collector": "synthetic recorder endpoint", "real_conversion_and_return_export": True,
              "next_phase": loop._phase(state)["name"], "receipt": converted,
              "fixture_only_conservative_charge": state["sim_s_charged"]}
    loop.write_atomic(out / "DRY_RUN.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.out)))
