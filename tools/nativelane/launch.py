"""Run the reused MAPPO/qualification tool with explicit native-lane treatment."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
on = os.environ.get("NATIVE_LANE", "0") == "1"
mode, tool = sys.argv[1:3]
sys.argv = [tool, *sys.argv[3:]]

if mode == "mappo":
    import wms.control.policy14 as P
    original = P.make_controller_factory

    def factory(*args, **kwargs):
        kwargs["native_lane_change"] = on
        return original(*args, **kwargs)
    P.make_controller_factory = factory
    import importlib.util
    import json
    import subprocess
    import time
    from datetime import datetime
    task_root = Path.home() / "nativelane"
    if on and "--horizon" in sys.argv and sys.argv[sys.argv.index("--horizon") + 1] == "300":
        early = task_root / "EARLY_REPAIRS.pid.json"
        if (task_root / "EARLY_REPAIRS_PLAN.json").exists():
            while not early.exists():
                assert time.time() < datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp()
                time.sleep(2)
            owner = json.loads(early.read_text())
            while True:
                try:
                    fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
                    alive = fields[0] != "Z" and fields[19] == owner["start_ticks"]
                except FileNotFoundError:
                    alive = False
                if not alive:
                    break
                assert time.time() < datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp()
                time.sleep(10)
        # The inherited training low-share cap silently shortened three old C
        # rollouts. Audit actual array clocks, archive short OFF outputs, and
        # repair both treatments at the requested duration before comparison.
        import numpy as np
        short = []
        for p in sorted((task_root / "off/wm_c").glob("w*_t*.npz")):
            with np.load(p) as z:
                if z["s_m"].shape[1] != 301:
                    short.append(p)
        if short:
            archive = task_root / "superseded_short_clock"
            archive.mkdir(exist_ok=True)
            meta = task_root / "off/wm_c/RUN_META.json"
            (archive / "RUN_META_original.json").write_bytes(meta.read_bytes())
            units = []
            for p in short:
                meta = p.with_suffix(".json")
                d = json.loads(meta.read_text())
                units.append(d["world"] + "@" + str(int(d["t0"])))
                p.rename(archive / p.name)
                meta.rename(archive / meta.name)
                fixed = task_root / "off/clock_early" / p.name
                fixed_meta = fixed.with_suffix(".json")
                if fixed.exists() and fixed_meta.exists() and "error" not in json.loads(fixed_meta.read_text()):
                    import shutil
                    with np.load(fixed) as z:
                        assert z["s_m"].shape[1] == 301
                    shutil.copyfile(fixed, p)
                    shutil.copyfile(fixed_meta, meta)
            argv = [sys.executable, str(Path(__file__).resolve()), "mappo", tool,
                    "--repo", str(ROOT), "--source", str(Path.home() / "mappodiag/src_cbase"),
                    "--units", ",".join(units), "--horizon", "300", "--samples", "4", "--workers", "3",
                    "--pad-max", "300", "--wm", str(Path.home() / "mappodiag/wm_13e_b2.pt"),
                    "--wm-config", "configs/c4_wm_v10round13e_stage_b.toml",
                    "--mappo", str(Path.home() / "coopmeas/mappo_v5_final.pt"),
                    "--out", str(task_root / "off/wm_c")]
            subprocess.run(argv, env={**os.environ, "NATIVE_LANE": "0"}, check=True)
            for name in units:
                world, tick = name.split("@")
                with np.load(task_root / "off/wm_c" / f"{world}_t{tick}.npz") as z:
                    assert z["s_m"].shape[1] == 301
            (task_root / "CLOCK_REPAIR.json").write_text(json.dumps({"repaired_off_units": units,
                 "actual_horizon_s": 300, "training_low_share_cap_disabled_for_measurement_only": True}, indent=1))
    repair = task_root / "REPAIR_ON60.json"
    if on and "--horizon" in sys.argv and sys.argv[sys.argv.index("--horizon") + 1] == "300" and repair.exists():
        request = json.loads(repair.read_text())
        argv = [sys.executable, str(Path(__file__).resolve()), "mappo", tool,
                "--repo", str(ROOT), "--source", str(Path.home() / "mappodiag/src_bsrc"),
                "--units", request["units"], "--horizon", "60", "--samples", "4", "--workers", "6",
                "--pad-max", "64", "--wm", str(Path.home() / "mappodiag/wm_13e_b2.pt"),
                "--wm-config", "configs/c4_wm_v10round13e_stage_b.toml",
                "--mappo", str(Path.home() / "coopmeas/mappo_v5_final.pt"),
                "--out", str(task_root / "on/wm_coop")]
        subprocess.run(argv, check=True)
        results = [json.loads(p.read_text()) for p in (task_root / "on/wm_coop").glob("w*_t*.json")]
        assert len(results) == 33 and all("error" not in r and r.get("finite") is True for r in results)
        repair.rename(task_root / "REPAIR_ON60_DONE.json")
    # Once the qualification pool has fully exited, use the released six slots.
    q = task_root / "QUAL_TASKS.json"
    if q.exists() and json.loads(q.read_text()).get("complete") and "--workers" in sys.argv:
        cap = int(os.environ.get("NATIVELANE_WORKER_CAP", "12"))
        assert 1 <= cap <= 12
        sys.argv[sys.argv.index("--workers") + 1] = str(cap)
    spec = importlib.util.spec_from_file_location("nativelane_mappo_tool", tool)
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    original_init = M._init

    def measurement_init(args):
        from dataclasses import replace
        original_init(args)
        cfg = replace(M.G["trainer"].config, low_penetration_horizon_cap=False)
        M.G["trainer"].config = cfg
        M.G["cfg"] = cfg
    M._init = measurement_init
    original_work = M.work

    def guarded_work(unit):
        if time.time() >= datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp():
            return {"unit": f"{unit[0]}_t{int(unit[1])}", "error": "09:00 deadline; inference not started"}
        result = original_work(unit)
        if "error" not in result:
            import hashlib
            import numpy as np
            with np.load(Path(M.G["a"]["out"]) / (result["unit"] + ".npz")) as z:
                actual = z["s_m"].shape[1] - 1
            result.update(actual_horizon_s=actual, native_lane_change=on,
                          measurement_horizon_cap_disabled=True,
                          measurement_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            if actual != M.G["a"]["horizon"]:
                result["error"] = f"actual elapsed horizon {actual} != requested {M.G['a']['horizon']}"
        return result
    M.work = guarded_work
    raise SystemExit(M.main())
elif mode == "qual":
    import torch
    import wms.control.bridge as B
    import wms.control.counterfactual as CF
    from wms.control.native_cav import head_native_accel

    CF.idm_native_accel = lambda state, **_: head_native_accel(state)
    sys.path.insert(0, str(Path.home() / "actqual/tools"))
    import wm_coop_k as W
    original_build = W.build_trainer
    import wms.control.train_cli as TC
    from wms.control.policy14_cli import load_members

    # The copied FINAL has no adjacent physics sidecar. Bind the exact same
    # committed 13E stage-B config as the MAPPO harness, never a physics fallback.
    TC._load_world_models = lambda paths, device: load_members(
        paths, [str(ROOT / "configs/c4_wm_v10round13e_stage_b.toml")] * len(paths), device)[0]

    def build(*args, **kwargs):
        trainer, model, payload = original_build(*args, **kwargs)

        def factory(policy, group, safety, record=None, **extra):
            # The scripted harness represents native traffic inside its policy object.
            # Explicitly expose intervention-only ownership to the same bridge in BOTH
            # treatments; targets release to native after their original 3 s command.
            acting = None
            outer_mask = extra.pop("act_mask", None)
            extra["native"] = head_native_accel
            # Allocate on first state, then keep one mutable ownership tensor per episode.
            inner = None

            def prepare(state):
                nonlocal inner, acting
                W.CUR["state"] = state
                if acting is None:
                    acting = torch.zeros_like(state["present"], dtype=torch.bool)
                    inner = B.make_cav_controller(policy, group, safety, record,
                                                  act_mask=acting, native_ramp_merge="wm_law",
                                                  native_lane_change=on, **extra)
                acting.zero_()
                scripted = trainer.policy
                tick = int(state.get("physics_step_index", 0))
                active = tick < 6 and getattr(scripted, "mode", "") in ("P", "A")
                if active:
                    for slot in scripted.targets:
                        if slot < acting.shape[-1]:
                            acting[:, slot] = True
                if outer_mask is not None:
                    acting.logical_and_(outer_mask.to(acting.device))

            def callback(state):
                prepare(state)
                return inner(state)

            if on:
                def native_mask(state):
                    prepare(state)
                    return inner.native_lane_mask(state)
                callback.native_lane_mask = native_mask
            return callback

        trainer.controller_factory = factory
        return trainer, model, payload
    W.build_trainer = build
    if Path(tool).name == "wm_coop_k.py":
        W.main()
        raise SystemExit(0)
else:
    raise SystemExit(f"unknown mode {mode}")
runpy.run_path(tool, run_name="__main__")
