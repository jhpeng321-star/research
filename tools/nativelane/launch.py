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
    import time
    from datetime import datetime
    spec = importlib.util.spec_from_file_location("nativelane_mappo_tool", tool)
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    original_work = M.work

    def guarded_work(unit):
        if time.time() >= datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp():
            return {"unit": f"{unit[0]}_t{int(unit[1])}", "error": "09:00 deadline; inference not started"}
        return original_work(unit)
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
