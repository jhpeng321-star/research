"""EXPLORATORY (S-RAMPSPEED, not the registered gate): run an unmodified S-ACTQUAL WM tool on the R16 code
with the 13E stage-B FINAL (b2) and the R14 physics (physics.json next to the checkpoint), ramp speed
adaptation OFF (r14) or ON at the TRAIN-fitted init (r16).

usage: python launch_r16.py VARIANT REPO TOOL.py [tool args...]
"""
import runpy
import sys

variant, repo, tool = sys.argv[1], sys.argv[2], sys.argv[3]
if variant not in ("r14", "r16"):
    raise SystemExit(f"unknown variant {variant}")
for p in reversed([f"{repo}/src", repo]):
    sys.path.insert(0, p)
import wms.control.train_cli as cli  # noqa: E402

original = cli._load_world_models


def patched(paths, device):
    models = original(paths, device)
    for m in models:
        ph = m.physics
        if not (ph.emergency_braking and ph.secure_gap_veto and ph.lane_change_secure_gap
                and ph.emergency_onset == "driver_b" and m.mainline_merge_yield_learnable):
            raise SystemExit("R14 physics / 13E yield not active on the loaded WM")
        if variant == "r16":
            m.enable_ramp_speed_adapt_learnable()
            for q in m.ramp_speed_adapt.parameters():
                q.requires_grad_(False)
        elif m.ramp_speed_adapt_learnable:
            raise SystemExit("r14 arm must have ramp speed adaptation off")
    return models


cli._load_world_models = patched
sys.argv = [tool] + sys.argv[4:]
runpy.run_path(tool, run_name="__main__")
