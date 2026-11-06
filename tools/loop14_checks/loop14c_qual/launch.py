"""EXPLORATORY (LOOP14C, not the registered gate): run an unmodified S-ACTQUAL WM tool with a CAV-side variant.

usage: python launch.py VARIANT REPO TOOL.py [tool args...]
  off        loop14c code, flags off (legacy 6 m forced ramp merge, IDM twin)
  wmlaw      bridge native_ramp_merge='wm_law' (every CAV ramp merge by the WM's HDV merge law)
  wmlaw_head wmlaw + the scripted native twin replaced by the learned native-CAV head (R1)
"""
import runpy
import sys

variant, repo, tool = sys.argv[1], sys.argv[2], sys.argv[3]
for p in reversed([f"{repo}/src", repo]):
    sys.path.insert(0, p)
import wms.control.bridge as bridge  # noqa: E402

if variant in ("wmlaw", "wmlaw_head"):
    original = bridge.make_cav_controller

    def patched(*args, **kwargs):
        kwargs.setdefault("native_ramp_merge", "wm_law")
        return original(*args, **kwargs)
    bridge.make_cav_controller = patched
if variant == "wmlaw_head":
    import wms.control.counterfactual as cf
    from wms.control.native_cav import head_native_accel

    def head_twin(state, **_):
        return head_native_accel(state)
    cf.idm_native_accel = head_twin
elif variant not in ("off", "wmlaw"):
    raise SystemExit(f"unknown variant {variant}")
sys.argv = [tool] + sys.argv[4:]
runpy.run_path(tool, run_name="__main__")
