"""Generate the INC1 hook patch on top of the frozen v2.2 post driver.

REGISTRATION_WM_N1_INC1_ADDENDUM_1.md A1.2: one new registered diff,
``patches/wall_driver_inc1_hook.diff``, applied on top of
``copies/post/wall_driver.py`` (the frozen v2.2 law + logging patch), producing
``copies/post_inc1/wall_driver.py``.  It adds ONLY two optional keyword
arguments to ``drive_wall_branch``:

1. ``post_restore_hook(api, t_s) -> dict | None`` - called exactly once right
   after the state restore and before the first control tick; its return value
   is logged in the branch record;
2. ``branch_length_s_override: float | None`` - the branch horizon.

With BOTH absent the driver is byte-for-byte behaviour-identical to the frozen
v2.2 driver (proven by tests/test_wall_driver_inc1_hook_identity.py).

Zero SUMO.  Writes only under output/wd01/walls_design/merge_release/ (and the
pre-approved temp scratch).  Never edits copies/post or the live tree.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
COPIES = HERE / "copies"
PATCHES = HERE / "patches"
BASE = COPIES / "post" / "wall_driver.py"
OUT_DIR = COPIES / "post_inc1"
FINAL = OUT_DIR / "wall_driver.py"
PATCH = PATCHES / "wall_driver_inc1_hook.diff"
SHAS_PATH = HERE / "PATCH_SHAS.json"
ADDENDUM = HERE / "WALL_V22_FREEZE_ADDENDUM_INC1.json"
REL = "output/wd01/walls_design/merge_release/copies/post/wall_driver.py"
SCRATCH = Path(r"C:\Users\jpeng\AppData\Local\Temp\opencode") / "wall_inc1_hook"

REPLACEMENTS = (
    (
        "                      hold_override_s: float | None = None,\n"
        "                      wide_pair_min_span_m: float | None = None) -> dict:\n",
        "                      hold_override_s: float | None = None,\n"
        "                      wide_pair_min_span_m: float | None = None,\n"
        "                      post_restore_hook=None,\n"
        "                      branch_length_s_override: float | None = None) -> dict:\n",
    ),
    (
        "    if horizon_ticks is not None:\n"
        "        horizon_s = t0_s + float(horizon_ticks)\n"
        "    else:\n",
        "    if horizon_ticks is not None:\n"
        "        horizon_s = t0_s + float(horizon_ticks)\n"
        "    elif branch_length_s_override is not None:\n"
        "        # INC1 hook patch (A1.2): optional branch horizon override\n"
        "        # (the INC1 300 s arm).  Absent, the pure v2.2 plan horizon\n"
        "        # path is unchanged.\n"
        "        horizon_s = t0_s + float(branch_length_s_override)\n"
        "    else:\n",
    ),
    (
        "    for tick in range(int(round(t0_s)), int(round(horizon_s))):\n"
        "        now = float(api.simulation.getTime())\n",
        "    # INC1 hook patch (A1.2): optional post-restore callback, invoked\n"
        "    # EXACTLY once right after the state restore and before the first\n"
        "    # control tick (INC1 uses it to insert the stopped obstacle).  Its\n"
        "    # return value is logged in the branch record.  Absent, the pure\n"
        "    # v2.2 driver behaviour is unchanged.\n"
        "    post_restore_result = None\n"
        "    if post_restore_hook is not None:\n"
        "        post_restore_result = post_restore_hook(api, t0_s)\n"
        "    for tick in range(int(round(t0_s)), int(round(horizon_s))):\n"
        "        now = float(api.simulation.getTime())\n",
    ),
    (
        '        "wall_spec_sha256": schedule_digest(wall_spec),\n'
        '        "created_utc": utc_now(),\n'
        "    }\n"
        '    write_json(Path(run_dir) / "branch_record.json", branch_record)\n',
        '        "wall_spec_sha256": schedule_digest(wall_spec),\n'
        '        "created_utc": utc_now(),\n'
        "    }\n"
        "    if post_restore_hook is not None:\n"
        "        # INC1 hook patch (A1.2): log the post-restore callback's\n"
        "        # return value in the branch record.\n"
        '        branch_record["post_restore_hook"] = post_restore_result\n'
        '    write_json(Path(run_dir) / "branch_record.json", branch_record)\n',
    ),
)


def lf_sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def entry(data: bytes) -> dict:
    lf = data.replace(b"\r\n", b"\n")
    return {"lf_sha256": lf_sha(lf), "bytes": len(lf), "lines": lf.count(b"\n")}


def _apply_patch_to_lf(base_bytes: bytes) -> bytes:
    text = base_bytes.replace(b"\r\n", b"\n").decode("utf-8")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise SystemExit(
                f"replacement anchor not unique (found {text.count(old)}): "
                f"{old[:60]!r}")
        text = text.replace(old, new)
    return text.encode("utf-8")


def main() -> None:
    base_raw = BASE.read_bytes()
    base_sha = lf_sha(base_raw)
    final_lf = _apply_patch_to_lf(base_raw)
    final_sha = hashlib.sha256(final_lf).hexdigest()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # the working copy keeps the base's physical line ending (CRLF)
    FINAL.write_bytes(final_lf.replace(b"\n", b"\r\n"))
    if FINAL.read_bytes().replace(b"\r\n", b"\n") != final_lf:
        raise SystemExit("post_inc1 working copy mismatch")

    # generate the patch from an LF scratch base vs the LF patched file
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "base").mkdir(parents=True)
    (SCRATCH / "final").mkdir(parents=True)
    scratch_base = SCRATCH / "base" / "wall_driver.py"
    scratch_final = SCRATCH / "final" / "wall_driver.py"
    scratch_base.write_bytes(base_raw.replace(b"\r\n", b"\n"))
    scratch_final.write_bytes(final_lf)
    proc = subprocess.run(
        ["git", "diff", "--no-index", "--", str(scratch_base), str(scratch_final)],
        capture_output=True, check=False)  # exit 1 = files differ
    lines = proc.stdout.decode("utf-8").splitlines()
    out = []
    for line in lines:
        if line.startswith("diff --git "):
            out.append(f"diff --git a/{REL} b/{REL}")
        elif line.startswith(("--- ", "+++ ")):
            out.append(f"{line[:4]}a/{REL}" if line.startswith("--- ")
                       else f"{line[:4]}b/{REL}")
        else:
            out.append(line)
    PATCH.write_bytes(("\n".join(out) + "\n").replace("\r\n", "\n").encode("utf-8"))
    diff_sha = hashlib.sha256(PATCH.read_bytes()).hexdigest()

    # round-trip: apply the patch on the LF base -> must equal the LF final
    apply_dir = SCRATCH / "apply"
    target = (apply_dir / "output" / "wd01" / "walls_design" /
              "merge_release" / "copies" / "post" / "wall_driver.py")
    target.parent.mkdir(parents=True)
    shutil.copyfile(scratch_base, target)
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(PATCH)],
        cwd=apply_dir, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"inc1 hook patch failed to apply: {proc.stderr}")
    if target.read_bytes().replace(b"\r\n", b"\n") != final_lf:
        raise SystemExit("inc1 hook patch round-trip mismatch")

    shas = json.loads(SHAS_PATH.read_text(encoding="utf-8"))
    if "inc1_hook" in shas:
        raise SystemExit("PATCH_SHAS.json already carries an inc1_hook key; "
                         "refusing to rewrite frozen bookkeeping")
    shas["inc1_hook"] = {
        "note": (
            "INC1 hook patch (REGISTRATION_WM_N1_INC1_ADDENDUM_1 A1.2): "
            "adds ONLY the optional kwargs post_restore_hook(api, t_s) "
            "(called once right after the restore, before the first control "
            "tick; return value logged in the branch record) and "
            "branch_length_s_override.  With both absent the driver is "
            "byte-for-byte behaviour-identical to the frozen post driver "
            "(tests/test_wall_driver_inc1_hook_identity.py).  No change to "
            "proposal, assembly, D_min, hold cap, release, safety aborts, "
            "ownership or speedMode."),
        "wall_driver.py": {
            "live_path": str(BASE),
            "base": "copies/post/wall_driver.py (v2.2 law + logging patch)",
            "base_lf_sha256": base_sha,
            "patch": "patches/wall_driver_inc1_hook.diff",
            "patched_file": "copies/post_inc1/wall_driver.py",
            "patched_lf_sha256": final_sha,
            **entry(final_lf),
            "diff_sha256": diff_sha,
            "round_trip_verified": True,
        },
    }
    SHAS_PATH.write_text(json.dumps(shas, indent=2) + "\n", encoding="utf-8")

    addendum = {
        "schema": "wall_v22_freeze_addendum_inc1/1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "output/wd01/wm_n1/REGISTRATION_WM_N1_INC1_ADDENDUM_1.md",
        "base": {"file": "copies/post/wall_driver.py", "lf_sha256": base_sha},
        "patch": {"file": "patches/wall_driver_inc1_hook.diff",
                  "sha256": diff_sha},
        "patched": {"file": "copies/post_inc1/wall_driver.py",
                    "lf_sha256": final_sha},
        "identity_test": {
            "status": "PENDING",
            "command": ("python -m pytest "
                        "output/wd01/walls_design/merge_release/tests/"
                        "test_wall_driver_inc1_hook_identity.py -q"),
            "detail": None,
        },
    }
    ADDENDUM.write_text(json.dumps(addendum, indent=2) + "\n", encoding="utf-8")

    print(f"base     {base_sha}")
    print(f"patched  {final_sha} ({entry(final_lf)['bytes']} bytes, "
          f"{entry(final_lf)['lines']} lines)")
    print(f"diff sha {diff_sha}")
    print(f"{PATCH.name}: {PATCH.stat().st_size} bytes")
    print(f"{FINAL}")
    print(f"{ADDENDUM.name}")


if __name__ == "__main__":
    main()
