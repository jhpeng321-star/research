"""Generate patches/wall_driver_v22_logging.diff + its PATCH_SHAS entry.

The v2.2 LOGGING patch (behaviour-neutral, smoke R2 prep) is kept SEPARATE
from the frozen v2.2 law patches: base = copies/pre/wall_driver.py with
patches/wall_driver.py.diff applied (the registered v2.2 law version,
whose LF sha must equal the recorded ``post`` entry), final =
copies/post/wall_driver.py (the same law code + this logging fix).

What the fix does (logging only, no control path):
  hold_cap.a_priori_effective_hold_s is computed AT ASSEMBLY -
  min(H, (MERGE_RELEASE_POINT_S - lead_s_road_at_assembly) / v_wall) -
  instead of from the release-tick lead position, and the row carries the
  new evidence field hold_cap.lead_s_road_at_assembly_m.

Zero SUMO; the live files (E:/research-dense, E:/rd-wt/wallmr) are only
hashed (READ-ONLY); scratch reconstruction happens in the pre-approved
temp directory, never in the repo.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
COPIES = HERE / "copies"
PATCHES = HERE / "patches"
PRE = COPIES / "pre" / "wall_driver.py"
POST = COPIES / "post" / "wall_driver.py"
LAW_PATCH = PATCHES / "wall_driver.py.diff"
LOG_PATCH = PATCHES / "wall_driver_v22_logging.diff"
SHAS_PATH = HERE / "PATCH_SHAS.json"
REL = "output/campaign_walls_v1/wall_driver.py"
LIVE = Path(r"E:\research-dense\output\campaign_walls_v1\wall_driver.py")
SCRATCH = Path(r"C:\Users\jpeng\AppData\Local\Temp\opencode") / "wallv22_logging"


def lf_sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def entry(data: bytes) -> dict:
    lf = data.replace(b"\r\n", b"\n")
    return {"lf_sha256": lf_sha(lf), "bytes": len(lf),
            "lines": lf.count(b"\n")}


def main() -> None:
    shas = json.loads(SHAS_PATH.read_text(encoding="utf-8"))
    recorded_law = shas["post"]["wall_driver.py"]["lf_sha256"]
    recorded_pre = shas["pre"]["wall_driver.py"]["lf_sha256"]

    live_raw = LIVE.read_bytes()
    live_sha = lf_sha(live_raw)
    # information only: the live target diverged after smoke run 1 (the
    # reconstructor owns that tree; E:/research-dense is READ-ONLY here).
    # The binding chain this script verifies is pre copy -> law patch ->
    # logging patch -> copies/post, with the recorded PATCH_SHAS shas.
    print(f"live target sha (informational): {live_sha}")

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    base_dir = SCRATCH / "base"
    apply_dir = SCRATCH / "apply"
    (base_dir / "output" / "campaign_walls_v1").mkdir(parents=True)
    (apply_dir / "output" / "campaign_walls_v1").mkdir(parents=True)

    law_path = base_dir / "output" / "campaign_walls_v1" / "wall_driver.py"
    shutil.copyfile(PRE, law_path)
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(LAW_PATCH)],
        cwd=base_dir, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"law patch failed to apply: {proc.stderr}")
    law_raw = law_path.read_bytes()
    law_sha = lf_sha(law_raw)
    if law_sha != recorded_law:
        raise SystemExit(
            f"reconstructed v2.2 law wall_driver.py sha {law_sha} != "
            f"recorded post {recorded_law}")

    final_raw = POST.read_bytes()
    final_sha = lf_sha(final_raw)
    if final_sha == law_sha:
        raise SystemExit("copies/post/wall_driver.py carries no logging "
                         "change vs the v2.2 law version")

    proc = subprocess.run(
        ["git", "diff", "--no-index", "--",
         str(base_dir / "output" / "campaign_walls_v1" / "wall_driver.py"),
         str(POST)],
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
    LOG_PATCH.write_bytes(
        ("\n".join(out) + "\n").replace("\r\n", "\n").encode("utf-8"))

    # verify: apply the logging patch on the law base -> must equal the copy
    target = apply_dir / "output" / "campaign_walls_v1" / "wall_driver.py"
    shutil.copyfile(law_path, target)
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(LOG_PATCH)],
        cwd=apply_dir, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"logging patch failed to apply: {proc.stderr}")
    if lf_sha(target.read_bytes()) != final_sha:
        raise SystemExit("logging patch round-trip mismatch")

    shas["logging_patch"] = {
        "note": (
            "v2.2 LOGGING patch (behaviour-neutral, smoke R2 prep): "
            "hold_cap.a_priori_effective_hold_s is computed AT ASSEMBLY "
            "(min(H, (2644.2 - lead_s_road_at_assembly_m)/wall_speed_mps)) "
            "instead of from the release-tick lead position, and the row "
            "adds lead_s_road_at_assembly_m. No control-path change. "
            "Apply AFTER patches/wall_driver.py.diff; the patched result "
            "is what copies/post/wall_driver.py contains (the 'post' "
            "entry above stays the pure v2.2 LAW version that the frozen "
            "registration binds)."),
        "wall_driver.py": {
            "live_path": str(LIVE),
            "base": "pre + patches/wall_driver.py.diff (v2.2 law)",
            "base_lf_sha256": law_sha,
            "patch": "patches/wall_driver_v22_logging.diff",
            "patched_lf_sha256": final_sha,
            **entry(final_raw),
            "round_trip_verified": True,
        },
    }
    SHAS_PATH.write_text(json.dumps(shas, indent=2) + "\n",
                         encoding="utf-8")
    print(f"law base   {law_sha}")
    print(f"patched    {final_sha} ({entry(final_raw)['bytes']} bytes, "
          f"{entry(final_raw)['lines']} lines)")
    print(f"recorded pre {recorded_pre}; live target (informational) "
          f"{live_sha}")
    print(f"{LOG_PATCH.name}: {LOG_PATCH.stat().st_size} bytes")
    print("PATCH_SHAS.json updated (logging_patch section added)")


if __name__ == "__main__":
    main()
