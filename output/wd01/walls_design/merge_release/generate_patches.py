"""Generate patches/*.diff (pre -> post, LF, live-target headers) and
PATCH_SHAS.json (pre + post LF sha256 per file).

Zero SUMO, zero writes outside output/wd01/walls_design/merge_release/.
The live files (E:/research-dense, E:/rd-wt/wallmr) are READ-ONLY and
are only hashed to re-verify the pre copies.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
COPIES = HERE / "copies"
PATCHES = HERE / "patches"
PATCHES.mkdir(exist_ok=True)

#: name -> (live path, repo-relative header path for the diff)
FILES = {
    "common.py": (r"E:\research-dense\output\campaign_walls_v1\common.py",
                  "output/campaign_walls_v1/common.py"),
    "wall_request.py": (
        r"E:\research-dense\output\campaign_walls_v1\wall_request.py",
        "output/campaign_walls_v1/wall_request.py"),
    "wall_driver.py": (
        r"E:\research-dense\output\campaign_walls_v1\wall_driver.py",
        "output/campaign_walls_v1/wall_driver.py"),
    "wall_group.py": (r"E:\rd-wt\wallmr\traffic_shepherd\wall_group.py",
                      "traffic_shepherd/wall_group.py"),
    "sensing.py": (r"E:\rd-wt\wallmr\traffic_shepherd\sensing.py",
                   "traffic_shepherd/sensing.py"),
}

UNCHANGED = {
    "sensing.py": "no functional change for v2.2: SensorFrame already "
                  "carries the lane id (edge identity) and the lane-local "
                  "position that frame_s_road reads; sha recorded so the "
                  "deployed file is verifiable",
}


def lf_sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def file_entry(live: str, data: bytes) -> dict:
    return {"live_path": live, "lf_sha256": lf_sha(data),
            "bytes": len(data.replace(b"\r\n", b"\n")),
            "lines": data.replace(b"\r\n", b"\n").count(b"\n")}


shas: dict = {"pre": {}, "post": {}, "unchanged": {}}
for name, (live, rel) in FILES.items():
    pre = (COPIES / "pre" / name).read_bytes()
    post = (COPIES / "post" / name).read_bytes()
    shas["pre"][name] = file_entry(live, pre)
    shas["post"][name] = file_entry(live, post)
    if lf_sha(pre) != lf_sha(post):
        proc = subprocess.run(
            ["git", "diff", "--no-index", "--",
             str(COPIES / "pre" / name), str(COPIES / "post" / name)],
            capture_output=True, check=False)  # exit 1 = files differ
        lines = proc.stdout.decode("utf-8").splitlines()
        out = []
        for line in lines:
            if line.startswith("diff --git "):
                out.append(f"diff --git a/{rel} b/{rel}")
            elif line.startswith(("--- ", "+++ ")):
                tag = line[:4]
                out.append(f"{tag}a/{rel}" if line.startswith("--- ")
                           else f"{tag}b/{rel}")
            else:
                out.append(line)
        diff = "\n".join(out) + "\n"
        (PATCHES / f"{name}.diff").write_bytes(diff.replace("\r\n", "\n").encode("utf-8"))
        shas["post"][name]["patch"] = f"patches/{name}.diff"
    else:
        shas["unchanged"][name] = UNCHANGED.get(
            name, "pre and post are identical")

(HERE / "PATCH_SHAS.json").write_text(
    json.dumps(shas, indent=2) + "\n", encoding="utf-8")
for name in FILES:
    p = PATCHES / f"{name}.diff"
    print(f"{name}: {'patch ' + str(p.stat().st_size) + ' bytes' if p.exists() else 'UNCHANGED'}")
print("PATCH_SHAS.json written")
