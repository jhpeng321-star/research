"""Copy the live wall-law v2.1 files into copies/pre as LF, record pre sha256.

Zero SUMO, zero writes outside output/wd01/walls_design/merge_release/.
The live files are read-only; this only reads them.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "copies", "pre")
os.makedirs(PRE, exist_ok=True)

LIVE = {
    "wall_request.py": r"E:\research-dense\output\campaign_walls_v1\wall_request.py",
    "wall_driver.py": r"E:\research-dense\output\campaign_walls_v1\wall_driver.py",
    "common.py": r"E:\research-dense\output\campaign_walls_v1\common.py",
    "wall_group.py": r"E:\rd-wt\wallmr\traffic_shepherd\wall_group.py",
    "sensing.py": r"E:\rd-wt\wallmr\traffic_shepherd\sensing.py",
}


def lf_sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


pre = {}
for name, src in LIVE.items():
    raw = open(src, "rb").read()
    data = raw.replace(b"\r\n", b"\n")
    with open(os.path.join(PRE, name), "wb") as fh:
        fh.write(data)
    pre[name] = {"live_path": src, "lf_sha256": lf_sha(data),
                 "bytes": len(data), "lines": data.count(b"\n")}
    print(f"{name}: {pre[name]['lf_sha256']} ({pre[name]['bytes']} bytes)")

with open(os.path.join(HERE, "PATCH_SHAS.json"), "w", encoding="utf-8") as fh:
    json.dump({"pre": pre}, fh, indent=2)
    fh.write("\n")
print("pre shas recorded")
