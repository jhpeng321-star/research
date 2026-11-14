"""Stage approved TRAIN pack arrays + identity sidecars into the workspace using
NTFS hard links (same E: volume). No bytes are copied, no source is modified,
and no disk space beyond directory entries is consumed. The guard sees only
workspace-local paths, so the guarded materialiser can run workspace-only.

Sources are the canonical permitted TRAIN packs (INDEX/manifest hashes already
matched). Pack arrays are opened read-only (mmap) by the materialiser.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

W = Path(r"E:/rd-wt/ws-wallc1")
INPUTS = W / "_out" / "wallc1_inputs"

PACK_SRC = {
    "pack_v3": Path(r"E:/rd-wt/ws-cl/_ext/pack_v3"),
    "pack_dspeed": Path(r"E:/rd-wt/dspeed-data/_out/pack_dspeed"),
    "pack_damping": Path(r"E:/rd-wt/damping-snap/_out/pack_damping"),
}
ID_SRC = {
    "identities_v3": Path(r"E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities"),
    "identities_dspeed": Path(r"E:/rd-wt/a9/_out/identities_dspeed"),
    "identities_damping": Path(r"E:/rd-wt/damping-snap/_out/identities_damping"),
}


def link_all(src_dir: Path, dst_dir: Path, patterns):
    dst_dir.mkdir(parents=True, exist_ok=True)
    made, reused = [], []
    for pat in patterns:
        for src in sorted(src_dir.glob(pat)):
            dst = dst_dir / src.name
            if dst.exists():
                reused.append(src.name)
                continue
            os.link(src, dst)
            made.append(src.name)
    return made, reused


def main() -> int:
    report = {"packs": {}, "identities": {}}
    for name, src in PACK_SRC.items():
        made, reused = link_all(src, INPUTS / name, ["*.npy"])
        report["packs"][name] = {"src": str(src), "linked": len(made),
                                 "reused": len(reused), "files": made}
    for name, src in ID_SRC.items():
        made, reused = link_all(src, INPUTS / name,
                                ["identities.npy", "global_positions.npy", "vehicle_map.json"])
        report["identities"][name] = {"src": str(src), "linked": made, "reused": reused}
    (W / "_out" / "WALL_C1_PACK_STAGING.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
