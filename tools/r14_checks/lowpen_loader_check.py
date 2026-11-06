"""S-R14 D3 data-loader check of the S-LOWPEN-20261025 packs (CPU, no training, no model).

For each pack: world roles are computed with ``wms.data.c4_pairs.world_roles`` over the packs AND
any extra manifests given (the v5 sources), exactly as the trainer's ``_add_c4_sources`` does; the
windows are built with the R14 recipe's loader settings (history 60 s / 1 s, entries_v2, ffill
junction fix, strict controller hygiene, first-passage merge-pivot history, K = 192, stride 10 s).
Checks: every TRAIN window comes from a TRAIN-role world, every VAL window from a VAL-role world,
no world on both sides, no world shared with the extra manifests; counts by design CAV share;
realized CAV share at the window's first row (all windows decoded, one process).

  python lowpen_loader_check.py --packs P1 P2 P3 --extra-manifests M1.json ... --horizons 120 300 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "truth", "vtype")


def guard(p):
    s = str(p).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {p}")
    return Path(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--extra-manifests", nargs="*", default=[])
    ap.add_argument("--horizons", nargs="+", type=float, default=[120.0, 300.0])
    ap.add_argument("--stride", type=float, default=10.0)
    ap.add_argument("--decode", type=int, default=-1, help="windows decoded per split (-1 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.name == "posix":
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    import numpy as np
    from wms.data.c4_episodes import build_c4_datasets, read_source_manifest
    from wms.data.c4_pairs import world_roles

    packs = [guard(p) for p in a.packs]
    docs = {p.name: read_source_manifest(p) for p in packs}
    extra = [json.loads(guard(m).read_text()) for m in a.extra_manifests]
    roles, overridden = world_roles([*docs.values(), *extra])
    pack_worlds = {w for d in docs.values() for e in d["episodes"] for w in [e["world"]]}
    extra_worlds = {e["world"] for d in extra for e in d["episodes"]}
    out = {"packs": {}, "n_roles": len(roles), "overridden_to_val": overridden,
           "pack_worlds_shared_with_extra": sorted(pack_worlds & extra_worlds),
           "started": time.strftime("%F %T"), "argv": sys.argv}
    ok = not (pack_worlds & extra_worlds)
    for p in packs:
        doc = docs[p.name]
        pen = {e["world"]: float(e["penetration"]) for e in doc["episodes"]}
        own = {}
        for e in doc["episodes"]:
            own.setdefault(e["world"], set()).add(e["split_role"])
        both_sides = sorted(w for w, r in own.items() if len(r) > 1)
        res = {"manifest_roles": dict(Counter(e["split_role"] for e in doc["episodes"])),
               "worlds_with_both_roles_in_manifest": both_sides, "horizons": {}}
        ok &= not both_sides
        for h in a.horizons:
            built = build_c4_datasets(p, horizon_steps=int(round(h / 0.5)), history_s=60.0, history_stride_s=1.0,
                                      entries_v2=True, stride_s=a.stride, k_slots=192, role_of=roles,
                                      junction_fix="ffill", command_known=True, allow_empty=True,
                                      merge_pivot_history=True)
            hres = {}
            for split in ("train", "val"):
                ds = built[split]
                if ds is None:
                    hres[split] = {"n": 0}
                    continue
                worlds = ds.receipt["worlds"]
                wrong = [w for w in worlds if roles[w] != split]
                ok &= not wrong
                n = len(ds)
                by_pen = Counter()
                for e_idx, _t in ds.positions:
                    by_pen[pen[ds._loader.entries[e_idx]["world"]]] += 1
                k = n if a.decode < 0 else min(n, a.decode)
                share = []
                for i in (range(n) if k == n else np.linspace(0, n - 1, k).round().astype(int)):
                    r = ds[int(i)]
                    pr, cav = np.asarray(r["present"][0], bool), np.asarray(r["is_cav"][0], bool)
                    share.append(float((pr & cav).sum() / max(1, pr.sum())))
                share = np.array(share)
                hres[split] = {"n": n, "worlds": len(worlds), "wrong_role_worlds": wrong,
                               "by_design_share": {str(k2): v for k2, v in sorted(by_pen.items())},
                               "decoded": int(len(share)),
                               "realized_share_q": [float(x) for x in np.quantile(share, [0, .05, .5, .95, 1])]
                               if len(share) else None,
                               "realized_share_gt_0p15": int((share > 0.15).sum())}
            res["horizons"][str(h)] = hres
            print(p.name, h, json.dumps(hres)[:400], flush=True)
        out["packs"][p.name] = res
    out["ok"] = bool(ok)
    out["finished"] = time.strftime("%F %T")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
