"""Build the C4 world-model window lists from the read-only data cache.

C4 trains the world model on NATIVE driving plus RANDOM direct-output CAV
actions only (owner rule: no designed controllers in WM training data):

* pack_v3 TRAIN windows of the ``native`` / ``native_base`` arms (the lpwo_v3
  lists already exclude PI and wall arms; the damping-``base`` arm of the same
  lists is additionally dropped here);
* all ``arm0..arm3`` (sampled target-speed trajectory) windows of pack_dspeed;
* the ``direct-mix`` arm of the switching pack's window list (the ``switching``
  arm items, whose CAVs contain damping/PI segments, are excluded).

Writes three list documents under ``runs/active/c4_wm_lists/`` and prints a
census (per source, per CAV penetration) as JSON.  Read-only inputs.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path("E:/research-dense/data_cache")
OUT = ROOT / "runs" / "active" / "c4_wm_lists"

NATIVE_ARMS = ("native", "native_base")
DSPEED_ARMS = ("arm0", "arm1", "arm2", "arm3")
_PEN = re.compile(r"p(\d+)")


def penetration(world: str) -> str:
    match = _PEN.search(str(world))
    return f"p{int(match.group(1))}" if match else "?"


def _write(doc: dict, windows: list[dict], name: str, rule: str) -> Path:
    out = {**doc, "windows": windows, "n_windows": len(windows), "rule": rule}
    path = OUT / name
    path.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return path


def census(rows: list[dict], label: str) -> dict:
    by_pen = Counter(penetration(w["world"]) for w in rows)
    return {"source": label, "n_windows": len(rows),
            "by_penetration": dict(sorted(by_pen.items())),
            "by_arm": dict(sorted(Counter(str(w.get("arm")) for w in rows).items()))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for name, filename in (("train", "windows_train_native_c4.json"),
                           ("val", "windows_val_native_c4.json")):
        doc = json.loads((DATA / "lpwo_v3" / f"windows_{name}_noPI_noWall.json")
                         .read_text(encoding="utf-8"))
        keep = [w for w in doc["windows"]
                if (str(w.get("pack")) == "pack_v3" and str(w.get("arm")) in NATIVE_ARMS)
                or (str(w.get("pack")) == "pack_dspeed" and str(w.get("arm")) in DSPEED_ARMS)]
        written[name] = _write(doc, keep, filename,
                               "C4 native selection: lpwo_v3 noPI/noWall list restricted to "
                               "pack_v3 {native, native_base} + pack_dspeed arm0..arm3 "
                               "(damping/base arm and the pack_damping windows dropped)")

    sw_doc = json.loads((DATA / "wmn1_switching_train" / "windows_train_hetdata.json")
                        .read_text(encoding="utf-8"))
    keep = [w for w in sw_doc["windows"] if str(w.get("arm")) == "direct-mix"]
    written["directmix"] = _write(sw_doc, keep, "windows_train_directmix.json",
                                  "C4 direct-output selection: switching-pack window list "
                                  "restricted to arm=direct-mix (arm=switching excluded)")
    # Explicit TRAIN-scoped world split of the direct-mix list (r3 semantics: 10% of
    # worlds, seed 0) so [data.switching] can name both lists.  Done here because the
    # library's implicit split labels its documents TRAIN_HOLDOUT_*, which its own
    # TRAIN-scope check refuses (unused since R3: R4/R5 chains have no switching pack).
    from wms.data.val_split import holdout_worlds
    held = set(holdout_worlds({"windows": keep}, frac=0.1, seed=0))
    written["directmix_train"] = _write(
        sw_doc, [w for w in keep if str(w["world"]) not in held],
        "windows_train_directmix_trainpart.json",
        "C4 direct-mix TRAIN part (world split frac 0.1, seed 0; TRAIN-scoped)")
    written["directmix_val"] = _write(
        sw_doc, [w for w in keep if str(w["world"]) in held],
        "windows_val_directmix.json",
        "C4 direct-mix monitoring part (world split frac 0.1, seed 0; TRAIN-scoped)")

    # ---- census: window lists + pack manifests + timeline index (metadata only)
    train = json.loads(written["train"].read_text(encoding="utf-8"))["windows"]
    val = json.loads(written["val"].read_text(encoding="utf-8"))["windows"]
    direct = json.loads(written["directmix_train"].read_text(encoding="utf-8"))["windows"]
    direct_val = json.loads(written["directmix_val"].read_text(encoding="utf-8"))["windows"]

    manifest_v3 = json.loads((DATA / "pack_v3" / "manifest.json").read_text(encoding="utf-8"))
    items = [i for i in manifest_v3["items"] if str(i.get("split")) == "TRAIN"]
    native_items = [i for i in items if str(i.get("arm")) in NATIVE_ARMS]
    dropped_cal = len([i for i in manifest_v3["items"]
                       if str(i.get("split")) != "TRAIN"])

    index = json.loads((DATA / "wmn1_switching_train" / "ACTION_TIMELINE_INDEX_train.json")
                       .read_text(encoding="utf-8"))
    dm_items = [i for i in index["items"] if str(i.get("arm")) == "direct-mix"]
    sw_items = [i for i in index["items"] if str(i.get("arm")) == "switching"]

    report = {
        "schema": "c4_wm_lists/1",
        "generated": [str(p.relative_to(ROOT)) for p in written.values()],
        "windows": {
            "native_train": census(train, "pack_v3 native/native_base + pack_dspeed arm0-3 (TRAIN)"),
            "native_val": census(val, "same selection, validation list"),
            "directmix_train": census(direct, "switching pack arm=direct-mix, TRAIN part (world split 0.1/0)"),
            "directmix_val": census(direct_val, "switching pack arm=direct-mix, monitoring part"),
        },
        "items": {
            "pack_v3_train_items": len(items),
            "pack_v3_native_arm_items": census(native_items, "pack_v3 TRAIN native arms")
            if native_items else {},
            "pack_v3_calibration_items_excluded": dropped_cal,
            "switching_index_direct_mix_items": len(dm_items),
            "switching_index_switching_items_excluded": len(sw_items),
            "direct_mix_by_penetration": dict(sorted(
                Counter(penetration(i["world"]) for i in dm_items).items())),
        },
    }
    print(json.dumps(report, indent=1, sort_keys=True))
    (OUT / "census.json").write_text(json.dumps(report, indent=1, sort_keys=True),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
