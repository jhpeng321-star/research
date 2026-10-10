"""Owner-approved pack_v3 fix #1: rebuild the pack_v3 identity sidecar and the
LPWO v3 window lists (CPU-only, metadata/mmap; no training, GPU, SUMO or wall).

Inputs (read-only):
  * E:/rd-wt/ws-wallc1/_out/lpwo_v2/{BUILD_RECEIPT,windows_train_noPI_noWall,
    windows_val_noPI_noWall,excluded_runs_PI_wall}.json
  * E:/research-dense/protocols/holdout_exclusions.json  (holdout guard)
  * E:/rd-wt/ws-wallc1/_out/wallc1_states_v2/{wall_same,wall_pilot,
    seen_controls}.json  (eval sets to stay disjoint from)
  * D:/rd-staging/wmn1_feat/pack_v3/manifest.json + outcomes.parquet (metadata)
  * E:/rd-wt/ws-p2/_ext/identities  (verified local pack_v3 identity source)

Outputs (new, under E:/rd-wt/ws-packv3fix/_out):
  * lpwo_v3/ window lists + BUILD_RECEIPT
  * sidecars_packv3_v2/identities + BUILD_RECEIPT

CAL / HOLDOUT / chunk-5 / J4B / INC1 / sealed contents are never opened: the
holdout exclusions are matched by world id only, and the sealed world ids are
not enumerated here.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path("E:/rd-wt/ws-packv3fix")
LPWO_V2 = Path("E:/rd-wt/ws-wallc1/_out/lpwo_v2")
WALLC1_STATES = Path("E:/rd-wt/ws-wallc1/_out/wallc1_states_v2")
HOLDOUT_EXCL = Path("E:/research-dense/protocols/holdout_exclusions.json")
PACK_V3 = Path("D:/rd-staging/wmn1_feat/pack_v3")
IDENTITY_SRC = Path("E:/rd-wt/ws-p2/_ext/identities")
OUTCOMES_SRC = Path("D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z")
OUT = WS / "_out"
LPWO_V3 = OUT / "lpwo_v3"
SIDECAR_V2 = OUT / "sidecars_packv3_v2"

WINDOW_SCHEMA = "wm_s_lpo_windows/1"
SIDECAR_SCHEMA = "wm_n1_train_identity_sidecar/1"
# Worlds whose contents must never be opened and which must be excluded.
SEALED_WORLD_TOKENS = ("chunk-5", "chunk5", "j4b", "inc1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prefix(record: dict) -> str:
    return str(record.get("batch") or record.get("chunk") or "")


def world_of(record: dict) -> str:
    return str(record.get("world"))


def counts(records: list[dict]) -> dict:
    by_pack = Counter(str(r.get("pack")) for r in records)
    by_policy = Counter(str(r.get("policy")) for r in records)
    by_batch = Counter(prefix(r) for r in records)
    by_world = Counter(world_of(r) for r in records)
    return {"total": len(records), "by_pack": dict(sorted(by_pack.items())),
            "by_policy": dict(sorted(by_policy.items())),
            "by_batch": dict(sorted(by_batch.items())),
            "n_worlds": len(by_world)}


def eval_worlds(path: Path) -> dict:
    """Worlds in one wms_eval_states file (regex; never materialises the payload)."""
    text = path.read_text(encoding="utf-8")
    worlds = sorted(set(re.findall(r'"world":"([^"]+)"', text)))
    keys = re.findall(r'"key":"([^"]+)"', text)
    return {"file": str(path), "sha256": sha256_file(path),
            "n_keys": len(keys), "n_worlds": len(worlds), "worlds": worlds}


def build_sidecar() -> dict:
    if SIDECAR_V2.exists():
        raise SystemExit(f"refusing to overwrite {SIDECAR_V2}")
    dst = SIDECAR_V2 / "identities"
    dst.mkdir(parents=True)
    names = ["identities.npy", "global_positions.npy", "vehicle_map.json",
             "TRAIN_IDENTITY_RECEIPT.json"]
    src_hashes, copied = {}, []
    for name in names:
        src = IDENTITY_SRC / name
        if not src.is_file():
            raise SystemExit(f"identity source missing: {src}")
        src_hashes[name] = sha256_file(src)
        shutil.copy2(src, dst / name)
        copied.append(name)
        # copy must be byte-identical
        if sha256_file(dst / name) != src_hashes[name]:
            raise SystemExit(f"copy hash mismatch: {name}")
    # TRAIN-only outcomes sidecar (the mixed pack_v3 table must never be opened).
    outc_dst = SIDECAR_V2 / "outcomes"
    outc_dst.mkdir()
    outc_names = ["base_train_outcomes.parquet", "overlay_outcomes.parquet",
                  "TRAIN_ISOLATION_RECEIPT.json", "SHA256SUMS"]
    outc_hashes, outc_copied = {}, []
    for name in outc_names:
        src = OUTCOMES_SRC / name
        if not src.is_file():
            raise SystemExit(f"outcomes source missing: {src}")
        outc_hashes[name] = sha256_file(src)
        shutil.copy2(src, outc_dst / name)
        outc_copied.append(name)
        if sha256_file(outc_dst / name) != outc_hashes[name]:
            raise SystemExit(f"outcomes copy hash mismatch: {name}")
    outc_receipt = load_json(OUTCOMES_SRC / "TRAIN_ISOLATION_RECEIPT.json")
    if (outc_receipt.get("split") != "TRAIN"
            or outc_receipt.get("pack", {}).get("manifest_sha256")
            != sha256_file(PACK_V3 / "manifest.json")):
        raise SystemExit("outcomes sidecar is not TRAIN or not bound to pack_v3")
    receipt = load_json(IDENTITY_SRC / "TRAIN_IDENTITY_RECEIPT.json")
    ids = np.load(dst / "identities.npy", mmap_mode="r", allow_pickle=False)
    positions = np.load(dst / "global_positions.npy", mmap_mode="r", allow_pickle=False)
    vehicle_map = load_json(dst / "vehicle_map.json")
    manifest = load_json(PACK_V3 / "manifest.json")
    train_pos = sorted(i for i, it in enumerate(manifest["items"])
                       if str(it.get("split")).upper() == "TRAIN")
    pos_set = set(int(p) for p in positions.tolist())
    coverage_ok = pos_set == set(train_pos)
    schema_ok = (receipt.get("schema") == SIDECAR_SCHEMA
                 and receipt.get("split") == "TRAIN")
    shape_ok = (ids.ndim == 2 and ids.dtype == np.int64
                and ids.shape[1] == int(manifest["k_max"])
                and positions.shape == (ids.shape[0],))
    if not (coverage_ok and schema_ok and shape_ok):
        raise SystemExit("sidecar validation failed: "
                         f"coverage={coverage_ok} schema={schema_ok} shape={shape_ok}")
    build = {
        "schema": "wm_n1_packv3_identity_build/1",
        "task": "PACKV3-FIX-01 correct pack_v3 TRAIN identity sidecar",
        "metadata_only": True,
        "source": {"path": str(IDENTITY_SRC),
                   "files_sha256": src_hashes,
                   "TRAIN_IDENTITY_RECEIPT_sha256":
                       sha256_file(IDENTITY_SRC / "TRAIN_IDENTITY_RECEIPT.json")},
        "pack": {"path": str(PACK_V3),
                 "manifest_sha256": sha256_file(PACK_V3 / "manifest.json"),
                 "k_max": int(manifest["k_max"]),
                 "n_manifest_items": len(manifest["items"])},
        "output": {"path": str(dst), "files": copied,
                   "files_sha256": {n: sha256_file(dst / n) for n in copied}},
        "outcomes": {
            "path": str(outc_dst), "source": str(OUTCOMES_SRC),
            "files": outc_copied,
            "source_files_sha256": outc_hashes,
            "output_files_sha256": {n: sha256_file(outc_dst / n)
                                    for n in outc_copied},
            "schema": outc_receipt.get("schema"), "split": outc_receipt.get("split"),
            "pack_manifest_sha256": outc_receipt.get("pack", {}).get("manifest_sha256"),
            "n_items_train": outc_receipt.get("pack", {}).get("n_items_train"),
            "n_rows": outc_receipt.get("sidecar", {}).get("n_rows"),
        },
        "rows": int(ids.shape[0]),
        "slot_width": int(ids.shape[1]),
        "n_vehicles": len(vehicle_map),
        "coverage": {"train_manifest_positions": len(train_pos),
                     "sidecar_positions": len(pos_set),
                     "train_positions_covered": len(pos_set & set(train_pos)),
                     "non_train_positions": len(pos_set - set(train_pos)),
                     "exact_match": coverage_ok},
        "vehicle_map_sample": vehicle_map[:2],
        "checks": {"schema": schema_ok, "shape": shape_ok, "coverage": coverage_ok},
    }
    (SIDECAR_V2 / "BUILD_RECEIPT.json").write_text(
        json.dumps(build, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return build


def build_lists() -> dict:
    if LPWO_V3.exists():
        raise SystemExit(f"refusing to overwrite {LPWO_V3}")
    LPWO_V3.mkdir(parents=True)
    lpwo_receipt = load_json(LPWO_V2 / "BUILD_RECEIPT.json")
    inputs = ["windows_train_noPI_noWall.json", "windows_val_noPI_noWall.json",
              "excluded_runs_PI_wall.json", "wall_removed_windows.json",
              "BUILD_RECEIPT.json"]
    input_sha = {n: sha256_file(LPWO_V2 / n) for n in inputs}
    # The LPWO receipt hashes the four artifacts it wrote.
    for name, want in lpwo_receipt["sha256"].items():
        if input_sha.get(name) != want:
            raise SystemExit(f"{name} sha256 {input_sha.get(name)} != LPWO receipt {want}")
    train = load_json(LPWO_V2 / "windows_train_noPI_noWall.json")
    val = load_json(LPWO_V2 / "windows_val_noPI_noWall.json")
    excluded = load_json(LPWO_V2 / "excluded_runs_PI_wall.json")
    removed = load_json(LPWO_V2 / "wall_removed_windows.json")

    holdout = load_json(HOLDOUT_EXCL)
    hold_excluded = {str(r["world"]) for r in holdout["excluded"]}
    hold_disclosed = {str(w) for w in holdout.get("kept_disclosed", [])}

    evals = {name: eval_worlds(WALLC1_STATES / f"{name}.json")
             for name in ("wall_same", "wall_pilot", "seen_controls")}
    eval_union = set()
    for info in evals.values():
        eval_union |= set(info["worlds"])

    excluded_keys = set(str(k) for k in excluded["keys"])

    def sealed_or_holdout(w: dict) -> str | None:
        world = world_of(w)
        low = world.lower()
        if any(tok in low for tok in SEALED_WORLD_TOKENS):
            return "sealed_world_token"
        if world in hold_excluded:
            return "holdout_excluded_world"
        if world in eval_union:
            return "eval_set_world"
        return None

    # Rule (a): keep the LPWO decision exactly (no PI/wall exposure). Re-assert it.
    PI_ACTIONS = frozenset({"pi", "fleet-pi", "pi-external-u", "fleet-pi-external-u"})

    def arm_violation(w: dict) -> bool:
        policy = str(w.get("policy") or "").strip().lower()
        arm = str(w.get("arm") or "").strip().lower()
        batch = prefix(w).lower()
        return (policy in PI_ACTIONS or arm in PI_ACTIONS
                or "wall" in policy or "wall" in arm or "wall" in batch)

    role_violations = [w for w in train["windows"] + val["windows"]
                       if str(w.get("split")) != "TRAIN"]

    keep_train, keep_val, drop = [], [], []
    for origin, payload in (("train", train), ("val", val)):
        for w in payload["windows"]:
            if arm_violation(w):
                raise SystemExit(f"LPWO list retains a PI/wall arm: {w['key']}")
            if str(w["key"]) in excluded_keys:
                raise SystemExit(f"LPWO list retains an excluded key: {w['key']}")
            reason = sealed_or_holdout(w)
            if reason:
                drop.append({"key": w["key"], "pack": w.get("pack"),
                             "world": world_of(w), "origin_list": origin,
                             "reason": reason})
                continue
            (keep_train if origin == "train" else keep_val).append(w)

    train_worlds = {world_of(w) for w in keep_train}
    val_worlds = {world_of(w) for w in keep_val}
    overlap = {
        "train_val": sorted(train_worlds & val_worlds),
        "train_eval": sorted(train_worlds & eval_union),
        "val_eval": sorted(val_worlds & eval_union),
        "train_holdout_excluded": sorted(train_worlds & hold_excluded),
        "val_holdout_excluded": sorted(val_worlds & hold_excluded),
        "train_eval_by_set": {name: sorted(train_worlds & set(i["worlds"]))
                              for name, i in evals.items()},
    }
    if overlap["train_val"] or overlap["train_eval"] or overlap["val_eval"]:
        raise SystemExit(f"disjointness violated: {overlap}")

    def out_list(name, records, rule):
        return {"schema": WINDOW_SCHEMA, "list": name, "rule": rule,
                "source": {"windows": "E:/rd-wt/ws-wallc1/_out/lpwo_v2",
                           "sha256": input_sha},
                "pack_names": sorted({str(w["pack"]) for w in records}),
                "n_windows": len(records), "counts": counts(records),
                "windows": records}

    train_out = out_list(
        "windows_train_noPI_noWall",
        keep_train,
        "LPWO v2 TRAIN windows minus HOLDOUT-excluded/CAL/sealed worlds minus "
        "wallc1 eval-set worlds; PI and wall exposure excluded exactly as LPWO v2")
    val_out = out_list(
        "windows_val_noPI_noWall", keep_val,
        "LPWO v2 TRAIN-scoped validation windows, same world exclusions")
    combined_keys = sorted(excluded_keys | {d["key"] for d in drop})
    excluded_out = {
        "schema": WINDOW_SCHEMA, "list": "excluded_runs_PI_wall_v3",
        "rule": "LPWO v2 PI+wall excluded run keys (unchanged) plus every window "
                "dropped from the v3 lists for a holdout/eval/sealed reason",
        "source": {"excluded_runs_PI_wall.json": input_sha["excluded_runs_PI_wall.json"]},
        "n_runs": len(combined_keys),
        "n_pi_wall_runs_inherited": len(excluded_keys),
        "n_holdout_eval_windows_dropped": len(drop),
        "n_dropped_keys_already_excluded": len(excluded_keys) + len(drop) - len(combined_keys),
        "keys": combined_keys,
    }
    dropped_out = {
        "schema": "wm_s_lpwo_v3_dropped_windows/1",
        "rule": "windows removed from the LPWO v2 lists for holdout/eval reasons",
        "n_windows": len(drop),
        "by_reason": dict(sorted(Counter(d["reason"] for d in drop).items())),
        "windows": drop,
    }
    paths = {
        "windows_train_noPI_noWall.json": train_out,
        "windows_val_noPI_noWall.json": val_out,
        "excluded_runs_PI_wall_v3.json": excluded_out,
        "holdout_eval_dropped_windows.json": dropped_out,
    }
    hashes = {}
    for name, payload in paths.items():
        (LPWO_V3 / name).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                                    encoding="utf-8")
        hashes[name] = sha256_file(LPWO_V3 / name)
    (LPWO_V3 / "eval_worlds.json").write_text(
        json.dumps({"schema": "wm_s_lpwo_v3_eval_worlds/1",
                    "sets": evals, "n_union_worlds": len(eval_union)},
                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    hashes["eval_worlds.json"] = sha256_file(LPWO_V3 / "eval_worlds.json")
    receipt = {
        "schema": "wm_s_lpwo_v3_receipt/1",
        "task": "PACKV3-FIX-01 LPWO v3 window lists (pack_v3 contributing)",
        "metadata_only": True,
        "simulation_s": 0,
        "sealed_never_read": ["CAL", "CAL_SPLIT_A4.json", "HOLDOUT", "chunk-5",
                              "J4B", "INC1"],
        "inputs_sha256": input_sha,
        "source_root": str(LPWO_V2),
        "holdout_exclusions": {"path": str(HOLDOUT_EXCL),
                               "sha256": sha256_file(HOLDOUT_EXCL),
                               "n_excluded_worlds": len(hold_excluded),
                               "n_kept_disclosed_worlds": len(hold_disclosed),
                               "exact_match": len(hold_excluded) == int(
                                   holdout.get("expected_excluded_count", -1))},
        "eval_sets": {name: {"path": i["file"], "sha256": i["sha256"],
                             "n_worlds": i["n_worlds"], "n_keys": i["n_keys"]}
                      for name, i in evals.items()},
        "n_train_before": int(train["n_windows"]),
        "n_val_before": int(val["n_windows"]),
        "n_train": len(keep_train), "n_val": len(keep_val),
        "train_counts": train_out["counts"], "val_counts": val_out["counts"],
        "n_dropped_holdout_eval": len(drop),
        "dropped_by_reason": dict(sorted(Counter(d["reason"] for d in drop).items())),
        "removed_by_reason_lpwo_v2": lpwo_receipt.get("removed_by_reason"),
        "overlap": overlap,
        "role_violations_non_train": len(role_violations),
        "excluded_keys": excluded_out["n_runs"],
        "sha256": hashes,
    }
    (LPWO_V3 / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv=None) -> int:
    which = (argv or sys.argv[1:] or ["all"])[0]
    out = {}
    if which in ("all", "sidecar"):
        out["sidecar"] = build_sidecar()
    if which in ("all", "lists"):
        out["lists"] = build_lists()
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
