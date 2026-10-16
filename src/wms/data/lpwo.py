"""Leave-PI-AND-WALL-out (LPWO) window lists for WM-S (metadata only).

Starts from the P5 leave-PI-out lists (``windows_train_noPI.json``,
``windows_val.json``, ``pi_excluded_runs.json``) and removes every window whose
run had ANY CAV under wall ownership, so a model trained on the result has
never seen the wall group controller.

Rule (schedule/event aware, fail closed):

* ``walls_pilot_chunk4`` / ``policy == wall`` runs: the run directory's
  ``branch_record.json`` + ``wall_events.jsonl`` are read
  (``read_wall_timeline``, injected).  Every wall-labelled run is removed - engaged,
  owned-but-unassembled, not-engaged (``not_feasible``: no CAV ever owned, but the
  controller label ``wall`` would still be seen) or unverifiable.  The reason is
  recorded per run so the engaged subset can be used as the unseen-wall test set.
* every other pack_v3 run: its ``switch_schedule.json`` (when recoverable) is
  scanned for any wall action and its run dir for ``wall_events.jsonl``; a
  fixed/switching run whose schedule cannot be recovered is removed fail-closed.
   every retained run requires a verified schedule; policy identity alone is
   insufficient. Missing or malformed schedules are excluded fail-closed.
* pack_dspeed / pack_damping: PI-free arm sets are re-asserted; ``wall`` in any
  policy/arm string removes the window.

Nothing here reads an array, an outcome row, CAL/HOLDOUT/chunk-5/J4B/INC1 or any
sealed source; no SUMO, no GPU.  One small json at a time.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from wms.config.presets import repo_root
from wms.data import window_lists as lpo_build  # leave-PI-out builder (schedule resolution)

LPWO_SCHEMA = lpo_build.LPO_SCHEMA            # same window-list schema (trainer-compatible)
RUNS_ROOT = repo_root()
WALL_BATCH_TOKENS = ("wall",)
WallTimelineReader = Callable[..., dict]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_wall_label(*values) -> bool:
    return any(any(tok in str(v or "").strip().lower() for tok in WALL_BATCH_TOKENS)
               for v in values)


def schedule_actions(path: Path) -> dict[str, int]:
    return dict(lpo_build.scan_schedule(path)["actions"])


def has_wall_action(actions: dict[str, int]) -> bool:
    return any("wall" in str(a).lower() for a in actions)


def load_pack_rows(index_path: str | Path, guard) -> dict[str, dict]:
    index = guard.read_json(index_path)
    return {str(r["key"]): r for r in index["runs"]}


def run_dir_of(row: dict, root: Path = RUNS_ROOT) -> Optional[Path]:
    root_rel = row.get("root_rel")
    if not root_rel:
        return None
    if Path(root_rel).is_absolute() or ".." in Path(root_rel).parts:
        raise ValueError("root_rel must be a relative campaign path")
    run = row.get("run") or str(row["key"]).split("/", 1)[1]
    if Path(run).name != run:
        raise ValueError("run must be a basename")
    in_tree = Path(root) / root_rel / run
    if not in_tree.exists() and Path(root_rel).name in lpo_build.DRIVE_ONLY_CAMPAIGNS:
        # Drive-only raw campaign (``python -m wms fetch get <campaign>``) lands in data_cache/
        fetched = lpo_build.drive_only_root(Path(root_rel).name) / run
        if fetched.exists():
            return fetched
    return in_tree


def classify_window(window: dict, rows: dict[str, dict], *, root: Path,
                    guard, read_wall_timeline: WallTimelineReader) -> dict:
    """Wall verdict for one window: ``{"wall": bool, "reason", "verified_by", ...}``.

    ``read_wall_timeline(run_dir, guard=guard)`` is injected (the reader lives with the
    wall evaluation code; the data layer must not import it).
    """
    pack = str(window.get("pack"))
    policy = str(window.get("policy") or "").strip().lower()
    arm = str(window.get("arm") or "")
    key = str(window.get("key"))
    wall_label = is_wall_label(policy, arm, window.get("batch"))
    guard.check_record(window)
    row = rows.get(key)
    # Additional packs carry their campaign binding in their own staged INDEX.
    if row is None:
        return {"wall": True, "reason": "unverifiable_missing_pack_row",
                "verified_by": None}
    guard.check_record(row)
    for field in ("world", "arm", "policy", "t0_s"):
        if row.get(field) != window.get(field):
            return {"wall": True, "reason": "unverifiable_window_index_mismatch",
                    "verified_by": None}
    run_dir = run_dir_of(row, root)
    if run_dir is not None:
        guard.check_path(run_dir)

    if wall_label:
        if run_dir is None or not run_dir.is_dir():
            return {"wall": True, "reason": "wall_unverifiable_no_run_dir",
                    "verified_by": None, "timeline": None}
        timeline = read_wall_timeline(run_dir, guard=guard)
        if timeline["status"] == "unverifiable":
            reason = "wall_unverifiable"
        elif timeline["engaged"]:
            reason = "wall_engaged"
        elif timeline["owned"]:
            reason = "wall_owned_not_assembled"
        else:
            reason = "wall_label_not_engaged"
        return {"wall": True, "reason": reason, "verified_by": "wall_events+branch_record",
                "timeline": timeline}

    if run_dir is None:
        return {"wall": True, "reason": "unverifiable_schedule_missing",
                "verified_by": None}
    for event_path in (run_dir / "run" / "wall_events.jsonl", run_dir / "wall_events.jsonl"):
        guard.check_path(event_path)
        if event_path.is_file():
            timeline = read_wall_timeline(run_dir, guard=guard)
            # An events file alone proves neither refusal nor absence of ownership.
            if timeline.get("owned", True):
                return {"wall": True, "reason": "wall_event_ownership_or_unverifiable",
                        "verified_by": "wall_events", "timeline": timeline}
    sched = None
    for candidate in (run_dir / "run" / "switch_schedule.json", run_dir / "switch_schedule.json"):
        guard.check_path(candidate)
        if candidate.is_file():
            sched = candidate
            break
    if sched is None:
        return {"wall": True, "reason": "unverifiable_schedule_missing", "verified_by": None}
    try:
        data = guard.read_json(sched)
        schedule = data.get("schedule", data)
        events = schedule["events"]
        if not isinstance(events, list):
            raise ValueError("events is not a list")
        actions = Counter()
        def visit(value):
            if isinstance(value, dict):
                for name, child in value.items():
                    if name in ("action", "controller", "owner", "ownership", "policy"):
                        if not isinstance(child, str) or not child.strip():
                            raise ValueError("invalid controller identity")
                        actions[child.strip().lower()] += 1
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(schedule)
        known = {"native", "idm", "damping", "circles", "retain", "direct-speed", "dspeed", "trajectory"}
        if has_wall_action(actions):
            return {"wall": True, "reason": "schedule_wall_action", "verified_by": str(sched),
                    "actions": dict(actions)}
        if (any(lpo_build.is_pi_action(a) for a in actions)
                or lpo_build.is_pi_policy(policy) or lpo_build.is_pi_action(arm)):
            return {"wall": True, "reason": "schedule_pi_action", "verified_by": str(sched)}
        if not actions or set(actions) - known:
            raise ValueError("missing or unknown controller identities")
        if pack in lpo_build.PI_FREE_PACKS and arm not in lpo_build.PI_FREE_PACKS[pack]:
            raise ValueError("unexpected campaign arm")
    except (OSError, ValueError, KeyError, TypeError):
        return {"wall": True, "reason": "unverifiable_schedule_malformed", "verified_by": str(sched)}
    return {"wall": False, "reason": None, "verified_by": str(sched),
            "schedule_sha256": sha256_file(sched)}


def _counts(records: list[dict]) -> dict:
    by_policy = Counter(str(r.get("policy")) for r in records)
    by_pack = Counter(str(r.get("pack")) for r in records)
    by_world = Counter(str(r.get("world")) for r in records)
    return {"total": len(records), "by_policy": dict(sorted(by_policy.items())),
            "by_pack": dict(sorted(by_pack.items())), "n_worlds": len(by_world),
            "by_world": dict(sorted(by_world.items()))}


def _write_json(path: Path, payload) -> str:
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
    return sha256_file(path)


def _timeline_row(window: dict, verdict: dict, origin: str) -> dict:
    row = {"key": window["key"], "pack": window.get("pack"), "run": window.get("run"),
           "world": window.get("world"), "arm": window.get("arm"),
           "policy": window.get("policy"), "batch": window.get("batch"),
           "chunk": window.get("chunk"), "t0_s": window.get("t0_s"),
           "n_ticks": window.get("n_ticks"), "split": window.get("split"),
           "origin_list": origin, "reason": verdict["reason"],
           "verified_by": verdict.get("verified_by")}
    tl = verdict.get("timeline")
    if tl:
        row["timeline"] = {k: v for k, v in tl.items() if k not in ("run_dir",)}
    return row


def build(lpo_dir: str | Path, out_dir: str | Path, *, pack_v3_index: str | Path,
          root: str | Path, guard, read_wall_timeline: WallTimelineReader,
          extra_indexes: dict | None = None) -> dict:
    """Build the LPWO artifacts; returns the receipt."""
    lpo_dir, out_dir = Path(lpo_dir), Path(out_dir)
    guard.check_path(out_dir)
    if out_dir.exists():
        raise ValueError("output directory must be new")
    lpo_receipt = guard.read_json(lpo_dir / "BUILD_RECEIPT.json")
    input_sha = {}
    for name in ("windows_train_noPI.json", "windows_val.json", "pi_excluded_runs.json"):
        guard.check_path(lpo_dir / name)
        got = sha256_file(lpo_dir / name)
        want = lpo_receipt["sha256"][name]
        if got != want:
            raise ValueError(f"{name} sha256 {got} != LPO receipt {want} (stale/modified input)")
        input_sha[name] = got

    train = guard.read_json(lpo_dir / "windows_train_noPI.json")
    val = guard.read_json(lpo_dir / "windows_val.json")
    pi_excl = guard.read_json(lpo_dir / "pi_excluded_runs.json")
    indexes = {"pack_v3": pack_v3_index, **(extra_indexes or {})}
    rows_by_pack = {pack: load_pack_rows(path, guard) for pack, path in indexes.items()}
    rows = rows_by_pack["pack_v3"]
    for window in train["windows"] + val["windows"]:
        guard.check_record(window)
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "BUILD_STARTED.json", {"schema": "wm_s_lpwo_started/1",
                "inputs_sha256": input_sha, "simulation_s": 0, "training_steps": 0})

    keep = {"train": [], "val": []}
    removed: list[dict] = []
    verified_counter: Counter = Counter()
    verdicts = {}
    for window in train["windows"] + val["windows"]:
        identity = (window["pack"], window["key"])
        verdict = classify_window(window, rows_by_pack.get(window["pack"], {}),
                                  root=Path(root), guard=guard,
                                  read_wall_timeline=read_wall_timeline)
        if identity not in verdicts or verdict["wall"]:
            verdicts[identity] = verdict
    inherited_excluded = set(pi_excl["keys"])
    for origin, payload in (("train", train), ("val", val)):
        for window in payload["windows"]:
            verdict = verdicts[(window["pack"], window["key"])]
            if window["key"] in inherited_excluded:
                verdict = {"wall": True, "reason": "inherited_pi_exclusion", "verified_by": "LPO"}
            if verdict["wall"]:
                removed.append(_timeline_row(window, verdict, origin))
            else:
                keep[origin].append(window)
                verified_counter[(window.get("pack"), verdict["verified_by"])] += 1

    # PI exclusions may already cover some wall-owned runs; retain their keys.
    corpus_wall = {k for k, r in rows.items()
                   if str(r.get("split")).upper() == "TRAIN"
                   and is_wall_label(r.get("policy"), r.get("arm"), r.get("batch"))}
    removed_keys = {r["key"] for r in removed if is_wall_label(r["policy"], r["arm"], r["batch"])}
    missing_wall = sorted(corpus_wall - removed_keys - set(pi_excl["keys"]))
    if missing_wall:
        raise ValueError(f"{len(missing_wall)} TRAIN wall runs of the corpus are in neither "
                         f"LPO list nor removed (e.g. {missing_wall[:3]})")

    excluded_keys = sorted(set(pi_excl["keys"]) | {r["key"] for r in removed}
                           | corpus_wall)
    train_out = {
        "schema": LPWO_SCHEMA, "list": "windows_train_noPI_noWall",
        "rule": ("TRAIN windows of the WM-S comparator corpus whose run had NO CAV on a "
                 "PI controller and NO CAV under wall ownership (LPO list minus every "
                 "wall-labelled/wall-owned/unverifiable-wall run)"),
        "source": {"windows_train_noPI.json": input_sha["windows_train_noPI.json"]},
        "corpus": train.get("corpus"),
        "n_windows": len(keep["train"]), "counts": _counts(keep["train"]),
        "windows": keep["train"],
    }
    val_out = {
        "schema": LPWO_SCHEMA, "list": "windows_val_noPI_noWall",
        "rule": ("LPO world-grouped validation windows minus wall windows (same "
                 "sha256(world) rule; worlds are not re-assigned)"),
        "source": {"windows_val.json": input_sha["windows_val.json"]},
        "n_windows": len(keep["val"]), "counts": _counts(keep["val"]),
        "windows": keep["val"],
    }
    excl_out = {
        "schema": LPWO_SCHEMA, "list": "excluded_runs_PI_wall",
        "rule": ("pack run keys to exclude from any WM-S TRAIN pack load: proven-PI + "
                 "unverifiable-schedule runs (P5) plus every wall-labelled/wall-owned run "
                 "of the pack_v3 TRAIN corpus"),
        "source": {"pi_excluded_runs.json": input_sha["pi_excluded_runs.json"]},
        "n_runs": len(excluded_keys), "keys": excluded_keys,
        "n_pi_unverifiable_runs": pi_excl.get("n_unverifiable"),
        "n_wall_runs": len({r["key"] for r in removed}),
        "wall_removed_by_reason": dict(sorted(Counter(r["reason"] for r in removed).items())),
        "unverifiable": pi_excl.get("unverifiable", []),
    }
    removed_out = {
        "schema": "wm_s_wall_removed_windows/1", "list": "wall_removed_windows",
        "rule": "windows removed from the LPO lists because of wall ownership/label",
        "n_windows": len(removed), "windows": removed,
    }
    hashes = {
        "windows_train_noPI_noWall.json": _write_json(out_dir / "windows_train_noPI_noWall.json", train_out),
        "windows_val_noPI_noWall.json": _write_json(out_dir / "windows_val_noPI_noWall.json", val_out),
        "excluded_runs_PI_wall.json": _write_json(out_dir / "excluded_runs_PI_wall.json", excl_out),
        "wall_removed_windows.json": _write_json(out_dir / "wall_removed_windows.json", removed_out),
    }
    receipt = {
        "schema": "wm_s_lpwo_receipt/1",
        "task": "WALL-C1 leave-PI-and-wall-out window lists",
        "metadata_only": True,
        "sealed_never_read": ["CAL", "CAL_SPLIT_A4.json", "HOLDOUT", "chunk-5", "J4B", "INC1"],
        "inputs_sha256": input_sha,
        "guard": guard.audit(),
        "schedule_evidence": {f"{w['pack']}:{w['key']}": verdicts[(w["pack"], w["key"])]
                              for w in keep["train"] + keep["val"]},
        "pack_v3_index": str(pack_v3_index),
        "n_train_before": train["n_windows"], "n_val_before": val["n_windows"],
        "n_train": len(keep["train"]), "n_val": len(keep["val"]),
        "n_removed_windows": len(removed),
        "n_removed_from_train": sum(1 for r in removed if r["origin_list"] == "train"),
        "n_removed_from_val": sum(1 for r in removed if r["origin_list"] == "val"),
        "train_by_policy_before": train["counts"]["by_policy"],
        "val_by_policy_before": val["counts"]["by_policy"],
        "train_by_policy_after": train_out["counts"]["by_policy"],
        "val_by_policy_after": val_out["counts"]["by_policy"],
        "train_by_pack_after": train_out["counts"]["by_pack"],
        "n_train_worlds_after": train_out["counts"]["n_worlds"],
        "n_val_worlds_after": val_out["counts"]["n_worlds"],
        "removed_by_reason": dict(sorted(Counter(r["reason"] for r in removed).items())),
        "removed_by_origin_and_reason": {
            f"{o}/{rs}": n for (o, rs), n in sorted(Counter(
                (r["origin_list"], r["reason"]) for r in removed).items())},
        "kept_verification": {f"{p}|{v}": n for (p, v), n in sorted(
            verified_counter.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))},
        "n_excluded_run_keys": len(excluded_keys),
        "n_pi_excluded_keys_inherited": pi_excl["n_runs"],
        "n_wall_run_keys_added": len(set(excluded_keys) - set(pi_excl["keys"])),
        "wall_runs_in_corpus_train": len(corpus_wall),
        "fail_closed_unverifiable_wall_runs": sorted(
            r["key"] for r in removed if "unverifiable" in str(r["reason"])),
        "sha256": hashes,
    }
    # train + val + removed must reproduce the LPO lists exactly
    assert receipt["n_train"] + receipt["n_removed_from_train"] == train["n_windows"]
    assert receipt["n_val"] + receipt["n_removed_from_val"] == val["n_windows"]
    _write_json(out_dir / "BUILD_RECEIPT.json", receipt)
    return receipt


def validate(out_dir: str | Path, lpo_dir: str | Path, guard) -> dict:
    """Post-hoc fail-closed check of a built LPWO directory (no data reads)."""
    out_dir, lpo_dir = Path(out_dir), Path(lpo_dir)
    for directory, names in ((out_dir, ("BUILD_RECEIPT.json", "windows_train_noPI_noWall.json",
                                       "windows_val_noPI_noWall.json", "excluded_runs_PI_wall.json")),
                             (lpo_dir, ("windows_train_noPI.json", "windows_val.json", "pi_excluded_runs.json"))):
        for name in names:
            guard.check_path(directory / name)
    receipt = guard.read_json(out_dir / "BUILD_RECEIPT.json")
    for name, want in receipt["sha256"].items():
        guard.check_path(out_dir / name)
        if sha256_file(out_dir / name) != want:
            raise ValueError(f"LPWO receipt hash mismatch: {name}")
    tr = json.loads((out_dir / "windows_train_noPI_noWall.json").read_text(encoding="utf-8"))
    va = json.loads((out_dir / "windows_val_noPI_noWall.json").read_text(encoding="utf-8"))
    ex = json.loads((out_dir / "excluded_runs_PI_wall.json").read_text(encoding="utf-8"))
    old_tr = json.loads((lpo_dir / "windows_train_noPI.json").read_text(encoding="utf-8"))
    old_va = json.loads((lpo_dir / "windows_val.json").read_text(encoding="utf-8"))
    excluded = set(ex["keys"])
    problems = []
    for name, payload in (("train", tr), ("val", va)):
        for w in payload["windows"]:
            guard.check_record(w)
            if is_wall_label(w.get("policy"), w.get("arm"), w.get("batch")):
                problems.append(f"{name}: wall-labelled window kept {w['key']}")
            if w["key"] in excluded:
                problems.append(f"{name}: excluded key kept {w['key']}")
    old_keys = {(w["pack"], w["key"]) for w in old_tr["windows"]} | \
        {(w["pack"], w["key"]) for w in old_va["windows"]}
    new_keys = {(w["pack"], w["key"]) for w in tr["windows"]} | \
        {(w["pack"], w["key"]) for w in va["windows"]}
    if not new_keys <= old_keys:
        problems.append("new lists contain windows not in the LPO lists")
    if {w["world"] for w in tr["windows"]} & {w["world"] for w in va["windows"]}:
        problems.append("train/val share a world")
    if not set(json.loads((lpo_dir / "pi_excluded_runs.json").read_text(
            encoding="utf-8"))["keys"]) <= excluded:
        problems.append("excluded list lost PI keys")
    return {"ok": not problems, "problems": problems,
            "n_train": tr["n_windows"], "n_val": va["n_windows"],
            "n_excluded": len(excluded)}


build_lpwo = build  # architecture name (docs/ARCHITECTURE_V4.md 2.3)
