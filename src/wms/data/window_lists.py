"""WM-S leave-PI-out (LPO) training-window builder (P5 generalisation prep).

Builds the training/validation window lists for the WM-S leave-PI-out
generalisation run from metadata only:

* every TRAIN window whose run had ANY CAV on the PI controller is excluded
  (fleet-PI arms, WM-72 ``__pi`` switching arms, and fixed/switching arms whose
  ``switch_schedule.json`` events contain a PI action for any CAV);
* native / damping / wall rows and PI-free fixed/switching rows are kept;
* the pack_dspeed (direct-speed replay, ``arm0..3``) and pack_damping
  (``base``/``native_base``) runs contain no PI arm by campaign construction
  and are asserted to keep exactly those arm sets;
* a deterministic world-grouped ~10% validation subset is held out of the
  PI-free TRAIN windows (worlds are assigned whole, so no world leaks between
  the training and validation lists);
* the excluded PI TRAIN runs are emitted as the held-out evaluation state
  list for later scoring.

Inputs are metadata only: pack ``INDEX.json`` files and per-run
``switch_schedule.json`` files (the same schedule metadata the committed
inventory build opened; see ``output/wd01/wm_n1/inventory/opened_paths.json``).
No outcome file, no physics rows, no sealed source (CAL/HOLDOUT/chunk-5/J4B/
INC1) is read, no SUMO is started, and every array-bearing pack file is left
untouched.  One schedule file is in memory at a time.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from ..config.presets import repo_root
from ..core.errors import ConfigError
from .registry import resolve as _resolve_dataset

LPO_SCHEMA = "wm_s_lpo_windows/1"

#: PI controller identities (mirrors inventory_lib.POLICY_ALIASES pi family).
PI_ACTIONS = frozenset({"pi", "fleet-pi", "pi-external-u", "fleet-pi-external-u"})

#: Campaigns whose arm sets contain no PI controller (asserted, not assumed).
PI_FREE_PACKS = {
    "pack_dspeed": {"arm0", "arm1", "arm2", "arm3"},
    "pack_damping": {"base", "native_base"},
}

PACK_ORDER = {"pack_v3": 0, "pack_dspeed": 1, "pack_damping": 2}

#: Raw campaign trees that live on Drive only; fetch one with ``python -m wms fetch get <name>``
#: (it lands in ``data_cache/<name>``).  Dataset name == campaign directory name.
DRIVE_ONLY_CAMPAIGNS = ("campaign_stage1_chunk3", "campaign_stage1_chunk4")
CHUNK1_P40_DATASET = "raw_chunk1_p40"


def drive_only_root(name: str) -> Path:
    """Local (fetched) location of the Drive-only raw dataset ``name``.

    Resolved through the data registry (``WMS_DATA_<NAME>``, ``configs/data_registry.toml``,
    ``tools/data_registry.json``); an unregistered name falls back to ``data_cache/<name>``.
    The path may not exist: nothing here downloads.
    """
    try:
        return _resolve_dataset(name)
    except ConfigError:
        return repo_root() / "data_cache" / name


#: Relocated raw chunk1-p40 runs (superseded batch; Drive-only dataset ``raw_chunk1_p40``).
CHUNK1_P40_OVERRIDE_ROOT = drive_only_root(CHUNK1_P40_DATASET)
DEFAULT_DATA_ROOTS = (repo_root(), CHUNK1_P40_OVERRIDE_ROOT)

VAL_WORLD_MODULO = 10


def is_pi_action(action: str) -> bool:
    """True when a schedule/plan action is a PI controller action."""
    return str(action).strip().lower() in PI_ACTIONS


def is_pi_policy(policy: str) -> bool:
    return str(policy).strip().lower() in PI_ACTIONS


def val_world(world: str, *, modulo: int = VAL_WORLD_MODULO) -> bool:
    """Deterministic world-grouped validation assignment (~1/modulo worlds)."""
    digest = hashlib.sha256(str(world).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo == 0


def load_opened_schedule_map(inventory_dir: Path) -> dict[tuple[str, str], str]:
    """(campaign dir, run) -> switch_schedule.json path, from the inventory.

    Run names collide across campaign batches (e.g. ``<world>__native`` exists
    in both chunk2 and truearms_chunk2), so the map is keyed by the campaign
    directory name, never by run name alone.
    """
    data = json.loads(
        (Path(inventory_dir) / "opened_paths.json").read_text(encoding="utf-8"))
    out: dict[tuple[str, str], str] = {}
    for path in data["paths"]:
        if "switch_schedule" not in path:
            continue
        parts = Path(path).parts
        run = parts[-3]
        campaign = parts[-4]
        out.setdefault((campaign, run), path)
    return out


def resolve_schedule(run: str, root_rel: Optional[str], opened_map: dict,
                     data_roots: tuple[Path, ...] = DEFAULT_DATA_ROOTS,
                     override_root: Path = CHUNK1_P40_OVERRIDE_ROOT) -> Optional[Path]:
    """Locate a run's switch_schedule.json; ``None`` when absent everywhere."""
    for root in data_roots:
        base = Path(root) / root_rel if root_rel else Path(root)
        for sub in (base / run / "run" / "switch_schedule.json",
                    base / run / "switch_schedule.json"):
            if sub.is_file():
                return sub
    campaign = Path(root_rel).name if root_rel else None
    if campaign in DRIVE_ONLY_CAMPAIGNS:
        # raw tree fetched from Drive (``wms fetch get <campaign>``) instead of runs/<campaign>
        cached = drive_only_root(campaign)
        for sub in (cached / run / "run" / "switch_schedule.json",
                    cached / run / "switch_schedule.json"):
            if sub.is_file():
                return sub
    if campaign is not None:
        candidate = opened_map.get((campaign, run))
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    if campaign == "campaign_stage1_chunk1":
        # relocated chunk1-p40 twins only; never borrowed across campaigns
        override = Path(override_root) / run / "run" / "switch_schedule.json"
        if override.is_file():
            return override
    return None


def scan_schedule(path: Path) -> dict:
    """Read one switch_schedule.json; returns arm/action summary (no outcomes)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sched = data.get("schedule", data)
    actions = Counter()
    for event in sched.get("events", []):
        actions[str(event.get("action", "")).strip().lower()] += 1
    return {
        "run": data.get("run") or Path(path).parent.parent.name,
        "schedule_arm": sched.get("arm"),
        "n_events": sum(actions.values()),
        "actions": dict(actions),
        "has_pi": any(is_pi_action(action) for action in actions),
    }


def classify_v3_run(row: dict, opened_map: dict[tuple[str, str], str],
                    data_roots: tuple[Path, ...] = DEFAULT_DATA_ROOTS) -> dict:
    """PI status of one pack_v3 run from policy identity + schedule events.

    ``pi == "unverifiable"`` marks a fixed/switching run whose schedule file is
    unrecoverable (raw run dirs deleted after the inventory build); such runs
    are excluded from training fail-closed but are NOT claimed as PI.
    """
    policy = str(row.get("policy", "")).strip().lower()
    arm = str(row.get("arm", "")).strip().lower()
    needs_schedule = policy in ("fixed", "switching")
    path = resolve_schedule(row["run"], row.get("root_rel"), opened_map, data_roots)
    if needs_schedule and path is None:
        return {"pi": "unverifiable", "source": "schedule-unverifiable",
                "schedule": None}
    scan = scan_schedule(path) if path is not None else None
    if is_pi_policy(policy):
        source = "fleet-pi" if scan is not None and scan["schedule_arm"] == "fleet-pi" \
            else "policy"
        if scan is not None and not scan["has_pi"] and scan["n_events"]:
            raise ValueError(
                f"{row['run']}: policy {policy!r} but schedule has no PI event")
        return {"pi": True, "source": source, "schedule": scan}
    if scan is not None and scan["has_pi"]:
        if arm == "pi":
            return {"pi": True, "source": "wm72-pi-arm", "schedule": scan}
        return {"pi": True, "source": "schedule-events", "schedule": scan}
    if arm == "pi":
        raise ValueError(f"{row['run']}: arm 'pi' but neither policy nor schedule is PI")
    return {"pi": False, "source": "no-pi", "schedule": scan}


def _with_run(row: dict) -> dict:
    """Pack rows key runs as ``<batch>/<run>``; expose ``run`` explicitly."""
    if "run" not in row:
        row = dict(row)
        row["run"] = str(row["key"]).split("/", 1)[1]
    return row


def _window_record(row: dict, pack: str, *, pi_note: Optional[str] = None) -> dict:
    rec = {
        "key": row["key"],
        "pack": pack,
        "run": row["run"],
        "world": row.get("world"),
        "arm": row.get("arm"),
        "policy": row.get("policy"),
        "split": row.get("split"),
        "batch": row.get("batch"),
        "chunk": row.get("chunk"),
        "t0_s": row.get("t0_s"),
        "n_ticks": row.get("n_ticks"),
    }
    if pi_note is not None:
        rec["pi_source"] = pi_note
    return rec


def _counts(records: list[dict]) -> dict:
    by_policy = Counter(str(r["policy"]) for r in records)
    by_world = Counter(str(r["world"]) for r in records)
    return {"total": len(records),
            "by_policy": dict(sorted(by_policy.items())),
            "n_worlds": len(by_world),
            "by_world": dict(sorted(by_world.items()))}


def _write_json(path: Path, payload: dict) -> str:
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(pack_indexes: dict[str, Path], out_dir: Path, *,
          inventory_dir: Optional[Path] = None,
          data_roots: tuple[Path, ...] = DEFAULT_DATA_ROOTS,
          val_modulo: int = VAL_WORLD_MODULO) -> dict:
    """Build all LPO artifacts; returns the receipt payload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opened_map = load_opened_schedule_map(inventory_dir) if inventory_dir else {}

    train, val, pi_eval, pi_excluded, unverifiable = [], [], [], [], []
    missing_drive_only: set[str] = set()
    n_by_pack = {}
    for pack, index_path in sorted(pack_indexes.items(), key=lambda kv: PACK_ORDER[kv[0]]):
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        runs = index["runs"]
        if pack in PI_FREE_PACKS:
            arms = {str(r.get("arm")) for r in runs}
            unexpected = sorted(arms - PI_FREE_PACKS[pack])
            if unexpected:
                raise ValueError(f"{pack}: unexpected arms {unexpected}; PI-free "
                                 f"campaign assertion failed")
        pack_train = [r for r in runs if str(r.get("split")).upper() == "TRAIN"]
        n_by_pack[pack] = len(pack_train)
        for row in pack_train:
            row = _with_run(row)
            if pack in PI_FREE_PACKS:
                record = _window_record(row, pack)
                if val_world(row["world"], modulo=val_modulo):
                    val.append(record)
                else:
                    train.append(record)
                continue
            verdict = classify_v3_run(row, opened_map, data_roots)
            if verdict["pi"] == "unverifiable":
                unverifiable.append(_window_record(
                    row, pack, pi_note=verdict["source"]))
                pi_excluded.append(row["key"])
                campaign = Path(row["root_rel"]).name if row.get("root_rel") else None
                if campaign in DRIVE_ONLY_CAMPAIGNS and not drive_only_root(campaign).is_dir():
                    missing_drive_only.add(campaign)
                continue
            if verdict["pi"] is True:
                pi_eval.append(_window_record(row, pack, pi_note=verdict["source"]))
                pi_excluded.append(row["key"])
                continue
            record = _window_record(row, pack)
            if val_world(row["world"], modulo=val_modulo):
                val.append(record)
            else:
                train.append(record)

    for name in sorted(missing_drive_only):
        warnings.warn(
            f"{name}: raw schedules not on disk, so its fixed/switching runs were excluded as "
            f"unverifiable; fetch them first with `python -m wms fetch get {name}` "
            f"(expected at {drive_only_root(name)})", stacklevel=2)

    order = lambda rec: (PACK_ORDER[rec["pack"]], rec["key"])  # noqa: E731
    train.sort(key=order)
    val.sort(key=order)
    pi_eval.sort(key=order)

    total = len(train) + len(val) + len(pi_eval) + len(unverifiable)
    hashes = {}
    payload_train = {
        "schema": LPO_SCHEMA, "list": "windows_train_noPI",
        "rule": ("TRAIN windows of the WM-S comparator corpus whose run had NO CAV "
                 "on a PI controller (fleet-PI, WM-72 __pi arms and any "
                 "fixed/switching schedule containing a PI action are excluded)"),
        "corpus": {p: str(i) for p, i in sorted(pack_indexes.items())},
        "n_windows": len(train), "counts": _counts(train), "windows": train,
    }
    payload_val = {
        "schema": LPO_SCHEMA, "list": "windows_val",
        "rule": (f"deterministic world-grouped ~1/{val_modulo} validation subset of "
                 f"the PI-free TRAIN windows (sha256(world)[:16] mod {val_modulo} "
                 f"== 0); worlds are assigned whole"),
        "corpus": {p: str(i) for p, i in sorted(pack_indexes.items())},
        "n_windows": len(val), "counts": _counts(val), "windows": val,
    }
    payload_eval = {
        "schema": LPO_SCHEMA, "list": "pi_eval_states",
        "rule": ("held-out PI evaluation states: every TRAIN run excluded for PI, "
                 "with the exclusion source; for post-run scoring only"),
        "n_windows": len(pi_eval), "counts": _counts(pi_eval), "windows": pi_eval,
    }
    payload_excl = {
        "schema": LPO_SCHEMA, "list": "pi_excluded_runs",
        "rule": ("pack run keys to exclude from any WM-S TRAIN pack load: proven-PI "
                 "runs plus fixed/switching runs whose schedule file is "
                 "unrecoverable (fail-closed; see unverifiable)"),
        "n_runs": len(pi_excluded), "keys": sorted(pi_excluded),
        "unverifiable": unverifiable,
        "n_unverifiable": len(unverifiable),
    }
    hashes["windows_train_noPI.json"] = _write_json(
        out_dir / "windows_train_noPI.json", payload_train)
    hashes["windows_val.json"] = _write_json(out_dir / "windows_val.json", payload_val)
    hashes["pi_eval_states.json"] = _write_json(
        out_dir / "pi_eval_states.json", payload_eval)
    hashes["pi_excluded_runs.json"] = _write_json(
        out_dir / "pi_excluded_runs.json", payload_excl)

    receipt = {
        "schema": "wm_s_lpo_receipt/1",
        "task": "WM-S leave-PI-out generalisation prep (P5)",
        "metadata_only": True,
        "sealed_never_read": ["CAL", "CAL_SPLIT_A4.json", "HOLDOUT", "chunk-5",
                              "J4B", "INC1"],
        "packs": {p: str(i) for p, i in sorted(pack_indexes.items())},
        "n_train_windows_by_pack": n_by_pack,
        "n_train": len(train), "n_val": len(val), "n_pi_eval": len(pi_eval),
        "n_pi_excluded_keys": len(pi_excluded),
        "n_unverifiable": len(unverifiable),
        "unverifiable_runs": [r["key"] for r in unverifiable],
        "n_corpus_train_windows": total,
        "val_rule": f"sha256(world)[:16] mod {val_modulo} == 0",
        "val_worlds": sorted({r["world"] for r in val}),
        "pi_eval_by_source": dict(sorted(Counter(
            str(r.get("pi_source")) for r in pi_eval).items())),
        "sha256": hashes,
    }
    (out_dir / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def iter_schedule_events(path: Path) -> Iterator[dict]:
    """Stream a schedule's events (kept for memory-bounded re-scans)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    yield from (data.get("schedule", data)).get("events", [])


build_lpo = build  # architecture name (docs/ARCHITECTURE_V4.md 2.3)
