"""WM-N1 piece (a): corpus inventory + size estimate (metadata only).

Enumerates every TRAIN/CAL run of the WM-N1 training corpus:

  * chunk 1-4 campaign runs,
  * the chunk-1 p40 speedMode-27 fix batch (chunk1_fix27),
  * the J3 truearms batches (chunk 2 and 3),
  * the J4 walls pilot (chunk 4), when present.

Splits are decided from each chunk's WORLD_PLAN.json BEFORE any run file is
touched.  HOLDOUT worlds are counted by NAME only; their directories are
never listed or opened.  The exclusion list from tools/holdout_guard.py +
protocols/holdout_exclusions.json is applied, and every opened path is
recorded and asserted clean.

Policy is classified from switch_schedule.json (schedule.arm) plus the run's
fleet controller for the schedule-less base arms.  Chunks 1-3 ``native`` /
``pi`` arms carry a switching schedule (WM-72) and are therefore labelled
``switching``; the genuine fleet-native / fleet-PI arms come from the J3
truearms batches.

Physics content is read ONLY for a timed micro-sample of at most three TRAIN
runs to estimate vehicles-per-tick, featurization throughput and the zstd
ratio.  ``dispatcher_rows.jsonl`` and ``insertions.jsonl`` are never read.
Zero simulation.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HOLDOUT = "HOLDOUT"
TRAIN = "TRAIN"
CAL = "CALIBRATION"
READABLE_SPLITS = (TRAIN, CAL)

SCHEDULE_NAME = "switch_schedule.json"
RECEIPT_NAME = "RUN_RECEIPT.json"
PHYSICS_NAME = "physics_steps.jsonl"
PHYSICS_SUBDIR = "run"

K_MAX = 128
N_FEATURES = 16
TICK_S = 0.5
FLOAT16_BYTES = 2

WALL_ACTIONS = ("wall", "wall-v2")
POLICY_ALIASES = {
    "native": "native",
    "damping": "damping",
    "pi": "pi",
    "fleet-pi": "pi",
    "pi-external-u": "pi",
    "fleet-pi-external-u": "pi",
    "switching": "switching",
    "fixed": "fixed",
    "wall": "wall",
    "wall-v2": "wall",
    "mixed-assignment": "mixed",
}
FLEET_SCHEDULE_ARMS = ("native", "fleet-pi", "fleet-pi-external-u")

PLAN_SPECS = (
    {
        "name": "chunk1",
        "chunk": "1",
        "plan_rel": "output/campaign_stage1_chunk1/results_plan_wm44/WORLD_PLAN.json",
        "root_rel": "runs/campaign_stage1_chunk1",
    },
    {
        "name": "chunk2",
        "chunk": "2",
        "plan_rel": "output/campaign_stage1_v2/results_plan_chunk2/WORLD_PLAN.json",
        "root_rel": "runs/campaign_stage1_chunk2",
    },
    {
        "name": "chunk3",
        "chunk": "3",
        "plan_rel": "output/campaign_stage1_v3/results_plan_chunk3/WORLD_PLAN.json",
        "root_rel": "runs/campaign_stage1_chunk3",
    },
    {
        "name": "chunk4",
        "chunk": "4",
        "plan_rel": "output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json",
        "root_rel": "runs/campaign_stage1_chunk4",
    },
    {
        "name": "chunk1_fix27",
        "chunk": "fix27",
        "plan_rel": "output/campaign_stage1_v3/results_plan_fix27/WORLD_PLAN.json",
        "root_rel": "runs/campaign_stage1_chunk1_fix27",
    },
    {
        "name": "truearms_chunk2",
        "chunk": "2",
        "plan_rel": "output/campaign_stage1_v3/results_plan_truearms_chunk2/WORLD_PLAN.json",
        "root_rel": "runs/campaign_stage1_chunk2_truearms",
    },
    {
        "name": "truearms_chunk3",
        "chunk": "3",
        "plan_rel": "output/campaign_stage1_v3/results_plan_truearms_chunk3/WORLD_PLAN.json",
        "root_rel": "runs/campaign_stage1_chunk3_truearms",
    },
    {
        "name": "walls_pilot_chunk4",
        "chunk": "4",
        "plan_rel": "output/campaign_walls_v1/results_plan/WORLD_PLAN.json",
        "root_rel": "runs/campaign_walls_pilot_chunk4",
    },
)


class SplitGuardError(RuntimeError):
    """A read was attempted on a sealed (HOLDOUT) or excluded world."""


def canonical_policy(raw):
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in POLICY_ALIASES:
        return POLICY_ALIASES[key]
    if key.startswith("w-") or key.startswith("w_"):
        return "wall"
    return key


class SplitGuard:
    """Split decision + opened-path audit for one inventory build."""

    def __init__(self, excluded, disclosed, external=None, exclusions_path=None):
        self.excluded = set(excluded)
        self.disclosed = list(disclosed)
        self.external = external
        self.exclusions_path = str(exclusions_path) if exclusions_path else None
        self.opened = []

    def assert_world_readable(self, world, split):
        if split == HOLDOUT:
            raise SplitGuardError(
                f"refusing to open sealed HOLDOUT world {world!r} (split from "
                f"WORLD_PLAN.json; counted by name only)")
        if world in self.excluded:
            raise SplitGuardError(
                f"refusing to open excluded world {world!r} "
                f"(protocols/holdout_exclusions.json; owner ruling)")

    def record_open(self, path):
        self.opened.append(str(path))

    def assert_clean(self, holdout_tokens, forbidden_names=("dispatcher_rows", "insertions")):
        for p in self.opened:
            for token in holdout_tokens:
                if token and token in p:
                    raise SplitGuardError(
                        f"opened-path audit: sealed HOLDOUT token {token!r} "
                        f"appears in opened path {p!r}")
            lowered = p.lower()
            for bad in forbidden_names:
                if bad in lowered:
                    raise SplitGuardError(
                        f"opened-path audit: forbidden file {bad!r} in {p!r}")

    def assert_external_agrees(self):
        if self.external is None:
            return
        ext_excluded = self.external.excluded_roots()
        mismatch = set(ext_excluded) ^ self.excluded
        if mismatch:
            raise SplitGuardError(
                f"holdout_guard.py exclusion set disagrees with our parse: "
                f"symmetric difference {sorted(mismatch)}")


def load_exclusions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "holdout_exclusions/1":
        raise SplitGuardError(f"unexpected exclusions schema {data.get('schema')!r}")
    excluded = {row["world"] for row in data["excluded"]}
    return data, excluded, list(data.get("kept_disclosed", []))


def load_external_guard(tools_dir):
    path = Path(tools_dir) / "holdout_guard.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("rd_holdout_guard", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_guard(data_root, exclusions_rel="protocols/holdout_exclusions.json",
               tools_rel="tools/holdout_guard.py"):
    exclusions_path = Path(data_root) / exclusions_rel
    _, excluded, disclosed = load_exclusions(exclusions_path)
    external = load_external_guard(Path(data_root) / tools_rel)
    guard = SplitGuard(excluded, disclosed, external=external,
                       exclusions_path=exclusions_path)
    if external is not None:
        try:
            external.load_holdout_exclusions(exclusions_path)
        except Exception:  # noqa: BLE001 - surfaced by assert_external_agrees
            external = None
            guard.external = None
    guard.assert_external_agrees()
    return guard


def read_plan(path, guard=None):
    if guard is not None:
        guard.record_open(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def world_index(plan):
    index = {}
    for w in plan.get("worlds", []):
        index[w["world"]] = w
    return index


def world_of_run(run):
    if run.get("world"):
        return run["world"]
    return str(run["run"]).split("__", 1)[0]


def run_t0(run, world_meta):
    for key in ("restores_state",):
        block = run.get(key)
        if isinstance(block, dict) and block.get("at_s") is not None:
            return float(block["at_s"])
    saves = run.get("saves_state")
    if isinstance(saves, dict) and saves.get("at_s") is not None:
        return float(saves["at_s"])
    if world_meta and world_meta.get("t0_s") is not None:
        return float(world_meta["t0_s"])
    return None


def plan_policy_hint(world_meta, run):
    arm = run.get("arm")
    if arm == "native_base":
        nb = (world_meta or {}).get("native_base_arm") or {}
        pol = nb.get("policy") or nb.get("controller_name") or "native"
        return canonical_policy(pol), True
    if arm == "base" or (run.get("mode") == "full_instrumented"
                         and not run.get("suffix_controller")):
        base = (world_meta or {}).get("base_arm") or {}
        if base.get("controller_name"):
            return canonical_policy(base["controller_name"]), True
    return None, False


def classify_policy(arm, schedule_arm, hint_policy, hint_fleet):
    if schedule_arm:
        policy = canonical_policy(schedule_arm) or "unknown"
        fleet = str(schedule_arm).strip().lower() in FLEET_SCHEDULE_ARMS
        wm72 = policy == "switching" and str(arm) in ("native", "pi")
        return policy, fleet, wm72
    if hint_policy:
        return hint_policy, bool(hint_fleet), False
    guess = canonical_policy(arm)
    if guess in ("damping", "fixed", "switching", "wall"):
        return guess, False, False
    return "unknown", False, False


def _read_schedule(run_dir, guard):
    for cand in (Path(run_dir) / PHYSICS_SUBDIR / SCHEDULE_NAME,
                 Path(run_dir) / SCHEDULE_NAME):
        if cand.is_file():
            guard.record_open(cand)
            data = json.loads(cand.read_text(encoding="utf-8"))
            sched = data.get("schedule", data)
            return sched.get("arm"), len(sched.get("events", []))
    return None, 0


def _read_receipt(run_dir, guard):
    cand = Path(run_dir) / RECEIPT_NAME
    if not cand.is_file():
        return None
    guard.record_open(cand)
    return json.loads(cand.read_text(encoding="utf-8"))


def _physics_path(run_dir):
    return Path(run_dir) / PHYSICS_SUBDIR / PHYSICS_NAME


def _receipt_rows(receipt):
    if not isinstance(receipt, dict):
        return None
    if receipt.get("physics_rows_written") is not None:
        return int(receipt["physics_rows_written"])
    suffix = receipt.get("suffix_summary") or {}
    if suffix.get("rows_written") is not None:
        return int(suffix["rows_written"])
    return None


def read_run(spec, run, world_meta, guard, data_root):
    world = world_of_run(run)
    split = world_meta["split"]
    guard.assert_world_readable(world, split)
    arm = run.get("arm")
    run_name = run["run"]
    run_dir = Path(data_root) / spec["root_rel"] / run_name
    present = run_dir.is_dir()
    hint_policy, hint_fleet = plan_policy_hint(world_meta, run)

    schedule_arm = None
    n_schedule_events = 0
    receipt = None
    if present:
        schedule_arm, n_schedule_events = _read_schedule(run_dir, guard)
        receipt = _read_receipt(run_dir, guard)

    policy, is_fleet, wm72 = classify_policy(arm, schedule_arm, hint_policy, hint_fleet)
    if schedule_arm:
        policy_source = "schedule"
    elif hint_policy:
        policy_source = "plan"
    elif policy != "unknown":
        policy_source = "arm"
    else:
        policy_source = "unknown"

    status = None
    rows_written = None
    saved_state_id = None
    if receipt is not None:
        status = receipt.get("exit_status")
        rows_written = _receipt_rows(receipt)
        saved_state_id = receipt.get("saved_state_id")
        if saved_state_id is None:
            ss = receipt.get("saved_state") or {}
            saved_state_id = ss.get("saved_state_id")
    physics_path = _physics_path(run_dir)
    physics_size = physics_path.stat().st_size if physics_path.is_file() else None

    complete = status == "complete"
    is_smoke = bool(run.get("smoke")) or "smoke" in run_name.lower()

    return {
        "batch": spec["name"],
        "chunk": spec["chunk"],
        "world": world,
        "split": split,
        "root_rel": spec["root_rel"],
        "loading_veh_h": (world_meta.get("loading_veh_h") or run.get("loading_veh_h")),
        "penetration": (world_meta.get("penetration") if world_meta.get("penetration") is not None else run.get("penetration")),
        "t0_s": run_t0(run, world_meta),
        "arm": arm,
        "policy": policy,
        "policy_source": policy_source,
        "is_fleet": bool(is_fleet),
        "wm72_relabel": bool(wm72),
        "schedule_arm": schedule_arm,
        "n_schedule_events": n_schedule_events,
        "duration_s": run.get("duration_s"),
        "present": bool(present),
        "receipt_status": status,
        "is_complete": bool(complete),
        "physics_rows_written": rows_written,
        "physics_file_bytes": physics_size,
        "saved_state_id": saved_state_id,
        "is_smoke": is_smoke,
        "mode": run.get("mode"),
        "run": run_name,
    }


def enumerate_spec(spec, data_root, guard):
    plan_path = Path(data_root) / spec["plan_rel"]
    if not plan_path.is_file():
        return [], [], []
    plan = read_plan(plan_path, guard=guard)
    world_meta = world_index(plan)
    holdout_names = []
    rows = []
    planned = []
    smoke_run = (plan.get("smoke") or {}).get("run") if isinstance(plan.get("smoke"), dict) else None
    for run in plan.get("runs", []):
        world = world_of_run(run)
        meta = world_meta.get(world)
        if meta is None:
            continue
        split = meta["split"]
        planned.append({"batch": spec["name"], "world": world, "run": run["run"],
                        "arm": run.get("arm"), "split": split})
        if split == HOLDOUT:
            holdout_names.append({"batch": spec["name"], "world": world,
                                  "run": run["run"], "arm": run.get("arm")})
            continue
        row = read_run(spec, run, meta, guard, data_root)
        if smoke_run is not None and run["run"] == smoke_run:
            row["is_smoke"] = True
        rows.append(row)
    return rows, holdout_names, planned


def build_rows(data_root, guard, specs=None):
    specs = PLAN_SPECS if specs is None else specs
    all_rows = []
    holdout = []
    planned = []
    for spec in specs:
        rows, hold, plan = enumerate_spec(spec, data_root, guard)
        all_rows.extend(rows)
        holdout.extend(hold)
        planned.extend(plan)
    return all_rows, holdout, planned


def build_pairs(df):
    if df is None or len(df) == 0:
        return []
    usable = df[df["present"] & df["is_complete"] & (~df["is_smoke"])]
    pairs = []
    for (world, t0), group in usable.groupby(["world", "t0_s"], dropna=False):
        native_rows = group[(group["policy"] == "native")]
        if len(native_rows) == 0:
            continue
        preferred = native_rows[native_rows["arm"] == "native"]
        native_ref = preferred.iloc[0] if len(preferred) else native_rows.iloc[0]
        for _, other in group.iterrows():
            if other["policy"] not in ("damping", "pi", "fixed", "switching", "wall"):
                continue
            a_id = native_ref["saved_state_id"]
            b_id = other["saved_state_id"]
            if a_id is None or b_id is None:
                same_base = None
            else:
                same_base = bool(a_id == b_id)
            pairs.append({
                "world": world,
                "t0_s": t0,
                "pair_key": f"{world}@{t0}",
                "category": f"{other['policy']}-native",
                "native_run": native_ref["run"],
                "native_batch": native_ref["batch"],
                "other_run": other["run"],
                "other_batch": other["batch"],
                "other_policy": other["policy"],
                "same_base_state": same_base,
            })
    return pairs


def _featurize(parsed_rows, k_max, n_features):
    n = len(parsed_rows)
    arr = np.zeros((n, k_max, n_features), dtype=np.float16)
    for i, row in enumerate(parsed_rows):
        vehicles = row.get("vehicles") or {}
        cav = set(row.get("cav_ids") or [])
        for j, (vid, v) in enumerate(list(vehicles.items())[:k_max]):
            lane_id = str(v.get("lane_id", ""))
            lane_tail = lane_id.rsplit("_", 1)[-1]
            try:
                lane_phys = int(lane_tail)
            except ValueError:
                lane_phys = 0
            arr[i, j, 0] = v.get("lane_position_m", 0.0)
            arr[i, j, 1] = v.get("speed_mps", 0.0)
            arr[i, j, 2] = v.get("acceleration_mps2", 0.0)
            arr[i, j, 3] = 1.0 if vid in cav else 0.0
            arr[i, j, 4] = 5.0
            arr[i, j, 5] = 1.0 if lane_phys == 0 else 0.0
            arr[i, j, 6] = 1.0 if lane_phys == 1 else 0.0
            arr[i, j, 7] = 1.0 if lane_phys == 2 else 0.0
    return arr


def micro_sample(rows, data_root, guard, n_runs=3, max_lines=400,
                 k_max=K_MAX, n_features=N_FEATURES):
    candidates = [r for r in rows if r["split"] == TRAIN and r["present"]
                  and r["is_complete"] and not r["is_smoke"]
                  and r["physics_file_bytes"]]
    candidates.sort(key=lambda r: (r["chunk"], r["run"]))
    chosen = []
    seen_chunks = set()
    for r in candidates:
        if r["chunk"] in seen_chunks:
            continue
        chosen.append(r)
        seen_chunks.add(r["chunk"])
        if len(chosen) >= n_runs:
            break
    if len(chosen) < n_runs:
        for r in candidates:
            if r in chosen:
                continue
            chosen.append(r)
            if len(chosen) >= n_runs:
                break

    per_tick = []
    parsed_all = []
    n_parsed = 0
    import time
    t0 = time.perf_counter()
    for entry in chosen:
        run_dir = Path(data_root) / entry.get("root_rel", "") / entry["run"]
        phys = _physics_path(run_dir)
        guard.record_open(phys)
        with open(phys, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                parsed_all.append(row)
                per_tick.append(len(row.get("vehicles") or {}))
                n_parsed += 1
    parse_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    feats = _featurize(parsed_all, k_max, n_features)
    feat_seconds = time.perf_counter() - t1

    import pyarrow as pa
    raw = feats.tobytes()
    compressed = pa.compress(raw, codec="zstd")
    ratio = (len(compressed) / len(raw)) if raw else 0.0

    total_seconds = parse_seconds + feat_seconds
    ticks_per_s = (len(parsed_all) / total_seconds) if total_seconds > 0 else 0.0

    return {
        "runs": [r["run"] for r in chosen],
        "lines_read": n_parsed,
        "vehicles_per_tick_mean": float(np.mean(per_tick)) if per_tick else 0.0,
        "vehicles_per_tick_max": int(np.max(per_tick)) if per_tick else 0,
        "vehicles_per_tick_p95": float(np.percentile(per_tick, 95)) if per_tick else 0.0,
        "parse_seconds": parse_seconds,
        "featurize_seconds": feat_seconds,
        "ticks_per_second": ticks_per_s,
        "zstd_ratio": ratio,
        "sample_raw_bytes": len(raw),
        "sample_compressed_bytes": len(compressed),
        "k_max": k_max,
        "n_features": n_features,
    }


def _run_root_for(entry):
    for spec in PLAN_SPECS:
        if spec["name"] == entry["batch"]:
            return spec["root_rel"]
    return ""


def estimate_sizes(rows, micro, workers=4, k_max=K_MAX, n_features=N_FEATURES):
    train_rows = [r for r in rows if r["split"] == TRAIN and r["present"] and not r["is_smoke"]]
    cal_rows = [r for r in rows if r["split"] == CAL and r["present"] and not r["is_smoke"]]

    def ticks_for(r):
        if r["physics_rows_written"]:
            return int(r["physics_rows_written"])
        if r["duration_s"]:
            return int(round(float(r["duration_s"]) / TICK_S))
        return 0

    def agg(subset):
        ticks = sum(ticks_for(r) for r in subset)
        bytes_ = sum(r["physics_file_bytes"] or 0 for r in subset)
        return ticks, bytes_

    train_ticks, train_phys = agg(train_rows)
    cal_ticks, cal_phys = agg(cal_rows)
    total_ticks = train_ticks + cal_ticks
    total_phys = train_phys + cal_phys

    veh_bytes = total_ticks * k_max * n_features * FLOAT16_BYTES
    mask_bytes = total_ticks * k_max * FLOAT16_BYTES
    n_dec = total_ticks // int(30.0 / TICK_S)
    plan_bytes = n_dec * k_max * 20 * 1
    counter_bytes = n_dec * 30 * 2 * FLOAT16_BYTES
    tensor_bytes = veh_bytes + mask_bytes + plan_bytes + counter_bytes
    zstd_bytes = int((veh_bytes + mask_bytes) * micro["zstd_ratio"])

    rate = micro["ticks_per_second"]
    if rate > 0:
        wall_h = total_ticks / (rate * workers) / 3600.0
    else:
        wall_h = None

    per_chunk = {}
    for r in rows:
        if r["split"] not in READABLE_SPLITS or not r["present"] or r["is_smoke"]:
            continue
        d = per_chunk.setdefault(r["chunk"], {"TRAIN": 0, "CALIBRATION": 0, "policy": {}})
        d[r["split"]] += 1
        d["policy"][r["policy"]] = d["policy"].get(r["policy"], 0) + 1

    per_policy = {}
    for r in rows:
        if r["split"] not in READABLE_SPLITS or not r["present"] or r["is_smoke"]:
            continue
        per_policy.setdefault(r["policy"], {"TRAIN": 0, "CALIBRATION": 0})
        per_policy[r["policy"]][r["split"]] += 1

    return {
        "train_runs": len(train_rows),
        "cal_runs": len(cal_rows),
        "train_ticks": train_ticks,
        "cal_ticks": cal_ticks,
        "total_ticks": total_ticks,
        "physics_raw_bytes": total_phys,
        "veh_bytes": veh_bytes,
        "mask_bytes": mask_bytes,
        "plan_bytes": plan_bytes,
        "counter_bytes": counter_bytes,
        "tensor_bytes": tensor_bytes,
        "zstd_bytes": zstd_bytes,
        "n_decisions": n_dec,
        "per_chunk": per_chunk,
        "per_policy": per_policy,
        "wall_hours_4_workers": wall_h,
        "workers": workers,
        "k_max": k_max,
        "n_features": n_features,
    }


def incomplete_runs(rows):
    out = []
    for r in rows:
        reasons = []
        if not r["present"]:
            reasons.append("missing")
        elif not r["is_complete"]:
            reasons.append(f"receipt={r['receipt_status']!r}")
        if r["is_smoke"]:
            reasons.append("smoke")
        if reasons:
            out.append({"batch": r["batch"], "world": r["world"], "run": r["run"],
                        "split": r["split"], "reason": ";".join(reasons)})
    return out
