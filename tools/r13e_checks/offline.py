"""Paired R13E CPU inference: S-R12C, off/emergency/emergency+gs.

Example (all paths must be inside this workspace)::

    python tools/r13e_checks/offline.py --checkpoint runs/example/wm_s.pt \
        --train-root data_ops/train --states _out/train_states.json.gz --out _out/r13e

Writes report.json and the BASE producer's one-step NPZ tables. No training, SUMO,
checkpoint rewriting, or result substitution. Runs serially (one CPU worker/thread);
on Linux the process must successfully set nice=19. Missing assets are fatal.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

# Set these before importing numpy/torch, including in direct CLI invocation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth", "vtype")
EMISSIONS = ("temporal_accel_off", "residual_accel_off", "follower_courtesy_off")
CLASSES = ("recorded_braking", "free_road", "slow_start", "all_lane1")
N_STATES, N_SEEDS, HORIZON_S, DT_S = 48, 2, 120.0, 0.5
ARMS = {"off": {"emergency_braking": False, "secure_gap_veto": False},
        "emergency": {"emergency_braking": True, "secure_gap_veto": False},
        "emergency+gs": {"emergency_braking": True, "secure_gap_veto": True}}
CONTRASTS = {"emergency_minus_off": ("emergency", "off"),
             "emergency+gs_minus_off": ("emergency+gs", "off"),
             "emergency+gs_minus_emergency": ("emergency+gs", "emergency")}
POST_S, STUCK_S, STUCK_SPEED = 6.0, 60.0, 1.0
MERGE_END_M, DELAY_END_M, V_REF = 75.44, 500.0, 31.29
BOOT, RNG_SEED = 1000, 130013
MANIFEST = "C4_EPISODES_MANIFEST.json"
DEFINITIONS = {
    "bias": "predicted final acceleration minus recorded next-row acceleration, m/s^2",
    "classes": "BASE row_classes: lane-1 HDV transitions; braking rec<-1; free road 5<v<20 "
               "and gap>60 or no leader; slow start unpartnered v<2 and 6<=gap<=30",
    "ci": "95% percentile cluster bootstrap, 1000 draws; one-step clusters=windows, "
           "rollout clusters=snapshots with both seeds together; paired differences formed "
           "within state/seed before resampling; CI unavailable with <2 clusters or <5 finite rows. "
           "Rollout estimates are equal-run means (shares averaged per run), not pooled row ratios",
    "rollout": "48 unique TRAIN snapshots x 2 seeds x 3 arms, 120 s, dt=0.5 s; t0 roster, entries off, "
               "recorded CAV teacher only, HDV recorded exits/junction overrides disabled; HDV lanes free",
    "hard_braking": "HDV transitions present at both ends with finite realized dv/0.5 < -3 m/s^2; "
                    "denominator=finite HDV transitions (all lanes)",
    "ttc": "post-step HDV follower rows, nearest same-lane present leader (HDV or CAV); "
           "bumper gap/positive closing speed <1.5 s, including negative gaps; "
           "denominator=finite on-road HDV rows, including rows without a closing leader",
    "overlap": "post-step unordered same-lane present pairs with at least one HDV and "
               "negative bumper gap; ties ordered by slot; count=pair-steps, strict gap<0",
    "blowup": "run flagged if any post-step present HDV has nonfinite position/speed/lane, "
              "abs(position)>100000 m, speed<0 or speed>100 m/s, lane not in {0,1,2}, "
              "or any continuous HDV transition has abs(dv/0.5)>100 m/s^2; "
              "bad rows counted separately and never silently treated as safe",
    "insertion": "continuous-present HDV lane change into lane 0 or 1; excludes t0. "
                 "Ramp merges are the subset lane 2->1; merges_per_run counts events, not vehicles",
    "post_insertion": "HDV inserter events only. Nearest vehicle strictly behind the inserter's "
                      "front bumper in its new lane at insertion is the fixed follower (ties by slot). "
                      "Count realized a<-3 on transitions ENDING 0.5..6 s strictly after insertion, "
                      "only while the measured HDV is continuously present in the insertion lane. "
                      "Both inserter-own and HDV-follower event-rows/shares are reported separately; "
                      "overlapping event windows may count a transition more than once. Events without "
                      "the full 6 s future are counted as truncated and excluded. Event-share denominator "
                      "is full-window events with >=1 finite eligible transition; numerator any hard brake",
    "stuck": "clock starts at zero at t0; increments by 0.5 s for consecutive valid post-step HDV "
             "rows in lane 2, 0<=s<=75.44, speed<1 m/s; resets otherwise. Stuck means clock>60 s "
             "(strict). Report stuck vehicle-steps, final stuck vehicles, max clock and time-mean "
             "ramp occupancy in that zone (regardless of speed). Not a pre-t0 waiting-time estimate",
    "delay": "speed-deficit proxy veh-s = sum 0.5*max(0,1-v/31.29) over valid post-step HDV rows "
             "in all lanes with s<500. Regions: upstream s<0, merge 0<=s<75.44, downstream "
             "75.44<=s<500. Report each region, total and exposure veh-s; invalid rows excluded "
             "and counted separately. This is not travel-time delay, and model exits end exposure",
    "pairing": "all three pairwise contrasts, later arm minus earlier; identical state/window "
               "and Python/numpy/torch/lane seeds; each snapshot has two distinct deterministic seeds",
}


def guard_text(value):
    text = str(value).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise ValueError(f"refusing restricted path/metadata: {value}")
    # Also catches world ids such as w964000m2700 before any data-file read.
    if any(964000 <= int(n) <= 964199 for n in re.findall(r"(?<!\d)\d{6}(?!\d)", text)):
        raise ValueError(f"refusing prohibited evaluation seed in {value}")


def operational_root():
    """The prepared deployment archive explicitly shares its registered run root.

    Everywhere else, including this workstation, paths stay in the worktree.
    """
    remote = Path("/root/autodl-tmp/c4/v5")
    if ROOT.parent == remote and re.fullmatch(r"repo_r13e_[0-9a-f]{8}", ROOT.name):
        return remote
    return ROOT.resolve()


def checked_path(value, *, exists=True):
    """Check spelling AND resolved target (including symlinks), before opening anything."""
    guard_text(value)
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    # Reject external/UNC spellings before resolve can inspect a remote target.
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(operational_root()):
        raise ValueError(f"path must be inside workspace {ROOT}: {value}")
    path = path.resolve()
    guard_text(path)
    if not path.is_relative_to(operational_root()):
        raise ValueError(f"path must be inside workspace {ROOT}: {value}")
    if exists and not path.exists():
        raise FileNotFoundError(f"missing required asset: {path}")
    return path


def audit_metadata(value, key=""):
    """Inspect inline metadata only; never follow source/provenance paths."""
    if isinstance(value, dict):
        for k, v in value.items():
            audit_metadata(v, str(k))
    elif isinstance(value, list):
        for v in value:
            audit_metadata(v, key)
    elif isinstance(value, str):
        guard_text(value)
    if "seed" in key.lower() and value is not None and not isinstance(value, (dict, list)):
        try:
            seed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"invalid seed metadata: {key}={value!r}") from None
        if 964000 <= seed <= 964199:
            raise ValueError(f"refusing prohibited evaluation seed {seed}")


def read_json(path):
    path = checked_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        doc = json.load(stream)
    audit_metadata(doc)
    return doc


def train_manifests(roots):
    docs, worlds = [], {}
    from wms.data.c4_episodes import ARRAYS, QUEUE_ARRAYS

    for root in roots:
        root = checked_path(root)
        doc = read_json(root / MANIFEST)
        if doc.get("schema") != "wms_c4_episode_source/1" or doc.get("split") != "TRAIN":
            raise ValueError(f"{root}: recorded source split must explicitly be TRAIN")
        entries = doc.get("episodes", [])
        if not entries:
            raise ValueError(f"{root}: no episodes in TRAIN source manifest")
        for e in entries:
            if e.get("split_role") not in ("train", "val"):
                raise ValueError(f"{root}: episode lacks recorded train/val split_role")
            world = e["world"]
            previous = worlds.setdefault(world, e["split_role"])
            if previous != e["split_role"]:
                raise ValueError(f"conflicting source split roles for world {world}")
            # Guard every declared data path before _one_split can open any array.
            directory = checked_path(root / e["dir"])
            if not directory.is_relative_to(root):
                raise ValueError(f"episode directory escapes source root: {directory}")
            for name in (*ARRAYS, *QUEUE_ARRAYS):
                checked_path(directory / f"{name}.npy", exists=name in ARRAYS)
        if not any(e["split_role"] == "train" for e in entries):
            raise ValueError(f"{root}: no recorded train-role episodes (no relabeling allowed)")
        docs.append((root, doc))
    return docs, worlds


def _split_claims(doc):
    claims = []
    for block in (doc, doc.get("meta", {}), doc.get("metadata", {})):
        if isinstance(block, dict) and "split" in block:
            claims.append(str(block["split"]).upper())
    return claims


def load_train_states(path, worlds):
    from wms.eval.normalize import initial_frame, recorded_for

    doc = read_json(path)
    envelope = doc if isinstance(doc, dict) else {}
    records = next((doc[k] for k in ("states", "records", "windows") if k in doc), None) \
        if isinstance(doc, dict) else doc
    if not isinstance(records, list) or len(records) < N_STATES:
        raise ValueError(f"need at least {N_STATES} TRAIN states")
    accepted, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every state must be an object")
        claims = _split_claims(envelope) + _split_claims(record)
        if any(c != "TRAIN" for c in claims):
            raise ValueError("state recorded split is not TRAIN")
        world = record.get("world", (record.get("meta") or {}).get("world"))
        role = worlds.get(world)
        if role == "val":
            raise ValueError(f"state world {world} is source validation, not train-role")
        if not claims and role != "train":
            raise ValueError("state lacks recorded TRAIN split or matching source training split")
        for block in (record, record.get("meta", {})):
            if isinstance(block, dict) and block.get("split_role", "train") != "train":
                raise ValueError("state has non-training split_role")
        if record.get("base_run_dir"):
            checked_path(record["base_run_dir"])
        arm = str(record.get("arm", "native"))
        initial = initial_frame(record)
        if not any(v["present"] and not v["is_cav"] and v["lane"] in (0, 1, 2)
                   for v in initial["vehicles"]):
            raise ValueError("each TRAIN state must contain a present on-road HDV")
        t0 = initial["time_s"]
        future = recorded_for(record, arm)
        times = np.array([f["time_s"] for f in future])
        expected = t0 + DT_S * np.arange(int(HORIZON_S / DT_S) + 1)
        if not all(np.any(np.isclose(times, t, atol=1e-6, rtol=0)) for t in expected):
            raise ValueError(f"state needs a full recorded {HORIZON_S:g} s CAV teacher on the 0.5 s grid")
        if not world and not record.get("key"):
            raise ValueError("state needs a world or snapshot key for unique identity")
        identity = (str(world or record["key"]), float(t0))
        if identity in seen:
            raise ValueError(f"duplicate state snapshot: {identity}")
        seen.add(identity)
        accepted.append(record)
    return accepted[:N_STATES]


def check_config(path, seen=()):
    """Preflight extends before the standard config loader follows it."""
    from wms.core import toml

    path = checked_path(path)
    if path in seen or len(seen) >= 16:
        raise ValueError("config extends cycle or excessive depth")
    raw = toml.load(path)
    if "extends" in raw:
        check_config(path.parent / raw["extends"], (*seen, path))
    return path


def cpu_limits():
    if sys.platform.startswith("linux"):
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
        if os.getpriority(os.PRIO_PROCESS, 0) != 19:
            raise RuntimeError("R13E requires Linux nice=19")
    import torch
    torch.set_num_threads(1)


def file_digest(path):
    h = hashlib.sha256()
    with checked_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    """Hash tensor key, shape, dtype, and actual bytes (including signed zeros)."""
    import torch
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(repr((key, tuple(tensor.shape), str(tensor.dtype))).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_exact_predictor(checkpoint):
    import torch
    from tools.wm_eval_common import load_predictor

    checkpoint = checked_path(checkpoint)
    # The strict loader reads these implicitly; preflight resolved targets first.
    for name in ("physics.json", "TRAIN_RECEIPT.json", "lane_calib.json"):
        path = checked_path(checkpoint.with_name(name), exists=False)
        if path.is_file():
            read_json(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "wm_s/1":
        raise ValueError("checkpoint must be wm_s/1")
    predictor, capture = load_predictor(str(checkpoint), "AS_IS", device="cpu", threads=1,
                                        hdv_exits="none", entries="off")
    model, stored = predictor.model, payload["model"]
    if model.physics.dt_s != DT_S:
        raise ValueError(f"R13E requires checkpoint physics dt_s={DT_S}")
    # The shared loader tolerates some legacy missing keys. R13E tolerates NONE.
    if set(model.state_dict()) != set(stored):
        raise ValueError("checkpoint must match exactly: missing or ignored state_dict keys")
    digest = state_digest(stored)
    if state_digest(model.state_dict()) != digest:
        raise ValueError("checkpoint tensor bytes changed during loading")
    model.load_state_dict(stored, strict=True)
    model.eval().requires_grad_(False)
    for key in EMISSIONS:
        if payload.get("config", {}).get(key) is not True or getattr(model, key, None) is not True:
            raise ValueError(f"S-R12C checkpoint must retain {key}=True")
    # load_predictor(hdv_exits='none') removes recorded HDV departures, but its
    # junction_seq still revives slots using recorded HDV outcomes. Only CAV
    # junction presence is exogenous in this free-HDV comparison.
    captured_rollout = model.rollout

    def free_hdv_rollout(state, **kwargs):
        if kwargs.get("junction_seq") is not None:
            kwargs["junction_seq"] = kwargs["junction_seq"] & state["is_cav"].bool()[:, None, :]
        return captured_rollout(state, **kwargs)

    model.rollout = free_hdv_rollout
    return predictor, capture, digest


def set_arm(model, arm):
    from tools.wm_eval_common import _physics_holders

    if arm not in ARMS:
        raise ValueError(f"unknown R13E arm: {arm}")
    changes = ARMS[arm]
    holders = _physics_holders(model)
    if not holders or any(not set(changes) <= getattr(o, n).__dataclass_fields__.keys()
                          for o, n in holders):
        raise ValueError("PhysicsConfig emergency_braking/secure_gap_veto unavailable; main physics implementation required")
    for owner, name in holders:
        setattr(owner, name, dataclasses.replace(getattr(owner, name), **changes))


def seed_all(seed):
    import random
    import torch
    if 964000 <= seed <= 964199:
        raise ValueError("prohibited evaluation seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_datasets(cfg, docs):
    from wms.data.c4_episodes import _one_split

    sets = []
    for root, doc in docs:
        ds, _ = _one_split(root, doc, "train", horizon_steps=int(HORIZON_S / DT_S),
                           history_s=cfg.data.history_s, history_stride_s=cfg.data.history_stride_s,
                           entries_v2=False, stride_s=30.0, start_s=None, junction_fix=cfg.data.junction_fix,
                           command_known=True, merge_pivot_history=cfg.model.merge_decision == "first_passage")
        if ds is None or not len(ds):
            raise ValueError(f"{root}: no TRAIN windows with a full {HORIZON_S:g} s horizon")
        sets.append(ds)
    return sets


def run_onestep(predictor, cfg, sets):
    """Use the BASE producer verbatim, supplying an already strictly loaded model."""
    from tools.r10b_checks import run_onestep as producer
    from wms.config.train import TrainConfig
    from wms.train.settings import TrainSettings

    model = predictor.model
    settings = TrainSettings.from_config(TrainConfig.from_legacy_flat(predictor.backend.config))
    jobs = []
    for si, ds in enumerate(sets):
        by_episode = {}
        for index, (episode, _tick) in enumerate(ds.positions):
            by_episode.setdefault(episode, []).append(index)
        rng = np.random.default_rng(RNG_SEED + si)
        for episode, indices in sorted(by_episode.items()):
            index = int(rng.choice(indices))
            jobs.append((si, index, f"{si}:{ds._loader.entries[episode]['episode']}:{index}"))
    if not jobs:
        raise ValueError("no TRAIN one-step windows")
    rows = {arm: [] for arm in ARMS}
    old = producer.G.copy()
    try:
        producer.G.clear()
        producer.G.update(sets=sets, n_steps=int(HORIZON_S / DT_S), replay_present="filled", base_cols=True)
        for win, job in enumerate(jobs):
            for arm in ARMS:
                set_arm(model, arm)
                seed_all(RNG_SEED + win)
                producer.G["arms"] = {arm: (cfg, settings, model, {"strict": True})}
                _, result = producer.one_window(job)
                arr = result[arm][0]
                arr[:, producer.COLS.index("win")] = win
                rows[arm].append(arr)
    finally:
        producer.G.clear()
        producer.G.update(old)
    cols = producer.COLS + producer.BASE_COLS
    packed = {arm: np.concatenate(values) for arm, values in rows.items()}
    # Geometry, supervision, and clustering must be identical across interventions.
    fixed = [i for i, c in enumerate(cols) if c not in ("fin", "pre", "off")]
    if any(not np.array_equal(packed["off"][:, fixed], packed[arm][:, fixed], equal_nan=True)
           for arm in ARMS):
        raise ValueError("one-step replay rows differ between paired arms")
    return packed, cols, [job[2] for job in jobs]


def _estimate(values, groups):
    from tools.r12_checks import base_calibration as base

    values, groups = np.asarray(values), np.asarray(groups)
    finite = np.isfinite(values)
    n, clusters = int(finite.sum()), len(np.unique(groups[finite]))
    # BASE's minimum-five rule is retained; a single window is not a useful CI.
    previous = base.A.RNG
    try:
        base.A.RNG = np.random.default_rng(RNG_SEED)
        mean, lo, hi, _ = base.A.boot_mean(values, groups, BOOT)
    finally:
        base.A.RNG = previous
    if clusters < 2:
        lo = hi = float("nan")
    return {"mean": mean, "lo": lo, "hi": hi, "n": n, "clusters": clusters,
            "ci_available": bool(n >= 5 and clusters >= 2)}


def analyze_onestep(rows, cols):
    from tools.r12_checks import base_calibration as base

    data = {arm: {c: arr[:, i].astype(float) for i, c in enumerate(cols)} for arm, arr in rows.items()}
    out = {}
    for arm, d in data.items():
        previous = base.A.RNG
        try:
            base.A.RNG = np.random.default_rng(RNG_SEED)
            stats = base.class_stats(d, BOOT)
        finally:
            base.A.RNG = previous
        out[arm] = {}
        for name, selected in base.row_classes(d).items():
            if name not in CLASSES:
                continue
            error = (d["fin"] - d["rec"])[selected]
            groups = d["win"][selected]
            if len(np.unique(groups[np.isfinite(error)])) < 2:
                stats[name]["lo"] = stats[name]["hi"] = float("nan")
            out[arm][name] = {"base": stats[name], "bias": _estimate(error, groups),
                              "mae": _estimate(np.abs(error), groups)}
    for contrast, (later, earlier) in CONTRASTS.items():
        out[contrast] = {}
        a, b = data[earlier], data[later]
        for name in CLASSES:
            select = base.row_classes(a)[name]
            ea, eb = a["fin"] - a["rec"], b["fin"] - b["rec"]
            out[contrast][name] = {
                "bias": _estimate((eb - ea)[select], a["win"][select]),
                "mae": _estimate((np.abs(eb) - np.abs(ea))[select], a["win"][select])}
    return out


def rollout_metrics(pos, speed, lane, present, is_cav, lengths):
    """All-lane free-HDV realized metrics; arrays include t0 in row zero."""
    pos, speed, lane = (np.asarray(a, float) for a in (pos, speed, lane))
    present, hdv = np.asarray(present, bool), ~np.asarray(is_cav, bool)
    if pos.shape != speed.shape or pos.shape != lane.shape or pos.shape != present.shape:
        raise ValueError("rollout array shapes differ")
    lengths = np.asarray(lengths, float)
    if pos.ndim != 2 or len(pos) < 2 or hdv.shape != (pos.shape[1],) or lengths.shape != hdv.shape:
        raise ValueError("rollout needs [T+1,N] arrays and [N] vehicle descriptors")
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError("vehicle lengths must be finite and positive")
    rows = present[1:] & hdv[None]
    continuous = present[:-1] & rows
    acceleration = np.diff(speed, axis=0) / DT_S
    finite_acc = continuous & np.isfinite(acceleration)
    invalid = (~np.isfinite(pos[1:]) | ~np.isfinite(speed[1:]) | ~np.isfinite(lane[1:])
               | (np.abs(pos[1:]) > 100000) | (speed[1:] < 0) | (speed[1:] > 100)
               | ~np.isin(lane[1:], (0, 1, 2))) & rows
    bad_acc = continuous & (~np.isfinite(acceleration) | (np.abs(acceleration) > 100))
    hard = int((finite_acc & (acceleration < -3)).sum())
    ttc_low, eligible, overlaps = 0, 0, 0
    for t in range(1, len(pos)):
        valid = present[t] & np.isfinite(pos[t]) & np.isfinite(speed[t]) & np.isin(lane[t], (0, 1, 2))
        for ln in (0, 1, 2):
            slots = np.flatnonzero(valid & (lane[t] == ln))
            slots = slots[np.argsort(pos[t, slots], kind="stable")]
            for i, follower in enumerate(slots):
                if hdv[follower]:
                    eligible += 1
                    if i + 1 < len(slots):
                        leader = slots[i + 1]
                        gap = pos[t, leader] - lengths[leader] - pos[t, follower]
                        closing = speed[t, follower] - speed[t, leader]
                        ttc_low += int(closing > 0 and gap / closing < 1.5)
                for leader in slots[i + 1:]:
                    if (hdv[follower] or hdv[leader]) and pos[t, leader] - lengths[leader] - pos[t, follower] < 0:
                        overlaps += 1
    metrics = {"hdv_rows": int(rows.sum()), "finite_hdv_transitions": int(finite_acc.sum()),
               "a_lt_minus3_rows": hard, "a_lt_minus3_share": hard / finite_acc.sum() if finite_acc.any() else float("nan"),
               "ttc_eligible_rows": eligible, "ttc_lt_1_5_rows": ttc_low,
               "ttc_lt_1_5_share": ttc_low / eligible if eligible else float("nan"),
               "overlap_pair_steps": overlaps, "invalid_hdv_rows": int(invalid.sum()),
               "bad_acceleration_rows": int(bad_acc.sum()), "blowup": bool(invalid.any() or bad_acc.any())}
    valid_hdv = rows & ~invalid
    zone = valid_hdv & (lane[1:] == 2) & (pos[1:] >= 0) & (pos[1:] <= MERGE_END_M)
    wait = np.zeros(pos.shape[1])
    stuck_rows, max_wait = 0, 0.0
    for t in range(len(zone)):
        wait = np.where(zone[t] & (speed[t + 1] < STUCK_SPEED), wait + DT_S, 0.0)
        stuck_rows += int((wait > STUCK_S).sum())
        max_wait = max(max_wait, float(wait.max(initial=0)))
    metrics.update(stuck_vehicle_steps=stuck_rows, stuck_final_vehicles=int((wait > STUCK_S).sum()),
                   stuck_max_wait_s=max_wait, stuck_mean_ramp_in_zone=float(zone.sum() / len(zone)))
    # Reuse the shared collector-compatible proxy, with invalid rows excluded explicitly.
    from tools.wm_eval_common import delay_proxy

    valid = np.concatenate((np.zeros_like(present[:1]), valid_hdv))
    for region, lo, hi in (("upstream", -np.inf, 0.0), ("merge", 0.0, MERGE_END_M),
                           ("downstream", MERGE_END_M, DELAY_END_M)):
        mask = valid & (pos >= lo) & (pos < hi)
        metrics[f"delay_{region}_veh_s"] = delay_proxy(speed, mask, hdv, (len(pos) - 1) * DT_S)
        metrics[f"exposure_{region}_veh_s"] = float(mask.sum() * DT_S)
    metrics["delay_total_veh_s"] = sum(metrics[f"delay_{r}_veh_s"] for r in ("upstream", "merge", "downstream"))
    metrics.update(insertion_metrics(pos, lane, present, hdv, finite_acc, acceleration))
    return metrics


def insertion_metrics(pos, lane, present, hdv, finite_acc, acceleration):
    """Strictly post-insertion event-rows for HDV inserters and their fixed HDV followers."""
    steps = int(POST_S / DT_S)
    continuous = present[:-1] & present[1:]
    changed = (continuous & hdv[None] & np.isin(lane[:-1], (0, 1, 2))
               & np.isin(lane[1:], (0, 1)) & (lane[:-1] != lane[1:]))
    result = {"merges_per_run": int((changed & (lane[:-1] == 2) & (lane[1:] == 1)).sum()),
              "hdv_insertion_events": int(changed.sum()), "post_insertion_truncated_events": 0}
    for who in ("inserter", "follower"):
        for name in ("events", "hard_brake_events", "finite_rows", "hard_brake_rows"):
            result[f"hdv_{who}_post_{name}"] = 0
    for transition, inserter in zip(*np.nonzero(changed)):
        t = transition + 1  # insertion's post-step frame; its own acceleration is excluded
        if t + steps >= len(pos):
            result["post_insertion_truncated_events"] += 1
            continue
        target_lane = lane[t, inserter]
        if not np.isfinite(pos[t, inserter]):
            continue
        behind = np.flatnonzero(present[t] & (lane[t] == target_lane) & np.isfinite(pos[t])
                                & (pos[t] < pos[t, inserter]))
        # lexsort selects greatest position, then lowest slot id at a tie.
        follower = behind[np.lexsort((behind, -pos[t, behind]))[0]] if len(behind) else None
        for who, slot in (("inserter", inserter), ("follower", follower)):
            if slot is None or not hdv[slot]:
                continue
            eligible = (finite_acc[t:t + steps, slot]
                        & (lane[t:t + steps, slot] == target_lane)
                        & (lane[t + 1:t + steps + 1, slot] == target_lane))
            hard = eligible & (acceleration[t:t + steps, slot] < -3)
            prefix = f"hdv_{who}_post_"
            result[prefix + "events"] += int(eligible.any())
            result[prefix + "hard_brake_events"] += int(hard.any())
            result[prefix + "finite_rows"] += int(eligible.sum())
            result[prefix + "hard_brake_rows"] += int(hard.sum())
    for who in ("inserter", "follower"):
        prefix = f"hdv_{who}_post_"
        for unit in ("events", "rows"):
            denominator = result[prefix + ("events" if unit == "events" else "finite_rows")]
            result[prefix + f"hard_brake_{unit}_share"] = (
                result[prefix + f"hard_brake_{unit}"] / denominator if denominator else float("nan"))
    return result


def run_rollouts(predictor, capture, records):
    from wms.eval.rollout_inputs import t0_vehicles

    result = []
    for index, record in enumerate(records):
        t0, ids, kin, cav = t0_vehicles(record)
        for replicate in range(N_SEEDS):
            seed = RNG_SEED + N_SEEDS * index + replicate
            pair = {"state": index, "key": record.get("key"), "world": record.get("world"),
                    "t0_s": t0, "replicate": replicate, "seed": seed, "vehicles": len(ids)}
            for arm in ARMS:
                set_arm(predictor.model, arm)
                seed_all(seed)
                predictor.flags = dataclasses.replace(predictor.flags, lane_seed=seed)
                capture.pop("last", None)
                predictor.predict_states(record, {**record.get("cav_controllers", {}),
                                          "arm": str(record.get("arm", "native"))}, HORIZON_S, seed)
                raw = capture["last"]
                arrays = []
                for name, initial in (("positions", kin[:, 0]), ("speeds", kin[:, 4]),
                                      ("lanes", kin[:, 1:4].argmax(-1)), ("masks", np.ones(len(ids), bool))):
                    arr = raw[name].detach().cpu().numpy()
                    if arr.shape != (1, int(HORIZON_S / DT_S), len(ids)):
                        raise ValueError(f"unexpected captured {name} shape {arr.shape}")
                    arrays.append(np.concatenate((initial[None], arr[0]), axis=0))
                pair[arm] = rollout_metrics(*arrays, cav, kin[:, 6])
            result.append(pair)
    if not result:
        raise ValueError("no TRAIN rollout states")
    summary = {}
    groups = np.array([r["state"] for r in result])
    for arm in (*ARMS, *CONTRASTS):
        summary[arm] = {}
        for metric in result[0]["off"]:
            later, earlier = CONTRASTS.get(arm, (arm, arm))
            values = [float(r[later][metric]) - float(r[earlier][metric])
                      if arm in CONTRASTS else float(r[arm][metric]) for r in result]
            summary[arm][metric] = _estimate(values, groups)
    totals = {}
    for arm in ARMS:
        total = {key: sum(int(r[arm][key]) for r in result)
                 for key in ("hdv_rows", "finite_hdv_transitions", "a_lt_minus3_rows", "ttc_eligible_rows",
                             "ttc_lt_1_5_rows", "overlap_pair_steps", "invalid_hdv_rows", "bad_acceleration_rows")}
        total["blowup_runs"] = sum(int(r[arm]["blowup"]) for r in result)
        total["blowup_states"] = len({r["state"] for r in result if r[arm]["blowup"]})
        total["states"], total["runs"] = len(records), len(result)
        for numerator, denominator, name in (("a_lt_minus3_rows", "finite_hdv_transitions", "a_lt_minus3_share"),
                                              ("ttc_lt_1_5_rows", "ttc_eligible_rows", "ttc_lt_1_5_share")):
            total[name] = total[numerator] / total[denominator] if total[denominator] else float("nan")
        totals[arm] = total
    return {"rows": result, "totals": totals, "state_cluster_run_mean": summary}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-root", action="append", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--config", default="c4_wm_v10round12c.toml")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Validate every supplied spelling before reading even the first input.
    for path in (args.checkpoint, *args.train_root, args.states, args.out, args.config):
        guard_text(path)
    checkpoint = checked_path(args.checkpoint)
    roots = [checked_path(p) for p in args.train_root]
    states = checked_path(args.states)
    out = checked_path(args.out, exists=False)
    if out.exists():
        raise ValueError(f"output must be a new directory: {out}")
    config = Path(args.config)
    if not config.is_absolute() and config.parent == Path("."):
        config = ROOT / "configs" / config
    config = check_config(config)
    cpu_limits()
    from wms.config.load import load_train_config

    cfg = load_train_config(config, ["runtime.device=cpu"])
    docs, worlds = train_manifests(roots)
    records = load_train_states(states, worlds)
    before = file_digest(checkpoint)
    predictor, capture, weights = load_exact_predictor(checkpoint)
    physics = dataclasses.asdict(predictor.model.physics)
    set_arm(predictor.model, "off")  # fail early if main-agent physics is absent
    packed, cols, windows = run_onestep(predictor, cfg, train_datasets(cfg, docs))
    one_step = analyze_onestep(packed, cols)
    rolls = run_rollouts(predictor, capture, records)
    if state_digest(predictor.model.state_dict()) != weights or file_digest(checkpoint) != before:
        raise ValueError("checkpoint/model weights changed during inference")
    report = {"status": "complete", "checkpoint": str(checkpoint), "checkpoint_sha256": before,
              "state_dict_sha256": weights, "strict_exact_weights": True,
              "device": "cpu", "workers": 1, "threads": 1,
              "linux_nice": 19 if sys.platform.startswith("linux") else None,
              "config": str(config), "train_roots": [str(p) for p in roots],
              "train_manifest_sha256": [file_digest(p / MANIFEST) for p in roots],
               "states_sha256": file_digest(states), "states": N_STATES, "seeds_per_state": N_SEEDS,
               "horizon_s": HORIZON_S, "dt_s": DT_S, "arms": ARMS,
              "emissions": {key: getattr(predictor.model, key) for key in EMISSIONS},
               "base_physics": physics, "override_only": ["emergency_braking", "secure_gap_veto"],
               "definitions": DEFINITIONS,
              "one_step_windows": windows, "one_step": one_step, "rollouts": rolls}
    out.mkdir(parents=True)
    for arm, rows in packed.items():
        np.savez_compressed(out / f"onestep_{arm}.npz", rows=rows, cols=np.array(cols))
    (out / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"R13E complete: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"R13E offline failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
