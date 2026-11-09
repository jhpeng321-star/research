"""Teacher shards (``distill_teacher/1``, E:/agent-work/distill/SCHEMA.md) -> flat BC dataset.

Reader is numpy only (``allow_pickle=False``); it does not import the producer.  One shard per episode
(``<data>/shards/distill_<seed>.npz``); a shard is used only if its receipt
(``<data>/receipts/<seed>.json``) says ``status: complete`` (``require_receipts=False`` for fixtures).

Flat layout (all episodes concatenated, frame order preserved):
* vehicle rows ``N``: ``s, v, a, lane, is_cav, length, cmd, intent`` = exactly the ``[1,K]`` actor state
  restricted to its PRESENT slots in slot order (= ``wms.control.mappo.agent.compact_present``, exact);
* frames ``F``: ``frame_off [F+1]`` row offsets, ``frame_ep [F]`` episode index;
* per vehicle row labels (only CAV label rows carry them): ``acc_tgt`` (applied accel, NaN = no command),
  ``acc_u0`` (U0 command before any plan override), ``lane_tgt`` (choice 0 stay / 1 left / 2 right, -100 =
  none), ``cat`` (0 no label, 1 plan_active, 2 pre-onset window, 3 other).

``cat == 2``: rows of a CAV in the ``pre_onset_s`` seconds before a plan that contains it starts (not
plan_active themselves), so the student also sees the moments just before cooperation begins.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "distill_teacher/1"
CAT_NONE, CAT_PLAN, CAT_PRE, CAT_OTHER = 0, 1, 2, 3
CAT_NAMES = {CAT_PLAN: "plan", CAT_PRE: "pre_onset", CAT_OTHER: "other"}
IGNORE = -100
DELTA_TO_CHOICE = {0: 0, -1: 1, 1: 2}       # wms.control.lanes.LANE_DELTA = (0, -1, +1)
VEH_KEYS = ("s_m", "v_mps", "a_mps2", "lane", "is_cav", "length_m", "cmd_accel_mps2", "lane_intent")
LABEL_KEYS = ("cav_step", "cav_row", "accel_applied", "accel_u0", "ctrl_src", "lane_label", "lane_valid",
              "lane_cmd", "plan_active", "plan_act", "plan_onset")


@dataclass
class Dataset:
    s: np.ndarray
    v: np.ndarray
    a: np.ndarray
    lane: np.ndarray
    is_cav: np.ndarray
    length: np.ndarray
    cmd: np.ndarray
    intent: np.ndarray
    frame_off: np.ndarray
    frame_ep: np.ndarray
    acc_tgt: np.ndarray
    acc_u0: np.ndarray
    lane_tgt: np.ndarray
    cat: np.ndarray
    episodes: list = field(default_factory=list)     # [{seed, file, sha256, frames, rows, K}]
    skipped: list = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(len(self.frame_ep))


def _sha(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_files(data_dir, *, require_receipts: bool = True):
    """``[(seed, path)]`` sorted by seed, plus the skipped list ``[(file, reason)]``."""
    data_dir = Path(data_dir)
    out, skipped = [], []
    for path in sorted((data_dir / "shards").glob("distill_*.npz")):
        seed = path.stem[len("distill_"):]
        if require_receipts:
            receipt = data_dir / "receipts" / f"{seed}.json"
            if not receipt.is_file():
                skipped.append((path.name, "no receipt"))
                continue
            status = json.loads(receipt.read_text(encoding="utf-8")).get("status")
            if status != "complete":
                skipped.append((path.name, f"receipt status {status!r}"))
                continue
        out.append((int(seed), path))
    return sorted(out), skipped


def read_shard(path, *, pre_onset_s: int = 10, lane_target: str = "observed") -> dict:
    """One shard -> per-episode arrays (vehicle rows in shard order + per-row labels)."""
    if lane_target not in ("observed", "cmd_for_plan"):
        raise ValueError("lane_target must be 'observed' or 'cmd_for_plan'")
    z = np.load(path, allow_pickle=False)
    schema = str(z["schema"])
    if schema != SCHEMA:
        raise ValueError(f"{path}: schema {schema!r} != {SCHEMA!r}")
    off = z["off"].astype(np.int64)
    t = z["t"].astype(np.int64)
    n = int(off[-1])
    veh = {}
    for key in VEH_KEYS:
        arr = z[key]
        if arr.shape[0] != n:
            raise ValueError(f"{path}: {key} has {arr.shape[0]} rows, off says {n}")
        veh[key] = arr
    # present slots in slot order inside each frame (the compact_present order)
    slot = z["slot"].astype(np.int64)
    for i in range(len(t)):
        seg = slot[off[i]:off[i + 1]]
        if seg.size > 1 and np.any(np.diff(seg) <= 0):
            raise ValueError(f"{path}: frame {i} slots not strictly increasing")
    lab = {k: z[k] for k in LABEL_KEYS}
    vid = z["vid"].astype(np.int64)
    m = len(lab["cav_row"])
    rows = lab["cav_row"].astype(np.int64)
    if m and (rows.min() < 0 or rows.max() >= n or not veh["is_cav"][rows].all()):
        raise ValueError(f"{path}: cav_row out of range or not a CAV row")
    step = lab["cav_step"].astype(np.int64)
    if m and np.any((rows < off[step]) | (rows >= off[step + 1])):
        raise ValueError(f"{path}: cav_row outside its step's row range")

    acc_tgt = np.full(n, np.nan, np.float32)
    acc_u0 = np.full(n, np.nan, np.float32)
    lane_tgt = np.full(n, IGNORE, np.int16)
    cat = np.zeros(n, np.int8)
    acc_tgt[rows] = lab["accel_applied"].astype(np.float32)
    acc_u0[rows] = lab["accel_u0"].astype(np.float32)

    delta = lab["lane_label"].astype(np.int64)
    if lane_target == "cmd_for_plan":
        cmd = lab["lane_cmd"].astype(np.int64)
        use_cmd = lab["plan_active"].astype(bool) & (cmd != 0)
        delta = np.where(use_cmd, np.clip(cmd, -1, 1), delta)
    valid = lab["lane_valid"].astype(bool) | ((lane_target == "cmd_for_plan")
                                              & lab["plan_active"].astype(bool) & (lab["lane_cmd"] != 0))
    choice = np.vectorize(lambda d: DELTA_TO_CHOICE.get(int(d), IGNORE), otypes=[np.int16])(delta) if m else delta
    lane_tgt[rows] = np.where(valid, choice, IGNORE).astype(np.int16)

    plan = lab["plan_active"].astype(bool)
    pre = np.zeros(m, bool)
    if pre_onset_s > 0 and m:
        row_vid = vid[rows]
        row_t = t[step]
        for j in np.flatnonzero(lab["plan_onset"].astype(bool)):
            t0, v0 = row_t[j], row_vid[j]
            pre |= (row_vid == v0) & (row_t >= t0 - pre_onset_s) & (row_t < t0)
    cat[rows] = np.where(plan, CAT_PLAN, np.where(pre, CAT_PRE, CAT_OTHER)).astype(np.int8)
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return dict(veh=veh, off=off, t=t, acc_tgt=acc_tgt, acc_u0=acc_u0, lane_tgt=lane_tgt, cat=cat,
                K=int(z["K"]), meta=meta, n_label_rows=m)


def load_dataset(data_dir, *, pre_onset_s: int = 10, lane_target: str = "observed",
                 require_receipts: bool = True, hash_files: bool = True) -> Dataset:
    files, skipped = shard_files(data_dir, require_receipts=require_receipts)
    if not files:
        raise FileNotFoundError(f"no usable shards under {data_dir}/shards")
    parts, episodes = [], []
    row_base = 0
    frame_off = [np.zeros(1, np.int64)]
    frame_ep = []
    for e, (seed, path) in enumerate(files):
        p = read_shard(path, pre_onset_s=pre_onset_s, lane_target=lane_target)
        n = int(p["off"][-1])
        frame_off.append(p["off"][1:] + row_base)
        frame_ep.append(np.full(len(p["t"]), e, np.int32))
        parts.append(p)
        row_base += n
        episodes.append(dict(seed=seed, file=path.name, sha256=_sha(path) if hash_files else None,
                             frames=int(len(p["t"])), rows=n, label_rows=p["n_label_rows"], K=p["K"]))
    cat = lambda key: np.concatenate([p["veh"][key] for p in parts])  # noqa: E731
    return Dataset(
        s=cat("s_m").astype(np.float32), v=cat("v_mps").astype(np.float32), a=cat("a_mps2").astype(np.float32),
        lane=cat("lane").astype(np.int8), is_cav=cat("is_cav").astype(bool),
        length=cat("length_m").astype(np.float32), cmd=cat("cmd_accel_mps2").astype(np.float32),
        intent=cat("lane_intent").astype(np.float32),
        frame_off=np.concatenate(frame_off), frame_ep=np.concatenate(frame_ep),
        acc_tgt=np.concatenate([p["acc_tgt"] for p in parts]),
        acc_u0=np.concatenate([p["acc_u0"] for p in parts]),
        lane_tgt=np.concatenate([p["lane_tgt"] for p in parts]),
        cat=np.concatenate([p["cat"] for p in parts]),
        episodes=episodes, skipped=skipped)


def split_episodes(seeds, val_fraction: float, seed: int):
    """Deterministic episode split by seed: ``(train_idx, val_idx)`` (indices into ``seeds``)."""
    seeds = list(seeds)
    n = len(seeds)
    n_val = int(np.ceil(val_fraction * n)) if n >= 2 else 0
    order = np.argsort(np.asarray(seeds), kind="stable")
    perm = np.random.default_rng(seed).permutation(n)
    val = sorted(int(order[i]) for i in perm[:n_val])
    train = sorted(set(range(n)) - set(val))
    return train, val


# ---------------------------------------------------------------- synthetic fixture (tests / dry runs)
def write_fixture(out_dir, *, n_episodes: int = 4, steps: int = 30, n_veh: int = 40, k_slots: int = 512,
                  seed: int = 0, cav_share: float = 0.3) -> Path:
    """Schema-conformant synthetic shards + receipts (random kinematics; NOT teacher data)."""
    out_dir = Path(out_dir)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    (out_dir / "receipts").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for e in range(n_episodes):
        ep_seed = 990000 + e
        slots = np.sort(rng.choice(k_slots, n_veh, replace=False))
        is_cav = rng.random(n_veh) < cav_share
        is_cav[:2] = True
        s0 = np.sort(rng.uniform(-500, 60, n_veh))
        lane0 = rng.integers(0, 3, n_veh)
        t = np.arange(300, 300 + steps, dtype=np.int32)
        cols = {k: [] for k in ("slot", "vid", "s_m", "v_mps", "a_mps2", "lane", "junction", "is_cav",
                                "length_m", "cmd_accel_mps2", "lane_intent", "dwell_age_s")}
        lab = {k: [] for k in ("cav_step", "cav_row", "accel_applied", "accel_u0", "ctrl_src", "lane_label",
                               "lane_valid", "lane_cmd", "plan_active", "plan_family", "plan_role", "plan_act",
                               "plan_onset", "trigger_prob", "a_next")}
        off = [0]
        lane = lane0.copy()
        s = s0.copy()
        cav_idx = np.flatnonzero(is_cav)
        plan_members = cav_idx[:2]
        onset = steps // 3
        plan_end = onset + steps // 3
        for i in range(steps):
            v = rng.uniform(5, 25, n_veh)
            a = rng.normal(0, 0.5, n_veh)
            new_lane = lane.copy()
            change = rng.random(n_veh) < 0.05
            new_lane[change] = np.clip(lane[change] + rng.choice([-1, 1], change.sum()), 0, 2)
            base = off[-1]
            for j in range(n_veh):
                row = dict(slot=slots[j], vid=j, s_m=s[j], v_mps=v[j], a_mps2=a[j], lane=lane[j], junction=False,
                           is_cav=is_cav[j], length_m=5.0,
                           cmd_accel_mps2=rng.normal(0, 0.3) if is_cav[j] else 0.0,
                           lane_intent=0.0, dwell_age_s=3.0)
                for key, value in row.items():
                    cols[key].append(value)
            for j in cav_idx:
                active = (onset <= i < plan_end) and j in plan_members
                u0 = float(np.clip(rng.normal(0.2, 0.4), -3, 1.2))
                row = dict(cav_step=i, cav_row=base + j, accel_applied=-0.8 if active else u0, accel_u0=u0,
                           ctrl_src=2 if active else 1, lane_label=int(np.clip(new_lane[j] - lane[j], -1, 1)),
                           lane_valid=i < steps - 1, lane_cmd=0, plan_active=active,
                           plan_family=0 if active else -1,
                           plan_role=int(np.flatnonzero(plan_members == j)[0]) if active else -1,
                           plan_act=0 if active else -1, plan_onset=active and i == onset,
                           trigger_prob=0.9 if active else np.nan, a_next=0.0)
                for key, value in row.items():
                    lab[key].append(value)
            off.append(off[-1] + n_veh)
            s = s + v * 1.0
            lane = new_lane
        meta = dict(seed=ep_seed, K=k_slots, warmup_s=300, horizon_s=300 + steps, synthetic=True)
        arrays = dict(schema=np.array(SCHEMA), meta_json=np.array(json.dumps(meta)), K=np.int64(k_slots),
                      families=np.array(["synthetic"]), t=t, off=np.asarray(off, np.int64),
                      vid_names=np.array([f"veh{j}" for j in range(n_veh)]))
        dt = dict(slot=np.int16, vid=np.int32, s_m=np.float32, v_mps=np.float32, a_mps2=np.float32, lane=np.int8,
                  junction=bool, is_cav=bool, length_m=np.float32, cmd_accel_mps2=np.float32,
                  lane_intent=np.float32, dwell_age_s=np.float32)
        arrays.update({k: np.asarray(v_, dt[k]) for k, v_ in cols.items()})
        ldt = dict(cav_step=np.int32, cav_row=np.int64, accel_applied=np.float32, accel_u0=np.float32,
                   ctrl_src=np.int8, lane_label=np.int8, lane_valid=bool, lane_cmd=np.int8, plan_active=bool,
                   plan_family=np.int16, plan_role=np.int8, plan_act=np.int32, plan_onset=bool,
                   trigger_prob=np.float32, a_next=np.float32)
        arrays.update({k: np.asarray(v_, ldt[k]) for k, v_ in lab.items()})
        np.savez_compressed(out_dir / "shards" / f"distill_{ep_seed}.npz", **arrays)
        (out_dir / "receipts" / f"{ep_seed}.json").write_text(json.dumps(dict(seed=ep_seed, status="complete")))
    return out_dir
