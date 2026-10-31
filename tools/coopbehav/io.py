"""Allowlisted E3DATA reader. Vehicle IDs, never reusable tape slots, join clocks."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re

import numpy as np

BAD = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype", re.I)
SEEDS = re.compile(r"(?<!\d)(?:9640\d\d|9641\d\d|962\d{3}|977\d{3})(?!\d)")


def safe(path):
    p = Path(path)
    for text in (str(p), str(p.resolve())):
        if BAD.search(text) or SEEDS.search(text):
            raise ValueError(f"Restricted path: {p}")
    return p


def check_seed(seed):
    n = int(seed)
    if 964000 <= n <= 964199 or 962000 <= n <= 962999 or 977000 <= n <= 977999:
        raise ValueError(f"Restricted seed: {n}")


def read_json(path):
    return json.loads(safe(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Vehicle:
    x: float
    v: float
    length: float
    lane: int
    cav: bool
    issued: float | None = None
    controller: int | str | None = None


@dataclass
class Tape:
    time: np.ndarray
    rows: list[dict[str, Vehicle]]
    meta: dict = field(default_factory=dict)
    eps: float = np.finfo(np.float64).eps
    hashes: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        if len(self.time) != len(self.rows) or len(self.time) < 2:
            raise ValueError("At least two matching trajectory frames required")
        if not np.isfinite(self.time).all() or not np.all(np.diff(self.time) > 0):
            raise ValueError("Clocks must be finite and strictly increasing")
        if "seed" in self.meta:
            check_seed(self.meta["seed"])
        for row in self.rows:
            for v in row.values():
                if not np.isfinite([v.x, v.v, v.length]).all() or v.v < 0 or v.length <= 0:
                    raise ValueError("Invalid observable vehicle geometry/speed")


def load_tape(root, metadata=None):
    root = safe(root)
    meta = read_json(root / "episode.json") if (root / "episode.json").is_file() else dict(metadata or {})
    if metadata:
        meta.update(metadata)
    check_seed(meta["seed"])
    hashes = {}

    def array(name, required=True):
        p = safe(root / (name + ".npy"))
        if not p.is_file() and not required:
            return None
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return np.load(p, allow_pickle=False, mmap_mode="r")

    t, kin, mask, ids, cav = (array(n) for n in ("time_s", "veh_kin", "veh_mask", "vid", "is_cav"))
    if kin.ndim != 3 or kin.shape[2] < 7 or mask.shape != kin.shape[:2]:
        raise ValueError("Invalid E3DATA array shapes")
    if ids.shape != mask.shape or cav.shape != mask.shape or t.shape != (len(kin),):
        raise ValueError("Identity/CAV/clock shape mismatch")
    ct, issued, controller = (array(n, False) for n in
                              ("control_time_s", "issued_acceleration_mps2", "controller"))
    if ct is not None:
        if ct.ndim != 1 or len(ct) < 1 or not np.isfinite(ct).all() or not np.all(np.diff(ct) > 0):
            raise ValueError("Invalid control clocks")
        for a in (issued, controller):
            if a is not None and a.shape != (len(ct), kin.shape[1]):
                raise ValueError("Command ledger shape mismatch")
    elif issued is not None or controller is not None:
        raise ValueError("Command arrays require control clocks")
    branch = meta.get("branch", {})
    start = float(branch.get("branch_t_s", branch.get("t0_s", meta.get("warmup_s", t[0]))))
    end = start + float(branch.get("horizon_s", meta.get("horizon_s", t[-1] - start)))
    # A command is held only until the next recorded command (or one final cadence).
    cadence = float(np.median(np.diff(ct))) if ct is not None and len(ct) > 1 else 0.
    rows, times = [], []
    missing_commands = junction = 0
    for i, clock in enumerate(t):
        if not start <= clock <= end:
            continue
        row = {}
        ci = int(np.searchsorted(ct, clock, side="right") - 1) if ct is not None else -1
        ci_valid = ci >= 0 and (ci < len(ct) - 1 or clock < ct[-1] + cadence)
        for k in np.flatnonzero(mask[i]):
            identity = str(ids[i, k].item())
            if identity in row:
                raise ValueError("Duplicate identity at one clock")
            lane_bits = kin[i, k, 1:4]
            lane = int(np.argmax(lane_bits)) if np.sum(lane_bits) > 0 else -1
            junction += int(lane < 0)
            command = control = None
            if ci_valid:
                # The slot must contain this identity at the command clock too.
                si = int(np.searchsorted(t, ct[ci]))
                if si < len(t) and t[si] == ct[ci] and mask[si, k] and ids[si, k] == ids[i, k]:
                    if issued is not None and np.isfinite(issued[ci, k]):
                        command = float(issued[ci, k])
                    if controller is not None:
                        control = controller[ci, k].item()
            missing_commands += int(bool(cav[i, k]) and command is None)
            row[identity] = Vehicle(float(kin[i, k, 0]), float(kin[i, k, 4]),
                                    float(kin[i, k, 6]), lane, bool(cav[i, k]), command, control)
        rows.append(row)
        times.append(float(clock))
    return Tape(np.array(times), rows, meta, np.finfo(kin.dtype).eps, hashes,
                dict(missing_cav_command_frames=missing_commands, junction_vehicle_frames=junction,
                     command_ledger_available=issued is not None, branch_start_s=start, branch_end_s=end,
                     trace_scope="full_observable_fleet"))


def manifest_entries(root, arms=("mappo", "ctrl0")):
    doc = read_json(safe(root) / "C4_EPISODES_MANIFEST.json")
    result = []
    for e in doc["episodes"]:
        # Never open a restricted episode, even if it appears in an input manifest.
        if e.get("arm") not in arms:
            continue
        check_seed(e["seed"])
        path = safe(Path(root) / e["dir"])
        if not path.resolve().is_relative_to(safe(root).resolve()):
            raise ValueError("Episode escapes source root")
        result.append((path, e))
    return result
