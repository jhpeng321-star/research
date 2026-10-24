"""CPU-only R13Y inference, reusing the audited BLOWUP factorial worker.

Default is preflight ONLY; --run opts into the complete experiment. No fitting,
SUMO invocation, episode loading, oracle HDV parameters, or model-code patches.
The approved TRAIN export is pinned by --train-states-sha256. If a separate
300-second export is supplied, pin it with --train-spot-sha256 as well; its twelve
records must match the predetermined short-run states, including pre-t0 inputs.

Example (run from the updated parent repository, after reviewing preflight):
  python -B scripts/diag_r13y_offline.py --repo . --factorial-dir FACTORIAL \
    --checkpoint CK --states TRAIN.json.gz --train-states-sha256 SHA256 \
    --spot-states TRAIN300.json.gz --train-spot-sha256 SHA256 --out NEWDIR
Append --check-model to preflight the parent's API; append --run for evaluation.

Outputs: manifest.json, progress.jsonl, raw/{short,spot}/ARM/sNNN_kS.npz,
runs.json and summary.json. All intervals are paired percentile bootstrap 95%
CIs, resampling whole worlds (all states/seeds) and, separately, whole states
(both seeds). Worlds are source/episode identities, not t0-suffixed window IDs.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

# Set before importing numpy/torch, including in spawned CPU workers.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

DT, END, FREE, ROAD_END, LENGTH = 0.5, 75.44, 31.29, 553.33, 5.0
ARMS = ("off", "yield", "yield+emergency9")
LABELS = ("r13yOff_Tasis_Dasis_Con", "r13yYield_Tasis_Dasis_Con",
          "r13yEmergency_Tasis_Dasis_Con")
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth", "vtype")
BAD_SEED = re.compile(r"(?<!\d)964(?:0\d\d|1\d\d)(?!\d)")
# Exact source inspected in c4-blowup/scripts/factworker.py, also present on gpuR.
WORKER_SHA256 = "a18cff004d8a9296f12807f6576bacb94366ef9c75857a3616dbabd82103fe11"
G = {}


def safe_text(value):
    text = str(value).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN) or BAD_SEED.search(text):
        raise ValueError("restricted path/metadata refused")


def guard_path(value):
    """Check lexical spelling AND each resolved prefix before following links."""
    safe_text(value)
    path = Path(value).absolute()
    for part in reversed((path, *path.parents)):
        safe_text(part)
        safe_text(part.resolve())
    return path.resolve()


def install_access_guard(out=None):
    """Enforce restrictions on transitive Python opens, not just CLI arguments."""
    root = guard_path(out) if out else None

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = guard_path(os.fsdecode(args[0]))
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+"))
            writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR))
            if writing and (root is None or not path.is_relative_to(root)):
                raise PermissionError("write outside the diagnostic output refused")
        elif event == "os.mkdir":
            path = guard_path(os.fsdecode(args[0]))
            if root is None or not path.is_relative_to(root):
                raise PermissionError("directory creation outside diagnostic output refused")
        elif event == "import" and args[0].split(".")[0] in ("traci", "libsumo", "sumolib"):
            raise PermissionError("simulator import refused")
        elif event in ("subprocess.Popen", "os.system"):
            raise PermissionError("external commands are unavailable in this diagnostic")
    sys.addaudithook(audit)


def sha256(path):
    h = hashlib.sha256()
    with guard_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_metadata(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            safe_text(key)
            if key.lower() in ("split", "split_role") and str(value).lower() != "train":
                raise ValueError("non-TRAIN split refused")
            if key.lower() in ("seed", "world_seed") and value is not None:
                if 964000 <= int(value) <= 964199:
                    raise ValueError("restricted seed refused")
            check_metadata(value)
    elif isinstance(obj, list):
        for value in obj:
            check_metadata(value)
    elif isinstance(obj, str):
        safe_text(obj)


def read_states(path, approved_sha):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", approved_sha):
        raise ValueError("a SHA256 of the approved TRAIN export is required")
    if sha256(path) != approved_sha.lower():
        raise ValueError("TRAIN export checksum mismatch")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(guard_path(path), "rt", encoding="utf-8") as stream:
        # Reject forbidden metadata keys before decoding their associated values.
        text = stream.read()
    for key in re.findall(r'"([^"\\]*)"\s*:', text):
        safe_text(key)
    records = json.loads(text)
    if not isinstance(records, list) or not records:
        raise ValueError("expected a nonempty list of materialized TRAIN states")
    for rec in records:
        check_metadata(rec)
        if rec.get("source") not in ("v5_native", "genrand_plain"):
            raise ValueError("only the BLOWUP TRAIN export sources are accepted")
        if rec.get("seed") is None or not rec.get("episode"):
            raise ValueError("each state needs source, episode and an explicit world seed")
        if not isinstance(rec.get("history"), dict):
            raise ValueError("materialized pre-t0 history required")
        if rec.get("cav_controllers"):
            raise ValueError("expected native recorded-CAV replay, without controller overrides")
    return records


def identity(rec):
    return (rec["source"], rec["episode"], int(rec["seed"]), float(rec["t0_s"]))


def world_id(rec):
    return json.dumps([rec["source"], rec["episode"], int(rec["seed"])], separators=(",", ":"))


def recording_problem(rec, seconds):
    """Refuse nearest-frame extrapolation, gaps, and a future-offset t0."""
    raw = rec.get("recorded", {}).get("native")
    if isinstance(raw, dict):
        raw = raw.get("frames")
    if not isinstance(raw, list):
        return "missing native recorded frames"
    times = np.array([float(f["time_s"]) for f in raw])
    target = float(rec["t0_s"]) + DT * np.arange(int(seconds / DT) + 1)
    if len(times) < len(target) or not np.allclose(times[:len(target)], target, rtol=0, atol=1e-6):
        return f"native recording does not cover every 0.5s tick through {seconds}s"
    return None


def load_worker(repo, factorial):
    worker = guard_path(Path(factorial) / "factworker.py")
    if sha256(worker) != WORKER_SHA256:
        raise ValueError("factorial worker differs from the inspected BLOWUP source")
    for path in (repo, Path(repo) / "src", Path(repo) / "tools"):
        sys.path.insert(0, str(guard_path(path)))
    spec = importlib.util.spec_from_file_location("r13y_factorial_worker", worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_yield_settings(module):
    """Accept explicit physical properties; never interpret unconstrained logits."""
    values = {}
    for label, names in (("decel_mps2", ("decel_mps2", "decel")),
                         ("distance_m", ("distance_m", "distance"))):
        for name in names:
            value = getattr(module, name, None)
            if value is not None:
                value = value() if callable(value) else value
                values[label] = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                break
        else:
            raise RuntimeError(f"parent merge_yield must expose a physical {label} property")
    if not (math.isclose(values["decel_mps2"], 2, abs_tol=1e-5)
            and math.isclose(values["distance_m"], 60, abs_tol=1e-4)):
        raise RuntimeError(f"expected fresh S-R12C yield initialization 2/60, got {values}")
    parameters = list(module.named_parameters())
    if len(parameters) != 2 or any(p.numel() != 1 or not p.requires_grad for _, p in parameters):
        raise RuntimeError("merge_yield must have exactly two shared trainable scalars")
    values["parameters"] = {n: float(p.detach().cpu()) for n, p in parameters}
    return values


def init_models(repo, factorial, checkpoint, out, expected_checkpoint_sha=None):
    install_access_guard(out)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    fw = load_worker(repo, factorial)
    import wm_eval_common as wc
    original = wc.load_predictor
    settings = []

    def loader(path, mode, **kwargs):
        arm = ARMS[len(settings)]
        if guard_path(path) != guard_path(checkpoint) or mode != "AS_IS":
            raise RuntimeError("unexpected factorial checkpoint/mode")
        if expected_checkpoint_sha is not None and sha256(path) != expected_checkpoint_sha:
            raise RuntimeError("checkpoint changed after preflight")
        p, cap = original(str(guard_path(path)), mode, device="cpu", threads=1, **kwargs)
        m = p.model
        if not hasattr(m, "mainline_merge_yield_learnable"):
            raise RuntimeError("parent R13Y model API is not installed")
        if m.mainline_merge_yield_learnable:
            raise RuntimeError("S-R12C must load with learnable yield OFF")
        if arm != "off":
            enable = getattr(m, "enable_mainline_merge_yield_learnable", None)
            if not callable(enable):
                raise RuntimeError("parent enable_mainline_merge_yield_learnable() is missing")
            enable()
            if not m.mainline_merge_yield_learnable or getattr(m, "merge_yield", None) is None:
                raise RuntimeError("parent yield enable hook did not activate the module")
            values = physical_yield_settings(m.merge_yield)
        else:
            values = {}
        if arm == "yield+emergency9":
            holders = wc._physics_holders(m)
            if not holders:
                raise RuntimeError("no physics holders for emergency9 intervention")
            for owner, name in holders:
                setattr(owner, name, dataclasses.replace(getattr(owner, name), emergency_decel_mps2=9.0))
        m.to("cpu").eval()
        if any(p.device.type != "cpu" for p in m.parameters()):
            raise RuntimeError("non-CPU model refused")
        # Check the trainable interface above, then freeze everything for inference.
        m.requires_grad_(False)
        settings.append({"arm": arm, "yield": values,
                         "emergency_decel_mps2": m.physics.emergency_decel_mps2})
        return p, cap

    wc.load_predictor = loader
    try:
        fw.init_arms([(label, checkpoint) for label in LABELS], entries="off")
    finally:
        wc.load_predictor = original
    G.update(fw=fw, settings=settings, out=out)


def trajectory_metrics(pos, lane, mask, speed, is_cav):
    """t0 roster; BLOWUP zone/safety grid (0,H], factorial delay grid [0,H).

    HDV zone/standing/stuck/merge/safety metrics. Same-lane safety includes CAV
    leaders; overlap is unordered vehicle pairs involving >=1 HDV. Delay uses
    all t0 vehicles on the factorial 1s grid, with additional HDV-only totals.
    Missing zone speed/head/TTC is NaN (JSON null), never silently zero.
    """
    pos, speed = np.asarray(pos, float), np.asarray(speed, float)
    lane = np.rint(lane).astype(int)
    mask = np.asarray(mask, bool) & (lane >= 0) & (lane <= 2)
    if pos.shape != speed.shape or pos.shape != lane.shape or pos.shape != mask.shape:
        raise ValueError("trajectory array shapes disagree")
    if not np.isfinite(pos[mask]).all() or not np.isfinite(speed[mask]).all():
        raise ValueError("nonfinite present-vehicle kinematics")
    hdv = ~np.asarray(is_cav, bool)
    steps = len(pos) - 1
    valid = mask[1:] & hdv[None]
    zone = valid & (lane[1:] == 1) & (pos[1:] >= 0) & (pos[1:] < END)
    standing = zone & (speed[1:] < 2)
    heads = np.where(standing, pos[1:], -np.inf).max(1)
    has_head = np.isfinite(heads)
    out = {"zone_occupancy_veh": float(zone.sum() / steps),
           "zone_speed_mps": float(speed[1:][zone].mean()) if zone.any() else np.nan,
           "standing_column_veh": float(standing.sum() / steps),
           "standing_any_fraction": float(standing.any(1).mean()),
           "standing_head_m": float(heads[has_head].mean()) if has_head.any() else np.nan,
           "hdv_exposure_veh_s": float(valid.sum() * DT)}
    episodes = []
    for j in range(pos.shape[1]):
        changes = np.diff(np.r_[False, standing[:, j], False].astype(int))
        episodes.extend((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)) * DT)
    out["standing_episode_mean_s"] = float(np.mean(episodes)) if episodes else np.nan
    ramp = mask[0] & hdv & (lane[0] == 2)
    merged = (mask & ((lane == 0) | (lane == 1))).any(0) & ramp
    out.update(n_ramp_hdv=int(ramp.sum()), merges_per_run=int(merged.sum()),
               stuck_ramps=int((ramp & ~merged).sum()),
               stuck_present_end=int((ramp & ~merged & mask[-1] & (lane[-1] == 2)).sum()),
               ramp_censored_unmerged=int((ramp & ~merged & ~mask[-1]).sum()))
    transition = mask[1:] & mask[:-1] & hdv[None]
    braking = (np.diff(speed, axis=0) / DT < -3) & transition
    out["hard_brake_veh_s"] = float(braking.sum() * DT)
    out["hard_brake_events"] = int((braking & ~np.vstack([np.zeros_like(hdv), braking[:-1]])).sum())
    min_ttc, low_ttc, overlap_time, overlap_events = np.inf, 0, 0, 0
    previous_pairs = set()
    previous_hdv_pairs = set()
    hdv_overlap_events, hdv_overlap_time = 0, 0.0
    for i in range(steps + 1):
        pairs = set()
        for ln in range(3):
            ids = np.flatnonzero(mask[i] & (lane[i] == ln) & (pos[i] < ROAD_END))
            ids = ids[np.argsort(pos[i, ids], kind="stable")]
            if len(ids) < 2:
                continue
            f, lead = ids[:-1], ids[1:]
            gap = pos[i, lead] - pos[i, f] - LENGTH
            closing = speed[i, f] - speed[i, lead]
            eligible = hdv[f] & (closing > 0) & (gap >= 0)
            ttc = gap[eligible] / closing[eligible]
            if i > 0 and len(ttc):
                min_ttc = min(min_ttc, float(ttc.min()))
                low_ttc += int((ttc < 1.5).sum())
            # All pairs, including exact position ties and multi-car overlaps.
            for k, follower in enumerate(ids[:-1]):
                for leader in ids[k + 1:]:
                    if pos[i, leader] - pos[i, follower] >= LENGTH:
                        break
                    if hdv[follower] or hdv[leader]:
                        pairs.add(tuple(sorted((int(follower), int(leader)))))
        hdv_pairs = {pair for pair in pairs if hdv[pair[0]] and hdv[pair[1]]}
        if i > 0:
            overlap_time += len(pairs) * DT
            overlap_events += len(pairs - previous_pairs)
            hdv_overlap_time += len(hdv_pairs) * DT
            hdv_overlap_events += len(hdv_pairs - previous_hdv_pairs)
        previous_pairs = pairs
        previous_hdv_pairs = hdv_pairs
    out.update(ttc_min_s=min_ttc if np.isfinite(min_ttc) else np.nan,
               ttc_lt1p5_veh_s=float(low_ttc * DT), overlap_pair_s=float(overlap_time),
               overlap_events=int(overlap_events),
               overlap_hdv_hdv_events=int(hdv_overlap_events),
               overlap_hdv_hdv_pair_s=float(hdv_overlap_time))
    for prefix, selection in (("delay", np.ones_like(hdv)), ("hdv_delay", hdv)):
        p, region_lane, v = pos[:steps:2], lane[:steps:2], speed[:steps:2]
        m = mask[:steps:2] & selection[None] & (p < ROAD_END)
        deficit = np.maximum(0, 1 - v / FREE)
        regions = {"ramp": (region_lane == 2) & (p >= -378.66) & (p < END),
                   "up_near": (region_lane < 2) & (p >= -800) & (p < 0),
                   "merge": (region_lane < 2) & (p >= 0) & (p < END),
                   "downstream": (region_lane < 2) & (p >= END)}
        for region, select in regions.items():
            out[f"{prefix}_{region}_veh_s"] = float(deficit[m & select].sum())
    # BLOWUP armsum.py's primary 60-120s window plus 30s detail and long tail.
    horizon = int(steps * DT)
    spans = [(start, min(start + 30, horizon)) for start in range(0, horizon, 30)]
    spans += [(a, b) for a, b in ((60, 120), (120, 300)) if b <= horizon]
    for start, stop in spans:
        sl = slice(int(start / DT), int(stop / DT))
        z, st = zone[sl], standing[sl]
        suffix = f"_{start}_{stop}s"
        out["zone_occupancy_veh" + suffix] = float(z.sum() / len(z))
        out["zone_speed_mps" + suffix] = float(speed[1:][sl][z].mean()) if z.any() else np.nan
        out["standing_column_veh" + suffix] = float(st.sum() / len(st))
    return out


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    guard_path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def work(task):
    phase, idx, rec, sample, seconds = task
    import torch
    fw = G["fw"]
    rec = dict(rec, _fact_idx=idx)
    rows = []
    for arm, label in zip(ARMS, LABELS):
        started = time.perf_counter()
        # Reset every random stream for each arm, independent of worker scheduling.
        crn = 100003 * idx + 7919 * sample + 17
        random.seed(crn)
        np.random.seed(crn)
        torch.manual_seed(crn)
        with torch.inference_mode():
            res = fw.roll_once(label, rec, sample, int(seconds / DT), capture=False)
        n = res["n"]
        if res["pos"].shape != (int(seconds / DT) + 1, n):
            raise RuntimeError("factorial returned the wrong horizon/roster")
        wm = trajectory_metrics(res["pos"], res["lane"], res["mask"], res["v"], res["is_cav"])
        recorded = trajectory_metrics(res["rec_pos"], res["rec_lane"], res["rec_mask"],
                                      res["rec_v"], res["is_cav"])
        for metric in list(wm):
            if not metric.startswith(("zone_occupancy_veh", "zone_speed_mps", "standing_column_veh")):
                continue
            wm[metric + "_ratio_sumo"] = wm[metric] / recorded[metric] if recorded[metric] > 0 else np.nan
        command = np.stack([c["accel"][:n] for c in res["calls"]])
        command_present = res["mask"][1:] & ~res["is_cav"][None]
        hard = (command < -3) & command_present
        wm["command_hard_brake_veh_s"] = float(hard.sum() * DT)
        wm["command_hard_brake_events"] = int((hard & ~np.vstack([np.zeros(n, bool), hard[:-1]])).sum())
        cav = res["is_cav"][None] & res["mask"] & res["rec_mask"]
        err = np.abs(res["pos"] - res["rec_pos"])[cav]
        cav_error = float(err.max()) if err.size else 0.0
        if cav_error > 1e-3:
            raise RuntimeError(f"recorded CAV replay failed, max position error {cav_error}")
        target = guard_path(Path(G["out"]) / "raw" / phase / arm / f"s{idx:03d}_k{sample}.npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: res[key] for key in ("pos", "lane", "mask", "v", "is_cav", "rec_pos",
                                           "rec_lane", "rec_mask", "rec_v")}
        arrays["accel"] = command
        arrays["phys"] = np.stack([c["phys"][:n] for c in res["calls"]])
        np.savez_compressed(target, **arrays)
        rows.append({"phase": phase, "state": idx, "world": world_id(rec), "identity": identity(rec),
                     "sample": sample, "arm": arm, "crn_seed": crn, "seconds": seconds,
                     "runtime_s": time.perf_counter() - started, "cav_max_error_m": cav_error,
                     "metrics": wm, "sumo": recorded})
    # Protect the paired comparison against arm-dependent reference construction.
    if any(clean_json(r["sumo"]) != clean_json(rows[0]["sumo"]) for r in rows[1:]):
        raise RuntimeError("recorded SUMO reference changed across arms")
    return rows


def clustered_ci(values, groups, draws, seed):
    """Resample whole clusters; retain every state/seed within a selected cluster."""
    x = np.asarray(values, float)
    g = np.asarray(groups)
    valid = np.isfinite(x)
    x, g = x[valid], g[valid]
    if not len(x):
        return {"mean": None, "ci95": [None, None], "n_pairs": 0, "n_clusters": 0}
    labels, inverse = np.unique(g, return_inverse=True)
    sums = np.bincount(inverse, weights=x)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(labels), size=(draws, len(labels)))
    means = sums[picks].sum(1) / counts[picks].sum(1)
    return {"mean": float(x.mean()), "ci95": np.percentile(means, [2.5, 97.5]).tolist(),
            "n_pairs": len(x), "n_clusters": len(labels)}


def clustered_ratio_ci(numerators, denominators, groups, draws, seed):
    """Ratio of cohort means, preserving pairing and zero-occupancy runs."""
    a, b = np.asarray(numerators, float), np.asarray(denominators, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[valid], b[valid], np.asarray(groups)[valid]
    if not len(a) or b.sum() <= 0:
        return {"ratio": None, "ci95": [None, None], "n_pairs": len(a),
                "n_clusters": len(np.unique(g)), "valid_bootstrap_draws": 0}
    labels, inverse = np.unique(g, return_inverse=True)
    sa, sb = np.bincount(inverse, weights=a), np.bincount(inverse, weights=b)
    picks = np.random.default_rng(seed).integers(len(labels), size=(draws, len(labels)))
    den = sb[picks].sum(1)
    samples = sa[picks].sum(1)[den > 0] / den[den > 0]
    return {"ratio": float(a.sum() / b.sum()),
            "ci95": np.percentile(samples, [2.5, 97.5]).tolist() if len(samples) else [None, None],
            "n_pairs": len(a), "n_clusters": len(labels), "valid_bootstrap_draws": len(samples)}


def summarize(rows, draws, seed):
    summary = {"ci": "paired cluster percentile bootstrap, 95%", "draws": draws,
               "seed": seed, "sign": "left minus right", "phases": {}}
    for phase in ("short", "spot"):
        phase_rows = [r for r in rows if r["phase"] == phase]
        indexed = {(r["state"], r["sample"], r["arm"]): r for r in phase_rows}
        pairs = sorted({(r["state"], r["sample"]) for r in phase_rows})
        expected = 48 if phase == "short" else 12
        if (len(pairs) != expected * 2 or len(indexed) != expected * 2 * 3
                or len(phase_rows) != expected * 2 * 3):
            raise RuntimeError(f"incomplete or duplicate paired runs for {phase}")
        comparisons = [(arm, "sumo") for arm in ARMS]
        comparisons += [("yield", "off"), ("yield+emergency9", "off"), ("yield+emergency9", "yield")]
        report = {}
        for left, right in comparisons:
            metrics = sorted(indexed[(*pairs[0], left)]["metrics"])
            result = {}
            for metric in metrics:
                if right == "sumo" and metric not in indexed[(*pairs[0], left)]["sumo"]:
                    continue
                a, b, worlds, states = [], [], [], []
                for idx, sample in pairs:
                    lr = indexed[(idx, sample, left)]
                    rr = lr["sumo"] if right == "sumo" else indexed[(idx, sample, right)]["metrics"]
                    av, bv = lr["metrics"][metric], rr[metric]
                    a.append(av if av is not None else np.nan)
                    b.append(bv if bv is not None else np.nan)
                    worlds.append(lr["world"])
                    states.append(idx)
                a, b = np.asarray(a), np.asarray(b)
                delta = a - b
                finite = np.isfinite(delta)
                result[metric] = {"left_mean": float(a[finite].mean()) if finite.any() else None,
                                  "right_mean": float(b[finite].mean()) if finite.any() else None,
                                  "world_cluster": clustered_ci(delta, worlds, draws, seed),
                                  "state_cluster": clustered_ci(delta, states, draws, seed)}
            report[f"{left} minus {right}"] = result
        # Absolute per-arm values (including ratio-to-SUMO metrics), with clustered CIs.
        absolute = {}
        for arm in (*ARMS, "sumo"):
            rr = [indexed[(*pair, ARMS[0] if arm == "sumo" else arm)] for pair in pairs]
            key = "sumo" if arm == "sumo" else "metrics"
            absolute[arm] = {metric: {
                "world_cluster": clustered_ci([r[key][metric] for r in rr], [r["world"] for r in rr], draws, seed),
                "state_cluster": clustered_ci([r[key][metric] for r in rr], [r["state"] for r in rr], draws, seed)}
                for metric in rr[0][key]}
        ratios = {}
        for arm in ARMS:
            rr = [indexed[(*pair, arm)] for pair in pairs]
            ratios[arm] = {metric: {
                "world_cluster": clustered_ratio_ci([r["metrics"][metric] for r in rr],
                    [r["sumo"][metric] for r in rr], [r["world"] for r in rr], draws, seed),
                "state_cluster": clustered_ratio_ci([r["metrics"][metric] for r in rr],
                    [r["sumo"][metric] for r in rr], [r["state"] for r in rr], draws, seed)}
                for metric in rr[0]["sumo"]
                if metric.startswith(("zone_occupancy_veh", "zone_speed_mps", "standing_column_veh"))}
        summary["phases"][phase] = {"comparisons": report, "absolute": absolute,
                                     "ratio_of_means_to_sumo": ratios}
    return clean_json(summary)


def preflight(args):
    repo, factorial = guard_path(args.repo), guard_path(args.factorial_dir)
    checkpoint, states, out = map(guard_path, (args.checkpoint, args.states, args.out))
    if not str(states).endswith((".json", ".json.gz")):
        raise ValueError("states must be a materialized JSON or JSON.gz TRAIN export")
    if args.spot_states and not str(args.spot_states).endswith((".json", ".json.gz")):
        raise ValueError("spot states must be a materialized JSON or JSON.gz TRAIN export")
    if not checkpoint.is_file() or not repo.is_dir() or not factorial.is_dir():
        raise ValueError("repo, factorial directory and checkpoint must exist")
    if out.exists():
        raise ValueError("output must be a fresh directory (no mixed/resumed experiments)")
    if not out.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    # Guard every sidecar the shared checkpoint loader can consult.
    for name in ("physics.json", "TRAIN_RECEIPT.json", "lane_calib.json"):
        guard_path(checkpoint.with_name(name))
    worker = factorial / "factworker.py"
    if sha256(worker) != WORKER_SHA256:
        raise ValueError("factorial worker differs from the audited BLOWUP source")
    records = read_states(states, args.train_states_sha256)
    if len(records) != 192:
        raise ValueError("BLOWUP protocol expects exactly 192 TRAIN states before stride=4")
    selected = records[::4]
    if len({identity(r) for r in selected}) != 48:
        raise ValueError("duplicate TRAIN state identities")
    short = [("short", idx, r, sample, 120) for idx, r in enumerate(selected) for sample in (0, 1)]
    spot_indices = list(range(0, 48, 4))  # fixed before any outcome is read
    spot_map = {identity(r): r for r in selected}
    if args.spot_states:
        if not args.train_spot_sha256:
            raise ValueError("separate TRAIN spot export needs --train-spot-sha256")
        longer = read_states(args.spot_states, args.train_spot_sha256)
        spot_map = {identity(r): r for r in longer}
        if len(spot_map) != len(longer):
            raise ValueError("duplicate identities in TRAIN spot export")
    problems, spot = [], []
    for idx, rec in enumerate(selected):
        problem = recording_problem(rec, 120)
        if problem:
            problems.append(f"short state {idx}: {problem}")
    for idx in spot_indices:
        rec = spot_map.get(identity(selected[idx]))
        if rec is None:
            problems.append(f"spot state {idx}: matching TRAIN identity missing")
            continue
        for key in ("initial_state", "history", "counter", "demand_vph", "penetration"):
            if rec.get(key) != selected[idx].get(key):
                raise ValueError(f"spot state {idx}: pre-t0 input {key} changed")
        problem = recording_problem(rec, 300)
        if problem:
            problems.append(f"spot state {idx}: {problem}")
        else:
            native = rec["recorded"]["native"]
            native = native["frames"] if isinstance(native, dict) else native
            original = selected[idx]["recorded"]["native"]
            original = original["frames"] if isinstance(original, dict) else original
            if native[:len(original)] != original:
                raise ValueError(f"spot state {idx}: original 120s recorded prefix changed")
        spot.extend(("spot", idx, rec, sample, 300) for sample in (0, 1))
    sources = [Path(__file__).resolve(), worker, repo / "tools/wm_eval_common.py",
               repo / "src/wms/eval/predictor.py", repo / "src/wms/eval/rollout_inputs.py",
               repo / "src/wms/model/checkpoint.py", repo / "src/wms/model/wms.py"]
    files = {str(p): sha256(p) for p in sources}
    sidecars = {name: sha256(checkpoint.with_name(name)) for name in
                ("physics.json", "TRAIN_RECEIPT.json", "lane_calib.json") if checkpoint.with_name(name).is_file()}
    manifest = {"protocol": "R13Y/BLOWUP/1", "repo": str(repo), "factorial": str(factorial),
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "states": str(states), "states_sha256": args.train_states_sha256.lower(),
                "spot_states": str(guard_path(args.spot_states)) if args.spot_states else str(states),
                "spot_sha256": args.train_spot_sha256 or args.train_states_sha256,
                "train_provenance": "operator-pinned approved TRAIN exports; v5_native/genrand_plain only",
                "source_sha256": files, "checkpoint_sidecar_sha256": sidecars,
                "selection": [{"state": i, "source_index": i * 4, "identity": identity(r),
                               "world_cluster": world_id(r)} for i, r in enumerate(selected)],
                "spot_indices": spot_indices, "arms": ARMS, "samples": [0, 1],
                "short_seconds": 120, "spot_seconds": 300, "arm_rollouts": 360,
                "equivalent_120s_arm_rollouts": 468, "workers": args.workers,
                "device": "cpu", "threads_per_worker": 1, "entries": "off", "hdv_exits": "sumo",
                "cav_arm": "native", "crn": "100003*state_index + 7919*sample + 17; lane_seed=sample",
                "bootstrap": {"draws": args.bootstrap_draws, "seed": args.bootstrap_seed,
                              "clusters": ["source/episode/seed world (all states+seeds)", "state (both seeds)"]},
                "metric_conventions": {"zone": "HDV lane1 0<=s<75.44m; time-mean count, vehicle-time mean speed",
                    "primary_zone_window": "60_120s (exact BLOWUP armsum convention); whole horizon and 30s windows also reported",
                    "standing": "v<2m/s in zone; episode durations are window-censored",
                    "stuck": "t0 ramp HDVs without an observed mainline merge; exit censoring reported separately",
                    "safety": "same-lane bumper gaps, s<553.33m, production length 5m; HDV follower TTC; realized and commanded hard brakes <-3",
                    "delay": "factorial all-t0-roster 1s grid and additional HDV-only; vref=31.29m/s",
                    "integration": "zone/safety (0,H], delay [0,H) at 1s; initial overlap pairs are not new events",
                    "ratios": "ratio_of_means_to_sumo retains zero-reference runs; per-run ratios also reported",
                    "missing": "null, finite paired observations only; n_pairs/n_clusters in each CI"},
                "audit": "pinned factworker; neutral r13y labels; no oracle/transplant/D/T interventions",
                "recording_problems": problems}
    return manifest, short + spot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--factorial-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--train-states-sha256", required=True, help="SHA256 of operator-approved TRAIN export")
    parser.add_argument("--spot-states")
    parser.add_argument("--train-spot-sha256")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=130013)
    parser.add_argument("--check-model", action="store_true", help="preflight the parent's CPU model interface, no rollout")
    parser.add_argument("--run", action="store_true", help="execute all 360 inference rollouts (default: preflight only)")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4 or args.bootstrap_draws < 100 or not 0 <= args.bootstrap_seed < 2**32:
        parser.error("1<=workers<=4, bootstrap-draws>=100 and a uint32 bootstrap seed are required")
    if 964000 <= args.bootstrap_seed <= 964199:
        parser.error("restricted bootstrap seed")
    for key in ("repo", "factorial_dir", "checkpoint", "states", "spot_states", "out"):
        if getattr(args, key) is not None:
            setattr(args, key, str(guard_path(getattr(args, key))))
    install_access_guard(args.out if args.run else None)
    manifest, tasks = preflight(args)
    if args.check_model:
        init_models(args.repo, args.factorial_dir, args.checkpoint, args.out if args.run else None,
                    manifest["checkpoint_sha256"])
        manifest["arm_settings"] = G["settings"]
        if not args.run:
            G.clear()
    if not args.run:
        print(json.dumps(clean_json(manifest), indent=2, allow_nan=False))
        return 2 if manifest["recording_problems"] else 0
    if manifest["recording_problems"]:
        raise ValueError("recording preflight failed: " + "; ".join(manifest["recording_problems"]))
    # Check all three parent interfaces before starting a pool or any rollouts.
    if not args.check_model:
        init_models(args.repo, args.factorial_dir, args.checkpoint, args.out, manifest["checkpoint_sha256"])
        manifest["arm_settings"] = G["settings"]
    out = guard_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest["python"] = sys.version
    manifest["numpy"] = np.__version__
    import torch
    manifest["torch"] = torch.__version__
    manifest["runtime_source_sha256"] = {
        name: sha256(module.__file__) for name, module in list(sys.modules.items())
        if name.startswith("wms.") and str(getattr(module, "__file__", "")).endswith(".py")}
    write_json(out / "manifest.json", manifest)
    started, rows = time.perf_counter(), []
    with (out / "progress.jsonl").open("w", encoding="utf-8") as log:
        if args.workers == 1:
            results = map(work, tasks)
            for batch in results:
                rows.extend(batch)
                log.write(json.dumps(clean_json(batch), allow_nan=False) + "\n")
                log.flush()
                print(f"{len(rows)}/360 arm runs, {time.perf_counter()-started:.1f}s", flush=True)
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            G.clear()
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=init_models,
                                     initargs=(args.repo, args.factorial_dir, args.checkpoint, args.out,
                                               manifest["checkpoint_sha256"])) as pool:
                for batch in pool.map(work, tasks, chunksize=1):
                    rows.extend(batch)
                    log.write(json.dumps(clean_json(batch), allow_nan=False) + "\n")
                    log.flush()
                    print(f"{len(rows)}/360 arm runs, {time.perf_counter()-started:.1f}s", flush=True)
    rows.sort(key=lambda r: (r["phase"], r["state"], r["sample"], r["arm"]))
    write_json(out / "runs.json", rows)
    summary = summarize(rows, args.bootstrap_draws, args.bootstrap_seed)
    summary["runtime_s"] = time.perf_counter() - started
    write_json(out / "summary.json", summary)
    print(f"complete: {out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
