"""Attention diagnostic for the trained C1 WM-S transformer (CPU only, no training, no SUMO).

QUESTION: does the HdvResidualNet (3 x _KinBlock, 4 heads, attention restricted to self +
vehicles <=150 m in the same/adjacent lane, plus a learned pair bias) attend mostly to the
car directly ahead, or does it put substantial weight on cars behind / far away?

The model code is NOT modified: forward hooks on each ``blocks[b].attn`` capture the
returned ``(output, attn)`` tuple (attn is ``[B, H, K, K]``; row = query i, column = key j,
softmax over keys).  The rollout is the scoring-harness rollout (``WmsCheckpointModel.
predict_states``, CAVs teacher-forced from the recorded arm, entries OFF, physics from the
checkpoint's TRAIN_RECEIPT), 60 s, seed 0.

Roles are assigned relative to the QUERY HDV at the sampled step (see ``roles_for_query``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORKSPACE = Path("E:/rd-wt/ws-attn")
WM_N1 = WORKSPACE / "output" / "wd01" / "wm_n1"
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

OUT_DIR = WORKSPACE / "_out" / "attention_check"
CKPT = OUT_DIR / "_ckpt" / "wm_s.pt"
RECEIPT = Path("E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_struct/TRAIN_RECEIPT.json")
CKPT_SOURCE = Path("E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_struct/wm_s.pt")
STATES_DIR = Path("E:/rd-wt/ws-wallc1/_out/wallc1_states_v2")
SETS = {"wall_same": STATES_DIR / "wall_same.json",
        "seen_controls": STATES_DIR / "seen_controls.json"}
STORED_SCORES = Path("E:/rd-wt/ws-wallc1/_out/wall_scores/struct/states__struct__wall_same.json")

HORIZON_S = 60.0
SEED = 0
EVERY = 5                      # sample every 5th 0.5 s step (= every 2.5 s)
LC_LOOKAHEAD_STEPS = 4         # 2 s
RADIUS_M = 150.0
VEH_LEN_M = 5.0                # wms_plugin.DEFAULT_LENGTH_M (all vehicles)
MERGE_S0, MERGE_S1, MERGE_PAD = 0.0, 75.44, 50.0   # TRAIN_RECEIPT physics merge zone
NB, NH = 3, 4

ROLES = ["self", "leader", "ahead_2nd", "ahead_3rd", "ahead_4th+", "follower",
         "behind_2nd+", "adj_ahead_near", "adj_ahead_far", "adj_behind_near",
         "adj_behind_far"]
NR = len(ROLES)
# distance bins (non-self candidates only): ahead then behind, by |ds|
BIN_EDGES = [0.0, 20.0, 50.0, 100.0, 150.0 + 1e-6]
BIN_NAMES = ["ahead 0-20", "ahead 20-50", "ahead 50-100", "ahead 100-150",
             "behind 0-20", "behind 20-50", "behind 50-100", "behind 100-150"]
NBIN = len(BIN_NAMES)
SCAL = ["behind", "ahead", "far50", "far100", "absds_num", "entropy", "effn"]
NS = len(SCAL)

GROUPS = ["all", "free", "follow", "jam", "merge", "lc_any", "lc_main", "lc_ramp",
          "no_lc", "free_plain", "follow_plain", "jam_plain", "lc_main_nomerge",
          "merge_no_lc"]
GROUP_DOC = {
    "all": "every HDV query",
    "free": "free flow: speed > 20 m/s and bumper gap to leader > 50 m (or no leader)",
    "follow": "following: not free, speed > 5 m/s",
    "jam": "slow / jam: speed <= 5 m/s",
    "merge": "in or near merge zone: on-ramp lane 2, or s in [-50, 125.44] m",
    "lc_any": "query changes lane within the next 2 s (4 steps)",
    "lc_main": "lane change within 2 s, query starts in lane 0/1",
    "lc_ramp": "lane change within 2 s, query starts on ramp lane 2 (the merge)",
    "no_lc": "no lane change within the next 2 s",
    "free_plain": "free flow, outside merge zone, no lane change in 2 s",
    "follow_plain": "following, outside merge zone, no lane change in 2 s",
    "jam_plain": "slow/jam, outside merge zone, no lane change in 2 s",
    "lc_main_nomerge": "lane change within 2 s, mainline lanes, outside merge zone",
    "merge_no_lc": "in/near merge zone, no lane change in 2 s",
}

# ----------------------------------------------------------------- feature layout
_LAYOUT: dict[str, tuple[int, tuple]] = {}
_off = 0
for _name, _shape in [("count", ()), ("cand", ()), ("count_ex", ()),
                      ("uni", (NR,)), ("cand_roles", (NR,)),
                      ("bin_uni", (NBIN,)), ("n_binq", ()),
                      ("mass", (NB, NH, NR)), ("mass_cav", (NB, NH, NR)),
                      ("mass_ex", (NB, NH, NR)),
                      ("scal", (NB, NH, NS)), ("bins", (NB, NH, NBIN)),
                      ("cav_cand_share", ()), ("cav_share_nq", ())]:
    _size = int(np.prod(_shape)) if _shape else 1
    _LAYOUT[_name] = (_off, _shape)
    _off += _size
FLAT = _off


def _view(vec: np.ndarray, name: str) -> np.ndarray:
    off, shape = _LAYOUT[name]
    size = int(np.prod(shape)) if shape else 1
    out = vec[off:off + size]
    return out.reshape(shape) if shape else out[0]


def _put(vec: np.ndarray, name: str, value) -> None:
    off, shape = _LAYOUT[name]
    size = int(np.prod(shape)) if shape else 1
    vec[off:off + size] = np.asarray(value, dtype=np.float64).reshape(-1)


# ------------------------------------------------------------------------ roles
def roles_for_query(i: int, s: np.ndarray, lane: np.ndarray):
    """Role id per key j (``-1`` = not a candidate) for query ``i``.

    Candidates replicate ``_KinAttention._neighbourhood``: self, plus keys with
    ``|lane_j - lane_i| <= 1`` and ``|s_j - s_i| <= 150 m``.  ``ds = s_j - s_i`` (>0 = ahead).
    """
    n = s.shape[0]
    ds = s - s[i]
    dl = lane - lane[i]
    cand = (np.abs(ds) <= RADIUS_M) & (np.abs(dl) <= 1) & (lane >= 0) & (lane[i] >= 0)
    cand[i] = True
    role = np.full(n, -1, dtype=np.int64)
    role[i] = 0
    same = cand & (dl == 0)
    same[i] = False
    ahead_idx = np.flatnonzero(same & (ds > 0))
    ahead_idx = ahead_idx[np.argsort(ds[ahead_idx])]
    for rank, j in enumerate(ahead_idx):
        role[j] = 1 + min(rank, 3)
    behind_idx = np.flatnonzero(same & (ds <= 0))
    behind_idx = behind_idx[np.argsort(-ds[behind_idx])]
    for rank, j in enumerate(behind_idx):
        role[j] = 5 if rank == 0 else 6
    for off in (-1, 1):
        target = lane[i] + off
        if target < 0:
            continue
        m = cand & (lane == target)
        a_idx = np.flatnonzero(m & (ds > 0))
        a_idx = a_idx[np.argsort(ds[a_idx])]
        for rank, j in enumerate(a_idx):
            role[j] = 7 if rank == 0 else 8
        b_idx = np.flatnonzero(m & (ds <= 0))
        b_idx = b_idx[np.argsort(-ds[b_idx])]
        for rank, j in enumerate(b_idx):
            role[j] = 9 if rank == 0 else 10
    return role, cand, ds


def bin_index(ds: np.ndarray, cand_nonself: np.ndarray) -> np.ndarray:
    """Signed distance bin per key (``-1`` where not a non-self candidate)."""
    absd = np.abs(ds)
    b = np.digitize(absd, BIN_EDGES[1:-1])          # 0..3
    b = np.where(ds > 0, b, b + 4)
    return np.where(cand_nonself, b, -1)


# -------------------------------------------------------------------- per state
class Capture:
    def __init__(self, every: int):
        self.every = every
        self.call = -1
        self.hist: list[tuple] = []
        self.samples: dict[int, dict[int, np.ndarray]] = {}

    def hook(self, block: int):
        def fn(module, args, output):
            _x, mask, geo = args
            attn = output[1]
            if block == 0:
                self.call += 1
                self.hist.append((mask[0].detach().cpu().numpy().astype(bool).copy(),
                                  geo["s"][0].detach().cpu().numpy().astype(np.float64).copy(),
                                  geo["lane"][0].detach().cpu().numpy().astype(np.int64).copy(),
                                  geo["v"][0].detach().cpu().numpy().astype(np.float64).copy()))
            if self.call % self.every == 0:
                self.samples.setdefault(self.call, {})[block] = \
                    attn[0].detach().cpu().numpy().astype(np.float64).copy()
        return fn


def group_flags(speed, gap, lane_i, s_i, lc):
    merge = bool(lane_i == 2 or (MERGE_S0 - MERGE_PAD <= s_i <= MERGE_S1 + MERGE_PAD))
    jam = speed <= 5.0
    free = (not jam) and speed > 20.0 and gap > 50.0
    follow = (not jam) and (not free)
    plain = (not merge) and (not lc)
    return {
        "all": True, "free": free, "follow": follow, "jam": jam, "merge": merge,
        "lc_any": lc, "lc_main": lc and lane_i != 2, "lc_ramp": lc and lane_i == 2,
        "no_lc": not lc,
        "free_plain": free and plain, "follow_plain": follow and plain,
        "jam_plain": jam and plain,
        "lc_main_nomerge": lc and lane_i != 2 and not merge,
        "merge_no_lc": merge and not lc,
    }


def analyse_capture(cap: Capture, is_cav: np.ndarray, n_steps: int):
    """Turn captured attention into per-group flat accumulators for one rollout."""
    groups = {g: np.zeros(FLAT) for g in GROUPS}
    diag = {"sampled_steps": 0, "queries": 0, "max_outside_mass": 0.0,
            "min_cand_mass": 1.0, "max_rowsum_err": 0.0, "isolated": 0}
    for c in sorted(cap.samples):
        if c + LC_LOOKAHEAD_STEPS >= len(cap.hist):
            continue
        blocks = cap.samples[c]
        if len(blocks) != NB:
            raise RuntimeError(f"step {c}: captured {len(blocks)} blocks")
        mask, s_all, lane_all, v_all = cap.hist[c]
        idx = np.flatnonzero(mask)
        s, lane, v = s_all[idx], lane_all[idx], v_all[idx]
        cav = is_cav[idx]
        attn = np.stack([blocks[b][:, idx][:, :, idx] for b in range(NB)])   # [NB,H,n,n]
        diag["sampled_steps"] += 1
        # lane change within the next 2 s (state index c+1..c+4, same slot)
        future = [cap.hist[c + k] for k in range(1, LC_LOOKAHEAD_STEPS + 1)]
        for q in np.flatnonzero(~cav):
            slot = idx[q]
            lc = any(f[0][slot] and f[2][slot] != lane_all[slot] for f in future)
            role, cand, ds = roles_for_query(q, s, lane)
            ncand = int(cand.sum())
            roles_onehot = np.zeros((s.shape[0], NR))
            ok = role >= 0
            roles_onehot[np.flatnonzero(ok), role[ok]] = 1.0
            cav_f = cav.astype(np.float64)
            # sanity: attention is only on candidates
            rows = attn[:, :, q, :]                                  # [NB,H,n]
            outside = rows[:, :, ~cand].sum() if (~cand).any() else 0.0
            diag["max_outside_mass"] = max(diag["max_outside_mass"], float(outside))
            diag["max_rowsum_err"] = max(diag["max_rowsum_err"],
                                         float(np.abs(rows.sum(-1) - 1.0).max()))
            mass = np.einsum("bhn,nr->bhr", rows, roles_onehot)
            mass_cav = np.einsum("bhn,nr->bhr", rows, roles_onehot * cav_f[:, None])
            self_mass = mass[:, :, 0]
            nonself = np.ones(s.shape[0], dtype=bool)
            nonself[q] = False
            has_ex = ncand > 1
            if has_ex:
                diag["min_cand_mass"] = min(diag["min_cand_mass"],
                                            float(rows[:, :, cand].min()))
            else:
                diag["isolated"] += 1
            denom = np.maximum(1.0 - self_mass, 1e-9)
            mass_ex = mass / denom[..., None]
            mass_ex[:, :, 0] = 0.0
            absd = np.abs(ds)
            behind = rows[:, :, nonself & (ds <= 0)].sum(-1)
            ahead = rows[:, :, nonself & (ds > 0)].sum(-1)
            far50 = rows[:, :, nonself & (absd > 50.0)].sum(-1)
            far100 = rows[:, :, nonself & (absd > 100.0)].sum(-1)
            absds_num = (rows * np.where(nonself, absd, 0.0)[None, None, :]).sum(-1)
            safe = np.where(rows > 0, rows, 1.0)
            entropy = -(rows * np.log(safe)).sum(-1)
            scal = np.stack([behind, ahead, far50, far100, absds_num, entropy,
                             np.exp(entropy)], axis=-1)              # [NB,H,NS]
            cand_ns = cand & nonself
            bidx = bin_index(ds, cand_ns)
            bins = np.zeros((NB, NH, NBIN))
            bin_counts = np.zeros(NBIN)
            for bi in range(NBIN):
                sel = bidx == bi
                bin_counts[bi] = sel.sum()
                if sel.any():
                    bins[:, :, bi] = rows[:, :, sel].sum(-1)
            role_counts = roles_onehot.sum(0)
            # situation
            leaders = np.flatnonzero(role == 1)
            gap = (ds[leaders[0]] - VEH_LEN_M) if leaders.size else math.inf
            flags = group_flags(float(v[q]), gap, int(lane[q]), float(s[q]), bool(lc))
            vec = np.zeros(FLAT)
            _put(vec, "count", 1.0)
            _put(vec, "cand", ncand)
            _put(vec, "count_ex", 1.0 if has_ex else 0.0)
            _put(vec, "uni", role_counts / ncand)
            _put(vec, "cand_roles", role_counts)
            if has_ex:
                _put(vec, "bin_uni", bin_counts / float(ncand - 1))
                _put(vec, "n_binq", 1.0)
                _put(vec, "cav_cand_share", float(cav[cand_ns].mean()))
                _put(vec, "cav_share_nq", 1.0)
            _put(vec, "mass", mass)
            _put(vec, "mass_cav", mass_cav)
            if has_ex:
                _put(vec, "mass_ex", mass_ex)
            _put(vec, "scal", scal)
            _put(vec, "bins", bins)
            diag["queries"] += 1
            for g, on in flags.items():
                if on:
                    groups[g] += vec
    return groups, diag


_W: dict = {}


def _init_worker():
    os.environ["WMS_EVAL_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from wms_eval.models import flags_for
    from wms_eval.wms_plugin import WmsCheckpointModel
    physics = json.loads(RECEIPT.read_text(encoding="utf-8"))["physics"]
    mdl = WmsCheckpointModel(CKPT, flags_for("wms"), physics=physics, model_kind="structured")
    net = mdl._load()
    _W["mdl"] = mdl
    _W["net"] = net


def _work(payload):
    set_name, index, record, arm, keep_frames = payload
    t_start = time.time()
    mdl, net = _W["mdl"], _W["net"]
    cap = Capture(EVERY)
    handles = [blk.attn.register_forward_hook(cap.hook(b))
               for b, blk in enumerate(net.residual_net.blocks)]
    try:
        controllers = record.get("cav_controllers", {})
        frames = mdl.predict_states(record, {**controllers, "arm": arm}, HORIZON_S, SEED)
    finally:
        for h in handles:
            h.remove()
    n_steps = int(round(HORIZON_S / 0.5))
    if cap.call + 1 != n_steps:
        raise RuntimeError(f"expected {n_steps} residual-net calls, saw {cap.call + 1}")
    _t0, ids, kin, is_cav = mdl._inputs(record, arm)
    if len(ids) != cap.hist[0][0].shape[0]:
        raise RuntimeError("slot count mismatch between inputs and hooked net")
    groups, diag = analyse_capture(cap, is_cav, n_steps)
    init_n = int(len(ids))
    out = {
        "set": set_name, "index": index, "world": str(record.get("world")),
        "key": record.get("key"), "t0_s": float(record["t0_s"]), "arm": arm,
        "n_slots": init_n, "n_cav": int(is_cav.sum()),
        "penetration_pct": (record.get("world_meta") or {}).get("penetration_pct"),
        "demand": (record.get("world_meta") or {}).get("demand"),
        "groups": groups, "diag": diag, "seconds": time.time() - t_start,
    }
    if keep_frames:
        from wms_eval.c1v2 import _car_compact
        from wms_eval.data import recorded_for
        from wms_eval.metrics import car_metrics
        recorded = recorded_for(record, arm)
        out["car"] = _car_compact(car_metrics(frames, recorded, t0_s=float(record["t0_s"]),
                                              horizons=(10.0, 30.0, 60.0)))
    return out


# ---------------------------------------------------------------- state selection
def select_states(path: Path, n_worlds: int = 20):
    """Sorted by (world, t0_s, key); first state per world; first ``n_worlds`` worlds."""
    from wms_eval.c1v2 import iter_windows
    best: dict[str, dict] = {}
    for record in iter_windows(path):
        w = str(record["world"])
        key = (float(record["t0_s"]), str(record["key"]))
        if w not in best or key < best[w][0]:
            best[w] = (key, record)
    worlds = sorted(best)
    chosen = [best[w][1] for w in worlds[:n_worlds]]
    return chosen, len(worlds)


def pick_arm(record: dict) -> str:
    rec = sorted((record.get("recorded") or {}).keys())
    own = str(record.get("arm"))
    return own if own in rec else rec[0]


# ------------------------------------------------------------------- aggregation
def unpack(vec: np.ndarray) -> dict:
    n = float(_view(vec, "count"))
    n_ex = float(_view(vec, "count_ex"))
    n_bq = float(_view(vec, "n_binq"))
    if n <= 0:
        return {"n": 0}
    mass = _view(vec, "mass") / n                       # [NB,H,NR]
    mass_ex = _view(vec, "mass_ex") / max(n_ex, 1.0)
    scal = _view(vec, "scal") / n                       # [NB,H,NS]
    bins = _view(vec, "bins") / n
    return {
        "n": int(n), "n_ex": int(n_ex),
        "cand_mean": float(_view(vec, "cand")) / n,
        "cand_nonself_mean": float(_view(vec, "cand")) / n - 1.0,
        "uniform_role_share": (_view(vec, "uni") / n).tolist(),
        "cand_role_count_mean": (_view(vec, "cand_roles") / n).tolist(),
        "bin_uniform_share": (_view(vec, "bin_uni") / max(n_bq, 1.0)).tolist(),
        "cav_cand_share": float(_view(vec, "cav_cand_share")) / max(float(_view(vec, "cav_share_nq")), 1.0),
        "mass": mass, "mass_cav": _view(vec, "mass_cav") / n,
        "mass_ex": mass_ex, "scal": scal, "bins": bins,
    }


def block_head_mean(arr):
    return arr.mean(axis=(0, 1))


def headline(vec: np.ndarray) -> dict:
    u = unpack(vec)
    if u["n"] == 0:
        return {}
    m = block_head_mean(u["mass"])
    mex = block_head_mean(u["mass_ex"])
    sc = block_head_mean(u["scal"])
    nonself = 1.0 - m[0]
    return {
        "self": m[0], "leader": m[1], "leader_ex_self": mex[1],
        "ahead_same_lane": m[1:5].sum(), "behind_same_lane": m[5:7].sum(),
        "adjacent_lane": m[7:11].sum(), "behind_any": sc[0], "far50": sc[2],
        "far100": sc[3], "mean_abs_ds_ex_self_m": sc[4] / max(nonself, 1e-9),
        "eff_n": sc[6],
    }


def bootstrap_ci(per_state: list[dict], group: str, nboot: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    vecs = np.stack([p["groups"][group] for p in per_state])
    n = vecs.shape[0]
    keys = None
    draws = []
    for _ in range(nboot):
        pick = rng.integers(0, n, n)
        h = headline(vecs[pick].sum(0))
        if not h:
            continue
        keys = list(h)
        draws.append([h[k] for k in keys])
    if not draws:
        return {}
    arr = np.asarray(draws)
    return {k: [float(np.percentile(arr[:, i], 2.5)), float(np.percentile(arr[:, i], 97.5))]
            for i, k in enumerate(keys)}


def to_jsonable(u: dict) -> dict:
    if u["n"] == 0:
        return {"n": 0}
    out = {k: v for k, v in u.items() if not isinstance(v, np.ndarray)}
    for name in ("mass", "mass_cav", "mass_ex"):
        out[name] = {f"block{b}": {f"head{h}": {ROLES[r]: float(u[name][b, h, r])
                                                for r in range(NR)}
                                   for h in range(NH)} for b in range(NB)}
    out["scalars"] = {f"block{b}": {f"head{h}": {SCAL[k]: float(u["scal"][b, h, k])
                                                 for k in range(NS)}
                                    for h in range(NH)} for b in range(NB)}
    out["dist_bins"] = {f"block{b}": {f"head{h}": {BIN_NAMES[k]: float(u["bins"][b, h, k])
                                                   for k in range(NBIN)}
                                      for h in range(NH)} for b in range(NB)}
    out["bin_uniform_share"] = dict(zip(BIN_NAMES, u["bin_uniform_share"]))
    out["uniform_role_share"] = dict(zip(ROLES, u["uniform_role_share"]))
    out["cand_role_count_mean"] = dict(zip(ROLES, u["cand_role_count_mean"]))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-worlds", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="debug: states per set")
    ap.add_argument("--out", default=str(OUT_DIR / "raw_results.json"))
    args = ap.parse_args(argv)
    if _sha256(CKPT) != _sha256(CKPT_SOURCE):
        raise SystemExit("staged checkpoint differs from source")
    tasks = []
    selection = {}
    for set_name, path in SETS.items():
        chosen, n_worlds_total = select_states(path, args.n_worlds)
        if args.limit:
            chosen = chosen[:args.limit]
        selection[set_name] = {"worlds_total": n_worlds_total,
                               "selected": [{"world": str(r["world"]), "t0_s": r["t0_s"],
                                             "key": r["key"], "arm": pick_arm(r)}
                                            for r in chosen]}
        for i, rec in enumerate(chosen):
            tasks.append((set_name, i, rec, pick_arm(rec), i < 2 and set_name == "wall_same"))
    print(f"{len(tasks)} states, workers={args.workers}", flush=True)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for res in pool.map(_work, tasks):
            results.append(res)
            print(f"[{len(results)}/{len(tasks)}] {res['set']} {res['world']} "
                  f"arm={res['arm']} slots={res['n_slots']} cav={res['n_cav']} "
                  f"queries={res['diag']['queries']} {res['seconds']:.1f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    payload = {
        "schema": "attention_check_raw/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection": selection,
        "results": [{**r, "groups": {g: v.tolist() for g, v in r["groups"].items()}}
                    for r in results],
    }
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
