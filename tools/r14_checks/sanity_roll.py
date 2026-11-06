"""S-R13E 300 s inference sanity (CPU only, no training): S-R12C rolled from TRAIN-role R13E panel states.

Arms use the NATIVE switches of the code under test (no monkeypatched yield or gs):
  base   physics.emergency_braking=False, physics.secure_gap_veto=False, learned yield off
  EgsyG  emergency_braking=True,  secure_gap_veto=True,  model.enable_mainline_merge_yield_learnable()
  EyG    emergency_braking=True,  secure_gap_veto=False, model.enable_mainline_merge_yield_learnable()
  E      emergency_braking=True only (smoke-test arm for code without the yield switch)
An arm whose switch is missing from the code under test is SKIPPED (logged), never emulated.

States: the first N states of diag_emerg/train_states48.json.gz (R13E panel; TRAIN role re-verified against
the panel receipt AND each source's C4_EPISODES_MANIFEST.json). The panel stores only 120 s of t0-roster future,
so each record is re-extended here to H s of recorded SUMO future from the same converted TRAIN episode arrays
(t0 = 450 s, rows 900..900+2H; ALL vehicles, entrants included; ids veh<vid> exactly as the panel builder).
The first 120 s of the extended t0-roster CAV teacher is checked equal to the panel's.
Rollout = COMBO300/LONGDIAG harness: entries ON (v2 + entry_queue_state, SUMO-like insertion, HDV entrants)
seeded with entry_queue = [backlog 0, 0, recorded entrance rate main, ramp] from the record's RSU counter
(the SUMO backlog is not recoverable from converted arrays; disclosed as backlog=0).
CRN: torch/numpy/random seed and lane seed = 130013 + state index, identical across arms.

  python sanity_roll.py --repo REPO --out OUT [--n 12 --horizon 300 --arms base,EgsyG,EyG --workers 4]
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import random
import re
import signal
import sys
import time
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

L5 = Path("/root/autodl-tmp/c4/v5")
CKPT = L5 / "runs/c4_wm_v10_round12c/wm_s.pt"
CKPT_SHA = "c76c338dea6355f85cfaee63cdb9077fb129ed25db0ab582d77eda2961822091"
STATES = L5 / "diag_emerg/train_states48.json.gz"
STATES_SHA = "32354ca9fc223a0a713c50d37556b7c2ac99bdfa922d7bd434f48fe47a0b80bd"
RECEIPT = L5 / "diag_emerg/panel_receipt.json"
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth", "truth", "vtype")
EXCLUDED_SEEDS = range(964000, 964200)
DT, T0, ROW0 = 0.5, 450.0, 900
SEED0 = 130013
ARMS = {"base": dict(phys={"emergency_braking": False, "secure_gap_veto": False}, yld=False),
        "E": dict(phys={"emergency_braking": True, "secure_gap_veto": False}, yld=False),
        "EgsyG": dict(phys={"emergency_braking": True, "secure_gap_veto": True}, yld=True),
        "EyG": dict(phys={"emergency_braking": True, "secure_gap_veto": False}, yld=True)}
G: dict = {}


def guard(path) -> str:
    s = str(path).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    if any(int(n) in EXCLUDED_SEEDS for n in re.findall(r"\d+", s)):
        raise SystemExit(f"refusing evaluation-seed path {path}")
    return str(path)


def sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(guard(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact(frames, t0_s: float, horizon_s: float):
    """frames -> ids, S/L/V [T+1, N] on the 1 s grid (NaN / -1 when absent), cav flag per id (LONGDIAG)."""
    nt = int(round(horizon_s))
    by = {}
    for f in frames:
        i2 = int(round((float(f["time_s"]) - t0_s) / DT))
        if i2 % 2 == 0 and 0 <= i2 // 2 <= nt:
            by[i2 // 2] = f
    ids: dict = {}
    cav: dict = {}
    rows = []
    for i in range(nt + 1):
        r = {}
        for v in (by.get(i) or {}).get("vehicles", []):
            if not v.get("present", True):
                continue
            vid = str(v["id"])
            ids.setdefault(vid, len(ids))
            if "is_cav" in v:
                cav[vid] = int(bool(v["is_cav"]))
            r[vid] = (float(v["s"]), int(v.get("lane", -1)), float(v["v"]))
        rows.append(r)
    n = len(ids)
    S = np.full((nt + 1, n), np.nan, np.float32)
    L = np.full((nt + 1, n), -1, np.int8)
    V = np.full((nt + 1, n), np.nan, np.float32)
    for i, r in enumerate(rows):
        for vid, (s, lane, v) in r.items():
            j = ids[vid]
            S[i, j], L[i, j], V[i, j] = s, lane, v
    idl = list(ids)
    return {"ids": np.array(idl), "S": S, "L": L, "V": V,
            "cav": np.array([cav.get(x, -1) for x in idl], np.int8), "covered_rows": len(by)}


def extend_record(rec: dict, data_root: Path, horizon: float) -> tuple[dict, dict]:
    """Re-extend one TRAIN panel state to `horizon` s of recorded future (all vehicles)."""
    src, ep = rec["source"], rec["episode"]
    d = Path(guard(data_root / src / "episodes" / ep))
    n = int(round(horizon / DT))
    t = np.load(d / "time_s.npy", mmap_mode="r")
    if not np.allclose(np.asarray(t[ROW0:ROW0 + n + 1]), T0 + DT * np.arange(n + 1)):
        raise ValueError(f"{ep}: no complete native grid {T0}..{T0 + horizon} s")
    kin = np.asarray(np.load(d / "veh_kin.npy", mmap_mode="r")[ROW0:ROW0 + n + 1])
    mask = np.asarray(np.load(d / "veh_mask.npy", mmap_mode="r")[ROW0:ROW0 + n + 1]).astype(bool)
    vid = np.asarray(np.load(d / "vid.npy", mmap_mode="r")[ROW0:ROW0 + n + 1])
    cav = np.asarray(np.load(d / "is_cav.npy", mmap_mode="r")[ROW0:ROW0 + n + 1]).astype(bool)
    hot = kin[..., 1:4]
    lanes = np.where(hot.sum(-1) > 0, hot.argmax(-1), -1)
    frames = []
    for k in range(n + 1):
        slots = np.flatnonzero(mask[k])
        if len(np.unique(vid[k, slots])) != len(slots):
            raise ValueError(f"{ep}: duplicate vehicle id in a frame")
        frames.append({"time_s": float(T0 + DT * k), "vehicles": [
            {"id": f"veh{int(vid[k, s])}", "s": float(kin[k, s, 0]), "lane": int(lanes[k, s]),
             "v": float(kin[k, s, 4]), "a": float(kin[k, s, 5]), "length_m": float(kin[k, s, 6]),
             "present": True, "is_cav": bool(cav[k, s])} for s in slots]})
    arm = rec["arm"]
    old = rec["recorded"][arm]
    # identity check: t0 frame equal; panel t0-roster CAV teacher is a subset of the extension for 120 s
    if {v["id"] for v in old[0]["vehicles"]} != {v["id"] for v in frames[0]["vehicles"]}:
        raise ValueError(f"{ep}: t0 frame differs from the panel record")
    for k, f in enumerate(old[: len(frames)]):
        new = {v["id"]: v for v in frames[k]["vehicles"]}
        for v in f["vehicles"]:
            if v["is_cav"]:
                w = new.get(v["id"])
                if w is None or abs(w["s"] - v["s"]) > 1e-4 or w["lane"] != v["lane"]:
                    raise ValueError(f"{ep}: extended CAV teacher differs from the panel at frame {k}")
    counter = np.asarray(rec["counter"], np.float64).reshape(2, 30)
    rate = counter.sum(-1) / 300.0
    out = dict(rec)
    out.update(horizon_s=float(horizon), recorded={arm: frames}, recorded_horizon_steps={arm: n},
               entry_queue=[0.0, 0.0, float(rate[0]), float(rate[1])])
    return out, {"episode": ep, "world": rec["world"], "frames": n + 1, "entry_rate_vps": rate.tolist()}


def verify_train(states, receipt, data_root):
    sel = {e["episode"]: e for e in receipt["selected_episodes"]}
    manifests = {}
    for r in states:
        if r.get("split") != "TRAIN" or r.get("split_role") != "train":
            raise SystemExit(f"non-train state {r.get('key')}")
        e = sel.get(r["episode"])
        if e is None or e["split_role"] != "train":
            raise SystemExit(f"state not train-role in the panel receipt: {r['episode']}")
        if r["source"] not in manifests:
            m = json.loads(Path(guard(data_root / r["source"] / "C4_EPISODES_MANIFEST.json")).read_text())
            manifests[r["source"]] = {x["episode"]: x for x in m["episodes"]}
        me = manifests[r["source"]].get(r["episode"])
        if me is None or me["split_role"] != "train":
            raise SystemExit(f"state not train-role in its manifest: {r['episode']}")
        if int(me["seed"]) in EXCLUDED_SEEDS:
            raise SystemExit("evaluation seed")


def init_worker(ckpt: str, repo: str) -> None:
    for p in (Path(repo), Path(repo) / "src", Path(repo) / "tools"):
        sys.path.insert(0, str(p))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    flags = RolloutFlags(entries=True, entries_v2=True, entry_queue_state=True)
    p = WmsPredictor.from_checkpoint(guard(ckpt), flags, device="cpu")
    m = p.model
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    from wms.physics.config import PhysicsConfig
    holders, seen = [], set()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for name, val in list(vars(obj).items()):
            if isinstance(val, PhysicsConfig):
                holders.append((obj, name))
        for sub in getattr(obj, "_modules", {}).values():
            if sub is not None:
                visit(sub)
    visit(m)
    G.update(p=p, m=m, flags0=flags, torch=torch, holders=holders, phys0=m.physics,
             fields=set(PhysicsConfig.__dataclass_fields__))


def arm_supported(arm: str) -> str | None:
    spec = ARMS[arm]
    missing = set(spec["phys"]) - G["fields"]
    if missing:
        return f"PhysicsConfig lacks {sorted(missing)}"
    if spec["yld"] and not hasattr(G["m"], "enable_mainline_merge_yield_learnable"):
        return "model lacks enable_mainline_merge_yield_learnable"
    return None


def set_arm(arm: str) -> None:
    spec, m = ARMS[arm], G["m"]
    phys = dataclasses.replace(G["phys0"], **spec["phys"])
    for owner, name in G["holders"]:
        setattr(owner, name, phys)
    if spec["yld"]:
        m.enable_mainline_merge_yield_learnable()
    elif hasattr(m, "mainline_merge_yield_learnable"):
        m.mainline_merge_yield_learnable = False


def process(task):
    idx, arm, out_dir, horizon = task
    rec = G["records"][idx]
    errs, t1 = [], time.time()
    fn = Path(out_dir) / f"p{idx:02d}_ref_{arm}_k0.npz"
    tmo = Path(out_dir) / f"p{idx:02d}_ref_{arm}_k0.timeout"
    if fn.exists() or tmo.exists():
        return idx, arm, "exists", errs, 0.0
    why = arm_supported(arm)
    if why:
        return idx, arm, f"SKIPPED ({why})", errs, 0.0
    try:
        from wms.eval.rollout_inputs import t0_vehicles
        torch, p = G["torch"], G["p"]
        t0_s, ids, kin, is_cav = t0_vehicles(rec)
        set_arm(arm)
        seed = SEED0 + idx
        if seed in EXCLUDED_SEEDS:
            raise ValueError("evaluation seed")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        p.flags = dataclasses.replace(G["flags0"], lane_seed=seed)
        limit = int(os.environ.get("SANITY_TIMEOUT_S", "0") or 0)
        if limit:
            def _alarm(*_a):
                raise TimeoutError("rollout timeout")
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(limit)
        try:
            fr = p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": rec["arm"]}, horizon, seed)
        except TimeoutError:
            tmo.write_text(f"timeout {limit} s")
            return idx, arm, "timeout", errs, time.time() - t1
        finally:
            if limit:
                signal.alarm(0)
            p.flags = G["flags0"]
            set_arm("base")
        c = compact(fr, float(t0_s), horizon)
        tmp = Path(out_dir) / f".{fn.name}.{os.getpid()}.npz"
        np.savez_compressed(tmp, roster=np.array([str(i) for i in ids]), roster_cav=np.asarray(is_cav, bool),
                            t0=float(t0_s), key=rec["key"], arm=arm, seed=seed, sec=time.time() - t1,
                            covered_rows=c.pop("covered_rows"), **c)
        os.replace(tmp, fn)
    except Exception:
        errs.append(traceback.format_exc())
    return idx, arm, "ERR" if errs else "ok", errs, time.time() - t1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--states", default=str(STATES))
    ap.add_argument("--receipt", default=str(RECEIPT))
    ap.add_argument("--data-root", default=str(L5 / "data"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--arms", default="base,EgsyG,EyG")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not 1 <= a.workers <= 4:
        raise SystemExit("workers must be 1..4")
    if os.name == "posix":
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    arms = a.arms.split(",")
    if any(x not in ARMS for x in arms):
        raise SystemExit(f"unknown arm in {arms}")
    repo = Path(guard(a.repo)).resolve()
    out = Path(guard(a.out))
    out.mkdir(parents=True, exist_ok=True)
    if sha256(a.ckpt) != CKPT_SHA:
        raise SystemExit("checkpoint is not the registered S-R12C FINAL")
    if sha256(a.states) != STATES_SHA:
        raise SystemExit("panel is not the registered R13E train-role panel")
    with gzip.open(guard(a.states), "rt", encoding="utf-8") as f:
        doc = json.load(f)
    states = doc["states"][: a.n]
    receipt = json.loads(Path(guard(a.receipt)).read_text())
    verify_train(states, receipt, Path(a.data_root))
    records, info = [], []
    for r in states:
        rec, meta = extend_record(r, Path(a.data_root), a.horizon)
        records.append(rec)
        info.append(meta)
    del doc
    G["records"] = records
    for idx, rec in enumerate(records):
        fn = out / f"p{idx:02d}_ref_sumo.npz"
        if not fn.exists():
            t0_s = float(rec["t0_s"])
            from_ids = [v["id"] for v in rec["initial_state"]["vehicles"]]
            c = compact(rec["recorded"][rec["arm"]], t0_s, a.horizon)
            np.savez_compressed(fn, roster=np.array(from_ids),
                                roster_cav=np.array([bool(v["is_cav"]) for v in rec["initial_state"]["vehicles"]]),
                                t0=t0_s, key=rec["key"], arm="sumo", covered_rows=c.pop("covered_rows"), **c)
    meta = {"argv": sys.argv, "pid": os.getpid(), "repo": str(repo), "ckpt_sha256": CKPT_SHA,
            "states_sha256": STATES_SHA, "n": a.n, "horizon_s": a.horizon, "arms": arms,
            "states": info, "started": time.strftime("%F %T"),
            "entries": "v2 + entry_queue_state, backlog=0, rates from record counter",
            "seed_rule": f"{SEED0} + state index (CRN across arms)"}
    (out / f"run_meta_{os.getpid()}.json").write_text(json.dumps(meta, indent=1))
    # interleave arms per state so partial results stay paired
    todo = [(i, arm, str(out), a.horizon) for i in range(len(records)) for arm in arms]
    print(f"todo {len(todo)} workers {a.workers}", flush=True)
    t0 = time.time()
    if a.workers == 1:
        init_worker(a.ckpt, str(repo))
        it = map(process, todo)
    else:
        from multiprocessing import get_context
        pool = get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(a.ckpt, str(repo)),
                                        maxtasksperchild=4)
        it = pool.imap_unordered(process, todo, chunksize=1)
    for idx, arm, status, errs, sec in it:
        print(f"[{time.time() - t0:6.0f}s] p{idx:02d} {arm} {status} {sec:.0f}s", flush=True)
        if errs:
            print(errs[0], flush=True)
    (out / f"DONE_{os.getpid()}").write_text(time.strftime("%F %T"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
