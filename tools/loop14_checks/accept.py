#!/usr/bin/env python3
"""LOOP14 phase 4: paired acceptance of a fine-tuned WM against the previous ACCEPTED WM.

CPU only, no training, no SUMO.  Both checkpoints are evaluated with the CANDIDATE's config
(owner B1: the same switches - D1, emergency, gs, learned yield - on the same panel in the
same invocation), so a stop isolates the fine-tune, not a flag.

    python tools/loop14_checks/accept.py --candidate C.pt --reference R.pt --config CFG.toml \\
        --roots <TRAIN source> [<TRAIN source> ...] [--pairs <converted collected pairs>] \\
        --panel PANEL.json --out <new dir> [--onestep-windows 24 --rollout-windows 24]

Measurements (all on a FIXED TRAIN panel; ``--panel`` is written on first use and re-read
verbatim afterwards, so every cycle is judged on identical windows):

* one-step: the existing BASE producer ``tools/r10b_checks/run_onestep.py`` (filled replay,
  base columns, 120 s windows; arms ``r10b`` = candidate, ``r9`` = reference) -> all lane-1
  one-step MAE via ``round13c_waiter.paired_mae``; ``base_calibration.py`` and
  ``midrun_check.py`` reports are written as context (read-only, non-gating);
* rollout: 60 s free rollouts (recorded CAV commands, common seeds) -> full-road lane-1 HDV
  hard-brake share (a < -3 m/s^2 over continuous lane-1 transitions, the 13C/13E definition)
  and the r13e safety metrics (``rollout_metrics``: blowups, overlap pair-steps, TTC<1.5 share);
* paired difference (reported, non-gating in v1): for every collected POLICY/B pair group,
  |(WM delay_POLICY - WM delay_B) - (SUMO delay_POLICY - SUMO delay_B)| of the t0 HDVs.

Decision (registered LOOP14 rule = the 13E stop rule + a safety non-regression):
  REJECT if candidate lane-1 hard-brake share > 2 x reference share, OR candidate all-lane1
  one-step MAE > reference MAE + 0.1 m/s^2 (``round13c_waiter.stop_reasons``), OR candidate
  blowup runs > reference, OR candidate overlap pair-steps > reference, OR candidate TTC<1.5
  share > 2 x reference share, OR any measurement error (fail closed).  ACCEPT otherwise.
Writes ``ACCEPT.json`` (decision, reasons, all numbers, hashes) and exits 0 on ACCEPT, 3 on
REJECT, 2 on a measurement error.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "tools" / "r10b_checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth", "vtype")
DT_S = 0.5
ROLL_STEPS = 120               # 60 s, the 13C/13E full-road lane-1 hard-brake horizon
ONESTEP_HORIZON_S = 120.0
PAIR_STEPS = 240               # 120 s paired-difference horizon (branches are >= 120 s)
SEED0 = 140014
RULE = {"hard_brake_share_ratio_gt": 2.0, "onestep_mae_delta_gt_mps2": 0.1,
        "blowup_runs_increase": True, "overlap_pair_steps_increase": True,
        "ttc_lt_1_5_share_ratio_gt": 2.0, "measurement_error": "REJECT",
        "paired_difference": "reported, non-gating (v1)",
        "reference": "previous ACCEPTED WM evaluated with the candidate config (same flags, same panel)"}


def guard(path) -> Path:
    text = str(path)
    if any(t in text.lower() for t in FORBIDDEN):
        raise ValueError(f"restricted path refused: {text}")
    resolved = Path(path).resolve()
    if any(t in str(resolved).lower() for t in FORBIDDEN):
        raise ValueError(f"restricted resolved path refused: {resolved}")
    return resolved


def digest(path) -> str:
    with guard(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
        return h.hexdigest()


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_roots(roots) -> list[dict]:
    """TRAIN sources only: restricted tokens and evaluation seeds refuse the whole root."""
    out = []
    for root in roots:
        manifest = guard(Path(root) / "C4_EPISODES_MANIFEST.json")
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        for e in doc["episodes"]:
            if any(t in json.dumps(e).lower() for t in FORBIDDEN):
                raise ValueError(f"restricted reference in {manifest}")
            if 964000 <= int(e["seed"]) <= 964199:
                raise ValueError(f"evaluation seed in {manifest}")
            if e.get("split_role") not in ("train", "val"):
                raise ValueError("unregistered manifest split role")
        out.append({"root": str(guard(root)), "manifest_sha256": digest(manifest),
                    "episodes": len(doc["episodes"])})
    return out


#: R13Y learned-yield scalars: a pre-13E reference (e.g. S-R12C) lacks them and is evaluated
#: with them at their documented initialisation, exactly the owner-B1 same-flags reference.
YIELD_TENSORS = frozenset(("merge_yield.log_decel", "merge_yield.log_distance"))


def load_arm(config, ckpt, *, reference=False):
    """Strict load with the candidate config (no missing/ignored tensors)."""
    import torch
    from tools.r10_attribution import load_model
    payload = torch.load(guard(ckpt), map_location="cpu", weights_only=True)
    stored = payload["model"].get("follower_courtesy.net.0.weight")
    extra = ("model.follower_courtesy_version=2",) if stored is not None and stored.shape[-1] == 13 else ()
    cfg, settings, model, report = load_model(str(guard(config)), str(guard(ckpt)), extra)
    allowed_new = YIELD_TENSORS if reference else frozenset()
    if set(report["new"]) - allowed_new or report["unused"]:
        raise ValueError(f"{ckpt}: not an exact load under the candidate config "
                         f"(new={report['new'][:5]}, unused={report['unused'][:5]})")
    model.loop14_new_tensors = sorted(report["new"])
    return cfg, settings, model


def build_sets(cfg, roots, horizon_steps):
    from tools.r10_attribution import datasets
    return datasets(cfg, [str(guard(r)) for r in roots], horizon_steps, 30.0)


def make_panel(sets, roots, n, seed) -> dict:
    """One window per episode (fixed RNG), the first ``n`` in episode order; identity by name."""
    items = []
    for si, ds in enumerate(sets):
        by_ep: dict = {}
        for i, (ep, _tick) in enumerate(ds.positions):
            by_ep.setdefault(ep, []).append(i)
        rng = np.random.default_rng(seed + si)
        for ep in sorted(by_ep):
            pick = int(rng.choice(by_ep[ep]))
            items.append({"root": str(guard(roots[si])), "episode": ds._loader.entries[ep]["episode"],
                          "tick": int(ds.positions[pick][1])})
    order = np.random.default_rng(seed).permutation(len(items))
    return {"schema": "loop14_panel/1", "seed": seed, "windows": [items[i] for i in sorted(order[:n])]}


def resolve_panel(sets, roots, panel) -> list[tuple[int, int]]:
    roots = [str(guard(r)) for r in roots]
    out = []
    for w in panel["windows"]:
        si = roots.index(w["root"])
        ds = sets[si]
        names = [e["episode"] for e in ds._loader.entries]
        hits = [i for i, (ep, tick) in enumerate(ds.positions)
                if names[ep] == w["episode"] and int(tick) == int(w["tick"])]
        if len(hits) != 1:
            raise ValueError(f"panel window not found exactly once: {w}")
        out.append((si, hits[0]))
    return out


def free_rollout(model, settings, record, n_steps, seed) -> dict:
    """WM rollout from a recorded window: recorded CAV commands (timeline), t0 roster."""
    import torch
    from wms.data.collate import collate
    from wms.train.inputs import prepare_inputs

    batch = collate([record])
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n_steps)
    torch.manual_seed(seed)
    with torch.no_grad():
        roll = model.rollout(
            inp.initial, n_steps=inp.n_steps,
            cav_teacher=None if inp.teacher is None else inp.teacher[:, :inp.n_steps + 1],
            present_seq=inp.present_seq[:, :inp.n_steps + 1], is_cav_seq=inp.is_cav[:, :inp.n_steps + 1],
            history=inp.hist, history_mask=inp.hist_mask, controller_id=inp.controller_id,
            action_timeline=inp.action_timeline, cav_controller=inp.cav_controller,
            driver_history=inp.observed_hist, driver_history_mask=inp.observed_hist_mask,
            lane_seed=int(seed))
    s0, v0 = inp.initial["s_m"][0].numpy(), inp.initial["v_mps"][0].numpy()
    lane0 = inp.initial["lane"][0].numpy() if "lane" in inp.initial else inp.states[0, 0, :, 1:4].argmax(-1).numpy()
    m0 = inp.present[0, 0].numpy()
    cat = lambda first, key: np.concatenate([first[None], roll[key][0].detach().numpy()], 0)  # noqa: E731
    rec = inp.states[0, :inp.n_steps + 1].numpy()
    return {"pos": cat(s0, "positions"), "speed": cat(v0, "speeds"),
            "lane": cat(lane0.astype(float), "lanes").astype(float),
            "mask": np.concatenate([m0[None], roll["masks"][0].detach().numpy().astype(bool)], 0),
            "is_cav": inp.is_cav[0, 0].numpy().astype(bool), "length": rec[0, :, 6],
            "rec_speed": rec[:, :, 4], "rec_mask": inp.present_seq[0, :inp.n_steps + 1].numpy().astype(bool),
            "t0_present": m0.astype(bool)}


def lane1_hard_brake(r) -> tuple[int, int]:
    """Continuous lane-1 HDV transitions and those with a < -3 m/s^2 (13C ``hard_brakes``)."""
    v, lane, mask, cav = r["speed"], r["lane"], r["mask"], r["is_cav"]
    sel = mask[:-1] & mask[1:] & (lane[:-1] == 1) & (lane[1:] == 1) & ~cav[None, :]
    accel = np.diff(v.astype(float), axis=0) / DT_S
    if not np.isfinite(accel[sel]).all():
        raise ValueError("nonfinite lane-1 rollout acceleration")
    return int(sel.sum()), int((accel[sel] < -3).sum())


def safety_metrics(r) -> dict:
    roll = _module("r13e_offline", ROOT / "tools/r13e_checks/offline.py")
    present = r["mask"] & r["t0_present"][None, :]
    lengths = np.where(np.isfinite(r["length"]) & (r["length"] > 0), r["length"], 5.0)
    keep = r["t0_present"]
    m = roll.rollout_metrics(r["pos"][:, keep], r["speed"][:, keep], r["lane"][:, keep],
                             present[:, keep], r["is_cav"][keep], lengths[keep])
    return {k: m[k] for k in ("blowup", "overlap_pair_steps", "ttc_lt_1_5_rows", "ttc_eligible_rows",
                              "a_lt_minus3_rows", "finite_hdv_transitions")}


def hdv_delay(speed, mask, sel, steps) -> float:
    from tools.wm_eval_common import delay_proxy
    return delay_proxy(speed, mask, sel, steps * DT_S)


def onestep(args, out: Path) -> dict:
    """The existing BASE producer, verbatim, as a subprocess (arms r10b=candidate, r9=reference)."""
    os_dir = out / "onestep"
    cmd = [sys.executable, "-B", str(ROOT / "tools/r10b_checks/run_onestep.py"), "--repo", str(ROOT),
           "--config", str(guard(args.config)), "--var", f"r10b={guard(args.candidate)}",
           "--var", f"r9={guard(args.reference)}", "--replay-present", "filled", "--base-cols",
           "--horizon-s", str(ONESTEP_HORIZON_S), "--roots", *[str(guard(r)) for r in args.roots],
           "--stride-eps", str(args.stride_eps), "--limit", str(args.onestep_windows),
           "--workers", str(args.workers), "--out", str(os_dir)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(map(str, (ROOT / "src", ROOT, ROOT / "tools",
                                                    ROOT / "tools/r10b_checks"))))
    with (out / "onestep.log").open("x", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=4 * 3600, check=False).returncode
    if code != 0:
        raise ValueError(f"one-step producer exit {code} (see onestep.log)")
    done = json.loads((os_dir / "ONESTEP_DONE.json").read_text(encoding="utf-8"))
    if done.get("replay_present") != "filled" or not done.get("base_cols") or not done.get("windows"):
        raise ValueError("one-step completion/presence/columns missing")
    waiter = _module("round13c_waiter", ROOT / "scripts/round13c_waiter.py")
    mae = waiter.paired_mae(os_dir, ("r10b", "r9"))
    context = {}
    for tool, extra in (("base_calibration.py", ["--os", str(os_dir), "--out", str(out / "BASE.md"),
                                                 "--json", str(out / "BASE.json")]),
                        ("midrun_check.py", ["--os-dir", str(os_dir), "--out-json", str(out / "MIDRUN.json"),
                                             "--out-md", str(out / "MIDRUN.md")])):
        code = subprocess.run([sys.executable, "-B", str(ROOT / "tools/r12_checks" / tool), *extra],
                              cwd=ROOT, env=env, capture_output=True, timeout=3600, check=False).returncode
        context[tool] = {"exit": code}
    for name in ("BASE.json", "MIDRUN.json"):
        if (out / name).is_file():
            doc = json.loads((out / name).read_text(encoding="utf-8"))
            context[name] = doc.get("verdict", doc.get("decision"))
    return {"windows": done["windows"], "mae": {"candidate": mae["r10b"], "reference": mae["r9"]},
            "context_non_gating": context}


def rollouts(arms, sets, panel_idx) -> dict:
    if os.environ.get('WMS_LOOPLONG_SPEED') == '1':
        # Window-major evaluation decodes each immutable observable record once
        # for both arms. Per-arm window order, seeds, model work and sums are
        # unchanged; at most one record is live, irrespective of panel size.
        stats = {name: {'n': 0, 'hard': 0, 'blowup_runs': 0, 'overlap_pair_steps': 0,
                        'ttc_lt_1_5_rows': 0, 'ttc_eligible_rows': 0} for name in arms}
        for w, (si, idx) in enumerate(panel_idx):
            record = sets[si][idx]
            for name, (_cfg, settings, model) in arms.items():
                r = free_rollout(model, settings, record, ROLL_STEPS, SEED0 + w)
                dn, dh = lane1_hard_brake(r)
                s = safety_metrics(r)
                row = stats[name]
                row['n'] += dn
                row['hard'] += dh
                row['blowup_runs'] += int(s['blowup'])
                for k in ('overlap_pair_steps', 'ttc_lt_1_5_rows', 'ttc_eligible_rows'):
                    row[k] += int(s[k])
        result = {}
        for name, row in stats.items():
            n, hard = row.pop('n'), row.pop('hard')
            if n == 0:
                raise ValueError('empty lane-1 rollout panel')
            result[name] = {'lane1_transitions': n, 'lane1_a_lt_minus3': hard,
                'hard_brake_share': hard/n, **row,
                'ttc_lt_1_5_share': row['ttc_lt_1_5_rows']/row['ttc_eligible_rows'] if row['ttc_eligible_rows'] else 0.}
        return result
    out = {}
    for name, (_cfg, settings, model) in arms.items():
        n = hard = 0
        safety = {"blowup_runs": 0, "overlap_pair_steps": 0, "ttc_lt_1_5_rows": 0, "ttc_eligible_rows": 0}
        for w, (si, idx) in enumerate(panel_idx):
            r = free_rollout(model, settings, sets[si][idx], ROLL_STEPS, SEED0 + w)
            dn, dh = lane1_hard_brake(r)
            n, hard = n + dn, hard + dh
            s = safety_metrics(r)
            safety["blowup_runs"] += int(s["blowup"])
            for k in ("overlap_pair_steps", "ttc_lt_1_5_rows", "ttc_eligible_rows"):
                safety[k] += int(s[k])
        if n == 0:
            raise ValueError("empty lane-1 rollout panel")
        out[name] = {"lane1_transitions": n, "lane1_a_lt_minus3": hard, "hard_brake_share": hard / n,
                     **safety, "ttc_lt_1_5_share": (safety["ttc_lt_1_5_rows"] / safety["ttc_eligible_rows"]
                                                    if safety["ttc_eligible_rows"] else 0.0)}
    return out


def paired_difference(arms, pairs_root, max_groups) -> dict:
    """WM vs SUMO policy-minus-native t0-HDV delay over the collected TRAIN pairs (non-gating)."""
    cfg = next(iter(arms.values()))[0]
    sets = build_sets(cfg, [pairs_root], PAIR_STEPS)
    ds = sets[0]
    entries = ds._loader.entries
    groups: dict = {}
    for i, (ep, tick) in enumerate(ds.positions):
        e = entries[ep]
        branch = e.get("branch") or {}
        if e.get("pair_key") and abs(float(ds._loader.episode(ep).array("time_s")[tick]) - float(e["warmup_s"])) < 1e-6:
            groups.setdefault(e["pair_key"], {})[branch.get("kind")] = i
    keys = sorted(k for k, g in groups.items() if {"POLICY", "B"} <= set(g))[:max_groups]
    if not keys:
        raise ValueError("no complete POLICY/B pair groups in the collected source")
    result = {}
    for name, (_cfg, settings, model) in arms.items():
        errors, signs, rows = [], [], []
        for g, key in enumerate(keys):
            d = {}
            for kind in ("POLICY", "B"):
                r = free_rollout(model, settings, ds[groups[key][kind]], PAIR_STEPS, SEED0 + 7919 * g)
                sel = r["t0_present"] & ~r["is_cav"]
                d[kind] = (hdv_delay(r["speed"], r["mask"], sel, PAIR_STEPS),
                           hdv_delay(r["rec_speed"], r["rec_mask"], sel, PAIR_STEPS))
            wm, sumo = d["POLICY"][0] - d["B"][0], d["POLICY"][1] - d["B"][1]
            errors.append(abs(wm - sumo))
            signs.append(bool(np.sign(wm) == np.sign(sumo)))
            rows.append({"pair_key": key, "wm_dJ": wm, "sumo_dJ": sumo})
        result[name] = {"groups": len(keys), "mean_abs_error_veh_s": float(np.mean(errors)),
                        "sign_agreement": float(np.mean(signs)), "rows": rows}
    return result


def decide(onestep_res, roll) -> list[str]:
    waiter = _module("round13c_waiter", ROOT / "scripts/round13c_waiter.py")
    c, r = roll["candidate"], roll["reference"]
    reasons = waiter.stop_reasons(c["hard_brake_share"], r["hard_brake_share"],
                                  onestep_res["mae"]["candidate"]["mae"], onestep_res["mae"]["reference"]["mae"])
    if c["blowup_runs"] > r["blowup_runs"]:
        reasons.append("candidate rollout blowups > reference")
    if c["overlap_pair_steps"] > r["overlap_pair_steps"]:
        reasons.append("candidate overlap pair-steps > reference")
    if c["ttc_lt_1_5_share"] > RULE["ttc_lt_1_5_share_ratio_gt"] * r["ttc_lt_1_5_share"]:
        reasons.append("candidate TTC<1.5 s share > 2 x reference")
    return reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--config", required=True, help="the candidate's (fine-tune) WM config")
    ap.add_argument("--roots", nargs="+", required=True, help="TRAIN sources of the fixed panel")
    ap.add_argument("--pairs", default=None, help="converted collected pairs (paired difference)")
    ap.add_argument("--panel", required=True, help="fixed panel JSON (created on first use)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--onestep-windows", type=int, default=24)
    ap.add_argument("--rollout-windows", type=int, default=24)
    ap.add_argument("--pair-groups", type=int, default=24)
    ap.add_argument("--stride-eps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    out = guard(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    doc = {"schema": "loop14_accept/1", "rule": RULE, "decision": "REJECT", "reasons": []}
    code = 2
    try:
        doc["inputs"] = {"candidate": str(guard(args.candidate)), "candidate_sha256": digest(args.candidate),
                         "reference": str(guard(args.reference)), "reference_sha256": digest(args.reference),
                         "config": str(guard(args.config)), "config_sha256": digest(args.config),
                         "roots": check_roots(args.roots)}
        if doc["inputs"]["candidate_sha256"] == doc["inputs"]["reference_sha256"]:
            doc["note"] = "candidate == reference bytes (no-op update)"
        arms = {"candidate": load_arm(args.config, args.candidate),
                "reference": load_arm(args.config, args.reference, reference=True)}
        doc["inputs"]["reference_tensors_at_init"] = arms["reference"][2].loop14_new_tensors
        cfg = arms["candidate"][0]
        sets = build_sets(cfg, args.roots, ROLL_STEPS)
        panel_path = guard(args.panel)
        if panel_path.is_file():
            panel = json.loads(panel_path.read_text(encoding="utf-8"))
        else:
            panel = make_panel(sets, args.roots, args.rollout_windows, SEED0)
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            with panel_path.open("x", encoding="utf-8") as stream:
                json.dump(panel, stream, indent=1)
        doc["panel"] = {"path": str(panel_path), "sha256": digest(panel_path), "windows": len(panel["windows"])}
        doc["rollout"] = rollouts(arms, sets, resolve_panel(sets, args.roots, panel))
        doc["one_step"] = onestep(args, out)
        if args.pairs:
            check_roots([args.pairs])
            doc["paired_difference"] = paired_difference(arms, args.pairs, args.pair_groups)
        doc["reasons"] = decide(doc["one_step"], doc["rollout"])
        doc["decision"] = "REJECT" if doc["reasons"] else "ACCEPT"
        code = 3 if doc["reasons"] else 0
    except Exception as error:  # noqa: BLE001 - measurement error: fail closed (REJECT)
        doc["error"] = f"{type(error).__name__}: {error}"
        doc["reasons"] = ["measurement error (fail closed)"]
    doc["wall_s"] = time.time() - started

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return float(v) if math.isfinite(v) else None
        if isinstance(v, (np.integer, np.bool_)):
            return v.item()
        return v

    (out / "ACCEPT.json").write_text(json.dumps(clean(doc), indent=1), encoding="utf-8")
    print(json.dumps({"decision": doc["decision"], "reasons": doc["reasons"],
                      **({"error": doc["error"]} if "error" in doc else {})}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
