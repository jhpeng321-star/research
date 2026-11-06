"""R12 offline recount of the merge law's counted-exclusion cohort under the round-12 base (CPU, TRAIN only).

The round-9 counted exclusion (``merge_pivot_support.counted_exclusion``, owner decision 2026-10-22) is recomputed in
EVERY training batch by ``MergePivotTerm.prepare`` from the recorded state and the WM's own one-step geometry
(``build_recorded_geometry``).  Round 12 changes the base car-following (physical driver settings from the learned
centre, re-initialised residual read-out), so the cohort can change.  It is never reused from round 9: this tool runs
the identical ``prepare`` on the same TRAIN windows with (a) the round-9 model under the round-9 config and (b) the
round-12 INIT (round-12 config + the R12 init switches applied as in ``WmSTrainer._r12_reinit``), and reports the
support counts by class and per window.  Geometry-only classes (``event_vetoed_explained``) must not depend on the
base; the WM-motion classes may.  Optional (c): any further checkpoint under the round-12 config (``--extra``, e.g. a
post-run round-12 checkpoint on gpuR/cpuN) for the same recount after training.
Usage: python tools/r12_support_recount.py --roots R1 R2 --windows 48 --out docs/r12_support_recount.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def load(config: str, ckpt: str, r12_init: bool):
    from tools.r10_attribution import load_model
    cfg, settings, model, report = load_model(config, ckpt)
    if r12_init:
        with torch.no_grad():
            enc = model.driver_encoder
            n_long = int(enc.physical_lower.numel())
            enc.head[-1].weight[:n_long].zero_()
            enc.head[-1].bias[:n_long].zero_()
            model.residual_net.accel[-1].weight[0].zero_()
            model.residual_net.accel[-1].bias[0].zero_()
    return cfg, settings, model


def pivot_term(settings):
    from wms.train.losses.compose import build_loss
    composer = build_loss(settings)
    for term in (*composer.chunk_terms, *composer.unroll_terms):
        if getattr(term, "name", "") == "merge_pivot":
            return term
    raise RuntimeError("merge_pivot term not built by this config")


def recount(arm, jobs, sets, n_steps):
    from wms.data.collate import collate
    from wms.train.inputs import build_targets, prepare_inputs
    from wms.train.losses.merge_pivot_support import CLASSES
    cfg, settings, model = arm
    term = pivot_term(settings)
    per_window, totals = [], Counter()
    for si, i in jobs:
        batch = collate([sets[si][i]])
        with torch.no_grad():
            inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n_steps)
            tg = build_targets(inp, batch, settings, float(model.physics.dt_s), torch.device("cpu"),
                               physics=model.physics)
            tg.prepared["merge_pivot_source"] = (model, inp)
            term.prepare(tg)
        counts = {k: (int(v) if isinstance(v, (int, bool, np.integer)) else v)
                  for k, v in tg.prepared[term.name]["support_counts"].items()}
        per_window.append(counts)
        for k, v in counts.items():
            if isinstance(v, int):
                totals[k] += v
    return {"totals": dict(totals), "per_window": per_window, "classes": list(CLASSES)}


def merge_shards(paths, out_path):
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    out = {"tool": docs[0]["tool"], "windows": sum(d["windows"] for d in docs), "horizon_s": docs[0]["horizon_s"],
           "shards": [str(p) for p in paths], "seconds": max(d["seconds"] for d in docs), "arms": {}}
    for name in docs[0]["arms"]:
        totals, per_window, diff = Counter(), [], Counter()
        for d in docs:
            arm = d["arms"][name]
            per_window.extend(arm["per_window"])
            totals.update({k: v for k, v in arm["totals"].items() if k != "exclusion_version"})
            diff.update(arm.get("windows_differing_from_round9", {}))
        out["arms"][name] = {"totals": dict(totals), "per_window": per_window,
                             "classes": docs[0]["arms"][name]["classes"]}
        if name != "round9":
            out["arms"][name]["windows_differing_from_round9"] = dict(diff)
    Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, res in out["arms"].items():
        print(name, {k: v for k, v in res["totals"].items() if v})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--r9-ckpt", default=str(ROOT / "tmp_r12/ck/r9_stage300.pt"))
    p.add_argument("--extra", default=None, help="optional checkpoint evaluated under the round-12 config as is")
    p.add_argument("--windows", type=int, default=48)
    p.add_argument("--horizon-s", type=float, default=120.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="docs/r12_support_recount.json")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--merge", nargs="*", default=None, help="merge shard JSONs into --out and exit")
    a = p.parse_args(argv)
    if a.merge:
        merge_shards(a.merge, a.out)
        return
    torch.set_num_threads(a.threads)
    from tools.r10b_courtesy_offline import datasets
    t0 = time.time()
    arms = {"round9": load("c4_wm_v10round9.toml", a.r9_ckpt, False),
            "round12_init": load("c4_wm_v10round12.toml", a.r9_ckpt, True)}
    if a.extra:
        arms["round12_extra"] = load("c4_wm_v10round12.toml", a.extra, False)
    n_steps = int(round(a.horizon_s / 0.5))
    sets = datasets(arms["round9"][0], a.roots, n_steps, 30.0)
    rng = np.random.default_rng(12)
    jobs = []
    for si, ds in enumerate(sets):
        by_ep = {}
        for i, (ep, _tick) in enumerate(ds.positions):
            by_ep.setdefault(ep, []).append(i)
        jobs.extend((si, int(rng.choice(idx))) for idx in by_ep.values())
    rng.shuffle(jobs)
    jobs = jobs[:a.windows][a.shard::a.n_shards]
    out = {"tool": "tools/r12_support_recount.py", "windows": len(jobs), "horizon_s": a.horizon_s,
           "arms": {name: recount(arm, jobs, sets, n_steps) for name, arm in arms.items()}}
    ref = out["arms"]["round9"]["per_window"]
    for name, res in out["arms"].items():
        if name == "round9":
            continue
        diff = Counter()
        for w9, w in zip(ref, res["per_window"]):
            for k, v in w.items():
                if isinstance(v, int) and isinstance(w9.get(k), int) and v != w9[k]:
                    diff[k] += 1
        res["windows_differing_from_round9"] = dict(diff)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, res in out["arms"].items():
        print(name, {k: v for k, v in res["totals"].items() if "exclu" in k or "scored" in k or "events" in k})


if __name__ == "__main__":
    main()
