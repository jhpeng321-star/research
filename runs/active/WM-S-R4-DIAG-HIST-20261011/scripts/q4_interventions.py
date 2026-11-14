"""Q4: inference interventions on R4 over the real-history quick-20.

Variants (patched driver-encoder outputs at rollout step 0; the embedding is then
fixed for the whole rollout exactly as in the unmodified model):
  base        unmodified R4 (must reproduce the round-4 cache bit-exactly)
  paramclamp  9 inferred IDM/MOBIL parameters clamped to the TRAIN 5-95 % range
  paramprior  those parameters replaced by the population prior (vType defaults)
  embclamp    fused style embedding clamped per-dim to the TRAIN 5-95 % range
  embprior    embedding replaced by the TRAIN per-dim median (population prior)

For R4 (rollout_mode=residual) the 9 parameters do not feed the rollout at all
(physics uses the defaults; the residual is conditioned on the embedding only),
so paramclamp/paramprior are expected to be no-ops -- verified here numerically.
Writes out/q4_interventions.json.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_lib2 import (DIAG, OUT, R4_SCORE, SETS, dump_json, ensure_env, load_json,
                       make_predictor, paired, quick20, summary_for)

ensure_env()
import numpy as np  # noqa: E402

WORKERS = 5
VARIANTS = ("base", "paramclamp", "paramprior", "embclamp", "embprior")
_W: dict = {}


def _patch(predictor, variant: str, ranges: dict):
    if variant == "base":
        return
    import torch
    enc = predictor.model.driver_encoder
    orig = enc.forward
    p_lo = torch.as_tensor(np.asarray(ranges["param_lo"]), dtype=torch.float32)
    p_hi = torch.as_tensor(np.asarray(ranges["param_hi"]), dtype=torch.float32)
    e_lo = torch.as_tensor(np.asarray(ranges["emb_lo"]), dtype=torch.float32)
    e_hi = torch.as_tensor(np.asarray(ranges["emb_hi"]), dtype=torch.float32)
    e_med = torch.as_tensor(np.asarray(ranges["emb_med"]), dtype=torch.float32)

    def forward(history, history_mask, interaction, defaults, _orig=orig):
        out = _orig(history, history_mask, interaction, defaults)
        out = dict(out)
        if variant == "paramclamp":
            out["values"] = torch.maximum(torch.minimum(out["values"], p_hi), p_lo)
        elif variant == "paramprior":
            out["values"] = defaults.expand_as(out["values"]).clone()
        elif variant == "embclamp":
            out["embedding"] = torch.maximum(torch.minimum(out["embedding"], e_hi), e_lo)
        elif variant == "embprior":
            out["embedding"] = e_med.expand_as(out["embedding"]).clone()
        else:
            raise ValueError(variant)
        return out

    enc.forward = forward


def _init(ranges: dict):
    ensure_env(1)
    import torch
    torch.set_num_threads(1)
    _W["predictor"] = make_predictor("R4")
    _W["ranges"] = ranges


def _task(payload):
    variant, record = payload
    from wms.eval.score import ScoreSpec, evaluate_state
    predictor = _W["predictor"]
    _patch(predictor, variant, _W["ranges"])
    try:
        spec = ScoreSpec(horizons_s=(10.0, 30.0, 60.0, 120.0),
                         predict_horizon_s=120.0, seed=0)
        row = evaluate_state(record, predictor.predict_states, spec)
    finally:
        enc = predictor.model.driver_encoder
        try:
            del enc.forward
        except AttributeError:
            pass
    return variant, row


def main() -> None:
    import torch
    t0 = time.time()
    q2 = load_json(OUT / "q2_encoder.json")
    pn = list(q2["train"]["params"])
    ranges = {
        "param_lo": [q2["train"]["params"][n]["quantiles"]["5"] for n in pn],
        "param_hi": [q2["train"]["params"][n]["quantiles"]["95"] for n in pn],
        "emb_lo": q2["train"]["embedding"]["q05"],
        "emb_hi": q2["train"]["embedding"]["q95"],
        "emb_med": q2["train"]["embedding"]["median"],
    }
    records = quick20(SETS["real_q20"])
    tasks = [(v, r) for v in VARIANTS for r in records]
    rows = {v: [] for v in VARIANTS}
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init,
                             initargs=(ranges,)) as pool:
        for i, (variant, row) in enumerate(pool.map(_task, tasks)):
            rows[variant].append(row)
            if (i + 1) % 20 == 0:
                print(f"[q4] {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    for v in VARIANTS:
        rows[v].sort(key=lambda r: (str(r["world"]), float(r["t0_s"])))

    # integrity: base must equal the round-4 cache bit-exactly
    cache = load_json(R4_SCORE / "cache" / "R4-last__wall_same_real.json")["states"]
    c1rows = load_json(R4_SCORE / "cache" / "C1__wall_same_real.json")["states"]

    def val(row, h):
        b = ((row.get("arms") or {}).get("wall") or {})
        if b.get("error"):
            return None
        x = b["car"][h]["hdv"]["ds_med"]
        return float(x) if x is not None else None

    max_diff = 0.0
    diffs = []
    for mine, ref in zip(rows["base"], cache):
        for h in ("10.0", "30.0", "60.0", "120.0"):
            a, b = val(mine, h), val(ref, h)
            if a is not None and b is not None:
                max_diff = max(max_diff, abs(a - b))
                diffs.append(abs(a - b))
    integrity = {"max_abs_diff_vs_cache": max_diff, "n_compared": len(diffs),
                 "bit_identical": max_diff == 0.0}

    out = {"integrity_base_vs_round4_cache": integrity,
           "variant_summaries": {v: summary_for(rows[v]) for v in VARIANTS},
           "paired_variant_minus_base": {},
           "paired_variant_minus_c1": {},
           "paired_base_minus_c1": paired(rows["base"], c1rows),
           "per_state": {v: [{"world": r["world"], "t0_s": r["t0_s"],
                              "ds60": val(r, "60.0"), "ds120": val(r, "120.0")}
                             for r in rows[v]] for v in VARIANTS},
           "c1_cache_summary": summary_for(c1rows)}
    for v in VARIANTS:
        if v == "base":
            continue
        out["paired_variant_minus_base"][v] = paired(rows[v], rows["base"])
        out["paired_variant_minus_c1"][v] = paired(rows[v], c1rows)
    dump_json(out, OUT / "q4_interventions.json")

    # no-op verification detail for param variants
    for v in ("paramclamp", "paramprior"):
        dmax = 0.0
        for mine, ref in zip(rows[v], rows["base"]):
            for h in ("10.0", "30.0", "60.0", "120.0"):
                a, b = val(mine, h), val(ref, h)
                if a is not None and b is not None:
                    dmax = max(dmax, abs(a - b))
        print(f"[q4] {v} max |diff| vs base: {dmax}", flush=True)
    print(f"[q4] integrity {integrity}", flush=True)
    for v in VARIANTS:
        s = out["variant_summaries"][v]
        print(f"[q4] {v:11s} 60s {s['ds']['60']['median']:.3f}  "
              f"120s {s['ds']['120']['median']:.3f}  "
              f"blow {s['blowup']['n_blowup']}/{s['blowup']['n_eval']}", flush=True)
    print(f"q4 done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
