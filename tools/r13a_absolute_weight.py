"""R13A weight rule for ``loss.follower_courtesy.absolute_weight`` (gradient-norm parity; CPU, offline).

Rule (same principle as the R10 cut-in weight and the R11 contrast weight, docs/r11_weight_rule.json):
the ADDED absolute term gets the gradient norm of the term it sits next to.  On the R11 offline cache
(teacher-forced TRAIN windows, the same 8 batches of 4 fit windows as the R11 weight rule: held-out
episodes = every 4th episode, ``numpy.random.default_rng(7)``), at the courtesy head's launch init (a
fresh zero-init version-2 head, as rounds 12 / 13A / 13F start), measure per batch

    G_contrast = || d huber(a_hat, a_rec - matched control shift) / d head ||   (the R11 contrast loss)
    G_abs      = || d huber(a_hat, a_rec) / d head ||                           (the R13A absolute loss)

with the SAME head instance for both, and set

    absolute_weight = w_contrast * geomean(G_contrast) / geomean(G_abs),   w_contrast = 9.067 (round 12).

The ratio is reported for several head seeds and for the cached intent column (lane-v5 P(merge), the
cache's input) and with the intent column zeroed (R13A intent 'none'), which bounds the effect of the
pivot intent (not cached) on the number.  The tool trains nothing.

    python tools/r13a_absolute_weight.py --cache E:/agent-work/c4-v10r11/tmp_r11/cache --out docs/r13a_absolute_weight_rule.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tools.r11_courtesy_offline import batch_of, head_loss, load_windows  # noqa: E402
from wms.model.follower_courtesy import COURTESY_PAIR_FIELDS, FollowerCourtesyHead  # noqa: E402

W_CONTRAST = 9.067          # loss.follower_courtesy.weight of round 12 (docs/r11_weight_rule.json)
INTENT_COLUMN = COURTESY_PAIR_FIELDS.index("intent")


def _gnorm(loss, head) -> float:
    grads = torch.autograd.grad(loss, list(head.parameters()), allow_unused=True)
    return float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))


def measure(windows, *, seeds, delta, amin, amax, zero_intent: bool) -> dict:
    eps = sorted({w["episode"] for w in windows})
    held = set(eps[::4])
    fit_w = [w for w in windows if w["episode"] not in held]
    rng = np.random.default_rng(7)
    picks = [rng.choice(len(fit_w), size=4, replace=False) for _ in range(8)]
    batches = []
    for pick in picks:
        b = batch_of([fit_w[i] for i in pick])
        if zero_intent:
            b = dict(b)
            b["x"] = b["x"].copy()
            b["x"][:, INTENT_COLUMN] = 0.0
        batches.append(b)
    per_seed = {}
    for seed in seeds:
        torch.manual_seed(int(seed))
        gc, ga = [], []
        for b in batches:
            head = FollowerCourtesyHead(version=2)
            gc.append(_gnorm(head_loss(head, b, {"contrast": True}, delta, amin, amax), head))
            ga.append(_gnorm(head_loss(head, b, {"contrast": False}, delta, amin, amax), head))
        g_c, g_a = (float(np.exp(np.mean(np.log(v)))) for v in (gc, ga))
        per_seed[str(seed)] = {"g_contrast_per_batch": gc, "g_abs_per_batch": ga,
                               "g_contrast": g_c, "g_abs": g_a, "ratio": g_c / g_a,
                               "per_batch_ratio": [c / a for c, a in zip(gc, ga)]}
    ratios = [v["ratio"] for v in per_seed.values()]
    return {"per_seed": per_seed, "ratio_geomean_over_seeds": float(np.exp(np.mean(np.log(ratios)))),
            "ratio_min_max": [min(ratios), max(ratios)], "n_fit_windows": len(fit_w),
            "n_partner_rows_in_batches": int(sum(int(b["has"].sum()) for b in batches))}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="R11 offline cache prefix (tools/r11_courtesy_offline.py cache)")
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--config", default="configs/c4_wm_v10round12.toml")
    args = p.parse_args(argv)
    from wms.cli.assemble import build_physics
    from wms.config.load import load_train_config
    cfg = load_train_config(ROOT / args.config, ["runtime.device=cpu"])
    phys = build_physics(cfg)
    delta = float(cfg.loss.follower_courtesy.huber_delta_mps2)
    w_contrast = float(cfg.loss.follower_courtesy.weight)
    amin, amax = float(phys.accel_min_mps2), float(phys.accel_max_mps2)
    windows = load_windows(args.cache, args.n_shards)
    res = {"rule": "absolute_weight = w_contrast * geomean_batches(G_contrast) / geomean_batches(G_abs), "
                   "zero-init version-2 head (launch init), same head per batch for both terms, the 8 "
                   "R11 weight-rule batches (rng 7, 4 fit windows each); reported = geometric mean of the "
                   "ratio over head seeds with the cached (lane-v5) intent column",
           "cache": str(args.cache), "n_windows": len(windows),
           "n_episodes": len({w["episode"] for w in windows}), "config": args.config,
           "w_contrast": w_contrast, "huber_delta": delta, "plant_bounds": [amin, amax]}
    res["cached_intent"] = measure(windows, seeds=args.seeds, delta=delta, amin=amin, amax=amax,
                                   zero_intent=False)
    res["intent_zeroed"] = measure(windows, seeds=args.seeds, delta=delta, amin=amin, amax=amax,
                                   zero_intent=True)
    ratio = res["cached_intent"]["ratio_geomean_over_seeds"]
    res["absolute_weight"] = float(f"{w_contrast * ratio:.4g}")
    res["absolute_weight_intent_zeroed"] = float(
        f"{w_contrast * res['intent_zeroed']['ratio_geomean_over_seeds']:.4g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_windows", "absolute_weight", "absolute_weight_intent_zeroed")}
                     | {"ratio_min_max": res["cached_intent"]["ratio_min_max"],
                        "g_contrast_seed0": res["cached_intent"]["per_seed"][str(args.seeds[0])]["g_contrast"]}))


if __name__ == "__main__":
    main()
