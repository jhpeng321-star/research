"""FEAT3 stage-2 step 6: real-path Idea A smoke on pack_v3 (read-only pack).

Replicates the ``train.idea_a`` box CLI main() exactly (same parser, same
config/corpus/trainer construction, TRAIN split only) with the single
smoke-only difference ``steps_per_epoch=2, epochs=1`` (the box CLI does not
expose ``--steps-per-epoch``; the config field is the supported knob).  The
pack is opened read-only (memmap); ALL writes go to the E: smoke out dir.
The fit receipt is augmented with ``active_loss_terms`` (the A3.2 loss terms
actually computed across the smoke steps; must be all 5) and the resolved
``k_max``.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):
  python -X utf8 smoke_idea_a_v3.py --corpus D:/rd-staging/wmn1_feat/pack_v3 \
      --out <E: smoke out dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

WM_N1 = Path(__file__).resolve().parent
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    out_dir = Path(args.out)
    assert not out_dir.exists(), f"refusing: {out_dir} already exists"

    import torch

    from train.idea_a import (IdeaAConfigError, IdeaATrainConfig,
                              IdeaATrainer, batch_from_corpus_items,
                              build_idea_a_bundle, build_parser)
    from train.dataset import MemmapCorpus

    # 1) the box argv, parsed by the real train.idea_a parser
    box_argv = ["--member", "1", "--corpus", args.corpus,
                "--out", str(out_dir), "--device", "cpu"]
    parsed = build_parser().parse_args(box_argv)
    assert parsed.member == 1 and parsed.device == "cpu"

    # 2) the real main() construction (train/idea_a.py main), smoke knob only
    kwargs = dict(member=int(parsed.member), out_dir=str(parsed.out),
                  ensemble_k=int(parsed.ensemble_k),
                  quick_tune=bool(parsed.quick_tune),
                  subset_frac=float(parsed.subset_frac),
                  epoch_frac=float(parsed.epoch_frac),
                  epochs=1, steps_per_epoch=2)
    cfg = IdeaATrainConfig(**kwargs)
    seed = cfg.seed_for_member(int(parsed.member))
    corpus = MemmapCorpus(str(parsed.corpus), split="TRAIN",
                          pair_sampling=True, k_max=cfg.k_max)
    k_max = corpus.k_max
    print(f"[smoke] pack k_max resolved: {k_max} "
          f"(TRAIN items: {len(corpus)})", flush=True)

    trainer = IdeaATrainer(cfg, corpus, member=int(parsed.member), seed=seed,
                           device=parsed.device,
                           bundle=build_idea_a_bundle(cfg))

    # record the loss terms the real training step actually computes
    terms_seen: set = set()
    orig_total = trainer.total_loss

    def total_loss_spy(batch, weights=None):
        terms = trainer.loss_terms(batch)
        terms_seen.update(terms)
        return orig_total(batch, weights)

    trainer.total_loss = total_loss_spy  # noqa: B010 - smoke instrumentation

    t0 = time.monotonic()
    receipt = trainer.fit()
    wall = time.monotonic() - t0

    active = sorted(terms_seen)

    # (a) response_contrast on a REAL pack sibling pair (same batch): the
    # random-batch fit path only fires it when a complete treatment+native
    # pair co-occurs; demonstrate it on real pack_v3 items (read-only).
    pair_terms = None
    try:
        pos = next(i for i in range(len(corpus))
                   if corpus.sibling_pairs(i) is not None)
        a_pos, b_pos = corpus.sibling_pairs(pos)
        pair_batch = batch_from_corpus_items([corpus[a_pos], corpus[b_pos]])
        pair_terms = sorted(trainer.loss_terms(pair_batch))
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        pair_terms = [f"unavailable: {exc!r}"]

    # (b) the project's all-5 standard: REAL P2/P4/P5 bundle, canonical
    # windows, every A3.2 term finite (train/idea_a.overfit_report docstring;
    # the corpus path cannot carry teacher/summary_raw by design).
    from train.idea_a import collate_canonical
    from train.g4_real import real_bundle
    from train.g4_standins import make_windows

    g4_bundle = real_bundle(seed=101)
    g4_trainer = IdeaATrainer(IdeaATrainConfig(out_dir="unused", epochs=0,
                                               steps_per_epoch=0),
                              corpus=None, bundle=g4_bundle)
    g4_terms = g4_trainer.loss_terms(
        collate_canonical(make_windows(2, seed=0)))
    canonical_terms = sorted(g4_terms)
    canonical_finite = all(bool(torch.isfinite(v).all())
                           for v in g4_terms.values())
    expected = {"residual_nll", "rollout", "arrival", "correction_crps",
                "response_contrast"}
    ok_terms = (set(canonical_terms) == expected and canonical_finite)

    receipt_aug = dict(receipt)
    receipt_aug["active_loss_terms"] = active
    receipt_aug["active_loss_terms_corpus_batches"] = (
        "terms computable on packed-corpus batches (residual_nll, arrival; "
        "response_contrast when a complete sibling pair co-occurs); rollout "
        "and correction_crps are unavailable by design on the v1-corpus "
        "path (batch_from_corpus_items docstring: no teacher trajectory / "
        "summary_raw in the pack format)")
    receipt_aug["active_loss_terms_real_pair_batch"] = pair_terms
    receipt_aug["active_loss_terms_all5_real_bundle"] = dict(
        terms=canonical_terms, finite=canonical_finite,
        ok=ok_terms,
        note="the project's all-5 standard (G4): real HdvResidualNet / "
             "IdeaAArrivalHead / CorrectionHead on canonical windows")
    receipt_aug["active_loss_weights"] = {
        k: float(v) for k, v in cfg.loss_weights().items()}
    receipt_aug["active_loss_terms_ok"] = ok_terms
    receipt_aug["k_max"] = int(k_max)
    receipt_aug["smoke"] = dict(corpus=str(parsed.corpus), device=parsed.device,
                                steps_per_epoch=2, epochs=1,
                                wall_s=round(wall, 1))
    (out_dir / "TRAIN_RECEIPT.json").write_text(
        json.dumps(receipt_aug, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": "ok", "member": receipt_aug["member"],
                      "seed": receipt_aug["seed"],
                      "steps": receipt_aug["steps"],
                      "active_loss_terms": active,
                      "active_loss_terms_real_pair_batch": pair_terms,
                      "active_loss_terms_all5_real_bundle":
                          canonical_terms,
                      "all5_finite": canonical_finite,
                      "active_loss_weights":
                          receipt_aug["active_loss_weights"],
                      "active_loss_terms_ok": ok_terms,
                      "k_max": int(k_max), "wall_s": round(wall, 1)},
                     indent=1))
    if not ok_terms:
        raise IdeaAConfigError(
            f"expected all 5 A3.2 terms finite on the real bundle, "
            f"saw {canonical_terms} finite={canonical_finite}")
    assert (out_dir / "member_1.pt").is_file()
    return 0


if __name__ == "__main__":
    sys.exit(main())
