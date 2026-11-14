"""Damping-snap CPU check: build pair-dense batches over ``pack_v3`` + the
re-aligned ``pack_damping`` and report the damping/native sibling pairs.

Materializes one re-aligned damping item end-to-end to prove its overlay
outcome supervision is non-empty, and reports per-batch trajectory/damping
counts plus the number of complete eligible pairs.  No training, no GPU.

Usage::

    python -m dspeed.cpu_check_damping \
        --base D:/rd-staging/wmn1_feat/pack_v3 \
        --extra _out/pack_damping --overlay _out/labels_overlay_damping \
        --pairs 4 --batches 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402

from . import damping as DM  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))


def run(base_pack, extra_pack, overlay_dir, pairs_per_batch=4, seed=0,
        epoch=0, batches=2) -> dict:
    from dspeed.composite import ConcatCorpus
    from train.dataset import (contrast_pairs, is_outcome_supervision_excluded,
                               pair_dense_batches)
    corpus = ConcatCorpus([base_pack, extra_pack], split="TRAIN",
                          pair_sampling=True, k_max=192,
                          extra_overlay_dir=overlay_dir)
    items = [corpus._items[j] for j in corpus._order]
    pairs = contrast_pairs(items)

    def is_extra(pos):
        return items[pos].get("_pack") == 1

    def is_damping_native(a, b):
        ia, ib = items[a], items[b]
        arms = {str(ia["arm"]), str(ib["arm"])}
        return ia["pair_key"] == ib["pair_key"] and arms in (
            {"base", "native_base"}, {"base", "native"}, {"native_base", "base"},
            {"native", "base"})

    damping_native = [(a, b) for a, b in pairs if is_damping_native(a, b)]
    extra_pairs = [(a, b) for a, b in pairs if is_extra(a) or is_extra(b)]
    all_batches = pair_dense_batches(items, pairs_per_batch, seed, epoch,
                                     pairs=pairs, merge_coverage=True)
    partners = {}
    for a, b in pairs:
        partners.setdefault(int(a), set()).add(int(b))
        partners.setdefault(int(b), set()).add(int(a))
    report = []
    for bi, batch in enumerate(all_batches[:batches]):
        present = set(batch)
        npair = sum(1 for p in present for q in partners.get(p, ())
                    if p < q and q in present)
        extra = [p for p in batch if is_extra(p)]
        dn = sum(1 for p in present for q in partners.get(p, ())
                 if p < q and q in present and is_damping_native(p, q))
        report.append(dict(
            batch=bi, n_items=len(batch), n_damping=len(extra),
            n_pairs=npair, n_damping_native_pairs=dn,
            damping=[items[p]["world"] + "__" + items[p]["arm"]
                     for p in extra[:6]]))
    ex_pos = next(p for p in range(len(items)) if is_extra(p))
    example = corpus[ex_pos]
    ex = dict(world=items[ex_pos]["world"], arm=items[ex_pos]["arm"],
              t0_s=float(items[ex_pos]["t0_s"]),
              damping_snap=items[ex_pos].get("damping_snap"),
              excluded=is_outcome_supervision_excluded(items[ex_pos]),
              outcome_mask_nonempty=bool(np.asarray(
                  example["outcome_mask"]).any()),
              n_outcome_taus=int(np.asarray(example["outcome_mask"]).sum()),
              label_t0_rows=int(example.get("label_t0_rows", 0)),
              pair_key=example["pair_key"], policy=example["policy"])
    return dict(pairs_per_batch=pairs_per_batch, total_pairs=len(pairs),
                total_items=len(items),
                n_extra_items=sum(1 for it in items if it.get("_pack") == 1),
                n_damping_native_pairs=len(damping_native),
                n_extra_pairs=len(extra_pairs), batches=report, example=ex)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dspeed.cpu_check_damping")
    ap.add_argument("--base", default="D:/rd-staging/wmn1_feat/pack_v3")
    ap.add_argument("--extra", default=str(DM.PACK_ROOT))
    ap.add_argument("--overlay", default=str(DM.OVERLAY_ROOT))
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--batches", type=int, default=2)
    args = ap.parse_args(argv)
    rep = run(args.base, args.extra, args.overlay, args.pairs, args.seed,
              args.epoch, args.batches)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
