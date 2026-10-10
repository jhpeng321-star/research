"""A9 step 6: CPU-only 3-batch check over ``pack_v3`` + the dspeed pack.

Builds the composite corpus (no base outcome sidecar is needed to *batch*;
trajectory items read their labels from the dspeed overlay) and reports, for
the first 3 pair-dense batches, how many items, trajectory items and complete
eligible pairs each holds.  Also materializes one trajectory item end-to-end
to prove its overlay outcome supervision is non-empty.

Usage::

    python -m dspeed.cpu_check --base D:/rd-staging/wmn1_feat/pack_v3 \\
        --extra _out/pack_dspeed --overlay _out/labels_overlay_dspeed --pairs 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402

from . import common as D  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))


def run(base_pack, extra_pack, overlay_dir, pairs_per_batch=4, seed=0,
        epoch=0, batches=3) -> dict:
    from dspeed.composite import ConcatCorpus
    from train.dataset import contrast_pairs, pair_dense_batches
    corpus = ConcatCorpus([base_pack, extra_pack], split="TRAIN",
                          pair_sampling=True, k_max=192,
                          extra_overlay_dir=overlay_dir)
    items = [corpus._items[j] for j in corpus._order]
    pairs = contrast_pairs(items)
    partners = {}
    for a, b in pairs:
        partners.setdefault(int(a), set()).add(int(b))
        partners.setdefault(int(b), set()).add(int(a))
    all_batches = pair_dense_batches(items, pairs_per_batch, seed, epoch,
                                     pairs=pairs, merge_coverage=True)
    report = []
    for bi, batch in enumerate(all_batches[:batches]):
        present = set(batch)
        npair = sum(1 for p in present for q in partners.get(p, ())
                    if p < q and q in present)
        traj = [p for p in batch
                if str(items[p].get("policy")).lower() == "trajectory"]
        report.append(dict(batch=bi, n_items=len(batch),
                           n_trajectory=len(traj),
                           n_pairs=npair,
                           trajectories=[items[p]["world"] + "__"
                                         + items[p]["arm"] for p in traj[:4]]))
    # end-to-end: materialize one trajectory item and check the outcome mask
    traj_pos = next(p for p, it in enumerate(items)
                    if str(it.get("policy")).lower() == "trajectory")
    example = corpus[traj_pos]
    ex = dict(world=items[traj_pos]["world"], arm=items[traj_pos]["arm"],
              t0_s=float(items[traj_pos]["t0_s"]),
              outcome_mask_nonempty=bool(example["outcome_mask"].any()),
              n_outcome_taumask=int(example["outcome_mask"].sum()),
              delay_targets=[float(x) for x in
                             np.asarray(example["outcome_quantiles"])[:, 0]],
              label_t0_rows=int(example.get("label_t0_rows", 0)),
              pair_key=example["pair_key"], policy=example["policy"])
    return dict(pairs_per_batch=pairs_per_batch, total_pairs=len(pairs),
                total_items=len(items), batches=report, example=ex)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dspeed.cpu_check")
    ap.add_argument("--base", default="D:/rd-staging/wmn1_feat/pack_v3")
    ap.add_argument("--extra", default=str(D.PACK_ROOT))
    ap.add_argument("--overlay", default=str(D.OVERLAY_ROOT))
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--batches", type=int, default=3)
    args = ap.parse_args(argv)
    rep = run(args.base, args.extra, args.overlay, args.pairs, args.seed,
              args.epoch, args.batches)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
