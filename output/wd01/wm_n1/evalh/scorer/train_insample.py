"""IN-SAMPLE TRAIN scoring of a trained Idea A member (diagnostic, T1).

Scores the trained outcome head on the TRAIN pack in teacher mode -- the exact
input construction training used (``train.idea_a.IdeaACorpus.window`` lines
1031-1096, reused here through
:func:`evalh.scorer.predictors._training_window_summary`) -- against the
corrected TRAIN-only label sidecar, to separate "does not fit TRAIN" from
"fits TRAIN but not CAL".

Design constraints honoured here:

* CAL / HOLDOUT / chunk-5 / J4B / INC1 data is never referenced; only the TRAIN
  pack (``pack_v3``) and the corrected TRAIN-only outcome sidecar are read.
* Owner rule (b): ``base`` / ``native_base`` arms are excluded from supervision
  (``train.dataset.OUTCOME_SUPERVISION_EXCLUDED_ARMS``) before any prediction.
* The t0 mapper is reused: only sidecar rows with ``t == t0_s`` and
  ``tau_available > 0`` are scored (``train.dataset._t0_row_pick`` semantics).
* The pack arrays are concatenated over runs, so the item's GLOBAL ``tick`` is
  used (training's indexing), not the per-shard local tick of the CAL scorer.

The model runs the SAME correction-summary + outcome-head path the scorer uses
(``training_summary_raw`` style, via the shared window-summary helper), so the
numbers are directly comparable to the CAL-tune scorer output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from evalh.metrics import CONTRASTS, median_index
from evalh.protocol import QUANTILE_LEVELS, OutcomeRow
from evalh.scorer.baselines import (ZeroEffectBaseline,
                                    TrainClimatologyBaseline)
from evalh.scorer.metrics_loop import (DEFAULT_DV_SAMPLES,
                                       DV_METHOD_COMONOTONE,
                                       DV_METHOD_INDEPENDENT, score)

SCHEMA = "wm_n1_train_insample/1"
CLIM_SCHEMA = "wm_n1_train_climatology/1"

TAUS_S = (30.0, 60.0, 120.0, 300.0, 570.0)
KNOWN_POLICIES = ("native", "damping", "pi", "wall")


class InSampleRefusal(RuntimeError):
    """A required TRAIN input is absent or inconsistent (fail closed)."""


def _canon(arm_kind) -> str:
    from evalh.idea_a import canon_action

    token = str(arm_kind or "").strip().lower().split("_")[0]
    return canon_action(token)


def load_index(pack_dir):
    pack_dir = Path(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text("utf-8"))
    index = json.loads((pack_dir / "INDEX.json").read_text("utf-8"))
    runs = {str(r["key"]): r for r in index["runs"]}
    items = list(manifest["items"])
    return manifest, index, runs, items


def row_to_item(items):
    """``packed outcome row id -> (position, item)`` for TRAIN items only.

    The manifest's ``outcome_rows`` map is the authoritative run identity, so a
    world/arm/t0 that appears more than once (p40 variants) is disambiguated by
    the row id rather than guessed.
    """
    mapping: dict = {}
    for pos, item in enumerate(items):
        if str(item.get("split", "")).upper() != "TRAIN":
            continue
        for rid in item.get("outcome_rows") or []:
            rid = int(rid)
            if rid in mapping and mapping[rid][0] != pos:
                raise InSampleRefusal(f"outcome row {rid} claimed by two items")
            mapping[rid] = (pos, item)
    return mapping


def _run_end(runs, item):
    key = f"{item['batch']}/{item['world']}__{item['arm']}"
    run = runs.get(key)
    if run is None:
        raise InSampleRefusal(f"pack index has no run for {key}")
    return run, int(run["tick_start"]) + int(run["n_ticks"])


def train_label_groups(pack_dir, sidecar_dir, items, *, taus=TAUS_S):
    """TRAIN t0-aligned label rows grouped by their authoritative item position."""
    from train.dataset import OUTCOME_SUPERVISION_EXCLUDED_ARMS
    from train.label_isolation import ROW_ID_COLUMN, load_train_outcomes

    frame = load_train_outcomes(str(pack_dir), str(sidecar_dir))
    frame = frame.reset_index()
    if ROW_ID_COLUMN not in frame.columns:
        raise InSampleRefusal(f"sidecar frame lacks {ROW_ID_COLUMN}")
    r2i = row_to_item(items)
    tau_set = {round(float(t), 6) for t in taus}
    excluded = {str(a).strip().lower()
                for a in OUTCOME_SUPERVISION_EXCLUDED_ARMS}
    groups: dict = {}
    counts = dict(n_input_rows=int(len(frame)), n_excluded_arm=0,
                  n_excluded_tau=0, n_excluded_unavailable=0,
                  n_excluded_unknown_policy=0, n_excluded_no_item=0,
                  n_duplicate=0)
    for rec in frame.to_dict("records"):
        source = str(rec["source"])
        if source.strip().lower() in excluded:
            counts["n_excluded_arm"] += 1
            continue
        tau = round(float(rec["tau"]), 6)
        if tau not in tau_set:
            counts["n_excluded_tau"] += 1
            continue
        if float(rec["tau_available"]) <= 0.0:
            counts["n_excluded_unavailable"] += 1
            continue
        t0 = round(float(rec["t0_s"]), 6)
        if round(float(rec["t"]), 6) != t0:
            counts["n_excluded_tau"] += 1
            continue
        canon = _canon(rec.get("arm_kind"))
        if canon not in KNOWN_POLICIES:
            counts["n_excluded_unknown_policy"] += 1
            continue
        entry = r2i.get(int(rec[ROW_ID_COLUMN]))
        if entry is None:
            counts["n_excluded_no_item"] += 1
            continue
        pos, item = entry
        groups.setdefault(pos, []).append(
            dict(tau=tau, canon=canon, y=float(rec["total_delay_ref_s"]),
                 world=str(rec["world"]),
                 source=source))
    for pos, rows in groups.items():
        seen = set()
        for row in rows:
            if row["tau"] in seen:
                raise InSampleRefusal(f"duplicate t0 label row for item "
                                      f"{pos}/tau {row['tau']}")
            seen.add(row["tau"])
    return groups, counts


def _array(pack_dir, name):
    path = Path(pack_dir) / f"{name}.npy"
    if not path.is_file():
        raise InSampleRefusal(f"TRAIN pack lacks {name}.npy")
    return np.load(path, mmap_mode="r")


def item_head_quantiles(bundle, pack_dir, item, run_end, *, arrays, n_steps):
    """Teacher-mode ``[n_tau, n_targets, n_quantiles]`` head output for one item."""
    import torch

    from evalh.scorer.predictors import (_training_window_summary,
                                         head_quantiles_from_summary)
    from features.constants import IS_CAV_IDX
    from model.constants import TAUS

    kin, mask, target, plan, step, counter = arrays
    tick = int(item["tick"])
    horizon = max(0, min(int(n_steps), run_end - tick - 1))
    if horizon < 1:
        raise InSampleRefusal(
            f"item {item['world']}__{item['arm']} has no in-run window "
            f"(tick {tick}, run end {run_end})")
    stop = tick + horizon + 1
    kin_seq = torch.as_tensor(np.array(kin[tick:stop], np.float32))
    plan_seq = torch.as_tensor(np.array(plan[tick:stop], np.float32))
    present0 = torch.as_tensor(np.asarray(mask[tick]).astype(bool))
    tm = torch.as_tensor(np.asarray(target[tick:tick + horizon], bool))
    is_cav = torch.as_tensor(
        np.asarray(step[tick, :, IS_CAV_IDX] > 0.5).astype(bool))
    raw = _training_window_summary(kin_seq, plan_seq, present0, tm, is_cav)
    t0_kin = torch.as_tensor(np.array(kin[tick], np.float32)).unsqueeze(0)
    t0_mask = torch.as_tensor(np.asarray(mask[tick]).astype(bool)).unsqueeze(0)
    t0_counter = torch.as_tensor(np.asarray(counter[tick], np.float64)
                                 ).unsqueeze(0)
    head_q = head_quantiles_from_summary(
        bundle, raw, t0_kin=t0_kin, t0_mask=t0_mask, t0_counter=t0_counter)
    assert int(head_q.shape[0]) == len(TAUS)
    return head_q


def build_rows_and_quantiles(bundle, pack_dir, sidecar_dir, *,
                             n_steps=None, taus=TAUS_S, max_groups=None):
    """In-sample (rows, q_a, abs, census) over the TRAIN pack."""
    from model.constants import CORRECTION_WINDOW_STEPS, TAUS

    if n_steps is None:
        n_steps = CORRECTION_WINDOW_STEPS
    manifest, index, runs, items = load_index(pack_dir)
    groups, counts = train_label_groups(pack_dir, sidecar_dir, items,
                                        taus=taus)
    keys = sorted(groups)
    if max_groups is not None:
        keys = keys[:int(max_groups)]
    arrays = (_array(pack_dir, "veh_kin"), _array(pack_dir, "veh_mask"),
              _array(pack_dir, "target_mask"), _array(pack_dir, "cav_plan_ch"),
              _array(pack_dir, "veh_step"), _array(pack_dir, "counter"))
    delay_i = median_index(QUANTILE_LEVELS)
    taus_arr = np.asarray(TAUS, dtype=float)
    rows, q_a, abs_rows = [], [], []
    skipped_no_item = 0
    for pos in keys:
        item = items[pos]
        run, run_end = _run_end(runs, item)
        try:
            head_q = item_head_quantiles(bundle, pack_dir, item, run_end,
                                         arrays=arrays, n_steps=n_steps)
        except InSampleRefusal:
            skipped_no_item += 1
            continue
        for rec in sorted(groups[pos], key=lambda r: r["tau"]):
            ti = int(np.argmin(np.abs(taus_arr - rec["tau"])))
            qrow = np.asarray(head_q[ti, 0, :], dtype=float)
            t0 = round(float(item["t0_s"]), 6)
            rows.append(OutcomeRow(
                world=rec["world"], t0_s=t0, decision_t_s=t0, tau_s=rec["tau"],
                policy=rec["canon"], y_delay_s=rec["y"],
                features=dict(pack_item_position=pos, arm=rec["source"])))
            q_a.append(qrow)
            abs_rows.append(dict(world=rec["world"], arm=rec["canon"],
                                 tau=rec["tau"], q50=float(qrow[delay_i]),
                                 y=rec["y"]))
    census = dict(counts)
    census.update(n_items_scored=len({r.features.get("pack_item_position")
                                      for r in rows}),
                  n_rows_scored=len(rows),
                  n_groups_skipped_no_item=skipped_no_item)
    return rows, np.asarray(q_a, dtype=float), abs_rows, census


def in_sample_climatology(rows):
    """Empirical per-(contrast,tau) dV quantiles from the SAME scored pairs."""
    key = {}
    for r in rows:
        key[(r.world, round(r.t0_s, 6), round(r.tau_s, 6), r.policy)] = \
            float(r.y_delay_s)
    cells: dict = {}
    for name, (pol_a, pol_b) in CONTRASTS.items():
        by_tau: dict = {}
        worlds_by_tau: dict = {}
        for (world, t0, tau, canon) in sorted(key):
            if canon != pol_a:
                continue
            ka = (world, t0, tau, pol_a)
            kb = (world, t0, tau, pol_b)
            if kb not in key:
                continue
            by_tau.setdefault(tau, []).append(float(key[ka] - key[kb]))
            worlds_by_tau.setdefault(tau, set()).add(world)
        if by_tau:
            cells[name] = {
                f"{tau:g}": dict(
                    n_pairs=len(v), n_worlds=len(worlds_by_tau[tau]),
                    quantiles=[float(x) for x in
                               np.quantile(np.asarray(v, float),
                                           QUANTILE_LEVELS)])
                for tau, v in sorted(by_tau.items())}
    return dict(schema=CLIM_SCHEMA, reference_policy="native",
                contrasts=sorted(cells), cells=cells,
                note="IN-SAMPLE climatology: built from the same scored pairs")


def absolute_bias(abs_rows, *, taus=TAUS_S):
    """Per-(arm,tau) and per-arm absolute-delay bias of q50 vs the truth."""
    per_arm_tau: dict = {}
    per_arm: dict = {}
    y_by_arm_tau: dict = {}
    for row in abs_rows:
        arm, tau = row["arm"], round(float(row["tau"]), 6)
        per_arm_tau.setdefault((arm, tau), []).append(row["q50"] - row["y"])
        y_by_arm_tau.setdefault((arm, tau), []).append(row["y"])
        per_arm.setdefault(arm, []).append(row["q50"] - row["y"])
    out_tau = {}
    for (arm, tau), v in sorted(per_arm_tau.items()):
        mean_y = float(np.mean(y_by_arm_tau[(arm, tau)]))
        mean_bias = float(np.mean(v))
        out_tau[f"{arm}@{tau:g}"] = dict(
            arm=arm, tau=tau, n=len(v), mean_y=mean_y, mean_bias=mean_bias,
            median_bias=float(np.median(v)),
            mean_bias_pct=(float("nan") if mean_y == 0 else
                           100.0 * mean_bias / mean_y))
    out_arm = {arm: dict(arm=arm, n=len(v), mean_bias=float(np.mean(v)),
                         median_bias=float(np.median(v)))
               for arm, v in sorted(per_arm.items())}
    return dict(per_arm_tau=out_tau, per_arm=out_arm)


def run_insample(pack_dir, sidecar_dir, checkpoint, out_dir, *,
                 dv_samples=DEFAULT_DV_SAMPLES, n_boot=2000, seed=20261001,
                 max_groups=None):
    from evalh.scorer.checkpoint import load_scoring_checkpoint
    from model.constants import CORRECTION_WINDOW_STEPS

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise InSampleRefusal(f"output dir {out_dir} exists; refusing to "
                              "overwrite")
    bound = load_scoring_checkpoint(str(checkpoint))
    rows, q_a, abs_rows, census = build_rows_and_quantiles(
        bound.bundle, pack_dir, sidecar_dir, max_groups=max_groups)
    if not rows:
        raise InSampleRefusal("no scorable TRAIN rows after all filters")

    clim_doc = in_sample_climatology(rows)
    clim_path = out_dir / "train_climatology_insample.json"
    out_dir.mkdir(parents=True)
    clim_path.write_text(json.dumps(clim_doc, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")
    clim_sha = hashlib.sha256(clim_path.read_bytes()).hexdigest()
    dv_models = {
        "zero": ZeroEffectBaseline(),
        "train_climatology": TrainClimatologyBaseline(
            clim_path, expected_sha256=clim_sha),
    }
    metrics_com = score(rows, {"A": q_a}, dv_models=dv_models, taus=TAUS_S,
                        n_boot=n_boot, seed=seed,
                        dv_method=DV_METHOD_COMONOTONE)
    metrics_ind = score(rows, {"A": q_a}, dv_models=dv_models, taus=TAUS_S,
                        n_boot=n_boot, seed=seed,
                        dv_method=DV_METHOD_INDEPENDENT,
                        dv_samples=dv_samples)
    doc = dict(
        schema=SCHEMA, checkpoint_sha256=bound.sha256, member=bound.member,
        cav_motion="teacher", window_steps=int(CORRECTION_WINDOW_STEPS),
        taus=[float(t) for t in TAUS_S],
        reference_policy="native", exclusion_arms=["base", "native_base"],
        census=census,
        metrics_comonotone=metrics_com, metrics_independent=metrics_ind,
        absolute_bias=absolute_bias(abs_rows),
        climatology=dict(path=str(clim_path), sha256=clim_sha,
                         in_sample=True))
    (out_dir / "RESULTS.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _write_abs_csv(out_dir / "absolute_bias.csv", abs_rows)
    return doc


def _write_abs_csv(path: Path, abs_rows) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["world", "arm", "tau_s", "q50", "y_delay_s", "bias"])
        for r in abs_rows:
            writer.writerow([r["world"], r["arm"], repr(r["tau"]),
                             repr(r["q50"]), repr(r["y"]),
                             repr(r["q50"] - r["y"])])


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m evalh.scorer.train_insample",
        description="In-sample TRAIN scoring of checkpoint 94984cc0 "
                    "(teacher mode; TRAIN pack + corrected sidecar).")
    p.add_argument("--pack", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dv-samples", dest="dv_samples", type=int, default=4096)
    p.add_argument("--n-boot", dest="n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--max-groups", dest="max_groups", type=int, default=None,
                   help="smoke-test only: score at most this many items")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        doc = run_insample(args.pack, args.sidecar, args.checkpoint, args.out,
                           dv_samples=args.dv_samples, n_boot=args.n_boot,
                           seed=args.seed, max_groups=args.max_groups)
    except InSampleRefusal as exc:
        print(f"[train-insample refused] {exc}", file=sys.stderr)
        return 87
    print(json.dumps(dict(schema=SCHEMA, out=args.out,
                          census=doc["census"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
