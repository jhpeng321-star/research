"""Shared differentiable, native-lane TRAIN rollout; no simulator invocation."""
from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

from tools.wmfixmerge.data import COMPONENTS, WM2_SHA, digest, safe, verify
from wms.model.entries import NOMINAL_LENGTH_M, SUMO_V2_MIN_GAP_M
from wms.train.merge_response import ADMISSION_WIDTH_M

_WMREQUAL = Path(__file__).resolve().parents[1] / "wmrequal"
if str(_WMREQUAL) not in sys.path:
    # grouped_feedback imports sibling modules absolutely (labels, ...);
    # every entry point (feasibility, train) must survive without a manual shim.
    sys.path.insert(0, str(_WMREQUAL))


class Runtime:
    def __init__(self, prepared, config, checkpoint, controllers, device="cpu", *,
                 smooth_events=False, admission_width_m=None, merge_refine=(0, 0)):
        import torch
        from tools.wmcombined.e3_score import load_for_scoring
        from wms.data.c4_pairs import PairedBranchSource
        self.doc = verify(prepared)
        if digest(checkpoint) != WM2_SHA:
            raise ValueError("base is not registered WM2")
        self.device = torch.device(device)
        self.learned_admission = True
        # S-WMFIX-C fallback 1 (default OFF = exact identity): TRAINING-ONLY
        # smoothed admission relaxation.  When False (every inference/verdict
        # path) no surrogate is constructed and the differentiable stack is
        # bit-identical to the registered behaviour.
        self.smooth_events = bool(smooth_events)
        self.admission_width_m = (ADMISSION_WIDTH_M if admission_width_m is None
                                  else float(admission_width_m))
        # S-WMFIX-C fallback 2 (default (0, 0) = the registered 4-basis grid):
        # geometry-concentrated merge/ramp centres of the merge-response field.
        self.merge_refine = (int(merge_refine[0]), int(merge_refine[1]))
        self.config, settings, self.model = load_for_scoring(safe(config), safe(checkpoint))
        specs = json.loads(safe(controllers).read_text())
        specs = specs.get("controllers", specs)
        if not all(a in specs for a in ("ours_c1", "coopoff_c1")):
            raise ValueError("missing hash-bound entrant controller specifications")
        self.settings = replace(settings, seed=140014, entrant_control=True,
            entrant_controllers={a: specs[a] for a in ("ours_c1", "coopoff_c1")},
            entry_queue_state=True, entry_pad_auto=True, entry_pad_max=1024,
            entry_insert_rule="sumo_like_v2")
        self.model = self.model.to(self.device).eval()
        self.model.cav_exec_cap = "idm"
        self.model.enable_merge_response(merge_bases=self.merge_refine[0],
                                         ramp_bases=self.merge_refine[1])
        self.model.checkpoint_forward = True
        manifest = json.loads(safe(safe(self.doc["root"]) / "C4_EPISODES_MANIFEST.json").read_text())
        worlds = {row["world"] for row in self.doc["episodes"]}
        self.source = PairedBranchSource(self.doc["root"], horizon_steps=600,
            history_s=60., history_stride_s=1., entries_v2=True, junction_fix="ffill",
            command_known=True, arm_kinds=("B", "POLICY"), role="train",
            role_of={e["world"]: "train" if e["world"] in worlds else "val"
                     for e in manifest["episodes"]}, name="wmfix_registered_fit",
            arm_names=tuple(sorted({e["arm"] for e in self.doc["episodes"]})))
        if {g["pair_key"] for g in self.source.groups} != {r["pair_key"] for r in self.doc["episodes"]}:
            raise ValueError("production loader dropped or added registered groups")
        if self.source.receipt["dropped_groups_t0_mismatch"] or self.source.horizon != 600:
            raise ValueError("production loader failed exact starts/300s support")
        self.scales = torch.tensor(self.doc["scales"], device=self.device, dtype=torch.float32)
        self.anchor_scales = torch.tensor(self.doc["anchor_scales"], device=self.device,
                                          dtype=torch.float32)

    def rollout(self, group_index, horizon, arm_index=0):
        import numpy as np
        import torch
        from tools.errsrc.inference import LearnedMix
        from tools.wmcombined.e3_score import group_rows
        from tools.wmrequal.native_lanes import repair
        from tools.wmrequal.grouped_feedback import feedback
        from wms.control.futures import SampledFutures, FuturesConfig
        from wms.data.collate import collate
        from wms.train.inputs import prepare_inputs
        from wms.train.merge_response import (entrance_gap_series, smooth_admission_source,
                                              smooth_crossings)
        from wms.train.scoring_random import scoring_rng, bind_merge_crn
        import wms.control.entrant as entrant
        entrant.feedback_from_specs = feedback
        key, ref, arms = group_rows(self.source, group_index)
        if not arms:
            raise ValueError("no intervention arm")
        arms = [arms[arm_index % len(arms)]]  # native + one intervention: two-row microbatch
        steps = int(horizon * 2)
        records = self.source.group_records(group_index, [ref] + [r for _, r in arms], base_row=0)
        labels = [r for r in self.doc["labels"] if r["pair_key"] == key]
        native = next(r for r in labels if r["native"])
        order = [native] + [next(r for r in labels if r["arm"] == arm) for arm, _ in arms]
        for record, label in zip(records, order):
            record["entry_queue"] = np.array([*label["queue0"], *label["demand_rate"],
                label["penetration"], label["penetration"]], dtype=np.float32)
        batch = collate(records)
        with scoring_rng(140014, self.device):
            inp = prepare_inputs(batch, self.model, self.settings, self.device, steps)
        inp.cav_controller = repair(inp.cav_controller)
        bind_merge_crn(inp)
        inp.entries.cfg = replace(inp.entries.cfg, conserve_queue=True)
        # TRAIN-only teacher forcing of recorded exogenous demand, never service.
        due_rows = [r["due_increments"][:steps] for r in order]
        inp.entries._emit_due = RecordedDemand(due_rows)
        # Shared driver/embedding draws from the native history, never driver labels.
        select = lambda t: t[:1] if torch.is_tensor(t) and t.ndim and t.shape[0] == len(records) else t
        initial = {k: select(v) for k, v in inp.initial.items()}
        hist = dict(driver_history=select(inp.observed_hist), driver_history_mask=select(inp.observed_hist_mask),
                    history=select(inp.hist), history_mask=select(inp.hist_mask))
        plan = SampledFutures(FuturesConfig(n_samples=1, driver_noise_scale=1., embedding_noise_std=.05),
                              140014).plan([self.model], initial, hist, steps)[0][0]
        def expand(x):
            if isinstance(x, dict):
                return {k: expand(v) for k, v in x.items()}
            return x.expand(len(records), *x.shape[1:]).clone()
        kw = {k: expand(v) for k, v in plan.items()}
        mix = LearnedMix(kw["driver_params"], inp.observed_hist_mask, 140014, [key] * len(records))
        from wms.model.learned_admission import LearnedAdmission
        if self.learned_admission:
            # WMFIX: entrant CAVs draw from the native row's observed t0 CAV settings.
            cav_rows = (inp.initial["is_cav"][0].bool() & inp.initial["present"][0].bool()).cpu()
            cav_pool = torch.stack([kw["driver_params"][n][0].cpu()[cav_rows] for n in mix.names], -1)
            mix.cav_pools = [cav_pool.clone() for _ in records]
            inp.entries.cfg = replace(inp.entries.cfg, learned_admission=LearnedAdmission(mix))
        entries = TrackedEntries(inp.entries, track_admission=self.smooth_events)
        with scoring_rng(140014, self.device):
            roll = self.model.rollout(inp.initial, n_steps=steps, entries=entries,
                cav_teacher=None if inp.teacher is None else inp.teacher[:, :steps + 1],
                present_seq=None, is_cav_seq=None, history=inp.hist, history_mask=inp.hist_mask,
                controller_id=inp.controller_id, action_timeline=inp.action_timeline,
                cav_controller=inp.cav_controller, driver_history=inp.observed_hist,
                driver_history_mask=inp.observed_hist_mask, entrant_driver_sampler=mix,
                lane_uniform=inp.paired_random.uniform("lane", range(steps)), **kw)
        present, pos = torch.stack(entries.present, 1), torch.stack(entries.pos, 1)
        road_tick = .5 * torch.where(present, (1 - roll["speeds"] / 31.29).clamp_min(0), 0.)
        road = road_tick.sum((1, 2))
        upstream = (road_tick * (pos <= 75.44)).sum((1, 2))
        # HARNESS_CONTRACT: labels sum the left boundary q_0..q_{n-1} of the
        # post-service queue samples (tools/wmrequal/labels.py recorded); the
        # tracked series is seeded with the pre-rollout Q0 and cut at ``steps``.
        source = .5 * torch.as_tensor(np.stack(entries.queues, 1)[:, :steps],
                                      device=self.device).sum((1, 2))
        # Pre-insertion samples establish fixed identity lifetimes, including reused slots.
        birth = torch.stack(entries.newborn, 1)
        crossing_pos = torch.cat((pos[:, :1], roll["positions"]), 1)
        crossing_mask = torch.cat((present[:, :1], roll["masks"]), 1)
        births = torch.cat((torch.zeros_like(birth[:, :1]), birth), 1)
        lanes = torch.cat((inp.initial["lane"][:, None], roll["lanes"]), 1)
        count = smooth_crossings(crossing_pos, crossing_mask, births, self.doc["crossing_width"], lanes=lanes)
        gradient_source = source
        receipt = dict(pair_key=key, arms=[r["arm"] for r in order],
            queue_conservation=True, source_gradient="zero off hard service events")
        if self.smooth_events:
            # S-WMFIX-C fallback 1 (TRAINING ONLY): replace the exact integer
            # source in the DIFFERENTIABLE stack by the smoothed admission
            # surrogate; the exact ledger value stays in the receipt.  Inference
            # and verdicts never set this flag (exact identity when OFF).
            min_gap = admission_threshold_series(entries.admission_thresholds,
                                                 len(order), steps, device=self.device)
            pre = [torch.cat((inp.initial[name][:, None], roll[series]), 1)[:, :steps]
                   for name, series in (("s_m", "positions"), ("v_mps", "speeds"),
                                         ("present", "masks"), ("lane", "lanes"))]
            entrant_pre = torch.cat((torch.zeros_like(birth[:, :1]),
                                     birth.cumsum(1)[:, :-1] > 0), 1)
            pre_length = torch.where(entrant_pre,
                                     pre[0].new_full(pre[0].shape, NOMINAL_LENGTH_M),
                                     inp.initial["length_m"][:, None, :].expand_as(pre[0])
                                     .to(pre[0].dtype))
            gap_end = entrance_gap_series(pre[0], pre[1], pre[2], pre[3], pre_length,
                                          dt_s=inp.entries.dt)
            due = torch.as_tensor(np.stack(due_rows, 0), dtype=torch.float32,
                                  device=self.device)
            queue0 = torch.as_tensor([r["queue0"] for r in order], dtype=torch.float32,
                                     device=self.device)
            gradient_source, _ = smooth_admission_source(
                queue0, due, gap_end, min_gap, width=self.admission_width_m,
                dt_s=inp.entries.dt)
            receipt.update(smooth_events=True, admission_width_m=self.admission_width_m,
                           exact_source=source.detach().cpu().tolist(),
                           surrogate_source=gradient_source.detach().cpu().tolist(),
                           source_gradient="smoothed admission surrogate (training only)")
        predicted = torch.stack((road + gradient_source, upstream, gradient_source,
                                 count, road), 1)
        recorded = torch.tensor([[r["values"][str(horizon)][c] for c in COMPONENTS] for r in order],
                                device=self.device, dtype=torch.float32)
        return predicted, recorded, [0] * len(order), receipt


def admission_threshold_series(steps_rows, rows, steps, device="cpu"):
    """[R, T, 2] hard-path tested admission min-gaps for the smoothed surrogate.

    ``steps_rows`` is :class:`TrackedEntries.admission_thresholds` (per-step
    ``(row, entrance) -> inferred min_gap`` dicts, last tested attempt).  Steps
    without an attempt carry the last recorded value forward; before any test
    the incumbent textbook default (SUMO_V2_MIN_GAP_M) is used.  Detached
    constants only -- the surrogate's gradient flows through the gap, never
    through the thresholds.
    """
    import numpy as np
    import torch
    out = np.full((int(rows), int(steps), 2), float(SUMO_V2_MIN_GAP_M[0]), np.float32)
    for row in range(int(rows)):
        for entrance in range(2):
            last = float(SUMO_V2_MIN_GAP_M[0])
            for t in range(int(steps)):
                if t < len(steps_rows or ()):
                    last = float(steps_rows[t].get((row, entrance), last))
                out[row, t, entrance] = last
    return torch.as_tensor(out, dtype=torch.float32, device=device)


class TrackedEntries:
    def __init__(self, process, track_admission=False):
        self.process = process
        self.present, self.pos, self.newborn = [], [], []
        # HARNESS_CONTRACT: sample the pre-rollout Q0 first so the rectangle is
        # the label's left boundary q_0..q_{n-1}, not q_1..q_n (one-tick shift).
        self.queues = [process.pending.copy()]
        # S-WMFIX-C fallback 1: collect the hard-path tested admission min-gap
        # per (row, entrance, step) for the smoothed surrogate.  OFF (default)
        # touches nothing on the process and performs no new work.
        self.admission_thresholds = None
        if track_admission:
            process.track_admission = True
            process.last_admission_min_gaps = {}
            self.admission_thresholds = []

    def insert(self, state):
        # Same after-service rectangle convention as corrected validation scorer.
        new = self.process.insert(state)
        self.present.append(state["present"].clone())
        self.pos.append(state["s_m"].clone())
        self.queues.append(self.process.pending.copy())
        self.newborn.append(new.clone())
        if self.admission_thresholds is not None:
            self.admission_thresholds.append(
                dict(getattr(self.process, "last_admission_min_gaps", {})))
        return new


class RecordedDemand:
    def __init__(self, counts):
        import numpy as np
        self.counts = np.asarray(counts)
        if self.counts.ndim != 3 or self.counts.shape[-1] != 2 or (self.counts < 0).any() \
                or not np.array_equal(self.counts, np.rint(self.counts)):
            raise ValueError("invalid recorded per-entrance requests")
        self.ticks = [0] * len(self.counts)

    def __call__(self, b):
        tick = self.ticks[b]
        if tick >= self.counts.shape[1]:
            raise ValueError("recorded TRAIN demand exhausted")
        self.ticks[b] += 1
        return [e for e in range(2) for _ in range(int(self.counts[b, tick, e]))]
