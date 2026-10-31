"""Shared differentiable, native-lane TRAIN rollout; no simulator invocation."""
from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

from tools.wmfixmerge.data import COMPONENTS, WM2_SHA, digest, safe, verify

_WMREQUAL = Path(__file__).resolve().parents[1] / "wmrequal"
if str(_WMREQUAL) not in sys.path:
    # grouped_feedback imports sibling modules absolutely (labels, ...);
    # every entry point (feasibility, train) must survive without a manual shim.
    sys.path.insert(0, str(_WMREQUAL))


class Runtime:
    def __init__(self, prepared, config, checkpoint, controllers, device="cpu",
                 arm_names=None):
        import torch
        from tools.wmcombined.e3_score import load_for_scoring
        from wms.data.c4_pairs import PairedBranchSource
        self.doc = verify(prepared)
        if digest(checkpoint) != WM2_SHA:
            raise ValueError("base is not registered WM2")
        self.device = torch.device(device)
        self.learned_admission = True
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
        self.model.enable_merge_response()
        self.model.checkpoint_forward = True
        manifest = json.loads(safe(safe(self.doc["root"]) / "C4_EPISODES_MANIFEST.json").read_text())
        worlds = {row["world"] for row in self.doc["episodes"]}
        names = tuple(sorted({e["arm"] for e in self.doc["episodes"]}))
        if arm_names is not None:
            # S-WMFIX-ES: restrict a multi-arm corpus to the registered paired
            # contrast (WMDATA.md: native + ctrl0); designed-controller arms
            # (circles/pi) stay outside this TRAIN rollout source entirely.
            missing = set(arm_names) - set(names)
            if missing:
                raise ValueError("arm restriction outside the corpus: " + repr(sorted(missing)))
            names = tuple(a for a in names if a in set(arm_names))
        self.source = PairedBranchSource(self.doc["root"], horizon_steps=600,
            history_s=60., history_stride_s=1., entries_v2=True, junction_fix="ffill",
            command_known=True, arm_kinds=("B", "POLICY"), role="train",
            role_of={e["world"]: "train" if e["world"] in worlds else "val"
                     for e in manifest["episodes"]}, name="wmfix_registered_fit",
            arm_names=names)
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
        from wms.train.merge_response import smooth_crossings
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
        inp.entries._emit_due = RecordedDemand([r["due_increments"][:steps] for r in order])
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
        entries = TrackedEntries(inp.entries)
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
        predicted = torch.stack((road + source, upstream, source, count, road), 1)
        recorded = torch.tensor([[r["values"][str(horizon)][c] for c in COMPONENTS] for r in order],
                                device=self.device, dtype=torch.float32)
        return predicted, recorded, [0] * len(order), dict(pair_key=key, arms=[r["arm"] for r in order],
            queue_conservation=True, source_gradient="zero off hard service events")


class TrackedEntries:
    def __init__(self, process):
        self.process = process
        self.present, self.pos, self.newborn = [], [], []
        # HARNESS_CONTRACT: sample the pre-rollout Q0 first so the rectangle is
        # the label's left boundary q_0..q_{n-1}, not q_1..q_n (one-tick shift).
        self.queues = [process.pending.copy()]

    def insert(self, state):
        # Same after-service rectangle convention as corrected validation scorer.
        new = self.process.insert(state)
        self.present.append(state["present"].clone())
        self.pos.append(state["s_m"].clone())
        self.queues.append(self.process.pending.copy())
        self.newborn.append(new.clone())
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
