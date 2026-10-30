"""Small POLICY14 adapters over the existing per-CAV Dreamer implementation."""
from __future__ import annotations

import torch

from .bridge import make_cav_controller
from .dreamer import DreamerTrainer
from .start_states import StartBatch


def controller_factory(*args, **kwargs):
    return make_cav_controller(*args, **kwargs, symmetric_exec_cap=True, mask_forced_merge=True)


def make_controller_factory(policy_ramp_merge: bool = False, native_ramp_merge: str = "gap6",
                            native_lane_change: bool = False,
                            policy_strategic_merge: bool = False):
    """LOOP14C ramp-merge switches (``[policy14]`` keys of the same names).  Both off returns the
    unchanged :func:`controller_factory` object (bit-identical legacy path)."""
    if not isinstance(policy_ramp_merge, bool):
        raise ValueError("policy14.policy_ramp_merge must be a boolean")
    if native_ramp_merge not in ("gap6", "wm_law"):
        raise ValueError("policy14.native_ramp_merge must be 'gap6' or 'wm_law'")
    if not isinstance(native_lane_change, bool):
        raise ValueError("policy14.native_lane_change must be a boolean")
    if not isinstance(policy_strategic_merge, bool):
        raise ValueError("policy14.policy_strategic_merge must be a boolean")
    if policy_strategic_merge and (not native_lane_change or native_ramp_merge != "wm_law"):
        raise ValueError("policy_strategic_merge requires native_lane_change and wm_law")
    if (not policy_ramp_merge and native_ramp_merge == "gap6"
            and not native_lane_change and not policy_strategic_merge):
        return controller_factory

    def factory(*args, **kwargs):
        return make_cav_controller(*args, **kwargs, symmetric_exec_cap=True,
                                   mask_forced_merge=True, policy_ramp_merge=policy_ramp_merge,
                                   native_ramp_merge=native_ramp_merge,
                                   native_lane_change=native_lane_change,
                                   policy_strategic_merge=policy_strategic_merge)
    factory.ramp_merge = {"policy_ramp_merge": policy_ramp_merge,
                           "native_ramp_merge": native_ramp_merge}
    if native_lane_change:
        factory.ramp_merge["native_lane_change"] = True
    if policy_strategic_merge:
        factory.ramp_merge["policy_strategic_merge"] = True
    return factory


def guarded_real_credit_weights(weights, multipliers, *, individual_floor=False):
    """Measured multipliers plus the registered individual-baseline guard.

    The optional floor is the unscaled imagination individual weight, never a
    new numerical threshold. Coop is only reduced; group/team are untouched.
    """
    raw = {k: v * float(multipliers.get(k, 1.0)) for k, v in weights.items()}
    effective = dict(raw)
    if individual_floor:
        effective['individual'] = max(effective.get('individual', 0.),
                                      weights.get('individual', 0.))
    if effective.get('coop', 0.) > effective.get('individual', 0.):
        effective['coop'] = effective.get('individual', 0.)
    return effective, {'registered': dict(weights), 'multipliers': dict(multipliers),
                       'raw': raw, 'effective': effective,
                       'individual_floor': individual_floor}


class Policy14Trainer(DreamerTrainer):
    """Team pathwise acceleration credit plus sampled per-CAV categorical credit.

    A detached subtraction is NOT advertised as new pathwise credit. Instead the
    sampled marginal advantage weights the removed CAV's own lane score-function
    term. Other sampled action noise, arrivals and driver draws stay paired.
    Synergy is diagnostic-only, and marginal loss is averaged, never CAV-count-scaled.
    """

    marginal_lane_weight = 0.25
    #: S-SUMOCREDIT: ``{individual, group, coop, team}`` multipliers in [0, 1] from real SUMO credit
    #: (``wms.control.real_credit14``); None = off (the weights are passed through untouched).
    credit_multipliers = None

    def actor_objective(self, imagined, scale, weights, *, gradient_components=None):
        capped = None
        credit_receipt = None
        if self.credit_multipliers:
            weights, credit_receipt = guarded_real_credit_weights(weights, self.credit_multipliers,
                individual_floor=getattr(self.config, 'real_credit_individual_floor', False))
            if credit_receipt['raw'].get('coop', 0.) != weights.get('coop', 0.):
                capped = {'coop_before': credit_receipt['raw']['coop'], 'coop_after': weights['coop']}
        loss, extra = super().actor_objective(imagined, scale, weights,
                                              gradient_components=gradient_components)
        if capped is not None:
            extra["credit_coop_capped_by_individual"] = capped
        if credit_receipt is not None:
            extra['real_credit_weights'] = credit_receipt
        variants, lanes = imagined.get("variants"), imagined.get("lanes")
        terms, marginal_values = [], []
        if (variants is not None and lanes is not None and self.marginal_lane_weight
                and not getattr(self, 'long_credit_requested', False)):
            for group in variants.groups:
                row = group["row"]
                for slot, baseline in group["flips"].items():
                    advantage = (imagined["returns"][..., row]
                                 - imagined["returns"][..., baseline]).detach() / scale
                    active = lanes["lane_active"][..., row, slot]
                    terms.append(-(lanes["lane_log_prob"][..., row, slot] * advantage
                                   * active).mean())
                    marginal_values.append(float(advantage.mean()))
        if terms:
            term = self.marginal_lane_weight * torch.stack(terms).mean()
            loss = loss + term
            if gradient_components is not None:
                gradient_components["sampled_marginal_lane"] = term
                gradient_components['_dense_objective'] = gradient_components['_dense_objective'] + term
        extra.update(sampled_marginals=len(terms), marginal_advantage=(
            sum(marginal_values) / len(marginal_values) if marginal_values else None))
        return loss, extra


class SmokeSampler:
    """Synthetic integration starts, not real traffic or scientific training data."""

    def __init__(self, model):
        self.model = model
        self.draws = 0

    def sample(self, batch_size, device="cpu"):
        self.draws += 1
        def tensor(row, dtype=torch.float32):
            return torch.tensor(row, dtype=dtype, device=device).expand(batch_size, -1).clone()
        state = {"s_m": tensor([25., 65., 0., -80., -60., 0.]),
                 "v_mps": tensor([4., 7., 6., 8., 8., 0.]),
                 "lane": tensor([2, 1, 1, 0, 1, -1], torch.long),
                 "a_mps2": tensor([-.4, 0., 0., 0., 0., 0.]),
                 "length_m": tensor([5.] * 6),
                 "present": tensor([1, 1, 1, 1, 1, 0], torch.bool),
                 "is_cav": tensor([1, 0, 0, 1, 1, 0], torch.bool),
                 "entry_backlog": tensor([20., 10.])}
        from wms.model.entries import EntranceHistory
        history = EntranceHistory.defaults()
        feat = torch.tensor(history.to_features(), dtype=torch.float32, device=device)
        from .policy14_probes import missing_history
        return StartBatch(state, rollout_kwargs=missing_history(self.model, state),
                          metadata=[{"split": "TRAIN", "synthetic": True}] * batch_size,
                          entry_feat=feat.expand(batch_size, -1).clone(),
                          entry_queue=tensor([20., 10., .12, .04, .2, .2]))

    def state_dict(self):
        return {"draws": self.draws}

    def load_state_dict(self, state):
        self.draws = state["draws"]


def independent_score(trainer, batch, seed=965149, *, native_reference=False):
    """Rescore accrued road+source J; optional matched-horizon CCT-1 paired score.

    No critic/entropy/constraints. With long credit, J_dense and J_long retain
    physical costs and J is the explicitly labelled dense + gated-long score.
    """
    from .counterfactual import build_variants
    from .dreamer import _restore_rng, _rng_state
    from .gated300 import scoring, gate, SIGMA
    rng = _rng_state(trainer.device)
    try:
        with torch.no_grad(), scoring(trainer) as matched_horizon:
            torch.manual_seed(seed)
            batch = trainer._entry_slots(batch, torch.Generator().manual_seed(seed))
            variants = build_variants(batch.state, trainer.counterfactual,
                                      torch.Generator().manual_seed(seed))
            imagined = trainer.imagine(batch, deterministic=True, variants=variants,
                                       future_seed=seed)
            variants = imagined['variants']
            n = imagined["n_base"]
            components = imagined["components"]
            road = components["delay"].sum(1).mean(0)
            source = components["source_wait"].sum(1).mean(0)
            if native_reference or matched_horizon:
                # Identical reduction geometry per row, independent of batch alignment.
                def totals(values):
                    return torch.stack([values[..., r].contiguous().sum(1).mean(0)
                                        for r in range(values.shape[-1])])
                road, source = totals(components["delay"]), totals(components["source_wait"])
            j = road + source
            gains = [float(j[base] - j[start]) for start, base in variants.base_row.items()]
            hdv = components["hdv_hard_brake_seconds"].sum(1).mean(0)
            result = {"J": float(j[:n].mean()), "road_delay": float(road[:n].mean()),
                    "source_wait": float(source[:n].mean()),
                    "native_relative_gain": sum(gains) / len(gains) if gains else None,
                    "hdv_hard_brake_seconds": float(hdv[:n].mean()),
                    "critic_tail_included": False}
            if native_reference:
                if set(variants.base_row) != set(range(n)):
                    raise ValueError("native acceptance requires a paired native row for every start")
                result["native_J"] = float(torch.stack([j[variants.base_row[i]] for i in range(n)]).mean())
            if matched_horizon:
                from .gated300 import reference_index, score_position
                from .counterfactual import KIND_CTRL0
                refs = reference_index(variants) if bool((variants.row_kind == KIND_CTRL0).any()) else None
                result.update(J_dense=result['J'], J_raw_dense=result['J'],
                    scoring_phase_position=score_position(trainer),
                    score_rule='raw accrued J at the phase-frozen training horizon')
                if refs is not None:
                    ctrl0_j = j[refs[:n]]
                    result.update(CTRL0_J_dense=float(ctrl0_j.mean()),
                        raw_gain_vs_CTRL0_dense=float((ctrl0_j - j[:n]).mean()),
                        raw_dJ_vs_CTRL0_dense=float((j[:n] - ctrl0_j).mean()))
                result['score_horizon_s'] = imagined['imagination_horizon_s']
                if 'gated300_long_difference' in imagined:
                    long = imagined['gated300_long_difference'][:, 0, :n]
                    gated, on, agreement, median = gate(long, SIGMA[imagined['gated300_horizon_s']],
                        gain_only=trainer.gated300['config'].get('mode') == 'gain_only',
                        sample_count=imagined.get('gated300_futures', 8))
                    gain = float(torch.quantile(gated, .1, dim=0).mean())
                    weighted = trainer.gated300['config']['w_long'] * gain
                    long_j = imagined['gated300_long_J']
                    result.update(J_dense=result['J'], long_horizon_s=imagined['gated300_horizon_s'],
                        J_long=float(long_j[:n].mean()),
                        native_J_long=float(torch.stack([long_j[variants.base_row[i]] for i in range(n)]).mean()),
                        long_gated_gain=gain, long_rows_on_share=float(on.float().mean()),
                        long_mean_abs_dJ=float(median.abs().mean()),
                        long_sign_agreement=float(agreement.float().mean()),
                        score_rule='J_dense - w_long * q10(gated paired improvement at H_long)')
                    result.update(J_raw_long=result['J_long'],
                        long_futures=imagined.get('gated300_futures', 8),
                        long_credit_reference=imagined.get('gated300_reference', 'learned_native'))
                    if refs is not None:
                        result.update(CTRL0_J_long=float(long_j[refs[:n]].mean()),
                            raw_gain_vs_CTRL0_long=float((long_j[refs[:n]] - long_j[:n]).mean()),
                            raw_dJ_vs_CTRL0_long=float((long_j[:n] - long_j[refs[:n]]).mean()))
                    result['J'] -= weighted
                    if result['native_relative_gain'] is not None:
                        result['native_relative_gain'] += weighted
                result['composite_gated_score'] = result['J']
            return result
    finally:
        _restore_rng(rng, trainer.device)
