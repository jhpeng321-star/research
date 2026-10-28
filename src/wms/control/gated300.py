"""CCT-1 bounded, critic-free paired credit. Every entry point is default OFF."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import torch

from .antiexploit import K, QUANTILE, checked_path, tail_credit
from .contract import PHYSICS_DT_S, STEPS_PER_CONTROL
from .counterfactual import baseline_index

SIGMA = {90: 25.38, 120: 41.22, 180: 42.11, 300: 43.91}
STEP_S = PHYSICS_DT_S * STEPS_PER_CONTROL


@dataclass
class LongCreditConfig:
    enabled: bool = False
    cct1_file: str = ''  # May contain {wm_sha256}, for accepted-WM swaps.
    w_long: float = 1.0  # Fixed prospectively; never tuned on observed outcomes.
    score_training_horizon: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.score_training_horizon) is not bool:
            raise ValueError('long_credit switches must be boolean')
        if not isinstance(self.cct1_file, str):
            raise ValueError('long_credit.cct1_file must be a path string')
        if isinstance(self.w_long, bool) or not math.isfinite(self.w_long) or self.w_long < 0:
            raise ValueError('long_credit.w_long must be finite and nonnegative')


def install(trainer, section, wm_sha256s):
    """Bind checkpoint bytes, not a friendly WM name or inherited 90s authority."""
    cfg = LongCreditConfig(**section)
    receipt = dict(config=asdict(cfg), wm_sha256s=list(wm_sha256s),
                   world_model_ids=list(trainer.world_model_ids), H_long=0, reason='disabled')
    if cfg.enabled and cfg.cct1_file:
        if len(wm_sha256s) != 1:
            raise ValueError('CCT-1 requires a single WM-specific authority')
        path = checked_path(cfg.cct1_file.format(wm_sha256=wm_sha256s[0]))
        if path.is_file():
            raw = path.read_bytes()
            doc = json.loads(raw)
            if (doc.get('schema') != 'gated300_cct1/1'
                    or doc.get('wm_sha256') != wm_sha256s[0]
                    or doc.get('panel') != 'pairs300' or doc.get('policy_authorized') is not True):
                raise ValueError('long credit requires this WM\'s pairs300 CCT-1 verdict authority')
            hc = doc.get('H_C')
            if type(hc) is not int or hc not in (0, *SIGMA):
                raise ValueError('invalid CCT-1 H_C')
            contiguous = 0
            for horizon, sigma in SIGMA.items():
                row = doc.get('horizons', {}).get(str(horizon), {})
                sign, coverage, wrong = (row.get(k) for k in
                                         ('sign_agreement', 'coverage', 'wrong_mean_abs_real'))
                passed = (row.get('sigma') == sigma and row.get('threshold') == 2 * sigma
                          and all(type(x) in (int, float) and math.isfinite(x)
                                  for x in (sign, coverage, wrong))
                          and .8 <= sign <= 1 and .25 <= coverage <= 1 and 0 <= wrong <= 2 * sigma
                          and row.get('verdict') == 'PASS')
                if not passed:
                    break
                contiguous = horizon
            if hc != contiguous:
                raise ValueError('CCT-1 H_C disagrees with registered contiguous passing prefix')
            if hc > 90:
                if (doc.get('entrant_control') is not trainer.config.entrant_control
                        or doc.get('deterministic_crn') is False
                        or not trainer.config.deterministic_crn):
                    raise ValueError('CCT-1 entrant/CRN inference mismatch')
                if trainer.config.future_samples != K or not trainer.config.pessimistic_credit:
                    raise ValueError('long credit requires existing q10/K8 machinery')
                receipt['H_long'] = min(300, hc)
            receipt.update(H_C=hc, artifact_sha256=hashlib.sha256(raw).hexdigest(),
                           reason='active' if hc > 90 else 'H_C <= 90')
        else:
            receipt['reason'] = 'missing CCT-1 file'
    elif cfg.enabled:
        receipt['reason'] = 'missing CCT-1 file'
    trainer.gated300 = receipt
    return receipt


def active(trainer):
    receipt = getattr(trainer, 'gated300', {})
    on = receipt.get('H_long', 0) > 90
    if on and receipt['world_model_ids'] != trainer.world_model_ids:
        raise ValueError('long credit WM changed without rebinding CCT-1')
    return on


def requested_rollout(trainer):
    return active(trainer) and (trainer._training_imagination
                               or getattr(trainer, '_gated300_scoring', False))


def rollout_config(trainer, cfg):
    if not requested_rollout(trainer):
        return cfg
    if cfg.horizon * STEP_S > 90:
        raise ValueError('gated300 dense horizon must be <= trusted 90s')
    return replace(cfg, horizon=int(trainer.gated300['H_long'] / STEP_S),
                   future_samples=K, uncertainty_weight=0.)


def gate(improvement, sigma, eligible=None):
    """K x ... paired improvements = -dJ. Gate decisions never carry a gradient.

    Improvement needs q10>0; harm needs q90<0. A negative coefficient does not
    reverse this test: it is applied after the paired treatment/native sign test.
    """
    if improvement.shape[0] != K:
        raise ValueError('confidence gate requires K=8 paired samples')
    samples = improvement.detach()
    median = torch.quantile(samples, .5, dim=0)
    q10 = torch.quantile(samples, QUANTILE, dim=0)
    q90 = torch.quantile(samples, 1 - QUANTILE, dim=0)
    agreement = ((median > 0) & (q10 > 0)) | ((median < 0) & (q90 < 0))
    on = (median.abs() > 2 * sigma) & agreement & torch.isfinite(samples).all(0)
    if eligible is not None:
        on = on & eligible
    # where (rather than multiply) also zeros nonfinite OFF samples.
    return torch.where(on.unsqueeze(0), improvement, 0.), on, agreement, median


def prefix(trainer, imagined, dense_steps):
    """One H_long trajectory per future: extract its existing dense prefix once.

    Long credit is accrued t0..H_long (not a value bootstrap, nor H_long-90).
    Dense credit retains the existing mean of finite-horizon suffix returns.
    """
    if not requested_rollout(trainer):
        return imagined
    rewards = imagined['rewards']
    variants = imagined['variants']
    if variants is None or set(variants.base_row) != set(range(imagined['n_base'])):
        raise ValueError('long credit needs a paired native row for every start')
    base = baseline_index(variants, rewards.shape[-1], device=rewards.device)
    totals = rewards.sum(1, keepdim=True)
    long_difference = totals - totals[..., base].detach()
    out = dict(imagined)
    components = imagined['components']
    long_cost = components['delay'] + components.get('source_wait', 0.)
    out['gated300_long_J'] = long_cost.detach().sum(1).mean(0)
    for name in ('rewards', 'costs', 'entropy', 'actions', 'action_mask', 'disagreement',
                 'entrance_zone', 'values', 'returns'):
        value = out.get(name)
        if torch.is_tensor(value):
            axis = 0 if name == 'disagreement' else 1
            stop = dense_steps + (name == 'values')
            out[name] = value.narrow(axis, 0, stop)
    for name in ('components', 'lanes', 'action_audit', 'antiexploit_commands'):
        if isinstance(out.get(name), dict):
            out[name] = {k: v[:, :dense_steps] for k, v in out[name].items()}
    out['states'] = [states[:dense_steps + 1] for states in imagined['states']]
    out['returns'] = out['rewards'].flip((1,)).cumsum(1).flip((1,))
    out.update(gated300_long_difference=long_difference,
               gated300_horizon_s=trainer.gated300['H_long'],
               imagination_horizon_s=dense_steps * STEP_S)
    return out


def actor_term(trainer, imagined, coefficients, dense_difference, scale):
    samples = imagined['gated300_long_difference']
    eligible = coefficients != 0
    gated, on, agreement, median = gate(samples, SIGMA[imagined['gated300_horizon_s']], eligible)
    # Reuse the signed JOINT q10 selector, including covariance/negative team rows.
    selected, _ = tail_credit(imagined, gated, coefficients, enabled=True)
    weight = trainer.gated300['config']['w_long']
    long_value = weight * (selected.mean((0, 1)) * coefficients).sum()
    dense_value = (dense_difference.mean((0, 1)) * coefficients).sum().detach()
    dense_mass = (dense_difference.detach().mean((0, 1)) * coefficients).abs().sum()
    long_mass = weight * (selected.detach().mean((0, 1)) * coefficients).abs().sum()
    mask = eligible.expand_as(on)
    denom = mask.sum().clamp_min(1)
    metrics = dict(gated300_horizon_s=imagined['gated300_horizon_s'],
        gated300_rows_on_share=float(on.sum() / denom),
        gated300_mean_abs_dJ_long=float(samples.detach().abs()[..., eligible].mean())
            if bool(mask.any()) else 0.,
        gated300_mean_abs_median_dJ_long=float(median.abs()[mask].mean()) if bool(mask.any()) else 0.,
        gated300_median_pessimistic_sign_agreement=float((agreement & mask).sum() / denom),
        gated300_median_vs_q10_sign_agreement=float(
            (((median.sign() == torch.quantile(samples.detach(), QUANTILE, dim=0).sign())
              & mask).sum()) / denom),
        gated300_long_credit_share=float(long_mass / (dense_mass + long_mass).clamp_min(1e-30)),
        gated300_dense_credit=float(dense_value), gated300_long_credit=float(long_value.detach()),
        gated300_w_long=weight)
    return -long_value / scale, metrics


@contextmanager
def scoring(trainer):
    """Same observed-share curriculum resolution, deterministic policy; no RNG use."""
    enabled = getattr(trainer, 'gated300', {}).get('config', {}).get('score_training_horizon', False)
    old = trainer._training_imagination, getattr(trainer, '_gated300_scoring', False)
    try:
        if enabled:
            trainer._training_imagination = True
            trainer._gated300_scoring = True
        yield enabled
    finally:
        trainer._training_imagination, trainer._gated300_scoring = old


def checkpoint_restore(trainer, payload):
    saved = payload.get('gated300')
    if saved is not None and not getattr(trainer, '_gated300_explicit', False):
        if saved['world_model_ids'] != trainer.world_model_ids:
            raise ValueError('refreshed WM needs an explicit new long_credit binding')
        receipt = install(trainer, saved['config'], saved['wm_sha256s'])
        if receipt != saved:
            raise ValueError('CCT-1 binding changed on resume')
