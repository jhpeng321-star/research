"""Explicit, fail-closed configuration migration for a new training phase."""
from dataclasses import asdict


OPTIMIZER_RATES = {'actor_optimizer': 'actor_lr', 'critic_optimizer': 'critic_lr',
                   'alpha_optimizer': 'alpha_lr', 'lane_alpha_optimizer': 'alpha_lr'}


def resume_diff(config, reward, payload):
    # Fill historical absent fields with their actual dataclass defaults.
    prior_config = type(config)(**payload['config'])
    prior_reward = type(reward)(**payload['reward_weights'])
    diff = {}
    for section, requested, prior in (('trainer', config, prior_config),
                                       ('reward', reward, prior_reward)):
        for name, value in asdict(requested).items():
            old = getattr(prior, name)
            if value != old:
                diff[f'{section}.{name}'] = {'checkpoint': old, 'requested': value}
    for optimizer, field in OPTIMIZER_RATES.items():
        rates = [g['lr'] for g in payload.get(optimizer, {}).get('param_groups', [])]
        if any(rate != getattr(config, field) for rate in rates):
            diff[f'optimizer.{optimizer}.lr'] = {
                'checkpoint': rates, 'requested': getattr(config, field)}
    return diff


def authorize_new_phase(diff, allow):
    allowed = set(allow)
    # No wildcards or blanket approval: each actual difference must be named.
    unknown = allowed - set(diff)
    refused = set(diff) - allowed
    if unknown or refused:
        raise ValueError(f'new-phase resume differences require exact --resume-allow keys; '
                         f'unapproved={sorted(refused)}, unused={sorted(unknown)}; diff={diff}')


def restore_requested_rates(trainer):
    for name, field in OPTIMIZER_RATES.items():
        optimizer = getattr(trainer, name, None)
        if optimizer is not None:
            for group in optimizer.param_groups:
                group['lr'] = getattr(trainer.config, field)
