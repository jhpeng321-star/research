"""Training audit startup/selection diagnostics; no training objective changes."""
import sys


def report_phase_selection(chosen, phase_start, *, resumed):
    """Keep the registered selector, but explicitly identify a retained resume baseline."""
    baseline = resumed and chosen <= phase_start
    row = dict(selected_update=chosen, phase_start_update=phase_start,
               selected_phase_updates=chosen - phase_start,
               resumed_baseline_selected=baseline, selected_new_phase_update=chosen > phase_start,
               source='resumed_phase_start' if baseline else 'new_phase_update')
    if baseline:
        print(f'WARNING: selection retained the resumed baseline at update {chosen}; '
              'no newly trained phase checkpoint was selected', file=sys.stderr, flush=True)
    return row


def warn_unused_settings(raw, *, native_init=False, boundary=None):
    """Report configured values bypassed by this CLI's actual mode, before setup."""
    rows = []

    def add(section, keys, reason):
        keys = sorted(set(keys) & set(raw.get(section, {})))
        if keys:
            row = dict(section=section, keys=keys, reason=reason)
            rows.append(row)
            print(f"WARNING: inactive config [{section}] {', '.join(keys)}: {reason}",
                  file=sys.stderr, flush=True)

    flags = (boundary or {}).get('flags', {})
    long_credit = flags.get('long_credit', False)
    if flags.get('auto_horizon'):
        add('horizon', ('horizons_s', 'curriculum_updates', 'entry_pad_max', 'gate_file'),
            'boundary auto_horizon rebuilds stages/pads/gate; TOML last-knot spacing only is reused')
    if 'uncertainty' in raw and not raw['uncertainty'].get('enabled', False):
        add('uncertainty', set(raw['uncertainty']) - {'enabled'},
            'uncertainty.enabled=false; trainer future_samples is a separate setting')
    if native_init:
        add('lanes', raw.get('lanes', {}), 'native-init uses WM native lane rules')
        add('policy', ('lane_control', 'init_zero_accel'), 'native-init constructs the native residual actor')
        add('policy14', ('policy_ramp_merge', 'sampled_marginal_lane_weight',
                         'entropy_floor', 'entropy_decay_on_improvement'),
            'native-init forces native lanes and deterministic residuals (no lane credit/entropy)')
    if long_credit:
        add('policy14', ('critic_warmup_updates', 'critic_value_scale'),
            'long credit disables critic warmup and value updates')
        add('trainer', ('real_return_mix',), 'critic step is disabled under long credit')
        if raw.get('paircrit', {}).get('enabled'):
            add('paircrit', raw['paircrit'],
                'setup may still load artifacts, but its value critic supplies no long-credit update')
    if raw.get('data', {}).get('focus_cell_fraction') is not None:
        add('counterfactual', ('start_tilt',), 'exact focus-cell sampling replaces rejection tilt')
    add('counterfactual', ('validation_uniform',), 'policy14 CLI does not call trainer.validate')
    add('run', ('seed', 'updates', 'validation_batches'),
        'CLI --seed/--max-updates take precedence; no trainer.validate call')
    return rows


def require_real_credit_effect(trainer):
    """Reject an explicitly supplied multiplier file erased by guards at every knot."""
    from .policy14 import guarded_real_credit_weights
    if trainer.credit_multipliers is None:
        return None
    rows = []
    for position in trainer.counterfactual.curriculum_updates:
        weights = trainer.counterfactual.weights(position)
        effective, receipt = guarded_real_credit_weights(
            weights, trainer.credit_multipliers,
            individual_floor=trainer.config.real_credit_individual_floor)
        rows.append({'position': position, **receipt})
    if (not trainer.counterfactual.enabled or
            all(row['registered'] == row['effective'] for row in rows)):
        raise ValueError('--real-credit has no effective weight change after guards at any '
                         'curriculum knot; remove --real-credit for unchanged registered credit')
    return rows
