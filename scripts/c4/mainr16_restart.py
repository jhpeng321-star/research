"""Pure state migration for the explicitly authorized round-16 cycle-zero restart."""
from __future__ import annotations

import copy


def restarted_state(state, *, registration, registration_sha256, base, initial_returns, phases):
    if state.get('stopped') or len(state['cycles']) != 1:
        raise ValueError('restart requires the single initial cycle and no scientific stop')
    old = state['cycles'][0]
    if old['cycle'] != 0 or old['phases']['policy']['status'] != 'done' or old['phases']['collect']['status'] != 'failed':
        raise ValueError('restart requires completed superseded policy and failed startup collection')
    if any(old['phases'][p]['status'] != 'pending' for p in ('convert', 'wm_ft', 'check')):
        raise ValueError('cannot supersede a cycle with downstream work')
    result = copy.deepcopy(state)
    result.setdefault('superseded_cycles', []).append(copy.deepcopy(old))
    result.setdefault('superseded_policies', []).append(copy.deepcopy(state['policy']))
    result['cycles'] = [{'cycle': 0, 'phases': {
        p: {'status': 'pending', 'attempt': old['phases'][p]['attempt']} for p in phases}}]
    result.update(registration=registration, registration_sha256=registration_sha256,
                  accepted_wm=copy.deepcopy(base), policy=None, real_returns=initial_returns)
    for key in ('policy_wm', 'real_credit', 'gate_table', 'gate_mean_table'):
        result.pop(key, None)
    result['mainstart_selected_base'] = {'sha256': base['sha256'], 'label': 'round16',
                                         'strict_ttc_ratio': 11.27}
    assert result['sim_s_charged'] == state['sim_s_charged']
    assert result['seeds_used'] == state['seeds_used']
    return result
