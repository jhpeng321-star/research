"""Pure, fail-closed cycle-2 boundary and resource decisions."""
import copy

POLICY_SHA = '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
BUDGET_STOP = 'registered SUMO budget cannot fund another collection'


def budget_need(charged=176473, collection=36933, cap=210000):
    return dict(charged=charged, cap=cap, remaining=cap-charged,
                collection_need=collection, projected_total=charged+collection,
                additional_cap_needed=max(0, charged+collection-cap))


def policy_boundary(state):
    """Authorize only policy after an ACCEPT; preserve the exact budget stop history."""
    s = copy.deepcopy(state)
    assert s['sim_s_charged'] == 176473
    assert s['policy']['sha256'] == POLICY_SHA
    assert len(s['cycles']) == 2 and s['cycles'][-1]['cycle'] == 1
    phases = s['cycles'][1]['phases']
    assert all(p['status'] == 'done' for p in phases.values())
    assert phases['check']['receipt']['decision'] == 'ACCEPT'
    assert s['accepted_wm']['sha256'] == phases['wm_ft']['receipt']['candidate_sha256']
    assert s.get('stopped') == BUDGET_STOP
    s['c2prep_prior_budget_stop'] = s['stopped']
    s['stopped'] = None
    s['cycles'].append({'cycle': 2, 'phases': {
        n: {'status': 'pending', 'attempt': 0} for n in ('policy', 'collect', 'convert', 'wm_ft', 'check')}})
    return s


def resource_fits(smoke, updates, remaining_seconds):
    """Compare measured resource use only against the registered cap and time budget."""
    if smoke.get('status') != 'PASS' and not smoke.get('resource_failure'):
        raise ValueError('non-resource smoke failure; no fallback: '+str(smoke.get('error')))
    return (smoke.get('status') == 'PASS' and len(smoke.get('updates', [])) == 3
            and smoke.get('horizon_s') in (10, 30, 60)
            and smoke['observed_training_horizons'] == [smoke['horizon_s']]
            and smoke['peak_reserved_bytes'] <= smoke['allocator_cap_bytes']
            and smoke['wall_seconds'] + max(r['seconds'] for r in smoke['updates']) * updates
            <= remaining_seconds)
