"""MAIN04:26 shared B/R bank rule; append-only revalidation of saved exact grades."""
import copy
import json
from pathlib import Path

RULE = 'MAIN2026-11-03 04:26'
DIRECTORY = 'confirmations_safety0426'


def verdict(deltas):
    old = all(v <= 0 for v in deltas.values())
    combined = deltas['hard_brake'] + deltas['ttc_below_1p5_events']
    new = deltas['collisions'] <= 0 and deltas['teleports'] <= 0 and combined <= 0
    return dict(old_no_metric_increase=old, new_combined_rule=new,
                safety_critical_events_delta=combined)


def reevaluate(original):
    out = copy.deepcopy(original)
    out.pop('confirmation_sha256', None)
    out['safety_rule'] = RULE
    out['original_confirmation_sha256'] = original['confirmation_sha256']
    out['old_admitted'] = original['admitted']
    out['old_checks'] = copy.deepcopy(original['checks'])
    if 'safety_deltas' in original:
        windows = {k: verdict(d) for k, d in original['safety_deltas'].items()}
        out['safety_verdicts'] = windows
        out['checks']['safety'] = all(v['new_combined_rule'] for v in windows.values())
        out['admitted'] = all(out['checks'].values())
    return out


def legacy(root, arm):
    from .exact_grade import verify_sealed
    out = {}
    for p in (Path(root)/'replay_bank'/arm/'confirmations').glob('*.json'):
        value = json.loads(p.read_text());verify_sealed(value, 'confirmation_sha256')
        out[value['verification_id']] = value
    return out


def confirmations(root, arm):
    from .exact_grade import verify_sealed
    out = legacy(root, arm)
    for p in (Path(root)/'replay_bank'/arm/DIRECTORY).glob('*.json'):
        value = json.loads(p.read_text());verify_sealed(value, 'confirmation_sha256')
        vid = value['verification_id']
        assert value['original_confirmation_sha256'] == out[vid]['confirmation_sha256']
        out[vid] = value
    return out


def reconcile(root, arm, original, snapshot, runner_for, phase, workers=2):
    """No replication/extended rerun; adopt existing solos or run unused precharged ones."""
    from . import replay_bank as rb
    from . import exact_grade as xg
    target = Path(root)/'replay_bank'/arm/DIRECTORY/(original['verification_id']+'.json')
    if target.exists():
        value = json.loads(target.read_text());xg.verify_sealed(value, 'confirmation_sha256')
        assert value['original_confirmation_sha256'] == original['confirmation_sha256']
        return value
    out = reevaluate(original)
    if out['checks'].get('replication') and out['checks'].get('no_later_harm'):
        out['synergy_diagnostic_required'] = True
        if 'cooperation' not in out:
            r = snapshot['first_seed_receipt']
            out['cooperation'] = rb._cooperation(root, arm, r, snapshot['snapshot'],
                runner_for, str(root), phase, workers, r['chosen_plan'], r['grade_veh_s'])
    else:
        out['synergy_diagnostic_required'] = False
    xg._write_new(target, xg._seal(out, 'confirmation_sha256'))
    return json.loads(target.read_text())


def quality(root, arm, sources):
    from . import replay_bank as rb
    original_reader = rb.confirmations
    try:
        rb.confirmations = confirmations
        value = rb.bank_quality(root, arm, sources)
    finally:
        rb.confirmations = original_reader
    old = legacy(root, arm)
    new = confirmations(root, arm)
    value['safety_rule'] = RULE
    value['safety_revalidation_pending'] = sum('old_admitted' not in c for c in new.values())
    value['old_rule_counts'] = dict(confirmed=len(old), admitted=sum(c['admitted'] for c in old.values()),
        safety_rejected=sum(not c['checks'].get('safety', True) for c in old.values()))
    value['old_new_verdicts'] = [dict(verification_id=vid, old_admitted=c['old_admitted'],
        new_admitted=c['admitted'], safety_verdicts=c.get('safety_verdicts'))
        for vid, c in new.items() if 'old_admitted' in c]
    value['synergy_diagnostic_twins'] = [dict(verification_id=vid, admitted=c['admitted'],
        group_advantage_veh_s=c['first_seed_advantage_veh_s'], **c['cooperation'])
        for vid, c in new.items() if c.get('synergy_diagnostic_required') and 'cooperation' in c]
    path = Path(root)/'replay_bank'/arm/'BANK_QUALITY.json'
    tmp = path.with_name(path.name+'.safety0426.publishing')
    tmp.write_text(json.dumps(value, indent=1));tmp.replace(path)
    return value
