"""Frozen panel, equal resource quotas, and preregistered model/decision rules."""
from copy import deepcopy

from tools.coopbandit.protocol import CELLS
from tools.coopplan.planner import decide

ARMS = ('expected', 'strict')
CREDIT = 2950
H_CREDIT = 120
PANEL_SHA = '72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6'


def panel():
    return [dict(seed=984000+i, cell=CELLS.index((p,d)), repeat=r, penetration=p, demand=d)
            for i,(p,d,r) in enumerate((p,d,r) for p in (.4,.2,.1,.05)
                                      for d in (2700,2400,1900) for r in range(2))]


def budget():
    return dict(episodes=24*2*1200, all_group_credit=24*2*CREDIT,
                total=199200, cap=200000, unallocated=800, new_baselines=0,
                mechanics=0)


def model_choice(v4, v5):
    def matched(rows):
        rows = [r for r in rows if r['horizon_s'] == 300]
        if len(rows) != 12 or len({r['state_id'] for r in rows}) != 12:
            raise ValueError('complete same12 H300 quality required')
        return {r['state_id']:r for r in rows}
    a, b = matched(v4), matched(v5)
    if a.keys() != b.keys() or any(a[k]['exact_D'] != b[k]['exact_D'] for k in a):
        raise ValueError('models must share states and exact targets')
    from tools.wmteach.labels import helpful_quality, quality
    qa, qb = helpful_quality(list(a.values())), helpful_quality(list(b.values()))
    all_a, all_b = quality(list(a.values()),12), quality(list(b.values()),12)
    agreement = qb['helpful_choice_agreement'] >= qa['helpful_choice_agreement']
    saving = all_b['exact_chosen_saving'] >= all_a['exact_chosen_saving']
    return dict(selected='FULL-v5' if agreement and saving else 'FULL-v4',
                rule='v5 iff H300 helpful agreement AND all12 chosen saving >= v4',
                v4_helpful=qa, v5_helpful=qb, agreement_not_worse=agreement,
                saving_not_worse=saving, v4_all=all_a, v5_all=all_b)


def decision(prediction, arm):
    if arm not in ARMS:
        raise ValueError('registered arm')
    result = decide(prediction)
    if arm == 'expected':
        means = result['expected_delay']
        eligible = [a for a in means if a != 'O0' and means[a] < means['O0']]
        result.update(selected=min(eligible,key=lambda a:(means[a],a)) if eligible else 'O0',
                      eligible=sorted(eligible), confidence=None,
                      semantics='positive_expected_saving_argmax')
    result['arm'] = arm
    return result


def admit(event, spent):
    cost = 4*(event['time_s']+H_CREDIT)
    proposed = deepcopy(event['executed_action'])
    event['proposed_action'] = proposed
    refused = proposed['id'] != 'O0' and spent+cost > CREDIT
    event['budget_pruned'] = refused
    if refused:
        event.update(executed_action=dict(id='O0',members=[]), group=[])
        return 0
    return cost if proposed['id'] != 'O0' else 0
