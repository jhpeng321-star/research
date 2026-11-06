"""clean4 FAIRAUDIT F5 / MAIN option (a): R family support may be a subset (incl. empty) of B's,
never padded; B capacity is its own pool; R follows B's sequence restricted to its own families.
Absent key = original equal-support rule. Plus the LEARNAUDIT supervision assert (pure)."""
import json

import pytest

from tools.coopmwm import bank_breadth as bb, clean4_receipts as rc4


def _pool(arm, families):
    return [dict(context_id=f'{arm}{i}_{f}', example_id=f'{arm}{i}', family=f, verification_id=f'v{arm}{i}')
            for i, f in enumerate(families)]


def _patch(monkeypatch, b, r):
    pools = dict(B=_pool('B', b), R=_pool('R', r))
    monkeypatch.setattr(bb, 'practice_pool', lambda cfg, arm, **k: pools[arm])
    monkeypatch.setattr(bb, 'inputs', lambda cfg: [])
    return pools


ON = dict(r_family_subset=bb.SUBSET_RULE)


def test_key_validation_and_default_unchanged(monkeypatch):
    assert not bb.subset_rule({})
    with pytest.raises(ValueError):
        bb.subset_rule(dict(r_family_subset='anything else'))
    _patch(monkeypatch, ['closing_pair', 'abreast_wall'], [])
    with pytest.raises(ValueError):
        bb.matched_pools({})                       # original equal-support rule refuses
    assert bb.matched_pools(ON)['R'] == []        # empty R allowed under the registered rule


def test_subset_allowed_superset_refused(monkeypatch):
    _patch(monkeypatch, ['closing_pair', 'abreast_wall'], ['abreast_wall'])
    assert len(bb.matched_pools(ON)['R']) == 1
    _patch(monkeypatch, ['abreast_wall'], ['abreast_wall', 'group_hold'])
    with pytest.raises(ValueError):
        bb.matched_pools(ON)


def test_practice_plan_b_own_capacity_r_restricted(monkeypatch, tmp_path):
    _patch(monkeypatch, ['closing_pair', 'abreast_wall', 'abreast_wall'], [])
    b = bb.practice_plan(tmp_path, 'B', 1, cfg=ON, fraction=.25, repeat_cap=2, count=4, training_seed=7)
    assert len(b['rows']) == 4 and b['breadth']['subset_rule'] == bb.SUBSET_RULE
    assert {r['family'] for r in b['rows']} == {'closing_pair', 'abreast_wall'}
    # R follows B's realized sequence restricted to its own (empty) support: zero practice rows.
    from tools.coopmwm import practice_starts as ps, exact_grade as xg
    path = ps.plan_path(tmp_path, 'B', 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(xg._seal(dict(b), 'practice_plan_sha256')), encoding='utf-8')
    r = bb.practice_plan(tmp_path, 'R', 1, cfg=ON, fraction=.25, repeat_cap=2, count=4, training_seed=7)
    assert r['rows'] == [] and r['registry']['empty_bank']
    # original rule: B capacity is the min over both pools -> no practice when R is empty
    with pytest.raises(ValueError):
        bb.practice_plan(tmp_path, 'B', 2, cfg={}, fraction=.25, repeat_cap=2, count=4, training_seed=7)


def _arms(b_contexts, r_contexts):
    return dict(B=dict(learning=dict(bank_trigger=dict(contexts=b_contexts))),
                R=dict(learning=dict(bank_trigger=dict(contexts=r_contexts))))


def test_supervision_check():
    good = [dict(family='lane_drop_initiate:closing_pair'), dict(family='abreast_wall'),
            dict(family='lane_drop_join'), dict(family='wall_negative')]
    out = rc4.supervision_check(_arms(good, []))
    assert out['R']['contexts'] == 0 and out['B']['positive_family_counts']['abreast_wall'] == 1
    for bad_b, bad_r in ((good[1:], []), (good[:1], []), (good+[dict(family='wall')], []),
                         (good+[dict(family='none')], []), (good+[dict(family='ease_off')], []),
                         (good, [dict(family='abreast_wall')])):
        with pytest.raises(ValueError):
            rc4.supervision_check(_arms(bad_b, bad_r))


def test_raw_newmap_hold_row_refused_exact_binding_accepted(monkeypatch):
    """N5 REVIEW_N9_2 / MAIN: raw hold rows without lane_drop_macro fail closed; N2's converter binds exactly."""
    from pathlib import Path
    from tools.coopmwm import moves, map_descriptor as md, lane_drop
    run = Path('E:/research-dense/runs/active/C4-V7FT-20261017')
    path = run/'PLANGEN_MOVES_B.jsonl'
    if not path.is_file():
        pytest.skip('laptop run dir only')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    spec = Path(moves.__file__).resolve().parents[2]/'assets/research/newmap_candidates/wz3to2'
    desc = md.load(spec)
    for row in rows:
        with pytest.raises(ValueError):
            moves.from_plan(row)
        stripped = {k: v for k, v in row.items() if k not in ('map',)}
        with pytest.raises(ValueError):
            moves.from_plan(stripped)
        plan = lane_drop.plan_from_mined_row(row, desc)
        move = moves.from_plan(plan)
        assert isinstance(move, lane_drop.LaneDropMove) and move.family in ('closing_pair', 'abreast_wall')
    # legacy merge-map plans keep their projection path (no 'map'/variant/member_lanes keys)
    legacy = dict(id='p', family='wall', members=['a', 'b'], duration_s=3,
                  perturbation_profile=[[0., -1., -1.], [0., -1., -1.]], policy_profile=[[0., 0., 0.]]*2,
                  profile=[[0., -1., -1.]]*2)
    assert moves.from_plan(legacy).family in moves.FAMILIES


def test_registered_stop_rule():
    from tools.coopmwm import launch_training as lt
    assert lt.registered_last_update({}) == 24
    assert lt.registered_last_update(dict(registered_stop=dict(after_update=12, rule='MAIN stop'))) == 12
    for bad in (dict(after_update=0, rule='x'), dict(after_update=25, rule='x'), dict(after_update=12),
                dict(after_update='12', rule='x'), 'u12'):
        with pytest.raises(ValueError):
            lt.registered_last_update(dict(registered_stop=bad))


def test_first_three_registered_states_in_order():
    from tools.coopmwm import launch_qualification as lq, track2_monitor as tm

    class Backend:
        def __init__(self, counts, rule=None):
            self.registration = dict(replicates=[dict(replicate=i, world_seed=1000+i) for i in range(len(counts))],
                                     **({'state_order_rule': rule} if rule else {}))
            self.counts = counts

        def resolve_states(self, row):
            return [(row['replicate'], k) for k in range(self.counts[row['replicate']])]

    legacy = lq.first_three_groups(Backend([3, 1, 1, 1, 1]), {})
    assert [(r['replicate'], s) for r, s in legacy] == [(0, [(0, 0), (0, 1), (0, 2)])]
    cfg = dict(registered_state_order=tm.STATE_ORDER_RULE)
    groups = lq.first_three_groups(Backend([2, 1, 1, 1, 1], tm.STATE_ORDER_RULE), cfg)
    assert [(r['replicate'], s) for r, s in groups] == [(0, [(0, 0), (0, 1)]), (1, [(1, 0)])]
    with pytest.raises(ValueError):          # the registration must carry the same rule
        lq.first_three_groups(Backend([2, 1, 1, 1, 1]), cfg)
