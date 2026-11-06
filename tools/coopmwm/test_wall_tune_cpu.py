"""W2 per-situation wall tuning job builder: caps, per-seed rule, executor ledger, emit (CPU fakes).

No SUMO: ``sumo_verify.verify`` / tapes / bootstrap are injected fakes. The builder itself
is pure file work and is exercised on its real code path.
"""
import json

import pytest

from . import exact_grade as xg, moves, replay_bank as rb, verify_perseed as vp
from . import wall_negatives as wn, wall_release as wr, wall_tune as wt
from .protocol import digest

MEMBERS = ['m1', 'm2', 'm3', 'm4']


def make_situation(tmp_path, t0=455, arm='B', members=MEMBERS):
    ctx = tmp_path/'ctx_t455'
    ctx.mkdir(parents=True, exist_ok=True)
    rows = [dict(vid=v, s_m=-300.-12*i, v_mps=10., lane=(0, 1, 1, 0)[i], is_cav=True, length_m=5.)
            for i, v in enumerate(members)]
    rows.append(dict(vid='h1', s_m=-280., v_mps=9., lane=0, is_cav=False, length_m=5.))
    (ctx/'CONTEXT.json').write_text(json.dumps(dict(
        state_id='cp_wall_B_train_986061_t455', world=dict(seed=986061), arm=arm, t0=t0, rows=rows,
        policy_profile={v: [0.05*i]*wt.PROFILE_S for i, v in enumerate(members)},
        snapshot_sha256='ab'*32)), encoding='utf-8')
    spec = tmp_path/'situation.json'
    spec.write_text(json.dumps(dict(situation_id='lead986061', ctx_dir=str(ctx), arm=arm,
                                    members=members)), encoding='utf-8')
    return spec


def test_registration_first_and_never_rewritten(tmp_path):
    spec = make_situation(tmp_path)
    with pytest.raises(ValueError, match='registered budget'):
        wt.build_screen(tmp_path/'tune', spec)
    first = wt.init(tmp_path/'tune', 250000)
    assert wt.init(tmp_path/'tune', 10) == first                    # never rewritten
    with pytest.raises(ValueError):
        wt.init(tmp_path/'bad', 1000)                               # stage caps above the root cap
    reg = json.loads((tmp_path/'tune'/'REGISTRATION.json').read_text())
    reg['caps']['screen_sim_s'] = 10**9
    (tmp_path/'tune'/'REGISTRATION.json').write_text(json.dumps(reg))
    with pytest.raises(ValueError, match='changed'):
        wt.registration(tmp_path/'tune')


def test_situation_refuses_eval_worlds_and_mismatched_arms(tmp_path):
    spec = make_situation(tmp_path)
    data = json.loads(spec.read_text())
    ctx = json.loads((tmp_path/'ctx_t455'/'CONTEXT.json').read_text())
    for world in (dict(seed=964010), dict(seed=977000), dict(seed=12)):
        (tmp_path/'ctx_t455'/'CONTEXT.json').write_text(json.dumps(dict(ctx, world=world)))
        with pytest.raises(ValueError, match='TRAIN'):
            wt.load_situation(data)
    (tmp_path/'ctx_t455'/'CONTEXT.json').write_text(json.dumps(ctx))
    with pytest.raises(ValueError, match='arm'):
        wt.load_situation(dict(data, arm='R'))
    with pytest.raises(ValueError):
        wt.load_situation(dict(data, members=MEMBERS+['h1']))


def test_screen_jobs_are_per_seed_identical_slates_with_registered_crn_salts(tmp_path):
    spec = make_situation(tmp_path)
    root = tmp_path/'tune'
    wt.init(root, 250000)
    path, plan = wt.build_screen(root, spec)
    jobs = plan['jobs']
    assert [j['seed'] for j in jobs] == list(wt.REGISTERED['screen']['seeds'])
    assert len({digest(j['groups']) for j in jobs}) == 1 and all(j['solos'] == [] for j in jobs)
    assert jobs[0]['salt'] is None and all(j['salt'].startswith('walltune#') for j in jobs[1:])
    for j in jobs[1:]:
        g = int(digest(dict(replication=j['salt']))[:15], 16)
        assert not any(lo <= g <= hi for lo, hi in wt.EVAL_RANGES)
    groups = jobs[0]['groups']
    assert len(groups) == 12
    shapes = {(moves.from_plan(p).release_shape, moves.from_plan(p).release_s) for p in groups}
    assert shapes == {('step', 0), ('linear', 20), ('cosine', 20)}
    for p in groups:
        assert moves.from_plan(p) == wr.from_plan(p)               # exact, executable wall macros
    from . import sumo_verify as sv
    for j in jobs:
        assert j['branch_sim_s'] == sv.branch_sim_s(min(455+max(p['duration_s'] for p in groups)+120, 1200))
        assert j['quote_sim_s'] == (1+len(groups))*j['branch_sim_s']
        assert j['job_sha256'] == digest({k: v for k, v in j.items() if k != 'job_sha256'})
    assert plan['quote_sim_s'] == sum(j['quote_sim_s'] for j in jobs) <= wt.REGISTERED['caps']['screen_sim_s']
    with pytest.raises(ValueError, match='never rebuilt'):
        wt.build_screen(root, spec)


def test_budget_caps_truncate_by_priority_then_block(tmp_path):
    spec = make_situation(tmp_path)
    root = tmp_path/'tune'
    caps = dict(situation_sim_s=20000, screen_sim_s=10000, refine_sim_s=10000, confirm_sim_s=10000)
    wt.init(root, 20000, caps)
    _, plan = wt.build_screen(root, spec)
    full = wt.screen_candidates(wt.load_situation(spec))
    groups = plan['jobs'][0]['groups']
    assert plan['quote_sim_s'] <= 10000 and 0 < len(groups) < len(full)
    assert [p['id'] for p in groups] == [p['id'] for p in full[:len(groups)]]   # lowest priority dropped
    assert wt.reserved(root) == plan['quote_sim_s']
    tiny = tmp_path/'tiny'
    wt.init(tiny, 1000, dict(situation_sim_s=1000, screen_sim_s=1000, refine_sim_s=1000, confirm_sim_s=1000))
    with pytest.raises(wt.BudgetBlocked):
        wt.build_screen(tiny, spec)
    assert not (tiny/'plans').exists()


def rows_for(plan, seeds, adv, synergy=None, stage='refine', status='graded', collisions=0.):
    out = []
    for k, a in zip(seeds, adv):
        r = dict(stage=stage, round=1, mode='synergy' if stage == 'refine' else 'group', situation_id='lead986061',
                 seed=k, plan_sha256=digest(plan), plan=plan, status=status, adv=a, adv_lb95=a-10.,
                 safety=dict(collisions=collisions, teleports=0., hard_brake=0.))
        if synergy is not None:
            s = synergy[seeds.index(k)]
            r.update(synergy=s, synergy_lb95=s-10.)
        out.append(r)
    return out


def test_per_seed_rule_never_pools_and_never_imputes(tmp_path):
    situation = wt.load_situation(make_situation(tmp_path))
    a, b, c, d = wt.screen_candidates(situation)[:4]
    seeds = [1, 2, 3]
    rows = (rows_for(a, seeds, [900., -300., -250.], [800., -260., -240.])      # big pooled mean, 1/3 seeds
            + rows_for(b, seeds, [60., 55., 40.], [45., 30., 25.])               # positive on every seed
            + rows_for(c, [1, 2], [500., 500.], [400., 400.])                    # missing seed 3
            + rows_for(d, seeds, [80., 90., 70.], [70., 80., 60.], collisions=1.))  # unsafe on a seed
    ranked = wt.ranked(rows, 'refine')
    assert [k for k, _ in ranked] == [digest(b), digest(a)]
    assert sum(r['synergy'] for r in rows_for(a, seeds, [900., -300., -250.], [800., -260., -240.])) > \
        sum(r['synergy'] for r in rows_for(b, seeds, [60., 55., 40.], [45., 30., 25.]))


def fake_verify_factory(J):
    """sv.verify stand-in: precharges every branch at the job's branch sim-s, never runs SUMO."""
    def verify(snap, cands, runner, charge, proposer, workers=None, root=None, tail_s=None):
        from . import sumo_verify as sv
        t0 = 455
        sim = sv.branch_sim_s(sv.window_end(t0, cands, tail_s=tail_s))
        vid = f'sv_fake_{digest([c["id"] for c in cands])[:8]}'
        try:
            charge(vid, [dict(sim_s=sim)]*(len(cands)+1))
        except xg.BudgetBlocked as error:
            return dict(status='budget_blocked', verification_id=vid, reason=str(error))
        branches = [dict(branch='policy', J_veh_s=J['policy'], window=dict(collisions=0, teleports=0, hard_brake=0))]
        for c in cands:
            branches.append(dict(branch=c['id'], J_veh_s=J.get(c['id'], J['default']),
                                 window=dict(collisions=0, teleports=0, hard_brake=0)))
        return dict(status='accepted', verification_id=vid, branches=branches, J_policy_veh_s=J['policy'],
                    window=dict(start_s=t0, end_s=t0+100))
    return verify


def fake_stats(pairs):
    (a, b, _), = pairs
    return dict(lower95_one_sided_veh_s=a['J_veh_s']-b['J_veh_s']-5.)


def test_executor_charges_quote_exactly_once_and_measures_per_seed_synergy(tmp_path):
    spec = make_situation(tmp_path)
    root = tmp_path/'tune'
    wt.init(root, 250000)
    situation = wt.load_situation(spec)
    plans = wt.screen_candidates(situation)[:2]
    job = wt.build_jobs(situation, 'refine', plans, round_=1)[1]
    J = dict(policy=1000., default=990., **{plans[0]['id']: 900.})
    for s in job['owner'][plans[0]['id']]:
        J[s] = 960.
    rows = wt.run_job(root, job, verify=fake_verify_factory(J), stats=fake_stats,
                      read_result=lambda d, vid, br: br, snapshot=object(), runner=object())
    first = next(r for r in rows if r['plan_sha256'] == digest(plans[0]))
    assert first['adv'] == 100. and first['adv_lb95'] == 95.
    assert first['synergy'] == 60. and first['synergy_lb95'] == 55. and first['seed'] == 2
    ledger = wt._ledger(root)
    assert len(ledger) == 1 and ledger[0]['sim_s'] == job['quote_sim_s']
    assert len(json.loads(json.dumps(job['solos']))) == len({s['id'] for s in job['solos']})
    # Replay of the same job: blocked, never charged twice.
    again = wt.run_job(root, job, verify=fake_verify_factory(J), stats=fake_stats,
                       read_result=lambda d, vid, br: br, snapshot=object(), runner=object())
    assert {r['status'] for r in again} == {'budget_blocked'} and len(wt._ledger(root)) == 1
    # A tampered job (e.g. an extra group) is refused before any charge.
    bad = dict(job, groups=job['groups']+plans[:1])
    with pytest.raises(ValueError, match='changed'):
        wt.run_job(root, bad, verify=fake_verify_factory(J), stats=fake_stats, snapshot=object(), runner=object())


def test_charge_caps_are_hard_at_run_time(tmp_path):
    spec = make_situation(tmp_path)
    root = tmp_path/'tune'
    wt.init(root, 250000, dict(situation_sim_s=30000, screen_sim_s=4000, refine_sim_s=4000,
                               confirm_sim_s=4000))
    situation = wt.load_situation(spec)
    jobs = wt.build_jobs(situation, 'screen', wt.screen_candidates(situation)[:2])
    wt.charge(root, jobs[0], 'a', jobs[0]['quote_sim_s'])
    with pytest.raises(wt.BudgetBlocked, match='quote'):
        wt.charge(root, jobs[0], 'b', 1)
    wt.charge(root, jobs[1], 'c', jobs[1]['quote_sim_s'])
    with pytest.raises(wt.BudgetBlocked, match='screen cap'):
        wt.charge(root, jobs[2], 'd', jobs[2]['quote_sim_s'])
    with pytest.raises(wt.BudgetBlocked, match='replay'):
        wt.charge(root, jobs[2], 'a', 1)


def test_refine_and_confirm_builders_follow_the_per_seed_ranking(tmp_path):
    spec = make_situation(tmp_path)
    root = tmp_path/'tune'
    wt.init(root, 250000)
    situation = wt.load_situation(spec)
    _, screen = wt.build_screen(root, spec)
    groups = screen['jobs'][0]['groups']
    rows = []
    for i, p in enumerate(groups):
        rows += rows_for(p, [1, 2, 3], [10.*i, 5.*i-20, -3.], stage='screen')
    (root/'results').mkdir()
    (root/'results'/'screen.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='admitted'):
        wt.build_confirm(root, spec)
    _, refine = wt.build_refine(root, spec)
    tops = [k for k, _ in wt.ranked(rows, 'screen')[:2]]
    refined = refine['jobs'][0]['groups']
    assert [digest(p) for p in refined[:2]] == tops
    assert len(refined) <= wt.REGISTERED['refine']['max_candidates']
    assert refine['jobs'][0]['solos'] and refine['quote_sim_s'] <= wt.available(root, 'lead986061', 'refine')+refine['quote_sim_s']
    assert all(len(j['owner']) == len(refined) for j in refine['jobs'])
    rrows = rows_for(refined[1], [1, 2, 3], [50., 40., 30.], [20., 25., 30.])
    rrows += rows_for(refined[0], [1, 2, 3], [70., -40., -30.], [60., -10., -20.])
    (root/'results'/'refine.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rrows))
    _, confirm = wt.build_confirm(root, spec)
    job = confirm['jobs'][0]
    assert [digest(p) for p in job['plans']] == [digest(refined[1])]
    assert job['quote_sim_s'] == wt.confirm_quote(situation, job['plans'], 4) <= wt.REGISTERED['caps']['confirm_sim_s']
    total = sum(json.loads(p.read_text())['quote_sim_s'] for p in (root/'plans').glob('*.json'))
    assert total == wt.reserved(root) <= wt.REGISTERED['caps']['situation_sim_s']


def test_neighbourhood_covers_registered_coordinates():
    situation = dict(situation_id='q', state_id='q', t0=455, members=MEMBERS,
                     lanes=dict(m1=0, m2=1, m3=1, m4=0), policy_profile={v: [0.]*130 for v in MEMBERS})
    base = wt.candidate(situation, MEMBERS, 0, 30, 1., 'linear', 20)
    moves_ = [moves.from_plan(p) for p in wt.neighbours(situation, base)]
    m = moves.from_plan(base)
    changed = {tuple(k for k in ('members', 'onset', 'hold', 'depth', 'release_s', 'release_shape')
                     if getattr(x, k) != getattr(m, k)) for x in moves_}
    assert {('release_shape',), ('release_s',), ('depth',), ('hold',), ('onset',), ('members',),
            ('release_s', 'release_shape')} <= changed
    assert all(2 <= len(x.members) <= 4 for x in moves_)
    assert all(wr.duration(x) <= wt.PROFILE_S for x in moves_)


def test_quote_matches_registered_expectation():
    q = wt.quote_situation(455, 4)
    assert q['screen_candidates'] == 12 and q['seeds'] == [1, 2, 3]
    assert q['total_sim_s'] == q['screen_sim_s']+q['refine_sim_s']+q['confirm_sim_s']
    assert q['total_capped_sim_s'] <= wt.REGISTERED['caps']['situation_sim_s']
    for s in wt.STAGES:
        assert q[f'{s}_sim_s'] <= wt.REGISTERED['caps'][f'{s}_sim_s']


def make_bank(tmp_path, arm='B'):
    """Native-format all8 record (sealed) with one positive, one advantage-loss, one synergy-loss seed."""
    from .test_track2_backend_cpu import rows as scene_rows
    bank_dir = tmp_path/'minebank'/'bank'
    bank = rb.bank_root(bank_dir, arm)
    move = moves.Move('wall', ('c1', 'c2'), 0, 20, .6, 15, 'cosine')
    plan = wr.plan(move, [[.1]*wr.duration(move), [.2]*wr.duration(move)])
    vid = 'sv_cp_wall_B_train_986061_t0300_abcd1234'

    def tape(seed):
        return dict(branches_dir=str(bank/'sumo_verification'/'branches'), verification_id=f'{vid}_s{seed}',
                    branch=dict(branch='cand1', window=dict(collisions=0, teleports=0)), seed_index=seed)
    examples = [
        dict(seed_index=1, example_id=f'{vid}#c1#s1', plan=plan, admitted=True, cooperative=True, safety_ok=True,
             advantage_veh_s=300., lower95_one_sided_veh_s=80., tape=tape(1),
             synergy=dict(pooled_total_veh_s=250., lower95_one_sided_veh_s=60., best_solo_member='c1')),
        dict(seed_index=2, example_id=f'{vid}#c1#s2', plan=plan, admitted=False, cooperative=False,
             advantage_veh_s=-170., lower95_one_sided_veh_s=-220., tape=tape(2)),
        dict(seed_index=3, example_id=f'{vid}#c1#s3', plan=plan, admitted=True, cooperative=False, safety_ok=True,
             advantage_veh_s=40., lower95_one_sided_veh_s=5., tape=tape(3),
             synergy=dict(pooled_total_veh_s=-30., lower95_one_sided_veh_s=-70., best_solo_member='c2')),
        dict(seed_index=4, example_id=f'{vid}#c1#s4', plan=plan, admitted=False, status='data_missing')]
    record = xg._seal(dict(verification_id=vid, t0=300, per_seed_v1=dict(examples=examples)), 'confirmation_sha256')
    (bank/vp.RECORDS).mkdir(parents=True)
    (bank/vp.RECORDS/(vid+'.json')).write_text(json.dumps(record))
    (bank/'sumo_verification'/'verifications').mkdir(parents=True)
    (bank/'sumo_verification'/'verifications'/(vid+'.json')).write_text(json.dumps(
        dict(episode=dict(id='cp_wall_B_train_986061', world=dict(seed=986061)))))
    frames = {str(t): wn.whitelist(scene_rows()) for t in (299, 300)}
    return bank_dir, plan, move, frames


def test_emit_writes_clean3_mined_rows_and_per_seed_negatives(tmp_path):
    from . import bank_breadth, group_lane_bank
    bank_dir, plan, move, frames = make_bank(tmp_path)
    seen = []

    def reader(trace, t0):
        seen.append((str(trace), t0))
        return frames
    mined, negatives, receipt = wt.emit(bank_dir, 'B', tmp_path/'emit', source='CPU fixture',
                                        frames_reader=reader)
    assert len(mined) == 1 and mined[0]['seed'] == 1
    row = mined[0]
    binding = group_lane_bank.binding(row)                  # the clean3 ingestion validator
    assert binding['members'] == plan['members'] and binding['duration_s'] == plan['duration_s']
    assert row['plan_sha256'] == digest(plan) and row['wall_macro'] == move.to_dict()
    assert bank_breadth.mined_move(plan) == move
    assert set(row['synergy_veh_s']) == {'1'} and row['n_reps'] == 4
    assert {(n['seed'], n['basis']) for n in negatives} == {(2, 'advantage'), (3, 'synergy')}
    assert all(wn.validate(n) is n for n in negatives)
    assert all(trace.endswith('policy'+__import__('os').sep+'observable.jsonl.gz') or
               trace.endswith('policy/observable.jsonl.gz') for trace, _ in seen)
    assert receipt['mined'] == 1 and receipt['negatives'] == 2 and receipt['examples'] == 4
    on_disk = (tmp_path/'emit'/'WALL_NEGATIVES_B.jsonl').read_text().splitlines()
    assert [json.loads(x) for x in on_disk] == negatives
    # Stage refinement rows with a measured negative synergy also become (own-seed) negatives.
    stage = [dict(arm='B', mode='synergy', status='graded', synergy=-12., synergy_lb95=-40., adv=-3.,
                  situation_id='lead', verification_id='sv_stage', plan=plan, seed=2, t0=300,
                  world=dict(seed=986061), branches_dir=str(tmp_path/'b'), stage='refine', tag='t')]
    _, more, _ = wt.emit(bank_dir, 'B', tmp_path/'emit2', source='CPU', stage_rows=stage, frames_reader=reader)
    assert len(more) == 3 and more[-1]['source'] == 'CPU:refine'
    # R reads only the R bank (none here): nothing is borrowed across arms.
    mined_r, neg_r, _ = wt.emit(bank_dir, 'R', tmp_path/'emitR', source='CPU', frames_reader=reader)
    assert mined_r == [] and neg_r == []


def test_executor_refuses_a_snapshot_of_another_arm_before_spending(tmp_path):
    import torch
    from . import sumo_verify as sv
    spec = make_situation(tmp_path)
    ctx = tmp_path/'ctx_t455'
    for arm, ok in (('R', False), ('B', True)):
        snap = dict(episode=dict(id='cp_wall_train_986061', arm=arm, world=dict(seed=986061)), t0=455)
        torch.save(snap, ctx/'SNAPSHOT.pt')
        data = json.loads((ctx/'CONTEXT.json').read_text())
        data['snapshot_sha256'] = sv.snapshot_sha(snap)
        (ctx/'CONTEXT.json').write_text(json.dumps(data))
        situation = wt.load_situation(spec)
        if ok:
            assert wt._snapshot(situation)['episode']['arm'] == 'B'
        else:
            with pytest.raises(ValueError, match='arm'):
                wt._snapshot(situation)
