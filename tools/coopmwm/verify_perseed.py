"""verify_mode='all8_both_seeds' (owner 2026-11-03 ~22:05; DEFAULT OFF, legacy byte-identical when off).

Plans must be judged on the seed they are used on; cooperation can differ per seed and scenario.
Per completed first-seed decision (accepted OR rejected - a plan may win only on seed 2):

  seed 1 = the existing first-seed receipt (K candidates + 'policy continues', already run);
  seed 2 = ALL the same K candidates + 'policy continues' on rb.replicate(snapshot, vid)
           (the existing seed-2 CRN salt convention: salt = first-seed verification_id);
  per seed, EVERY candidate whose within-seed per-second paired bootstrap lower bound
  (bank_statistics._stats: policy-continues minus plan, B=2000, RNG 20261103, circular 30 s
  blocks or per-vehicle units) is > 0 AND whose composite MAIN04:26 safety verdict passes
  is admitted - not only the best; seeds select their winners independently;
  seed 1 additionally keeps the existing later-harm check (one extended-window verification
  of all seed-1 LB>0 candidates, tail 600 s, also the seed-1 safety window - as per_seed_v1);
  seed 2 later-harm is NOT MEASURED (disclosed per example, never imputed);
  per admitted (plan, seed) with >= 2 members: ONE batched solo verification on THAT seed
  (one member follows the plan, the others policy-continue; one shared policy branch),
  within-seed cooperative label = best-solo minus group per-second bootstrap LB > 0.

Every verification is charged through the existing hooks before physics: sv.verify ->
runner.admit (disk guard) -> rb._confirm_charge(run_root, phase, kind) (looked up at call time,
so precharged/builder overrides apply); warm_health.assert_running(root, arm) is checked before
each verification. Receipts are ordinary sealed sv receipts in per-kind roots; the decision
record (bank/confirmations_all8/<vid>.json, sealed 'confirmation_sha256') carries the
per_seed_v1 example list (one example per (candidate, seed), each with its own plan, tape and
seed index) so the per_seed_v1 bank rule (c4/bank-perseed) can consume it directly.
"""
import copy
import json
from pathlib import Path
import time

ALL8 = 'all8_both_seeds'
LEGACY = 'legacy'
VERIFY_MODES = (LEGACY, ALL8)
SCHEMA = 'coopmwm/verify-perseed/v1'
RECORDS = 'confirmations_all8'
SEED2_ROOT = 'confirm_seed2_all8'
EXT_ROOT = 'confirm_extended_all8'
SOLO_ROOTS = {1: 'confirm_solo_all8_s1', 2: 'confirm_solo_all8_s2'}
KIND_SEED2, KIND_EXT = 'confirm_seed2_all8', 'confirm_extended_all8'
KIND_SOLO = {1: 'confirm_solo_all8_s1', 2: 'confirm_solo_all8_s2'}
RULE = ('all8_both_seeds (owner 2026-11-03 22:05): all K candidates + policy-continues on BOTH CRN seeds; '
        'per seed every candidate with within-seed per-second paired bootstrap lower95 > 0 and composite '
        'safety pass is admitted (seed 1 also needs the extended-window later-harm pass; seed 2 later-harm '
        'not measured); solos per admitted (plan, seed); within-seed cooperative label; each admitted '
        '(plan, seed) is its own imitation example from THAT seed\'s tape.')


def mode(value):
    """Normalise a verify_mode config value; None/'' -> legacy."""
    v = value or LEGACY
    if v not in VERIFY_MODES:
        raise ValueError(f'unknown verify_mode {value!r}')
    return v


# ---------------------------------------------------------------- cost quote

def _branch(t0, duration_s, tail_s, end_s=1200):
    from . import sumo_verify as sv
    return sv.branch_sim_s(min(int(t0)+int(duration_s)+int(tail_s), int(end_s)))


def quote_decision(t0, durations, members, admitted=None, first_seed_positive=None, end_s=1200):
    """Charged sim-s of ONE decision: today (legacy) vs all8_both_seeds.

    durations/members: per candidate (K entries). admitted: {1: [idx..], 2: [idx..]} (0-based
    candidate indices admitted per seed; seed-1 LB>0 set is assumed == admitted for the extended
    quote). Legacy: first-seed K+1 branches; if the first seed is positive, replication (2) +
    extended (2) + per-member solos (2n, run when replication + later-harm pass, here assumed)."""
    from .replay_bank import EXTENDED_TAIL_S
    from .sumo_verify import TAIL_S
    admitted = {int(k): list(v) for k, v in (admitted or {1: [], 2: []}).items()}
    K = len(durations)
    maxD = max(durations)
    first = (K+1)*_branch(t0, maxD, TAIL_S, end_s)
    best = (admitted.get(1) or [0])[0]
    positive = bool(admitted.get(1)) if first_seed_positive is None else bool(first_seed_positive)
    legacy_conf = 0
    if positive:
        n = len(members[best])
        legacy_conf = (2*_branch(t0, durations[best], TAIL_S, end_s)
                       + 2*_branch(t0, durations[best], EXTENDED_TAIL_S, end_s)
                       + (2*n*_branch(t0, durations[best], TAIL_S, end_s) if n >= 2 else 0))
    seed2 = (K+1)*_branch(t0, maxD, TAIL_S, end_s)
    s1 = admitted.get(1) or []
    ext = (len(s1)+1)*_branch(t0, max(durations[i] for i in s1), EXTENDED_TAIL_S, end_s) if s1 else 0
    solos = sum((len(members[i])+1)*_branch(t0, durations[i], TAIL_S, end_s)
                for k in (1, 2) for i in admitted.get(k) or [] if len(members[i]) >= 2)
    legacy = first+legacy_conf
    new = first+seed2+ext+solos
    return dict(t0=t0, K=K, first_seed_sim_s=first, legacy_confirmation_sim_s=legacy_conf,
                legacy_total_sim_s=legacy, all8_seed2_sim_s=seed2, all8_extended_sim_s=ext,
                all8_solo_sim_s=solos, all8_confirmation_sim_s=seed2+ext+solos, all8_total_sim_s=new,
                multiplier_total=new/legacy, multiplier_confirmation=((seed2+ext+solos)/legacy_conf
                                                                      if legacy_conf else None),
                admitted=admitted, note='branch sim-s charged from t=0 (warm-up included), as sv.quote')


def standard_quote():
    """Per-decision quote at representative windows (K=8, D=120 s, 2-member plans)."""
    D, M = [120]*8, [['a', 'b']]*8
    rows = {}
    for t0 in (420, 600, 900):
        rows[f't0_{t0}'] = dict(
            first_seed_negative_seed2_none=quote_decision(t0, D, M, {1: [], 2: []}),
            first_seed_negative_seed2_one=quote_decision(t0, D, M, {1: [], 2: [3]}),
            one_winner_each_seed=quote_decision(t0, D, M, {1: [0], 2: [0]}),
            different_winners_each_seed=quote_decision(t0, D, M, {1: [0], 2: [5]}),
            two_winners_each_seed=quote_decision(t0, D, M, {1: [0, 1], 2: [0, 2]}))
    return rows


# ---------------------------------------------------------------- helpers

def first_seed_candidates(receipt, branches_dir, candidates=None):
    """The K first-seed candidate plans, identity-checked against the sealed receipt.

    ``candidates`` (in-memory, e.g. from the collector) or the branch spec files written by
    SubprocessBranchRunner (<branches>/<vid>/cand<i>.spec.json, sealed 'branch_sha256')."""
    from .exact_grade import verify_sealed
    from .protocol import digest
    want = receipt['candidate_plan_sha256']
    if candidates is None:
        candidates = []
        d = Path(branches_dir)/receipt['verification_id']
        for i in range(1, len(want)+1):
            p = d/f'cand{i}.spec.json'
            if not p.exists():
                raise FileNotFoundError(f'first-seed candidate spec missing: {p}')
            spec = json.loads(p.read_text(encoding='utf-8'))
            verify_sealed(spec, 'branch_sha256')
            if spec['branch_sha256'] != receipt['branch_specs_sha256'][i]:
                raise ValueError(f'candidate spec {p} differs from the sealed receipt')
            candidates.append(spec['plan'])
    candidates = [copy.deepcopy(c) for c in candidates]
    if [digest(c) for c in candidates] != list(want):
        raise ValueError('candidate plans differ from the first-seed receipt identity')
    return candidates


def _pair_safety(receipt, branches_dir, index):
    """Candidate ``index`` (1-based branch) minus policy deltas; MAIN04:26 verdict."""
    from . import replay_bank as rb
    from .bank_safety import verdict
    pol, cand = receipt['branches'][0], receipt['branches'][index]
    deltas = {k: float(cand['window'][k])-float(pol['window'][k]) for k in rb.SAFETY_KEYS}
    w = receipt['window']
    tr = {b['branch']: Path(branches_dir)/receipt['verification_id']/b['branch']/'observable.jsonl.gz'
          for b in (pol, cand)}
    deltas['ttc_below_1p5_events'] = (rb.ttc_events(tr[cand['branch']], w['start_s'], w['end_s'])
                                      - rb.ttc_events(tr[pol['branch']], w['start_s'], w['end_s']))
    v = verdict(deltas)
    return deltas, bool(v['new_combined_rule']), bool(v['old_no_metric_increase'])


def _complete(r):
    return r is not None and r.get('status') in ('accepted', 'rejected')


def _candidate_stats(receipt, branches_dir, index):
    """Within-seed paired bootstrap (policy-continues minus candidate) -> stats or raises."""
    from . import bank_statistics as bs
    if receipt['branches'][index]['status'] != 'graded' or receipt['branches'][0]['status'] != 'graded':
        raise bs._Missing(f'branch {receipt["branches"][index]["branch"]} ungraded')
    pol = bs._read_result(branches_dir, receipt['verification_id'], receipt['branches'][0])
    alt = bs._read_result(branches_dir, receipt['verification_id'], receipt['branches'][index])
    return bs._stats([(pol, alt, receipt['window'])]), alt


def _health(root, arm, health):
    if health is None:
        from .warm_health import assert_running
        assert_running(root, arm)
    elif health:
        health()


def _verify(kind, snapshot, candidates, root_dir, runner_for, run_root, phase, proposer, workers, tail_s,
            charges, root, arm, health):
    """One sv.verify through the existing hooks; appends the charge row."""
    from . import replay_bank as rb
    from . import sumo_verify as sv
    _health(root, arm, health)
    r = sv.verify(snapshot, candidates, runner_for(root_dir), rb._confirm_charge(run_root, phase, kind),
                  proposer, workers=workers, root=root_dir, tail_s=tail_s)
    charges.append(dict(kind=kind, phase=phase, verification_id=r['verification_id'], status=r['status'],
                        charged_sim_s=r.get('charged_sim_s', 0), actual_sim_s=r.get('actual_sim_s'),
                        n_branches=len(r.get('branches') or []), receipt_sha256=r['receipt_sha256']))
    return r


def _cooperation(seed, plan, plan_tape, window, solo, solo_dir, reference_J, admitted):
    """Within-seed cooperation label from one batched solo verification (never pooled)."""
    from . import bank_statistics as bs
    if len(plan['members']) < 2:
        return dict(cooperative=False, cooperative_status='single_cav',
                    statistical_reason='single-CAV plan: cooperative=False by definition')
    if solo is None or not _complete(solo):
        return dict(cooperative=False, cooperative_status='incomplete',
                    statistical_reason=f'seed-{seed} solo verification incomplete; never imputed')
    graded = [i for i, J in enumerate(solo['J_candidates_veh_s']) if J is not None]
    if len(graded) != len(plan['members']):
        return dict(cooperative=False, cooperative_status='incomplete',
                    statistical_reason=f'seed-{seed} solo branch failed; never imputed')
    match = solo['J_policy_veh_s'] == reference_J
    advs = [solo['J_policy_veh_s']-J for J in solo['J_candidates_veh_s']]
    b = max(range(len(advs)), key=lambda i: (advs[i], -i))
    try:
        solo_tape = bs._read_result(solo_dir/'branches', solo['verification_id'], solo['branches'][b+1])
        common = bs.paired_window(window, solo['window'])
        st = bs._stats([(solo_tape, plan_tape, common)])
    except bs._Missing as e:
        return dict(cooperative=False, cooperative_status='data_missing', statistical_reason=str(e),
                    solo_advantages_veh_s=advs)
    st.update(best_solo_member=plan['members'][b], best_solo_advantage_veh_s=advs[b],
              paired_window=dict(group_window=window, solo_window=solo['window'], used=common))
    positive = st['lower95_one_sided_veh_s'] > 0
    status = 'cooperative' if positive else 'not_cooperative'
    reason = f'seed-{seed} synergy lower95 ' + ('> 0' if positive else '<= 0')
    if not match:
        status, positive = 'reference_mismatch', False
        reason = f'seed-{seed} solo policy branch J differs from the reference policy branch; not counted'
    return dict(cooperative=bool(admitted and positive), cooperative_status=status, statistical_reason=reason,
                synergy=st, solo_advantages_veh_s=advs, solo_verification_id=solo['verification_id'],
                solo_receipt_sha256=solo['receipt_sha256'], solo_policy_matches_reference=match)


# ---------------------------------------------------------------- confirmation (one decision)

def confirm_all8(root, arm, receipt, snapshot, runner_for, run_root, phase, workers=2, version=None,
                 candidates=None, branches_dir=None, health=None, later_harm_seed1=True):
    """all8_both_seeds confirmation of ONE completed first-seed decision -> sealed record."""
    from . import replay_bank as rb
    from . import bank_statistics as bs
    from . import exact_grade as xg
    from .protocol import digest
    if receipt['status'] not in ('accepted', 'rejected'):
        raise ValueError('all8 confirmation needs a completed first-seed decision')
    bank = Path(rb.bank_root(root, arm));vid = receipt['verification_id']
    target = bank/RECORDS/(vid+'.json')
    if target.exists():
        v = json.loads(target.read_text());xg.verify_sealed(v, 'confirmation_sha256');return v
    first_dir = Path(branches_dir if branches_dir is not None else receipt['_branches'])
    plans = first_seed_candidates(receipt, first_dir, candidates)
    rc = {k: v for k, v in receipt.items() if not k.startswith('_')}
    proposer = rc['proposer'];charges = []
    common = dict(root=root, arm=arm, health=health, runner_for=runner_for, run_root=run_root, phase=phase,
                  proposer=proposer, workers=workers, charges=charges)

    def verify(kind, snap, cands, root_dir, tail_s):
        return _verify(kind, snap, cands, root_dir, tail_s=tail_s, **common)

    # Seed 2: the SAME K candidates + policy continues, existing salt convention.
    s2_snapshot = rb.replicate(snapshot, vid)
    s2_dir = bank/SEED2_ROOT
    s2 = verify(KIND_SEED2, s2_snapshot, plans, s2_dir, rb.sv.TAIL_S)
    seeds = {1: (rc, first_dir), 2: (s2, s2_dir/'branches')}

    # Within-seed paired bootstrap of every candidate on each seed (independent selection).
    evals = {1: {}, 2: {}}
    for k, (r, bdir) in seeds.items():
        for i in range(len(plans)):
            if not _complete(r):
                evals[k][i] = dict(missing=f'seed-{k} decision {r.get("status")}: {r.get("reason")}')
                continue
            try:
                st, tape = _candidate_stats(r, bdir, i+1)
            except bs._Missing as e:
                evals[k][i] = dict(missing=str(e));continue
            deltas, safe, old_safe = _pair_safety(r, bdir, i+1)
            evals[k][i] = dict(stats=st, tape=tape, window_deltas=deltas, window_safe=safe,
                               window_safe_old=old_safe, advantage=r['J_policy_veh_s']-r['J_candidates_veh_s'][i])

    # Seed 1 later-harm: one extended-window verification of every seed-1 LB>0 candidate.
    ext, ext_index = None, {}
    lb1 = [i for i, e in evals[1].items() if 'stats' in e and e['stats']['lower95_one_sided_veh_s'] > 0]
    if later_harm_seed1 and lb1:
        ext = verify(KIND_EXT, snapshot, [plans[i] for i in lb1], bank/EXT_ROOT, rb.EXTENDED_TAIL_S)
        ext_index = {i: j+1 for j, i in enumerate(lb1)}

    examples = []
    for k in (1, 2):
        for i, plan in enumerate(plans):
            e = evals[k][i]
            branch = seeds[k][0]['branches'][i+1] if _complete(seeds[k][0]) else None
            ex = dict(seed_index=k, candidate_index=i, example_id=f'{vid}#c{i+1}#s{k}', plan=copy.deepcopy(plan),
                      plan_id=plan['id'], plan_sha256=digest(plan), duration_s=int(plan['duration_s']),
                      members=list(plan['members']), group_size=len(plan['members']))
            if 'missing' in e:
                ex.update(admitted=False, status='data_missing', reason='per-unit data missing: '+e['missing'],
                          cooperative=False, cooperative_status='data_missing', later_harm='not_measured')
                examples.append(ex);continue
            st = e['stats'];lower = st['lower95_one_sided_veh_s']
            later, safe, source = None, e['window_safe'], f'seed{k}_window'
            if k == 1 and later_harm_seed1 and lower > 0:
                j = ext_index[i]
                if not _complete(ext) or ext['J_candidates_veh_s'][j-1] is None:
                    later = None;ex['later_harm_missing'] = True
                else:
                    later = ext['J_policy_veh_s']-ext['J_candidates_veh_s'][j-1] > 0
                    ed, safe, _ = _pair_safety(ext, bank/EXT_ROOT/'branches', j)
                    ex['extended_deltas'] = ed;source = 'extended_window_seed1'
                    ex['extended_advantage_veh_s'] = ext['J_policy_veh_s']-ext['J_candidates_veh_s'][j-1]
            if k == 1 and later_harm_seed1 and lower > 0 and later is None:
                ok, reason = False, 'seed-1 extended-window branch missing; never imputed'
            elif later is False:
                ok, reason = False, 'seed-1 later-harm check failed (extended-window advantage <= 0)'
            elif not lower > 0:
                ok, reason = False, f'seed-{k} one-sided 95% lower bound {lower:.4g} <= 0'
            elif not safe:
                ok, reason = False, f'seed-{k} composite safety check failed ({source})'
            else:
                ok = True
                reason = f'seed-{k} lower95 > 0, safety pass' + (', later-harm pass' if later else
                                                                  ', later-harm NOT MEASURED')
            ex.update(admitted=ok, status='admitted' if ok else 'rejected', reason=reason, admission=st,
                      lower95_one_sided_veh_s=lower, total_veh_s=st['pooled_total_veh_s'],
                      mean_per_unit_veh_s=st['per_seed'][0]['mean_per_unit_veh_s'],
                      n_units=st['per_seed'][0]['n_units'], advantage_veh_s=e['advantage'],
                      later_harm=('pass' if later else 'fail') if later is not None else 'not_measured',
                      safety_ok=safe, safety_source=source, window_deltas=e['window_deltas'],
                      window_safety_old_rule_ok=e['window_safe_old'],
                      tape=dict(branches_dir=str(seeds[k][1]), verification_id=seeds[k][0]['verification_id'],
                                branch=dict(branch), window=dict(seeds[k][0]['window']), seed_index=k,
                                intent_entry=(f'{vid}/{branch["branch"]}' if k == 1
                                              else f'{vid}/seed2/{branch["branch"]}')))
            examples.append(ex)

    # Solos per admitted (plan, seed) on THAT seed: one batched verification, shared policy branch.
    snaps = {1: snapshot, 2: s2_snapshot}
    for ex in examples:
        if ex['status'] != 'admitted':
            if 'cooperative' not in ex:
                ex.update(cooperative=False, cooperative_status='not_admitted')
            continue
        k, plan = ex['seed_index'], ex['plan']
        solo = None
        if len(plan['members']) >= 2:
            solo = verify(KIND_SOLO[k], snaps[k], [rb.solo_candidate(plan, m) for m in plan['members']],
                          bank/SOLO_ROOTS[k], rb.sv.TAIL_S)
        ex.update(_cooperation(k, plan, evals[k][ex['candidate_index']]['tape'], ex['tape']['window'], solo,
                               bank/SOLO_ROOTS[k], seeds[k][0]['J_policy_veh_s'], True))

    adm = [e for e in examples if e['admitted']]
    winners = {str(k): [e['plan_id'] for e in adm if e['seed_index'] == k] for k in (1, 2)}
    coop = [e for e in examples if e.get('cooperative') is True]
    coop_seeds = sorted({e['seed_index'] for e in coop})
    by_plan = {}
    for e in coop:
        by_plan.setdefault(e['plan_id'], set()).add(e['seed_index'])
    replicated = sorted(p for p, s in by_plan.items() if s == {1, 2})
    group_sizes = [len(p['members']) for p in plans]
    out = dict(schema=SCHEMA, verify_mode=ALL8, bank_rule='per_seed_v1', rule=RULE, verification_id=vid, arm=arm,
               policy_version=version, t0=rc['t0'], K=len(plans), first_seed_status=rc['status'],
               first_seed_advantage_veh_s=rc.get('grade_veh_s'),
               first_seed=dict(verification_id=vid, receipt_sha256=rc['receipt_sha256'], branches_dir=str(first_dir)),
               seed2=dict(verification_id=s2['verification_id'], status=s2['status'], salt=vid,
                          snapshot_sha256=s2.get('snapshot_sha256'), receipt_sha256=s2['receipt_sha256'],
                          root=str(s2_dir), J_policy_veh_s=s2.get('J_policy_veh_s')),
               extended=(dict(verification_id=ext['verification_id'], status=ext['status'],
                              receipt_sha256=ext['receipt_sha256'], candidate_indices=lb1)
                         if ext is not None else None),
               candidates=[dict(index=i, plan_id=p['id'], plan_sha256=digest(p), members=list(p['members']),
                                duration_s=int(p['duration_s'])) for i, p in enumerate(plans)],
               winners=winners, distinct_winners_across_seeds=set(winners['1']) != set(winners['2']),
               admitted=bool(adm),
               mean_replicated_advantage_veh_s=(sum(e['advantage_veh_s'] for e in adm)/len(adm) if adm else None),
               per_seed_v1=dict(examples=examples, admitted_seeds=sorted({e['seed_index'] for e in adm}),
                                cooperative_seeds=coop_seeds, replicated=bool(replicated),
                                replicated_plans=replicated,
                                reason='; '.join(f'{e["example_id"]}: {e["reason"]}' for e in examples)),
               cooperation=dict(group_size=max(group_sizes) if group_sizes else 0, cooperative=bool(coop),
                                cooperative_seeds=coop_seeds, cooperative_examples=len(coop),
                                replicated=bool(replicated), needs_second_seed_solos=False,
                                rule='within-seed synergy per admitted (plan, seed); never pooled'),
               charges=charges, charged_sim_s=sum(c['charged_sim_s'] or 0 for c in charges),
               actual_sim_s=sum(c['actual_sim_s'] or 0 for c in charges), unix=time.time())
    xg._write_new(target, xg._seal(out, 'confirmation_sha256'))
    return json.loads(target.read_text())


# ---------------------------------------------------------------- readers (per_seed_v1 integration)

def records(root, arm):
    """{vid: sealed all8 record}; each record IS a per_seed_v1 view (bank_rule='per_seed_v1')."""
    from . import replay_bank as rb
    from .exact_grade import verify_sealed
    out = {}
    for p in sorted((Path(rb.bank_root(root, arm))/RECORDS).glob('*.json')):
        v = json.loads(p.read_text());verify_sealed(v, 'confirmation_sha256')
        out[v['verification_id']] = v
    return out


def overlay(conf, root, arm):
    """per_seed_v1 integration: all8 records replace that decision's legacy confirmation view
    (added for decisions without one, e.g. first-seed-rejected decisions with a seed-2 winner)."""
    out = dict(conf)
    out.update(records(root, arm))
    return out


def bank_examples(root, arm):
    """Admitted (plan, seed) imitation examples, standalone: each carries its own plan, t0,
    seed-specific tape location/digest, advantage and within-seed cooperative label."""
    rows = []
    for vid, c in sorted(records(root, arm).items()):
        for e in c['per_seed_v1']['examples']:
            if e['admitted']:
                rows.append(dict(e, verification_id=vid, t0=c['t0'], policy_version=c['policy_version']))
    return rows


def imitation_entry(example, t0, weight):
    """The (tape, entry) that per_seed_v1 bank_pairs turns into imitation pairs for one example
    (uses the example's OWN plan, not the first-seed chosen plan)."""
    from .accel_storage import read
    from .protocol import digest
    t = example['tape']
    tape = read(Path(t['branches_dir'])/t['verification_id']/t['branch']['branch']/'RESULT.json')
    if t['branch'].get('result_sha256') and digest(tape) != t['branch']['result_sha256']:
        raise ValueError('bank branch tape differs from its sealed verification receipt')
    return tape, dict(start_s=int(t0), duration_s=int(example['duration_s']), plan_id=example['plan_id'],
                      weight=weight)


# ---------------------------------------------------------------- pending driver

def pending(root, arm, sources=()):
    """Completed first-seed decisions (accepted AND rejected) without an all8 record, oldest first."""
    from . import replay_bank as rb
    done = set(records(root, arm))
    return [r for r in rb.bank_decisions(root, arm, sources=sources)
            if r['status'] in ('accepted', 'rejected') and r['verification_id'] not in done]


def confirm_pending_all8(root, arm, runner_for, run_root, phase, workers=2, sources=(), limit=None, health=None):
    from . import replay_bank as rb
    by_vid = {}
    for p in (Path(rb.bank_root(root, arm))/'decisions').glob('*.json'):
        v = json.loads(p.read_text());by_vid[v['verification_id']] = v
    out = []
    for r in pending(root, arm, sources):
        if limit is not None and len(out) >= limit:
            break
        snap = rb._decision_snapshot(root, arm, r, by_vid)
        out.append(confirm_all8(root, arm, r, snap, runner_for, run_root, phase, workers,
                                r['_policy_version'], health=health)['admitted'])
    return out


def main(argv=None):
    """python -m tools.coopmwm.verify_perseed quote  (no SUMO; prints the per-decision quote)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('quote',))
    ap.parse_args(argv)
    print(json.dumps(standard_quote(), indent=1))


if __name__ == '__main__':
    main()
