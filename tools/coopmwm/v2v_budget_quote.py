"""Preparation-only V2V workload quote; never reserves or dispatches physics.

Diagnostic entries are public TRAIN metadata: end_s includes the deterministic
prefix, and members is the cooperative group's size. Bank growth is unbounded,
so a frozen-slate quote is a scenario, never an all-future-bank upper bound.
"""


def quote(*, lessons, cooperative, updates=24, arms=('A', 'B', 'R'),
          paired_arms=(), original_synergy=False, reference_reuse=False):
    """Quote common-MAPPO-u0 V2V arms with explicit diagnostic slates.

    lessons/cooperative map A/B/R to lists of public {end_s, members} metadata.
    MAIN13:45 requires isolated synergy. The original variant is optional and
    separately disclosed. Conservatively charge a fresh reference per state;
    a future SHA-verified lesson/reference adoption may reduce actual spending.
    No reference or outcome is opened here, and no adoption is presumed.
    """
    if updates != 24:
        raise ValueError('MAIN13:43 fixes common MAPPO u0, then u1 through u24')
    arms = tuple(arms)
    if arms not in (('A', 'B', 'R'), ('A', 'B')):
        raise ValueError('MAIN13:45 permits all three arms or A/B fallback only')
    if len(set(paired_arms)) != len(paired_arms) or not set(paired_arms) <= set(arms):
        raise ValueError('paired branches require explicit selected-arm scope')
    if set(lessons) != set(arms) or set(cooperative) != set(arms):
        raise ValueError('explicit diagnostic slates required for every arm')

    def rows(items, coop=False):
        total = 0
        for item in items:
            end, members = item['end_s'], item.get('members')
            if type(end) is not int or not 300 < end <= 1200:
                raise ValueError('full-prefix branch end must be within episode')
            if (coop or members is not None) and (type(members) is not int or not 2 <= members <= 6):
                raise ValueError('registered group size is 2..6')
            # Isolated: all + n solos + background + fresh reference.
            # With original: add original all + n solos; reference shared.
            multiplier = (2 * members + 4 if original_synergy else members + 3) if coop else 1
            if coop and reference_reuse:
                multiplier -= 1
            total += end * multiplier
        return total

    episode_s = 1200
    training = len(arms) * updates * 24 * episode_s
    interim = len(arms) * 96 * episode_s
    final_episodes = len(arms) * 96 * episode_s
    # Existing paired-branch plan: four branch times, full-prefix reruns.
    # MAPPO references reused subject to their existing equivalence gate.
    paired_per_arm = 96 * sum(t + 300 for t in (300, 500, 700, 900))
    paired = len(paired_arms) * paired_per_arm
    lesson = {a: updates * rows(lessons[a]) for a in arms}
    synergy = {a: updates * rows(cooperative[a], True) for a in arms}
    fixed = training + interim + final_episodes + paired
    total = fixed + sum(lesson.values()) + sum(synergy.values())
    return dict(
        authority='MAIN2026-11-03 13:43/13:45 conditional launch authority',
        arms=[a+'-v2v' for a in arms], common_start='original MAPPO u0 plus zero-init V2V',
        first_update=1, final_update=24,
        phase_costs=dict(train=training, interim_u8_test=interim,
                         final_test=final_episodes, paired_branch=paired,
                         lessons=lesson, policy_synergy=synergy),
        fixed_sim_s=fixed,
        frozen_slate_total_sim_s=total,
        additional_cap_ceiling_sim_s=3000000,
        within_additional_ceiling=total <= 3000000,
        cap_if_funded_sim_s=14500000+total,
        diagnostic_slate_counts=dict(
            lessons={a: len(lessons[a]) for a in arms},
            cooperative={a: len(cooperative[a]) for a in arms}),
        paired_branch_assumption=dict(arms=list(paired_arms),
            rule='optional explicitly requested full-prefix final branches; existing MAPPO references reused'),
        diagnostic_rule=(('full prefix; original and isolated variants' if original_synergy
                         else 'full prefix; isolated all/solo/background')
                         + ('; SHA-qualified reference adoption' if reference_reuse else '; fresh reference')),
        reference_reuse=dict(enabled=reference_reuse,
            gate='only SHA-matched same state/window/prefix/CRN baseline; otherwise stop before physics'),
        growth_rule=('each additional lesson adds remaining_updates*end_s; each additional '
                     'cooperative state adds remaining_updates*(n+3)*end_s for isolated only '
                     'or remaining_updates*(2*n+4)*end_s with original variant'),
        unknown_costs=['future admitted-bank growth', 'new V2V bank verification or confirmation',
                       'MAPPO branch-reuse failure or failed/replacement jobs'],
        budget_authorized=False, launch_authorized=False, reservation_sim_s=0,
        new_sim_s=0, is_full_run_upper_bound=False)


def receipt_scenarios(reports):
    """Metadata-only workload scenarios; A is checked on B's diagnostic states.

    This is an explicit quote assumption for MAIN, not an activation or a change
    to the V2V integrator's eventual runner. Original branches are not included.
    """
    lessons, cooperative = {}, {}
    for a in ('B', 'R'):
        lesson, synergy = reports['lesson_'+a], reports['synergy_'+a]
        if lesson['arm'] != a or synergy['arm'] != a:
            raise ValueError('diagnostic arm identity mismatch')
        if any(e.get('status') != 'graded' for e in lesson['entries']):
            raise ValueError('complete frozen lesson slate required')
        if any(e.get('status', 'graded') != 'graded' for e in synergy['entries']):
            raise ValueError('complete cooperative metadata slate required; never omit failed states')
        lessons[a] = [dict(end_s=e['new_policy_branch']['window']['end_s'])
                      for e in lesson['entries']]
        cooperative[a] = [dict(end_s=e['window']['end_s'], members=len(e['group']))
                          for e in synergy['entries']]
    # A is a plain PPO arm. This budgets checks, never imitation from B's bank.
    lessons['A'], cooperative['A'] = lessons['B'], cooperative['B']
    return dict(
        three_arms=quote(lessons=lessons, cooperative=cooperative),
        two_arm_fallback=quote(lessons={a: lessons[a] for a in ('A', 'B')},
            cooperative={a: cooperative[a] for a in ('A', 'B')}, arms=('A', 'B')),
        two_arm_reference_reuse=quote(lessons={a: lessons[a] for a in ('A', 'B')},
            cooperative={a: cooperative[a] for a in ('A', 'B')}, arms=('A', 'B'), reference_reuse=True),
        scenario_assumptions=[
            'A diagnostics on the same frozen B slate; no B imitation for A',
            'isolated synergy only; all/solos/background plus fresh reference',
            'current cooperative slate held fixed for workload projection',
            'new bank admissions require further explicit guarded diagnostic room'],
        launch_authorized=False, phase_registration_complete=False, new_sim_s=0)


def reduced_schedule_quote(reports, *, reference_reuse=False, drop_r_u8=False,
                           b_diagnostic_updates=(6, 12, 18, 24)):
    """MAIN14:06: keep three arms, reduce diagnostics before dropping any tests.

    Costs use public TRAIN branch metadata only. Baseline reuse is a separate,
    conditional scenario; no qualification is inferred from a quoted saving.
    """
    schedule = tuple(b_diagnostic_updates)
    if schedule not in ((6, 12, 18, 24), (12, 24)):
        raise ValueError('only MAIN14:06 diagnostic fallback is authorised')
    # Reuse the legacy metadata validation and complete full-prefix arithmetic.
    legacy = receipt_scenarios(reports)['three_arms']
    per_lesson = {a: n // 24 for a, n in legacy['phase_costs']['lessons'].items()}
    per_synergy = {a: n // 24 for a, n in legacy['phase_costs']['policy_synergy'].items()}
    if reference_reuse:
        for arm in ('B', 'R'):
            per_synergy[arm] -= sum(e['window']['end_s']
                                   for e in reports['synergy_'+arm]['entries'])
    lessons = dict(A=0, B=len(schedule)*per_lesson['B'], R=0)
    synergy = dict(A=0, B=len(schedule)*per_synergy['B'], R=2*per_synergy['R'])
    interim = (2 if drop_r_u8 else 3)*96*1200
    fixed = 3*24*24*1200 + interim + 3*96*1200
    total = fixed + sum(lessons.values()) + sum(synergy.values())
    return dict(
        authority='MAIN2026-11-03 14:06; all three arms, reduced diagnostics',
        arms=['A-v2v', 'B-v2v', 'R-v2v'],
        common_start='original MAPPO u0 plus zero-init V2V', first_update=1, final_update=24,
        phase_costs=dict(train=2073600, interim_u8_test=interim, final_test=345600,
                         paired_branch=0, lessons=lessons, policy_synergy=synergy),
        diagnostic_updates=dict(lessons=dict(A=[], B=list(schedule), R=[]),
            isolated_synergy=dict(A=[], B=list(schedule), R=[12, 24]), composite_safety='every update'),
        episode_test_updates=dict(A=[8, 24], B=[8, 24], R=[24] if drop_r_u8 else [8, 24]),
        fixed_sim_s=fixed, diagnostic_sim_s=sum(lessons.values())+sum(synergy.values()),
        frozen_slate_total_sim_s=total, additional_cap_ceiling_sim_s=3000000,
        within_additional_ceiling=total <= 3000000, cap_if_funded_sim_s=14500000+total,
        diagnostic_slate_counts=dict(lessons=dict(A=0, B=len(reports['lesson_B']['entries']), R=0),
            cooperative=dict(A=0, B=len(reports['synergy_B']['entries']),
                             R=len(reports['synergy_R']['entries']))),
        reference_reuse=dict(enabled=reference_reuse,
            gate='only SHA-matched same state/window/prefix/CRN baseline; otherwise stop before physics'),
        diagnostic_rule='isolated all/solo/background; fresh reference unless explicitly SHA-qualified',
        fallback_drops=(['R-v2v u8 episode test'] if drop_r_u8 else [])
            + (['B-v2v u6/u18 lesson and isolated synergy'] if schedule == (12, 24) else []),
        growth_rule='before every scheduled diagnostic prequote complete current slate against its shared phase; no implicit cap raise or partial branch set',
        unknown_costs=['future cooperative-bank growth beyond quoted slate',
            'new V2V bank verification or confirmation', 'failed/replacement jobs'],
        budget_authorized=False, launch_authorized=False, reservation_sim_s=0, new_sim_s=0,
        is_full_run_upper_bound=False)


def reduced_schedule_scenarios(reports):
    """Ordered outcome-independent fallbacks; retain all three training arms."""
    scenarios = dict(
        full_schedule_fresh=reduced_schedule_quote(reports),
        full_schedule_reference_reuse=reduced_schedule_quote(reports, reference_reuse=True),
        drop_r_u8_fresh=reduced_schedule_quote(reports, drop_r_u8=True),
        drop_r_u8_reference_reuse=reduced_schedule_quote(reports, drop_r_u8=True, reference_reuse=True),
        drop_r_u8_and_b6_b18_fresh=reduced_schedule_quote(reports, drop_r_u8=True,
            b_diagnostic_updates=(12, 24)),
        drop_r_u8_and_b6_b18_reference_reuse=reduced_schedule_quote(reports, drop_r_u8=True,
            b_diagnostic_updates=(12, 24), reference_reuse=True))
    # Fresh references are immediately cost-qualified; reuse still needs an
    # independent identity gate. Never present a conditional scenario as funded.
    selected = next((name for name, q in scenarios.items()
        if q['within_additional_ceiling'] and not q['reference_reuse']['enabled']), None)
    return dict(scenarios=scenarios, cost_feasible_without_reuse=selected,
        launch_authorized=False, phase_registration_complete=False, new_sim_s=0)


def live_workload_metadata():
    """Read-only current TRAIN slate; no model/physics calls or outcome export."""
    import json
    from .ls_validity_ops import remote
    from .ls_collect_ops import ROOT, COLLECTION
    return json.loads(remote('cpuQ', f'''import json,sys,time
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/coopmwm20261103_compositereleasecpu1/pytools')
from tools.coopmwm import replay_bank as rb
from tools.coopmwm.protocol import sha
from tools.coopmwm.exact_grade import verify_sealed
r=Path({ROOT!r});out={{'reports':{{}},'source_sha256':{{}},'unix':time.time(),'count_checks':{{}}}}
for arm in ('B','R'):
 sources=[{COLLECTION!r}] if arm=='B' else []
 decisions=rb.bank_decisions(r,arm,sources=sources);conf=rb.confirmations(r,arm)
 slate=r/'replay_bank'/arm/('LESSON_LIST_'+arm+'.json');lesson=json.loads(slate.read_text())
 verify_sealed(lesson,'lesson_list_sha256');wanted=set(lesson['verification_ids'])
 chosen=[x for x in decisions if x['verification_id'] in wanted]
 if {{x['verification_id'] for x in chosen}}!=wanted:raise ValueError('lesson metadata incomplete')
 coop=[x for x in decisions if conf.get(x['verification_id'],{{}}).get('admitted') and conf[x['verification_id']].get('cooperation',{{}}).get('cooperative') is True]
 quality=json.loads((r/'replay_bank'/arm/'BANK_QUALITY.json').read_text())
 expected={{vid for vid,c in conf.items() if c.get('admitted') and c.get('cooperation',{{}}).get('cooperative') is True}}
 actual={{x['verification_id'] for x in coop}}
 out['count_checks'][arm]={{'quality_cooperative':quality['cooperative'],'metadata_cooperative':len(coop),'missing_ids':sorted(expected-actual),'quality_unix':quality['unix']}}
 if expected!=actual or quality['cooperative']!=len(coop):raise ValueError('current cooperative slate mismatch: '+json.dumps(out['count_checks']))
 out['reports']['lesson_'+arm]={{'arm':arm,'entries':[{{'status':'graded','new_policy_branch':{{'window':x['window']}}}} for x in chosen]}}
 out['reports']['synergy_'+arm]={{'arm':arm,'entries':[{{'status':'graded','window':x['window'],'group':x['chosen_plan']['members']}} for x in coop]}}
 out['source_sha256']['lesson_list_'+arm]=sha(slate)
 for x in coop:
  out['source_sha256'][x['verification_id']]=x['receipt_sha256']
out.update(new_sim_s=0,new_inference=0,scope='all currently admitted cooperative TRAIN entries; fixed own-arm lesson slates; metadata only')
print(json.dumps(out))''', timeout=60))


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from .protocol import checked, sha
    from .storage import read, write
    p = argparse.ArgumentParser()
    p.add_argument('--receipts', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scope', choices=('historical1345', 'main1406'), default='historical1345')
    p.add_argument('--update', type=int, default=10)
    p.add_argument('--live-metadata', action='store_true')
    a = p.parse_args()
    root = checked(Path(a.receipts))
    paths = {kind+'_'+arm: root/f'{stem}_u{a.update:03}_{arm}.json'
             for kind, stem in (('lesson', 'LESSON_CHECK'), ('synergy', 'POLICY_SYNERGY'))
             for arm in ('B', 'R')}
    builder = reduced_schedule_scenarios if a.scope == 'main1406' else receipt_scenarios
    if a.live_metadata:
        if a.scope != 'main1406':raise ValueError('live metadata only for latest reduced schedule')
        meta = live_workload_metadata()
        value = builder(meta['reports'])
        value['metadata_snapshot'] = meta
        value['source_sha256'] = meta['source_sha256']
    else:
        value = builder({k: read(checked(path)) for k, path in paths.items()})
        value['source_sha256'] = {k: sha(checked(path)) for k, path in paths.items()}
    target = checked(Path(a.out))
    if target.exists():
        if read(target) != value:
            raise ValueError('preserve existing quote; use a new receipt name')
    else:
        write(target, value)
    if a.scope == 'main1406':
        print(json.dumps(dict(receipt=str(target), sha256=sha(target),
            cost_feasible_without_reuse=value['cost_feasible_without_reuse'],
            scenarios={k:dict(total_sim_s=q['frozen_slate_total_sim_s'],
                cap_if_funded=q['cap_if_funded_sim_s'], within_ceiling=q['within_additional_ceiling'])
                for k,q in value['scenarios'].items()}, new_sim_s=0)))
    else:
        print(json.dumps(dict(receipt=str(target), sha256=sha(target),
        three_arms_sim_s=value['three_arms']['frozen_slate_total_sim_s'],
        two_arms_sim_s=value['two_arm_fallback']['frozen_slate_total_sim_s'],
        three_arms_within_ceiling=value['three_arms']['within_additional_ceiling'],
        two_arms_within_ceiling=value['two_arm_fallback']['within_additional_ceiling'],
        two_arms_reference_reuse_sim_s=value['two_arm_reference_reuse']['frozen_slate_total_sim_s'],
        reference_reuse_qualification_required=True,
            new_sim_s=0)))
