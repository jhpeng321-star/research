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


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from .protocol import checked, sha
    from .storage import read, write
    p = argparse.ArgumentParser()
    p.add_argument('--receipts', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = checked(Path(a.receipts))
    paths = {kind+'_'+arm: root/f'{stem}_u010_{arm}.json'
             for kind, stem in (('lesson', 'LESSON_CHECK'), ('synergy', 'POLICY_SYNERGY'))
             for arm in ('B', 'R')}
    value = receipt_scenarios({k: read(checked(path)) for k, path in paths.items()})
    value['source_sha256'] = {k: sha(checked(path)) for k, path in paths.items()}
    target = checked(Path(a.out))
    if target.exists():
        if read(target) != value:
            raise ValueError('preserve existing quote; use a new receipt name')
    else:
        write(target, value)
    print(json.dumps(dict(receipt=str(target), sha256=sha(target),
        three_arms_sim_s=value['three_arms']['frozen_slate_total_sim_s'],
        two_arms_sim_s=value['two_arm_fallback']['frozen_slate_total_sim_s'],
        three_arms_within_ceiling=value['three_arms']['within_additional_ceiling'],
        two_arms_within_ceiling=value['two_arm_fallback']['within_additional_ceiling'],
        two_arms_reference_reuse_sim_s=value['two_arm_reference_reuse']['frozen_slate_total_sim_s'],
        reference_reuse_qualification_required=True,
        new_sim_s=0)))
