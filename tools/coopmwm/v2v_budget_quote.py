"""Preparation-only V2V workload quote; never reserves or dispatches physics.

Diagnostic entries are public TRAIN metadata: end_s includes the deterministic
prefix, and members is the cooperative group's size. Bank growth is unbounded,
so a frozen-slate quote is a scenario, never an all-future-bank upper bound.
"""


def quote(*, lessons, cooperative, updates=21, include_a_branch=True):
    """Quote the three u3->u24 arms with explicit diagnostic slates.

    lessons/cooperative map A/B/R to lists of public {end_s, members} metadata.
    Original and isolated synergy variants are both included. Conservatively
    charge a fresh all-new and u3-reference branch for every cooperative state;
    a future SHA-verified lesson/reference adoption may reduce actual spending.
    No reference or outcome is opened here, and no adoption is presumed.
    """
    if updates != 21:
        raise ValueError('MAIN13:24 fixes u4 through u24')
    if set(lessons) != {'A', 'B', 'R'} or set(cooperative) != {'A', 'B', 'R'}:
        raise ValueError('explicit diagnostic slates required for every arm')

    def rows(items, coop=False):
        total = 0
        for item in items:
            end, members = item['end_s'], item['members']
            if type(end) is not int or not 300 < end <= 1200:
                raise ValueError('full-prefix branch end must be within episode')
            if type(members) is not int or not 2 <= members <= 6:
                raise ValueError('registered group size is 2..6')
            # Original: all + n solos + u3 ref. Isolated: all + n solos
            # + background, adopting that exact original u3 reference.
            total += end * (2 * members + 4 if coop else 1)
        return total

    episode_s = 1200
    training = 3 * updates * 24 * episode_s
    interim = 3 * 96 * episode_s
    final_episodes = 3 * 96 * episode_s
    # Existing paired-branch plan: four branch times, full-prefix reruns.
    # MAPPO references reused subject to their existing equivalence gate.
    paired_per_arm = 96 * sum(t + 300 for t in (300, 500, 700, 900))
    paired = (2 if include_a_branch else 0) * paired_per_arm
    lesson = {a: updates * rows(lessons[a]) for a in ('A', 'B', 'R')}
    synergy = {a: updates * rows(cooperative[a], True) for a in ('A', 'B', 'R')}
    fixed = training + interim + final_episodes + paired
    return dict(
        authority='MAIN2026-11-03 13:24/13:32 PREPARE ONLY',
        arms=['A-v2v', 'B-v2v', 'R-v2v'], first_update=4, final_update=24,
        phase_costs=dict(train=training, interim_u8_test=interim,
                         final_test=final_episodes, paired_branch=paired,
                         lessons=lesson, policy_synergy=synergy),
        fixed_sim_s=fixed,
        frozen_slate_total_sim_s=fixed + sum(lesson.values()) + sum(synergy.values()),
        diagnostic_slate_counts=dict(
            lessons={a: len(lessons[a]) for a in ('A', 'B', 'R')},
            cooperative={a: len(cooperative[a]) for a in ('A', 'B', 'R')}),
        paired_branch_assumption=('A-v2v and B-v2v final only; existing MAPPO '
                                  'references reused; R paired branches deferred'
                                  if include_a_branch else 'not included; scope decision pending'),
        diagnostic_rule='full prefix per branch; both original and isolated variants',
        growth_rule=('each additional lesson adds 21*end_s; each additional '
                     'cooperative state adds 21*(2*n+4)*end_s at most across all updates'),
        unknown_costs=['future admitted-bank growth', 'new V2V bank verification or confirmation',
                       'MAPPO branch-reuse failure or failed/replacement jobs'],
        budget_authorized=False, launch_authorized=False, reservation_sim_s=0,
        new_sim_s=0, is_full_run_upper_bound=False)
