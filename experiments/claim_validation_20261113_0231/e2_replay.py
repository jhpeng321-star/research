"""E2 stage 2: same-state timing replay WITHOUT saving a mid-episode state (zero SUMO here; specs only).

Idea. SUMO with a fixed world seed and the frozen base policy is reproducible (identity receipts A67 / A68: stored
J reproduced with difference 0.0, same activations, decompressed vehicle logs identical). The executor acts only
when a plan starts, so before the first plan launch of an episode the run IS the U0 run. Hence, for one world:

  discover   run the world from t = 0 under the ORIGINAL bank and read ONLY the executor receipts (launch time,
             family, members). First launch = (t*, P, members). Used only when t* <= 600 s.
  branches   three runs of the same world from t = 0, identical up to t*:
               U0      no plan at all                                    (the registered base arm)
               F1      ONE_<P>     = plan P with cooldown 100000 s, alone (runner arm A37_PLAN)
               S1      ONE_S_<P>   = the onset-synchronised copy of P with cooldown 100000 s, alone
             A plan alone launches at its first feasible second. P was not feasible before t* (it would have
             launched under the bank), and member selection ignores onsets, so F1 and S1 both launch at t* with the
             same members; the cooldown (> the 1200 s episode) forbids a second launch, and no other plan exists in
             the branch. So the three branches differ ONLY in what happens to that one group from t* on.

No executor change is needed: 'cooldown_s' is a registered descriptor field and A37_PLAN is a registered arm.
Changing the cooldown changes the plan's 16-hex key and name, nothing in its hold law (disclosed in the manifest).

State selection rule (no outcome is read): per world, the FIRST launch of the discover run if t* <= 600 s; a world
with no launch by 600 s contributes no state. At most one state per world, at most 24 worlds.

  plan     -> E2_ONESHOT_BANK.json (8 families), REPLAY_PLAN.json (discover waves)          [needs e2_build]
  select   DISCOVER_RESULTS.json (acts only) -> REPLAY_SELECTION.json + branch waves
  check    BRANCH_RESULTS.json -> mechanics verdict per world (one launch, same t* / members, identical prefix)
  analyse  BRANCH_RESULTS.json -> F1 - U0, S1 - U0, F1 - S1 (paired percentile bootstrap over the states)
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import e2_build as B  # noqa: E402
import stats as S  # noqa: E402

ONE_SHOT_COOLDOWN_S = 100000        # > HORIZON_S: the plan can start once per episode (not a behavioural threshold)
LAUNCH_MAX_T = 600                  # protocol: only launches at or before 600 s
MAX_STATES = 24
BLOCK = 'dev'       # Development set (first worlds of the already-viewed a35_D2800_p60 panel)


def one_shot(plan, prefix='ONE_'):
    s = copy.deepcopy(plan['strategy'])
    s['cooldown_s'] = ONE_SHOT_COOLDOWN_S
    s['name'] = f'{prefix}{plan["family"]}'
    return dict(family=s['name'], strategy=s, lb95_veh_s=plan['lb95_veh_s'], source_family=plan.get('source_family',
                                                                                                    plan['family']))


def one_shot_bank(bank, cell=L.CELL):
    """-> (plans F, bank doc, {original family: dict(F1=..., S1=...)}); checks that each pair differs only in onsets
    and that each one-shot differs from its source only in the cooldown."""
    plans = L.bank_strategies(bank, cell)
    rows, pairs = [], {}
    for p in plans:
        f1, s1 = one_shot(p), one_shot(B.sync_copy(p), prefix='ONE_')
        if B.diff_fields(p['strategy'], f1['strategy']) != ['cooldown_s']:
            raise L.Refused(f'{f1["family"]}: one-shot differs from the original in more than the cooldown')
        d = B.diff_fields(f1['strategy'], s1['strategy'])
        if {x.split('.')[-1] for x in d} - {'onset_s'}:
            raise L.Refused(f'{s1["family"]}: differs from {f1["family"]} in more than onsets')
        rows += [f1, s1]
        pairs[p['family']] = dict(F1=f1['family'], S1=s1['family'], onsets_differ=bool(d),
                                  planned=S.planned_resource(p['strategy']))
    return plans, L.bank_doc(rows, cell), pairs


def first_launch(acts, t_max=LAUNCH_MAX_T):
    """The first launch of one discover episode from its executor receipts: acts = [dict(t, family, members)] in
    executor order (time, then priority). None when nothing launched by ``t_max``. Reads nothing else."""
    if not acts:
        return None
    t0 = min(int(a['t']) for a in acts)
    a = next(a for a in acts if int(a['t']) == t0)       # same second: first in executor (priority) order
    if t0 > t_max:
        return None
    return dict(t=t0, family=a['family'], members=list(a['members']))


def select(discover, pairs, max_states=MAX_STATES):
    """discover = {world_id: dict(world_index, acts)} -> the selected states (sorted by world index, first
    ``max_states`` worlds that have one) and the worlds without a state."""
    states, none = [], []
    for wid in sorted(discover, key=lambda w: int(discover[w]['world_index'])):
        fl = first_launch(discover[wid]['acts'])
        if fl is None:
            none.append(wid)
        elif fl['family'] not in pairs:
            raise L.Refused(f'{wid}: launched {fl["family"]}, not a plan of the original bank')
        elif len(states) < max_states:
            states.append(dict(world_id=wid, world_index=int(discover[wid]['world_index']), t_star=fl['t'],
                               family=fl['family'], members=fl['members'], F1=pairs[fl['family']]['F1'],
                               S1=pairs[fl['family']]['S1'], onsets_differ=pairs[fl['family']]['onsets_differ']))
    return states, none


def branch_waves(states):
    """U0 on every selected world + one F1 and one S1 wave per launched plan on that plan's worlds."""
    idx = sorted(s['world_index'] for s in states)
    waves = [dict(wave='E2R_U0', kind='U0', block=BLOCK, world_indices=idx)]
    for fam in sorted({s['family'] for s in states}):
        sub = [s for s in states if s['family'] == fam]
        for br in ('F1', 'S1'):
            waves.append(dict(wave=f'E2R_{br}_{fam}', kind='PLAN', block=BLOCK, bank='E2_ONESHOT_BANK.json',
                              family=sub[0][br], branch=br, world_indices=sorted(s['world_index'] for s in sub)))
    return waves


def check_world(state, res):
    """Mechanics verdict of one replayed state. res = dict(U0=dict(activations, prefix_digest), F1=dict(acts,
    prefix_digest), S1=dict(...)); prefix_digest = stats.log_prefix_digest(vehicle log, t*)."""
    problems = []
    if int(res['U0'].get('activations') or 0) != 0:
        problems.append('U0 branch launched a plan')
    for br in ('F1', 'S1'):
        acts = res[br]['acts']
        if len(acts) != 1:
            problems.append(f'{br}: {len(acts)} launches (must be exactly 1)')
            continue
        a = acts[0]
        if a['family'] != state[br]:
            problems.append(f'{br}: launched {a["family"]}')
        if int(a['t']) != state['t_star']:
            problems.append(f'{br}: launched at {a["t"]}, discover run at {state["t_star"]}')
        if list(a['members']) != list(state['members']):
            problems.append(f'{br}: members differ from the discover run')
        if res[br]['prefix_digest'] != res['U0']['prefix_digest']:
            problems.append(f'{br}: state before t* differs from the U0 branch')
    return dict(world_id=state['world_id'], t_star=state['t_star'], family=state['family'], ok=not problems,
                problems=problems)


def analyse(states, results):
    ok = [s for s in states if check_world(s, results[s['world_id']])['ok']]
    out = dict(n_states=len(states), n_valid=len(ok), convention='diff = treatment - reference; negative = less delay',
               by_family={f: sum(1 for s in ok if s['family'] == f) for f in sorted({s['family'] for s in ok})})
    if len(ok) < 2:
        return dict(out, status='unavailable (fewer than 2 valid states)')
    J = {br: [float(results[s['world_id']][br]['J']) for s in ok] for br in ('U0', 'F1', 'S1')}
    out.update(F1_minus_U0=S.paired_bootstrap(J['F1'], J['U0']), S1_minus_U0=S.paired_bootstrap(J['S1'], J['U0']),
               F1_minus_S1=S.paired_bootstrap(J['F1'], J['S1']),
               primary='F1_minus_S1 (timing effect: the staggered onsets against their synchronised copy, same state)')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('plan', 'select', 'check', 'analyse'))
    p.add_argument('--bank', help='plan: the original four-plan bank (sha 7e7dc76b checked)')
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--n-worlds', type=int, default=MAX_STATES, help='plan: worlds of the registered replay block')
    p.add_argument('--results', help='select: DISCOVER_RESULTS.json; check / analyse: BRANCH_RESULTS.json')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    d = Path(a.out)/'e2'/'replay'

    def put(path, text):
        return 'dry-run' if a.dry_run else L.write_once(path, text)
    if a.command == 'plan':
        sha = L.sha_file(a.bank)
        if sha != L.ORIGINAL_BANK_SHA256:
            raise L.Refused(f'{a.bank}: not the registered original bank')
        if not 1 <= a.n_worlds <= MAX_STATES:
            raise L.Refused(f'the replay uses at most {MAX_STATES} worlds')
        plans, bank, pairs = one_shot_bank(L.read_json(a.bank))
        text = L.text_of(bank)
        doc = L.seal(dict(
            schema='claim_validation.e2.replay_plan.v1', run_id=L.RUN_ID, cell=L.CELL,
            original_bank=dict(name=Path(a.bank).name, sha256=sha),
            one_shot_bank=dict(name='E2_ONESHOT_BANK.json', sha256=L.sha_bytes(text.encode()),
                               cooldown_s=ONE_SHOT_COOLDOWN_S,
                               disclosure='cooldown changed -> new strategy key / name; hold law unchanged'),
            pairs=pairs, launch_max_t=LAUNCH_MAX_T, max_states=MAX_STATES,
            selection_rule='first launch of the discover run if t* <= 600 s; reads executor receipts only',
            discover_waves=[dict(wave='E2R_DISCOVER', kind='BANK', block=BLOCK, n_worlds=a.n_worlds,
                                 bank=Path(a.bank).name, bank_sha256=sha)],
            sim_s=dict(discover=a.n_worlds*L.HORIZON_S, branches_max=3*a.n_worlds*L.HORIZON_S,
                       total_max=4*a.n_worlds*L.HORIZON_S)))
        rep = dict(bank=put(Path(a.out)/'e2'/'banks'/'E2_ONESHOT_BANK.json', text),
                   plan=put(d/'REPLAY_PLAN.json', L.text_of(doc)), pairs=pairs, sim_s=doc['sim_s'])
    elif a.command == 'select':
        plan = L.verify_seal(L.read_json(d/'REPLAY_PLAN.json'))
        states, none = select(L.read_json(a.results), plan['pairs'])
        doc = L.seal(dict(schema='claim_validation.e2.replay_selection.v1', run_id=L.RUN_ID,
                          replay_plan_sha256=plan['doc_sha256'], discover_sha256=L.sha_file(a.results),
                          rule=plan['selection_rule'], states=states, worlds_without_state=none,
                          waves=branch_waves(states), sim_s=sum(len(w['world_indices']) for w in branch_waves(states))
                          * L.HORIZON_S))
        rep = dict(selection=put(d/'REPLAY_SELECTION.json', L.text_of(doc)), n_states=len(states),
                   no_state=len(none), by_family={f: sum(1 for s in states if s['family'] == f)
                                                  for f in sorted({s['family'] for s in states})}, sim_s=doc['sim_s'])
    else:
        sel = L.verify_seal(L.read_json(d/'REPLAY_SELECTION.json'))
        res = L.read_json(a.results)
        rows = [check_world(s, res[s['world_id']]) for s in sel['states']]
        if a.command == 'check':
            doc = L.seal(dict(schema='claim_validation.e2.replay_mechanics.v1', run_id=L.RUN_ID,
                              selection_sha256=sel['doc_sha256'], results_sha256=L.sha_file(a.results), worlds=rows,
                              verdict='PASS' if rows and all(r['ok'] for r in rows) else 'FAIL'))
            rep = dict(mechanics=put(d/'REPLAY_MECHANICS.json', L.text_of(doc)), verdict=doc['verdict'],
                       failed=[r for r in rows if not r['ok']])
        else:
            doc = L.seal(dict(schema='claim_validation.e2.replay_analysis.v1', run_id=L.RUN_ID,
                              selection_sha256=sel['doc_sha256'], results_sha256=L.sha_file(a.results),
                              excluded=[r for r in rows if not r['ok']], **analyse(sel['states'], res)))
            rep = dict(analysis=put(d/'REPLAY_ANALYSIS.json', L.text_of(doc)), n_valid=doc['n_valid'],
                       F1_minus_S1=doc.get('F1_minus_S1'))
    print(L.text_of(dict(stage=f'e2_replay {a.command}', dry_run=a.dry_run, **rep)), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
