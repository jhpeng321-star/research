"""DRY EVAL HARNESS ONLY (owner 2026-11-04, Track 1 T1D, branch c4/planner-eval): the registered
paired evaluation MAPPO vs MAPPO+planner as a BUILD + DRY-RUN harness. It produces the sealed
evaluation plan, the paired job specs (paired worlds: every seed runs in BOTH arms), the budget
quote, the sealed receipt schema, the safety counters (collisions, teleports, TTC, hard braking)
and the paired analysis. It NEVER dispatches, reserves, charges or runs SUMO: every quote reports
new_sim_s=0 and dry_run writes plan + jobs only. The seed list comes ONLY from the registration
file passed on the command line - nothing is hard-coded here and no seed registry is opened; any
path containing sealed/holdout/heldout/cal_split/chunk5/j4b/inc1, any hetdata_truth or
vType/driver/vehicle_params truth file is refused everywhere (guard_path). The planner values the
plan pins are the registered T1B numbers: configurable decision cadence (60 s default), fire only when the CALIBRATED
P(move beats MAPPO over the FULL look-ahead horizon) >= p_star = 0.8 (owner rule 10-18), horizon
never shortened. The analysis is the registered estimator reused from branch_eval (10,000
whole-world paired resamples, numpy default_rng(20261101), percentile 95% CI, two-sided CI
inversion p), plus the per-world safety gate (collisions/teleports hard constraints; the
registered MAIN 2026-11-03 04:26 combined hard_brake + TTC<1.5 s rule from replay_bank._safety).
Receipts seal with the exact_grade scheme and verify with exact_grade.verify_sealed unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

SCHEMA = 'coopmwm/planner_eval/v1'
REGISTRATION_SCHEMA = 'coopmwm/planner_eval/registration/v1'
RECEIPT_SCHEMA = 'coopmwm/planner_eval/receipt/v1'
ANALYSIS_SCHEMA = 'coopmwm/planner_eval/analysis/v1'
EXPERIMENT = 'S-COOPMWM-PLANNER-EVAL'

ARMS = ('MAPPO', 'PLANNER')       # comparator (control), treatment (MAPPO + WM planner gate)
TREATMENT, COMPARATOR = 'PLANNER', 'MAPPO'
METRIC = 'team_delay_vs'          # registered comparator J (veh-s); paired d = PLANNER - MAPPO

EPISODE_END_S = 1200              # registered episode length (rerun_cpus.EPISODE_SIM_S)
DECISION_INTERVAL_S = 60          # default planner decision cadence (Track 1 T1B; configurable)
P_STAR = 0.8                      # registered confidence gate (owner rule 10-18)
LOOKAHEAD_S = 300                 # registered full WM look-ahead horizon - NEVER shortened
MOVE_FAMILIES = ('ease_off', 'wall', 'gap_open', 'ramp_assist', 'none')  # T1A shared contract
MOVE_FAMILY_ALIASES = {'free': 'ease_off', 'smoothing': 'ease_off'}  # legacy variety/boundary tags
SAFETY_KEYS = ('collisions', 'teleports', 'ttc_below_3_vs', 'ttc_below_1p5_events', 'hard_brake')
COMBINED_KEYS = ('hard_brake', 'ttc_below_1p5_events')   # replay_bank MAIN 2026-11-03 04:26 total
HARD_KEYS = ('collisions', 'teleports')

BOOT_SEED = 20261101              # registered analysis RNG (branch_eval)
N_BOOT = 10000
ALPHA = 0.05

PLAN_FILE = 'PLANNER_EVAL_PLAN.json'
JOBS_FILE = 'PLANNER_EVAL_JOBS.jsonl'

FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth',
             'vtype', 'vehicle_params', 'driver')
_TRUTH = re.compile(r'(vtype|driver|vehicle_params)[^/]*truth|truth[^/]*(vtype|driver|vehicle_params)',
                    re.IGNORECASE)
REGISTRATION_KEYS = ('schema', 'worlds', 'planner', 'episode', 'analysis', 'calibration', 'note')
WORLD_KEYS = ('seed', 'demand', 'penetration')
PLANNER_KEYS = ('decision_interval_s', 'p_star', 'lookahead_s')
PLANNER_OPTIONAL = ('k_candidates', 'wm_samples', 'groups_per_decision')
EPISODE_KEYS = ('end_s',)
ANALYSIS_KEYS = ('n_boot', 'boot_seed', 'alpha')
CALIBRATION_KEYS = ('path', 'sha256')
TRACE_KEYS = ('t_s', 'group_id', 'p_better', 'fired', 'family', 'members', 'lookahead_s')
OUTCOME_KEYS = ('J', 'safety', 'planner_trace', 'executor', 'result_ref')
RECEIPT_KEYS = ('schema', 'experiment', 'plan_sha256', 'job_id', 'seed', 'arm', 'planner_enabled',
                'metric', 'J', 'safety', 'planner_trace', 'episode_end_s', 'executor', 'result_ref',
                'receipt_sha256')


# ---------------------------------------------------------------- canonical + sealing

def canonical(value):
    # Identical to tools.coopnative.protocol.canonical (registered digests, as in branch_eval).
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seal(record, key):
    body = {k: v for k, v in record.items() if k != key}
    return dict(body, **{key: digest(body)})


def verify_seal(record, key):
    body = {k: v for k, v in record.items() if k != key}
    if record.get(key) != digest(body):
        raise ValueError(f'{key} mismatch (edited after sealing?)')
    return True


# ---------------------------------------------------------------- path guard (hard rules)

def guard_path(path):
    """Never open sealed/ or any path containing holdout/heldout/cal_split/chunk5/j4b/inc1;
    never read hetdata_truth or vType/driver/vehicle_params truth files."""
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError('REFUSED: a non-empty path is required')
    # Check spelling AND resolved target before opening; aliases cannot bypass the guard.
    for candidate in (str(path), str(Path(path).resolve())):
        s = candidate.replace('\\', '/').lower()
        hit = [t for t in FORBIDDEN if t in s]
        if hit:
            raise ValueError('REFUSED: forbidden path token '+repr(hit)+': '+str(path))
        if _TRUTH.search(s):
            raise ValueError('REFUSED: truth file path refused: '+str(path))
    return path


# ---------------------------------------------------------------- registration file

def _check_registered(label, value, want):
    if type(value) is not type(want) or value != want:
        raise ValueError(f'REFUSED: registration {label} must be the registered {want!r} '
                         f'(got {value!r})')
    return value


def _walk_guard(value):
    """Guard every path-like string anywhere inside the registration (a sealed path hidden in
    any field is refused)."""
    if isinstance(value, str):
        if '/' in value or '\\' in value:
            guard_path(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and ('/' in k or '\\' in k):
                guard_path(k)
            _walk_guard(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_guard(v)
    return value


def validate_registration(reg):
    """Strict schema (unknown keys refused, registered values pinned). Optional k_candidates,
    wm_samples and groups_per_decision register the CPU rollout quote, with the latter an upper
    bound on groups scored per decision. Missing counts leave the total WM quote unknown.
    Normalizes: worlds sorted
    by seed, demand/penetration coerced to float, so registration_sha256 is stable across JSON
    formatting and int/float spellings. Seeds come ONLY from here - never a module constant."""
    if not isinstance(reg, dict) or reg.get('schema') != REGISTRATION_SCHEMA:
        raise ValueError(f'REFUSED: registration schema must be {REGISTRATION_SCHEMA}')
    unknown = set(reg)-set(REGISTRATION_KEYS)
    if unknown:
        raise ValueError(f'REFUSED: unregistered registration keys {sorted(unknown)}')
    missing = {'schema', 'worlds', 'planner', 'episode', 'analysis'}-set(reg)
    if missing:
        raise ValueError(f'REFUSED: registration missing {sorted(missing)}')
    _walk_guard(reg)

    if not isinstance(reg['worlds'], list):
        raise ValueError('REFUSED: worlds must be a list')
    worlds = []
    for w in reg['worlds']:
        if not isinstance(w, dict):
            raise ValueError('REFUSED: each world must be an object')
        bad = set(w)-set(WORLD_KEYS)
        if bad:
            raise ValueError(f'REFUSED: unregistered world keys {sorted(bad)}')
        if type(w.get('seed')) is not int or w['seed'] <= 0:
            raise ValueError('REFUSED: every world needs a unique positive int seed')
        worlds.append(dict(seed=w['seed']))
    if len(worlds) < 2:
        raise ValueError('REFUSED: the paired analysis needs at least two worlds')
    seeds = [w['seed'] for w in worlds]
    if len(set(seeds)) != len(seeds):
        raise ValueError('REFUSED: duplicate world seed')
    meta = [k for k in ('demand', 'penetration') if any(k in w for w in reg['worlds'])]
    if meta and len(meta) != 2:
        raise ValueError('REFUSED: demand and penetration are all-or-none world metadata')
    if meta:
        for w, out in zip(reg['worlds'], worlds):
            if not all(k in w for k in meta):
                raise ValueError('REFUSED: demand and penetration must appear on every world')
            for k in meta:
                v = w[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f'REFUSED: world {w["seed"]} {k} must be a finite number')
                out[k] = float(v)
            if out['demand'] <= 0 or not 0 <= out['penetration'] <= 1:
                raise ValueError('REFUSED: demand must be positive and penetration within [0, 1]')
    worlds.sort(key=lambda w: w['seed'])

    planner = reg['planner']
    if not isinstance(planner, dict) or set(planner)-set(PLANNER_KEYS)-set(PLANNER_OPTIONAL):
        raise ValueError(f'REFUSED: planner keys must be exactly '
                         f'{sorted(PLANNER_KEYS)+sorted(PLANNER_OPTIONAL)}')
    interval = planner.get('decision_interval_s', DECISION_INTERVAL_S)
    decisions_per_episode(EPISODE_END_S, interval)
    for k, want in (('p_star', P_STAR), ('lookahead_s', LOOKAHEAD_S)):
        _check_registered('planner.'+k, planner.get(k), want)
    optional = {}
    for k in PLANNER_OPTIONAL:
        if k in planner:
            if type(planner[k]) is not int or planner[k] < 1:
                raise ValueError(f'REFUSED: planner.{k} must be a positive int')
            optional[k] = planner[k]
    if optional.get('wm_samples', 2) < 2:
        raise ValueError('REFUSED: the planner needs multiple WM samples (wm_samples >= 2)')

    episode = reg['episode']
    if not isinstance(episode, dict) or set(episode)-set(EPISODE_KEYS):
        raise ValueError(f'REFUSED: episode keys must be exactly {sorted(EPISODE_KEYS)}')
    _check_registered('episode.end_s', episode.get('end_s'), EPISODE_END_S)

    analysis = reg['analysis']
    if not isinstance(analysis, dict) or set(analysis)-set(ANALYSIS_KEYS):
        raise ValueError(f'REFUSED: analysis keys must be exactly {sorted(ANALYSIS_KEYS)}')
    _check_registered('analysis.n_boot', analysis.get('n_boot'), N_BOOT)
    _check_registered('analysis.boot_seed', analysis.get('boot_seed'), BOOT_SEED)
    _check_registered('analysis.alpha', analysis.get('alpha'), ALPHA)

    cal = reg.get('calibration')
    if cal is not None:
        if not isinstance(cal, dict) or set(cal)-set(CALIBRATION_KEYS):
            raise ValueError(f'REFUSED: calibration keys must be exactly {sorted(CALIBRATION_KEYS)}')
        if not isinstance(cal.get('path'), str) or not cal['path']:
            raise ValueError('REFUSED: calibration.path must be a non-empty string')
        guard_path(cal['path'])
        sha256 = cal.get('sha256')
        if not isinstance(sha256, str) or re.fullmatch(r'[0-9a-f]{64}', sha256) is None:
            raise ValueError('REFUSED: calibration.sha256 must be a 64-char lowercase hex digest')
    note = reg.get('note')
    if note is not None and not isinstance(note, str):
        raise ValueError('REFUSED: note must be a string')

    out = dict(schema=REGISTRATION_SCHEMA, worlds=worlds,
               planner=dict(decision_interval_s=interval, p_star=P_STAR,
                            lookahead_s=LOOKAHEAD_S,
                            **optional),
               episode=dict(end_s=EPISODE_END_S),
               analysis=dict(n_boot=N_BOOT, boot_seed=BOOT_SEED, alpha=ALPHA))
    if cal is not None:
        out['calibration'] = dict(cal)
    if note is not None:
        out['note'] = note
    return out


def load_registration(path):
    """Read the registration file passed by the caller (the ONLY seed source) and validate it."""
    guard_path(path)
    p = Path(path)
    if not p.is_file():
        raise ValueError('REFUSED: registration file missing: '+str(p))
    try:
        reg = json.loads(p.read_text(encoding='utf-8'))
    except ValueError as e:
        raise ValueError('REFUSED: registration is not valid JSON: '+str(e))
    return validate_registration(reg)


# ---------------------------------------------------------------- budget quote (dry: 0 charged)

def decisions_per_episode(episode_end_s=EPISODE_END_S, decision_interval_s=DECISION_INTERVAL_S):
    if any(type(v) is not int or v <= 0 for v in (episode_end_s, decision_interval_s)):
        raise ValueError('REFUSED: episode length and decision interval must be positive ints')
    if episode_end_s % decision_interval_s:
        raise ValueError('REFUSED: episode_end_s must be a multiple of the decision interval')
    return episode_end_s // decision_interval_s


def budget_quote(n_worlds, k_candidates=None, episode_end_s=EPISODE_END_S,
                 decision_interval_s=DECISION_INTERVAL_S, lookahead_s=LOOKAHEAD_S,
                 wm_samples=None, groups_per_decision=None):
    """SUMO sim-s the REAL paired run would charge: both arms rerun the full paired world from
    t=0 (no prefix reuse - the planner can fire from its first decision). WM rollouts are CPU
    inference, quoted separately in WM-s and never charged as SUMO sim-s. The dry harness
    charges nothing: dry_run_sim_s = 0."""
    if type(n_worlds) is not int or n_worlds < 2:
        raise ValueError('REFUSED: the paired analysis needs at least two worlds')
    for k, v in dict(k_candidates=k_candidates, lookahead_s=lookahead_s,
                     wm_samples=wm_samples, groups_per_decision=groups_per_decision).items():
        if (v is not None and (type(v) is not int or v <= 0)) or (k == 'lookahead_s' and v is None):
            raise ValueError(f'REFUSED: {k} must be a positive int')
    decisions = decisions_per_episode(episode_end_s, decision_interval_s)
    per_arm = n_worlds*episode_end_s
    unit_wm_s = n_worlds*decisions*(k_candidates+1)*lookahead_s if k_candidates else None
    total_wm_s = (unit_wm_s*wm_samples*groups_per_decision
                  if all(v is not None for v in (unit_wm_s, wm_samples, groups_per_decision))
                  else None)
    out = dict(worlds=n_worlds, arms=list(ARMS), episode_end_s=episode_end_s,
               decision_interval_s=decision_interval_s, decisions_per_episode=decisions,
               lookahead_s=lookahead_s, per_arm_sim_s=per_arm, total_sim_s=2*per_arm,
               formula='2 arms x worlds x episode_end_s (paired worlds, no prefix reuse)',
               planner_wm_rollout_s=unit_wm_s, wm_samples=wm_samples,
               groups_per_decision=groups_per_decision, planner_wm_total_rollout_s=total_wm_s,
               wm_quote_complete=total_wm_s is not None,
               wm_quote_note=('planner_wm_rollout_s is PER group PER WM sample: worlds x decisions '
                              'x (k_candidates+1) x full lookahead_s. Total additionally multiplies '
                              'registered wm_samples and groups_per_decision (a group-count upper '
                              'bound); absent counts mean unknown total. CPU inference only; '
                              'not wall-clock seconds or SUMO sim-s, never charged'),
               dry_run_sim_s=0)
    return out


# ---------------------------------------------------------------- plan (sealed)

def plan_sha(plan):
    return digest({k: v for k, v in plan.items() if k != 'plan_sha256'})


def build_plan(reg):
    reg = validate_registration(reg)
    planner = dict(reg['planner'])
    if 'calibration' in reg:
        planner['calibration'] = dict(reg['calibration'])
    plan = dict(
        schema=SCHEMA, experiment=EXPERIMENT,
        worlds=reg['worlds'], n_worlds=len(reg['worlds']), panel_sha256=digest(reg['worlds']),
        paired=dict(rule='each world (same SUMO seed and demand/penetration) runs in BOTH arms '
                         'from t=0, with a common auxiliary noise seed; per-world paired '
                         'd = PLANNER - MAPPO; no cross-world pairing'),
        arms=list(ARMS),
        metric=dict(key=METRIC, definition='cumulative episode team_delay_vs (veh-s)',
                    difference='treatment - comparator (negative = planner improved)',
                    secondary=list(SAFETY_KEYS)),
        planner=dict(planner,
                     gate_rule='fire the best move iff the CALIBRATED P(move beats MAPPO over the '
                               'full look-ahead) >= p_star; otherwise MAPPO drives',
                     horizon_rule='the look-ahead horizon is NEVER shortened',
                     calibration_note=('arm PLANNER uses the T1C-calibrated probability; the '
                                       'artifact is referenced by path+sha256, never opened here')),
        episode=dict(reg['episode'], rule='both arms run the FULL episode from t=0 with the same '
                                          'world seed (paired worlds, same seeds both arms)'),
        analysis=dict(reg['analysis'],
                      unit='world cluster (one J per world and arm; the world is the resample unit)',
                      estimator='mean per-world d over whole-world resamples '
                                '(branch_eval.cluster_bootstrap, one branch per world)',
                      bootstrap=dict(resamples=N_BOOT,
                                     rng=f'numpy.random.default_rng({BOOT_SEED}).integers(0,n,(B,n))',
                                     seed=BOOT_SEED, interval='percentile 95%'),
                      p_value='two-sided CI inversion: min(1, 2*min((#boot<=0)+1, (#boot>=0)+1)/(B+1))',
                      safety='per-counter paired bootstrap CIs (descriptive) + per-world hard gate',
                      completeness='every world x arm exactly once; an incomplete panel => no inference'),
        safety=dict(keys=list(SAFETY_KEYS),
                    hard='per world: collisions and teleports must NOT increase (hard constraints)',
                    combined='per world: hard_brake + ttc_below_1p5_events must NOT increase '
                             '(registered MAIN 2026-11-03 04:26 combined rule, replay_bank._safety)'),
        budget=budget_quote(len(reg['worlds']), reg['planner'].get('k_candidates'),
                            reg['episode']['end_s'], reg['planner']['decision_interval_s'],
                            reg['planner']['lookahead_s'], reg['planner'].get('wm_samples'),
                            reg['planner'].get('groups_per_decision')),
        note=reg.get('note'))
    plan['registration_sha256'] = digest(reg)
    plan['plan_sha256'] = plan_sha(plan)
    return plan


def verify_plan(plan):
    if plan.get('schema') != SCHEMA or plan.get('plan_sha256') != plan_sha(plan):
        raise ValueError('plan SHA mismatch (edited after registration?)')
    if plan.get('arms') != list(ARMS) or plan.get('experiment') != EXPERIMENT:
        raise ValueError('REFUSED: arms/experiment differ from the registered paired evaluation')
    if plan.get('n_worlds') != len(plan.get('worlds', ())):
        raise ValueError('REFUSED: n_worlds differs from the world list')
    if plan.get('budget') != budget_quote(plan['n_worlds'], plan['planner'].get('k_candidates'),
                                          plan['episode']['end_s'],
                                          plan['planner']['decision_interval_s'],
                                          plan['planner']['lookahead_s'],
                                          plan['planner'].get('wm_samples'),
                                          plan['planner'].get('groups_per_decision')):
        raise ValueError('REFUSED: plan budget differs from the registered quote')
    reg = dict(schema=REGISTRATION_SCHEMA, worlds=plan['worlds'],
               planner={k: plan['planner'][k] for k in PLANNER_KEYS+PLANNER_OPTIONAL
                        if k in plan['planner']}, episode=dict(end_s=plan['episode']['end_s']),
               analysis={k: plan['analysis'][k] for k in ANALYSIS_KEYS})
    if 'calibration' in plan['planner']:
        reg['calibration'] = plan['planner']['calibration']
    if plan.get('note') is not None:
        reg['note'] = plan['note']
    if canonical(plan) != canonical(build_plan(reg)):
        raise ValueError('REFUSED: plan differs from its registered specification')
    return True


# ---------------------------------------------------------------- paired job specs

def noise_seed(world, arm):
    # Common random numbers across both arms, even if a later executor uses this seed.
    if arm not in ARMS:
        raise ValueError(f'REFUSED: unknown arm {arm}')
    return int(digest(dict(experiment=EXPERIMENT, seed=world['seed']))[:15], 16)


def jobs(plan):
    """One job per world x arm: PAIRED worlds - both jobs share the identical world dict and
    seed; only the PLANNER job carries the registered planner config (cadence, gate, horizon,
    calibration reference). No SUMO is run here."""
    verify_plan(plan)
    out = []
    for w in plan['worlds']:
        for arm in ARMS:
            treatment = arm == TREATMENT
            out.append(dict(
                id=f'planner_eval_{arm}_{w["seed"]}', world=dict(w), arm=arm,
                planner_enabled=treatment,
                planner=dict(plan['planner'], calibration=dict(plan['planner']['calibration']))
                if treatment and 'calibration' in plan['planner']
                else (dict(plan['planner']) if treatment else None),
                metric=METRIC, safety_keys=list(SAFETY_KEYS),
                episode_end_s=plan['episode']['end_s'], stop_s=plan['episode']['end_s'],
                sim_s=plan['episode']['end_s'], mode='run', deterministic=True,
                noise_seed=noise_seed(w, arm), plan_sha256=plan['plan_sha256'],
                dry_run=True))
    return out


def check_paired(plan, job_list):
    """Every world exactly once per arm with the IDENTICAL world dict and seed."""
    verify_plan(plan)
    want_seeds = {w['seed'] for w in plan['worlds']}
    seen = {}
    for j in job_list:
        seed = j['world']['seed']
        if seed not in want_seeds:
            raise ValueError(f'REFUSED: job outside the registered panel: {j["id"]}')
        if j['arm'] not in ARMS:
            raise ValueError(f'REFUSED: unknown arm {j["arm"]}')
        if seed in seen and seen[seed][j['arm']] is not None:
            raise ValueError(f'REFUSED: duplicate {j["arm"]} job for seed {seed}')
        seen.setdefault(seed, dict.fromkeys(ARMS))[j['arm']] = j
    for seed, pair in seen.items():
        if pair[TREATMENT] is None or pair[COMPARATOR] is None:
            raise ValueError(f'REFUSED: unpaired world (missing arm): seed {seed}')
        if canonical(pair[TREATMENT]['world']) != canonical(pair[COMPARATOR]['world']):
            raise ValueError(f'REFUSED: paired worlds differ: seed {seed}')
    missing = want_seeds-set(seen)
    if missing:
        raise ValueError(f'REFUSED: worlds without jobs: {sorted(missing)}')
    expected = {(j['world']['seed'], j['arm']): j for j in jobs(plan)}
    for j in job_list:
        if canonical(j) != canonical(expected[j['world']['seed'], j['arm']]):
            raise ValueError('REFUSED: job differs from its registered specification')
    return True


# ---------------------------------------------------------------- receipts (sealed schema)

def _check_trace(trace, arm, planner, episode_end_s):
    if not isinstance(trace, list):
        raise ValueError('REFUSED: planner_trace must be a list')
    if arm == COMPARATOR:
        if trace:
            raise ValueError('REFUSED: the MAPPO arm must record no planner trace')
        return []
    interval, p_star = planner['decision_interval_s'], planner['p_star']
    out, last, seen, groups = [], 0, set(), {}
    for x in trace:
        if not isinstance(x, dict):
            raise ValueError('REFUSED: trace entries must be objects')
        bad = set(x)-set(TRACE_KEYS)
        if bad:
            raise ValueError(f'REFUSED: unregistered trace keys {sorted(bad)}')
        t = x.get('t_s')
        if (type(t) is not int or t <= 0 or t > episode_end_s
                or (interval is not None and t % interval)):
            raise ValueError(f'REFUSED: trace t_s {t!r} is off the {interval} s decision cadence')
        group = x.get('group_id', '')
        if not isinstance(group, str):
            raise ValueError('REFUSED: trace group_id must be a string')
        if t < last or (t, group) in seen:
            raise ValueError('REFUSED: trace entries must be ordered and unique per time/group')
        last = t
        seen.add((t, group))
        groups[t] = groups.get(t, 0)+1
        if groups[t] > planner.get('groups_per_decision', math.inf):
            raise ValueError('REFUSED: trace exceeds the registered groups_per_decision bound')
        p = x.get('p_better')
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError(f'REFUSED: trace p_better {p!r} must be within [0, 1]')
        if type(x.get('fired')) is not bool:
            raise ValueError('REFUSED: trace fired must be a bool')
        if x['fired'] and p < p_star:
            raise ValueError(f'REFUSED: fired at t={t} with P(better)={p} < p_star={p_star}')
        family = x.get('family')
        if isinstance(family, str):
            family = MOVE_FAMILY_ALIASES.get(family, family)
        if family is not None and family not in MOVE_FAMILIES:
            raise ValueError(f'REFUSED: trace family {family!r} is not a registered move family')
        members = x.get('members')
        if members is not None and (not isinstance(members, list)
                                    or any(type(m) not in (int, str) for m in members)):
            raise ValueError('REFUSED: trace members must be a list of member ids')
        if members is not None and (not 1 <= len(members) <= 4 or len(set(members)) != len(members)):
            raise ValueError('REFUSED: trace members must contain 1..4 unique CAV ids')
        lookahead = x.get('lookahead_s')
        if lookahead is not None and lookahead != planner['lookahead_s']:
            raise ValueError('REFUSED: trace lookahead_s differs from the registered horizon '
                             '(the look-ahead is NEVER shortened)')
        if x['fired'] and (family in (None, 'none') or members is None or lookahead is None):
            raise ValueError('REFUSED: fired moves need a non-none family, members and full lookahead_s')
        normalized = {k: x[k] for k in TRACE_KEYS if k in x}
        if 'family' in normalized:
            normalized['family'] = family
        out.append(normalized)
    return out


def make_receipt(job, outcome):
    """Validate the executor outcome against the job and return the sealed receipt. An outcome
    is dict(J, safety, planner_trace, executor?, result_ref?); SUMO is never run here."""
    if not isinstance(outcome, dict):
        raise ValueError('REFUSED: outcome must be an object')
    _walk_guard(outcome)
    if set(outcome)-set(OUTCOME_KEYS):
        raise ValueError(f'REFUSED: unregistered outcome keys {sorted(set(outcome)-set(OUTCOME_KEYS))}')
    if {'J', 'safety'}-set(outcome):
        raise ValueError('REFUSED: outcome needs J and safety')
    J = outcome['J']
    if isinstance(J, bool) or not isinstance(J, (int, float)) or not math.isfinite(J):
        raise ValueError('REFUSED: J must be a finite number')
    safety_raw = outcome['safety']
    if not isinstance(safety_raw, dict) or set(safety_raw) != set(SAFETY_KEYS):
        raise ValueError(f'REFUSED: safety keys must be exactly {sorted(SAFETY_KEYS)}')
    safety = {}
    for k in SAFETY_KEYS:
        v = safety_raw[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f'REFUSED: safety.{k} must be a finite non-negative count')
        safety[k] = float(v)
    trace = _check_trace(outcome.get('planner_trace', []), job['arm'],
                         job['planner'] or job, job['episode_end_s'])
    rec = dict(schema=RECEIPT_SCHEMA, experiment=EXPERIMENT, plan_sha256=job['plan_sha256'],
               job_id=job['id'], seed=job['world']['seed'], arm=job['arm'],
               planner_enabled=job['planner_enabled'], metric=job['metric'], J=float(J),
               safety=safety, planner_trace=trace, episode_end_s=job['episode_end_s'],
               executor=str(outcome.get('executor', '')),
               result_ref=str(outcome.get('result_ref', '')))
    rec = seal(rec, 'receipt_sha256')
    validate_receipt(rec, job)
    return rec


def validate_receipt(record, job=None):
    """Validate receipt integrity and trace structure. A bound job additionally enforces its
    registered cadence and group bound; neither can be inferred from a plan hash alone."""
    if record.get('schema') != RECEIPT_SCHEMA or record.get('experiment') != EXPERIMENT:
        raise ValueError('REFUSED: not a planner_eval receipt')
    verify_seal(record, 'receipt_sha256')
    if set(record) != set(RECEIPT_KEYS):
        raise ValueError('REFUSED: receipt keys differ from the schema')
    _walk_guard(record)
    if record['arm'] not in ARMS:
        raise ValueError(f'REFUSED: unknown arm {record["arm"]}')
    if record['planner_enabled'] != (record['arm'] == TREATMENT):
        raise ValueError('REFUSED: planner_enabled must mark exactly the PLANNER arm')
    if type(record['planner_enabled']) is not bool or record.get('metric') != METRIC:
        raise ValueError('REFUSED: receipt metric/planner_enabled differs from its job specification')
    if type(record.get('episode_end_s')) is not int or record['episode_end_s'] != EPISODE_END_S:
        raise ValueError('REFUSED: receipt episode_end_s differs from its job specification')
    if type(record.get('seed')) is not int or record['seed'] <= 0:
        raise ValueError('REFUSED: receipt seed must be a positive int')
    J = record['J']
    if isinstance(J, bool) or not isinstance(J, (int, float)) or not math.isfinite(J):
        raise ValueError('REFUSED: receipt J must be a finite number')
    if set(record.get('safety', ()))-set(SAFETY_KEYS) or set(SAFETY_KEYS)-set(record.get('safety', ())):
        raise ValueError(f'REFUSED: receipt safety keys must be exactly {sorted(SAFETY_KEYS)}')
    for k in SAFETY_KEYS:
        v = record['safety'][k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f'REFUSED: receipt safety.{k} must be a finite non-negative count')
    planner = (job['planner'] if job is not None and job['planner'] else
               dict(decision_interval_s=None, p_star=P_STAR, lookahead_s=LOOKAHEAD_S))
    _check_trace(record.get('planner_trace', []), record['arm'], planner,
                 record['episode_end_s'])
    if job is not None:
        expected = dict(job_id=job['id'], seed=job['world']['seed'], arm=job['arm'],
                        planner_enabled=job['planner_enabled'], plan_sha256=job['plan_sha256'],
                        episode_end_s=job['episode_end_s'], metric=job['metric'])
        for k, v in expected.items():
            if record[k] != v:
                raise ValueError(f'REFUSED: receipt {k} differs from its job')
    return True


# ---------------------------------------------------------------- safety counters + gate

def safety_delta(treatment, comparator):
    return {k: float(treatment['safety'][k])-float(comparator['safety'][k]) for k in SAFETY_KEYS}


def safety_verdict(deltas):
    """Per-world gate: OLD disclosed rule (no metric may increase), the hard constraints
    (collisions/teleports) and the registered MAIN 2026-11-03 04:26 combined rule."""
    combined = sum(deltas[k] for k in COMBINED_KEYS)
    return dict(old_no_metric_increase=all(v <= 0 for v in deltas.values()),
                hard=all(deltas[k] <= 0 for k in HARD_KEYS),
                combined_rule=combined <= 0, safety_critical_events_delta=combined)


def safety_gate(plan, records):
    """Per paired world: PLANNER safety counters minus MAPPO; hard failures never pass."""
    pairs = pair_receipts(records, plan)
    per_world, hard, combined = {}, [], []
    for seed, (t, c) in sorted(pairs.items()):
        v = safety_verdict(safety_delta(t, c))
        per_world[str(seed)] = v
        if not v['hard']:
            hard.append(seed)
        if not v['combined_rule']:
            combined.append(seed)
    return dict(keys=list(SAFETY_KEYS), per_world=per_world, hard_failures=hard,
                combined_failures=combined,
                safety_ok=bool(pairs) and not hard and not combined,
                rule=dict(hard='collisions/teleports must NOT increase (hard constraints)',
                          combined='hard_brake + ttc_below_1p5_events must NOT increase '
                                   '(replay_bank MAIN 2026-11-03 04:26 rule, per paired world)'))


# ---------------------------------------------------------------- analysis (paired bootstrap)

def pair_receipts(records, plan):
    """Completeness check, then {seed: (PLANNER receipt, MAPPO receipt)}. Any missing arm,
    duplicate or outside-panel record refuses the inference."""
    verify_plan(plan)
    want = {w['seed'] for w in plan['worlds']}
    expected = {(j['world']['seed'], j['arm']): j for j in jobs(plan)}
    pairs = {}
    for r in records:
        validate_receipt(r)
        if r['plan_sha256'] != plan['plan_sha256'] or r['seed'] not in want:
            raise ValueError(f'REFUSED: receipt outside the registered plan: {r["job_id"]}')
        validate_receipt(r, expected[r['seed'], r['arm']])
        pair = pairs.setdefault(r['seed'], {})
        if r['arm'] in pair:
            raise ValueError(f'REFUSED: duplicate {r["arm"]} receipt for seed {r["seed"]}')
        pair[r['arm']] = r
    bad = [s for s, p in pairs.items() if set(p) != set(ARMS)]
    if bad or set(pairs) != want:
        raise ValueError(f'REFUSED: incomplete panel (every world x arm exactly once): '
                         f'missing/bad {sorted((want-set(pairs)) | set(bad))}')
    return {s: (p[TREATMENT], p[COMPARATOR]) for s, p in pairs.items()}


def _paired_ci(deltas, n_boot, boot_seed, alpha):
    """Same whole-world resampling as branch_eval, without meaningless safety percentages."""
    d = np.asarray(deltas, float)
    idx = np.random.default_rng(boot_seed).integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    p = min(1., 2*min(np.sum(boot <= 0)+1, np.sum(boot >= 0)+1)/(n_boot+1))
    return dict(n_worlds=len(d), mean_delta_vs=float(d.mean()), ci95=[float(lo), float(hi)],
                p_two_sided=float(p), worlds_improved=int(np.sum(d < 0)))


def analyze(records, plan):
    """Registered paired analysis: primary PLANNER-MAPPO whole-world bootstrap on J; per safety
    counter descriptive paired CIs; the hard per-world safety gate. Sealed."""
    verify_plan(plan)
    records = list(records)
    pairs = pair_receipts(records, plan)
    seeds = sorted(pairs)
    planner_J = [float(pairs[s][0]['J']) for s in seeds]
    mappo_J = [float(pairs[s][1]['J']) for s in seeds]
    cfg = plan['analysis']
    raw = _bootstrap(planner_J, mappo_J, cfg['n_boot'], cfg['boot_seed'], cfg['alpha'])
    primary = dict(contrast=f'{TREATMENT}-{COMPARATOR}', metric=METRIC,
                   n_worlds=raw['n_worlds'], mean_diff_vs=raw['mean_diff_vs'], ci95=raw['ci95'],
                   p_two_sided=raw['p_two_sided'], mean_planner_J=raw['mean_treatment_J'],
                   mean_mappo_J=raw['mean_comparator_J'],
                   pct_delay_reduction=raw['pct_delay_reduction'], pct_ci95=raw['pct_ci95'],
                   worlds_improved=raw['worlds_improved'],
                   note='negative diff = the planner improved team delay; the estimator is '
                        'branch_eval.cluster_bootstrap on one branch (episode) per world')
    safety_cis = {}
    deltas = {k: [float(pairs[s][0]['safety'][k])-float(pairs[s][1]['safety'][k]) for s in seeds]
              for k in SAFETY_KEYS}
    for k in SAFETY_KEYS:
        safety_cis[k] = _paired_ci(deltas[k], cfg['n_boot'], cfg['boot_seed'], cfg['alpha'])
    out = dict(schema=ANALYSIS_SCHEMA, experiment=EXPERIMENT, plan_sha256=plan['plan_sha256'],
               seeds=seeds, primary=primary, safety=dict(gate=safety_gate(plan, records),
                                                         paired_ci=safety_cis),
               n_boot=cfg['n_boot'], boot_seed=cfg['boot_seed'], alpha=cfg['alpha'],
               completeness=dict(worlds=len(seeds), receipts=2*len(seeds), complete=True))
    out['analysis_sha256'] = digest(out)
    return out


def _bootstrap(treat, comp, n_boot, boot_seed, alpha):
    from .branch_eval import cluster_bootstrap
    if not np.any(comp):
        ci = _paired_ci(np.asarray(treat)-np.asarray(comp), n_boot, boot_seed, alpha)
        return dict(n_worlds=len(treat), mean_diff_vs=ci['mean_delta_vs'], ci95=ci['ci95'],
                    p_two_sided=ci['p_two_sided'], mean_treatment_J=float(np.mean(treat)),
                    mean_comparator_J=0., pct_delay_reduction=None, pct_ci95=[None, None],
                    worlds_improved=ci['worlds_improved'])
    raw = cluster_bootstrap(np.asarray(treat, float)[:, None], np.asarray(comp, float)[:, None],
                            n_boot, boot_seed, alpha)
    raw['pct_delay_reduction'] = (raw['pct_delay_reduction']
                                  if math.isfinite(raw['pct_delay_reduction']) else None)
    raw['pct_ci95'] = [v if math.isfinite(v) else None for v in raw['pct_ci95']]
    return raw


def prepare_track2(*, registration, checkpoints, contract):
    """Three-arm adapter to the same paired TRAIN-world/full-window harness.

    Track 2 has learned policies instead of a WM selector. This explicit adapter
    preserves its own approval-bound panel and reuses this module's paired
    bootstrap, completeness convention and physical-vector comparisons.
    """
    from copy import deepcopy
    from .rerun_cpus import FINAL_EVALUATION, final_world_registration
    interim = dict(FINAL_EVALUATION, update=8, phase='v2v_interim')
    if contract not in (FINAL_EVALUATION, interim):
        raise ValueError('Track 2 final evaluation contract differs')
    worlds = final_world_registration(dict(final_evaluation=registration))
    if worlds != registration or set(checkpoints) != set(contract['arms']):
        raise ValueError('unbound/incomplete Track 2 paired world registration')
    out = []
    for arm in contract['arms']:
        cp = checkpoints[arm]
        guard_path(cp['path'])
        if (cp.get('update') != contract['update'] or cp.get('components') !=
                contract['checkpoint_components'][arm] or not re.fullmatch('[0-9a-f]{64}', cp.get('file_sha256', ''))):
            raise ValueError('complete immutable final policy checkpoint required')
        for row in worlds['worlds']:
            out.append((dict(branch='shared_final', arm=arm, world_id=row['world_id'],
                world=deepcopy(row['world']), checkpoint=deepcopy(cp), sim_s=contract['horizon_s'],
                window=dict(start_s=0, end_s=contract['horizon_s']),
                final_worlds_sha256=worlds['final_worlds_sha256']), {}))
    return out


def analyse_track2(*, registration, checkpoints, records, contract):
    """Refuse incomplete/unpaired results; use native whole-world paired CIs."""
    specs = prepare_track2(registration=registration, checkpoints=checkpoints, contract=contract)
    expected = {(s['arm'], s['world_id']): s for s, _ in specs}
    have = {}
    for row in records:
        key = (row.get('arm'), row.get('world_id'))
        if key not in expected or key in have or row.get('status') != 'graded':
            raise ValueError('incomplete/duplicate/outside Track 2 paired result')
        spec = expected[key]
        if any(row.get(k) != spec[k] for k in ('world', 'checkpoint', 'window', 'final_worlds_sha256')):
            raise ValueError('Track 2 result world/checkpoint/window binding differs')
        if not isinstance(row.get('J_veh_s'), (int, float)) or not math.isfinite(row['J_veh_s']):
            raise ValueError('finite realised source-inclusive delay required')
        safety = row.get('safety', {})
        if set(safety) != set(contract['safety_counters']) or any(
                type(v) is not int or v < 0 for v in safety.values()):
            raise ValueError('all realised physical-vector counters required')
        have[key] = row
    if set(have) != set(expected):
        raise ValueError('every registered world x arm must be graded exactly once')
    bootstrap, paired = contract['bootstrap'], {}
    for comparison in contract['comparisons']:
        a, b = comparison.split('-')
        deltas = [have[a, w['world_id']]['J_veh_s']-have[b, w['world_id']]['J_veh_s']
                  for w in registration['worlds']]
        ci = _paired_ci(deltas, bootstrap['reps'], bootstrap['seed'], 1-bootstrap['level'])
        paired[comparison] = dict(n=ci['n_worlds'], mean=ci['mean_delta_vs'],
                                  low=ci['ci95'][0], high=ci['ci95'][1], native=ci)
    safety = {a: {k: sum(have[a, w['world_id']]['safety'][k] for w in registration['worlds'])
                  for k in contract['safety_counters']} for a in contract['arms']}
    return dict(status='graded', harness='tools.coopmwm.planner_eval',
        bootstrap=bootstrap, final_worlds_sha256=registration['final_worlds_sha256'],
        checkpoints=checkpoints, paired_ci=paired, safety=safety,
        complete=True, physical_vector_rule='all counters retained; B-A and R-A descriptive')


def verify_analysis(record):
    if record.get('schema') != ANALYSIS_SCHEMA or record.get('experiment') != EXPERIMENT:
        raise ValueError('REFUSED: not a planner_eval analysis')
    return verify_seal(record, 'analysis_sha256')


# ---------------------------------------------------------------- dry-run (no physics)

def preflight_dry(registration):
    """Validate everything and quote; nothing is written and no SUMO is charged (new_sim_s=0)."""
    return _dry_summary(build_plan(load_registration(registration)))


def _dry_summary(plan):
    job_list = jobs(plan)
    check_paired(plan, job_list)
    return dict(dry_run=True, new_sim_s=0, worlds=plan['n_worlds'], jobs=len(job_list),
                seeds=[w['seed'] for w in plan['worlds']], plan_sha256=plan['plan_sha256'],
                budget=plan['budget'], receipt_schema=RECEIPT_SCHEMA,
                safety_keys=list(SAFETY_KEYS), analysis_schema=ANALYSIS_SCHEMA,
                calibration_pinned='calibration' in plan['planner'], dispatch_supported=False)


def dry_run(registration, out_dir):
    """BUILD + DRY-RUN ONLY: write the sealed plan and the paired job specs. Refuses a non-empty
    output directory and any forbidden path. Never runs, charges or dispatches SUMO."""
    plan = build_plan(load_registration(registration))
    summary = _dry_summary(plan)
    out = Path(guard_path(out_dir))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('REFUSED: output directory is not empty: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    job_list = jobs(plan)
    plan_path, jobs_path = out/PLAN_FILE, out/JOBS_FILE
    plan_path.write_text(json.dumps(plan, indent=1, sort_keys=True)+'\n', encoding='utf-8')
    with jobs_path.open('w', encoding='utf-8') as f:
        for j in job_list:
            f.write(canonical(j)+'\n')
    summary.update(plan_file=str(plan_path), jobs_file=str(jobs_path),
                   jobs_lines=len(job_list), sim_s_charged=0)
    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('quote', 'preflight', 'dry_run'))
    p.add_argument('--registration', default=None, help='registration file (the ONLY seed source)')
    p.add_argument('--out', default='planner_eval_dry')
    p.add_argument('--worlds', type=int, default=None, help='quote only: world count')
    p.add_argument('--k-candidates', type=int, default=None)
    p.add_argument('--wm-samples', type=int, default=None)
    p.add_argument('--groups-per-decision', type=int, default=None)
    a = p.parse_args()
    if a.cmd == 'quote':
        print(json.dumps(budget_quote(a.worlds or 0, a.k_candidates, wm_samples=a.wm_samples,
                                      groups_per_decision=a.groups_per_decision)))
    elif a.cmd == 'preflight':
        print(json.dumps(preflight_dry(a.registration)))
    else:
        print(json.dumps(dry_run(a.registration, a.out)))
