"""AMENDMENT 80 (A80-S2B; POST-HOC, exploratory): the plans of the reduced WM hold-form search A75-S2b (A79; file
TOP_OFFER_CONDITIONS_D2800_p60.json written by a75ss.main, top <= 12 activated non-comparator finals) tested directly
in SUMO at D2800_p60 on top of MAPPO60C, exactly as A75-D2 tested its seeded draw.

The candidates are a PARAMETER (env A80_TOP_FILE + A80_TOP_SHA256, both required by ``pin-top``); everything else is
the A75-D2 driver (ops/a75_d2forms.py, unchanged, imported) with this root's names: executor a75_holdforms d1af7e0f,
arm A75_PLAN / A75_BANK, the 24 A68 admission worlds 992500-992523 vs the STORED A68V_MAPPO60C wave (REF_PINS
79815fe9), ADMIT_RULE / BANK_RULE of A75-D2 (form families: FORM_RULE below), the bank on the 96 A70 worlds
996300-996395 vs the STORED A70 MAPPO60C (primary) and DUO_HOLD_60C (descriptive) waves.

Lossless mapping (MAPPING_RULE) is checked BEFORE anything is pinned: every row must be rebuilt by the executor
(a75_holdforms.spec_of) with the search's own strategy_key, accepted by executor_refusal / a75_bank_plans, and
round-trip byte-equal through a75_holdforms.bank_doc; any failing row stops the stage (A80_MAPPING_REFUSED.json),
nothing is approximated, dropped or edited.

Mechanics (MECH_RULE): only the executor feature tokens (form_tokens) that the A75-D2 mechanics fixture (the six
EXAMPLE_PLANS) did not exercise. Each plan carrying such a token runs its first 2 admission episodes (992500/992501;
the SAME specs / job ids as its verification wave, so they count once, under the verification cap) and is checked
from the receipts + vehicle log; PASS gates the rest. No uncovered token -> sealed PASS without episodes.

Stages (own root; ledger phase ``holdforms_a80s2b``; caps in code: verification 12 x 24 x 1,200 = 345,600, confirmation
96 x 1,200 = 115,200, phase 460,800):
  pin-top     zero SUMO: MAPPING_RULE; write-once byte copy of the top file + A80_CANDIDATES.json (sealed)
  prepare     zero SUMO: candidate bank (a75_holdforms.bank_doc)
  dry         zero SUMO: waves / sim-s / stored-reference bindings (--stage v | c)
  mech        uncovered-token plans x 2 admission worlds -> mech-compare (A80_MECH.json PASS / FAIL)
  verify      the rest of the 12 x 24 verification episodes (refuses unless mech PASS)
  admit       zero physics: receipt invariants on every verification record (any violation: STOP), ADMIT_RULE,
              BANK_RULE -> A80_ADMISSION_D2800_p60.json + bank
  confirm     A80_BANK_60C (A75_BANK) on the 96 A70 worlds; not run if nothing is admitted
  analyse     primary A80_BANK_60C - MAPPO60C; A80_BANK_60C - DUO_HOLD_60C and DUO_HOLD_60C - MAPPO60C descriptive
--drop-raw (verify / confirm; own a80v_ / a80c_ episodes, the A75-D2 DROP_RULE). No automatic replay.

Commands (python ops/a80_s2b.py <cmd> --final-root R --a68-root R68 --a66-root R66 --ref68-reports RR68
          [--a70-root R70 --ref70-reports RR70]): pin-top ; prepare ; dry --stage v|c ; mech ; mech-compare ; verify
          [--drop-raw] ; admit ; confirm [--drop-raw] ; analyse ; status --stage v|c ; drop-raw --stage v|c ;
          check-top --top F (zero SUMO, no root)
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a75_d2forms as D2  # noqa: E402  (the A75-D2 driver, unchanged; module names re-pointed below)

AMENDMENT = '80'
STAGE = 'S2B'
CELL = D2.CELL                                         # D2800_p60
PHASE = 'holdforms_a80s2b'
HORIZON = D2.HORIZON
A75_SHA256 = D2.A75_SHA256                             # a75_holdforms d1af7e0f
N_MAX, VERIFY_N, CONFIRM_N, MECH_WORLDS = 12, D2.VERIFY_N, D2.CONFIRM_N, 2
VERIFY_CAP, CONFIRM_CAP = N_MAX*VERIFY_N*HORIZON, CONFIRM_N*HORIZON
CAP_SIM_S = VERIFY_CAP+CONFIRM_CAP
assert (VERIFY_CAP, CONFIRM_CAP, CAP_SIM_S) == (345_600, 115_200, 460_800)
TOP_ENV, TOP_SHA_ENV = 'A80_TOP_FILE', 'A80_TOP_SHA256'
TOP_NAME = f'TOP_OFFER_CONDITIONS_{CELL}.json'
TOP_COPY = f'A80_{TOP_NAME}'
CANDS_DOC = 'A80_CANDIDATES.json'
REFUSED_DOC = 'A80_MAPPING_REFUSED.json'
WRITER = 'a75ss.main TOP_OFFER_CONDITIONS (a75ss cae54dc1, A79 S2b)'
ROW_KEYS = ('family', 'strategy_key', 'score', 'gain_vs_base', 'gain_vs_base_lcb', 'feasible', 'extended',
            'offer_condition')
COND_KEYS = ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'gate', 'start_gate', 'phases', 'situation', 'roles')
ROLE_OLD = ('sumo_lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s', 'lc_target_sumo')
FORMS = ('gap', 'pulse', 'taper', 'zone', 'rolling', 'short', 'other')
PREFIX = dict(v='a80v_', c='a80c_')
ARM = 'A80_BANK_60C'
STOP_FILE = 'STOP_A80'
HOST_BASE_ENV = 'A80_HOST_BASE'
FORM_RULE = ('form family of a plan (first match, fixed before any outcome): gap (a role with start gap) > pulse (a '
             'pulsed role) > taper (a tapered role) > zone (zone mid / merge / custom) > rolling (>= 2 roles with >= 2 '
             'distinct onset_s) > short (every hold_s <= 45) > other. A75-D2 drew one form per plan; WM plans can '
             'combine forms, hence the order and the extra family "other"')
MAPPING_RULE = (
    'per row of the pinned file: the row keys are exactly the a75ss writer keys; offer_condition keys within the '
    'writer keys (+ zone_d), situation = {unconditional: true}, gate / start_gate / phases = null (the search cfg is '
    'unconditional), every role key within the executor role fields; spec = a75_holdforms.spec_of(family, '
    'offer_condition) has strategy_key == the row strategy_key and executor_refusal None; a75_holdforms.bank_doc of '
    'the spec gives back the identical offer_condition (round trip) and a75_bank_plans accepts it with no refusal '
    'and the same extended flag; families and keys unique; 1 <= rows <= 12. Any failure: STOP (no approximation)')
MECH_RULE = (
    'tokens = form_tokens(plan) (trigger / zone of extended plans, per-role form combination and start kind, gap '
    'release / multi-event / drop window, delayed onsets, mixed gap+onset roles; "plain" for plans run by the '
    'inherited A44 path, exercised by the A75-D2 identity check). Uncovered = tokens of a candidate not produced by '
    'any EXAMPLE_PLANS strategy (A75-D2 mechanics PASS). Each candidate with an uncovered token runs its first 2 '
    'verification episodes (worlds 992500 / 992501). PASS iff every such record loaded executor MAPPO60C and '
    'a75_sha256 d1af7e0f with a75 receipts, no executor refusal, collisions = 0 and teleports = 0, no receipt '
    'invariant violation (plan_checks) and every activation x of every member (vehicle log s + S_OFF) inside the '
    "plan's zone (+-1 m). A token never exercised on those 2 worlds is reported NOT EXERCISED (not a failure); the "
    'receipt invariants are re-checked on every verification record before admission')
ADMIT_RULE = D2.ADMIT_RULE
BANK_RULE = D2.BANK_RULE.replace('gap / short / zone / pulse / taper / rolling', '/'.join(FORMS) + ' (FORM_RULE)')
PRIMARY_RULE = D2.PRIMARY_RULE.replace('A75D2_BANK_60C', ARM)
DROP_RULE = D2.DROP_RULE.replace('a75d2v_* / a75d2c_*', 'a80v_* / a80c_*')
AUTHORITY = ('MAIN AMENDMENT 80 (A80-S2B, 2026-11-13; POST-HOC, exploratory): the A79 S2b WM search top plans '
             '(TOP_OFFER_CONDITIONS_D2800_p60, pinned file) each verified in SUMO on the 24 A68 admission worlds vs '
             'the '
             'stored MAPPO60C; A75-D2 admission / bank rules; one bank on the A70 panel; phase holdforms_a80s2b, caps '
             '345,600 + 115,200')
DISCLOSURE = ('POST-HOC and exploratory. Both panels were already used: the 24 A68 admission worlds by A68 / A71 / '
              'A75-D1 / A75-D2 (admission), the 96 A70 worlds by A70 / A75-D1 / A75-D2 (confirmation); D1 and D2 '
              'results were known at registration. The plans were chosen by the WM on WM start states, not on these '
              'worlds. Multiplicity: up to 12 plans each tested at 95% on the same 24 worlds; admission is a screen '
              'and '
              'its bounds are selection-biased. Never pooled with other panels')
NOT_EXERCISED = 'NOT EXERCISED'


# ================================================================================================ pure helpers

def _H():
    return D2._H()


def _N():
    return D2._mods()[0]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def check_sha(text):
    t = str(text or '').strip().lower()
    if len(t) != 64 or any(c not in '0123456789abcdef' for c in t):
        raise SystemExit(f'not a sha256: {text!r}')
    return t


def role_kind(r):
    return 'gap' if r.get('start') == 'gap' else 'onset'


def role_letters(r):
    out = []
    for k, ch in (('dv_rel', 'd'), ('taper_in_s', 'i'), ('taper_out_s', 'o'), ('pulse_on_s', 'p')):
        if k in r:
            out.append(ch)
    return ''.join(out)


def form_tokens(spec):
    """Executor feature tokens of a plan (canonical a75 spec)."""
    H = _H()
    s = H.canonical(spec)
    if not H.is_extended(s):
        return {('plain',)}
    roles = s['roles']
    t = {('ext_trigger', s['trigger']), ('ext_zone', 'custom' if s.get('zone_d') is not None else s['zone']),
         ('ext_roles', '+'.join(sorted({role_kind(r) for r in roles})))}
    for r in roles:
        k = role_kind(r)
        t.add(('role', k, role_letters(r)))
        if int(r['onset_s']) > 0:
            t.add(('onset_delay', k))
        if k == 'gap':
            t.add(('gap_release', r.get('gap_release', 'timer')))
            t.add(('gap_events_multi', int(r.get('gap_events', 1)) > 1))
            t.add(('gap_drop', r.get('gap_drop_m') is not None))
    return t


def form_family(spec):
    """FORM_RULE."""
    roles = spec['roles']
    if any(r.get('start') == 'gap' for r in roles):
        return 'gap'
    if any('pulse_on_s' in r for r in roles):
        return 'pulse'
    if any('taper_in_s' in r or 'taper_out_s' in r for r in roles):
        return 'taper'
    if spec['zone'] in ('mid', 'merge', 'custom'):
        return 'zone'
    if len(roles) >= 2 and len({int(r['onset_s']) for r in roles}) >= 2:
        return 'rolling'
    if max(int(r['hold_s']) for r in roles) <= 45:
        return 'short'
    return 'other'


def covered_tokens():
    strat, _ = D2.examples()
    out = set()
    for s in strat.values():
        out |= form_tokens(s)
    return out


def _canon_json(x):
    return json.dumps(x, sort_keys=True)


def map_row(i, row):
    """MAPPING_RULE for one row -> (candidate dict, None) or (None, reason)."""
    H = _H()
    if not isinstance(row, dict) or tuple(sorted(row)) != tuple(sorted(ROW_KEYS)):
        return None, f'row keys {sorted(row) if isinstance(row, dict) else type(row).__name__} are not the writer keys'
    fam, cond = row['family'], row['offer_condition']
    if not isinstance(fam, str) or not fam or not isinstance(cond, dict):
        return None, 'family / offer_condition malformed'
    extra = set(cond) - set(COND_KEYS) - {'zone_d'}
    if extra or set(COND_KEYS) - set(cond):
        return None, f'offer_condition keys: extra {sorted(extra)}, missing {sorted(set(COND_KEYS) - set(cond))}'
    if cond['situation'] != {'unconditional': True}:
        return None, f'situation {cond["situation"]} is not unconditional'
    for k in ('gate', 'start_gate', 'phases'):
        if cond[k] is not None:
            return None, f'{k} = {cond[k]} (the registered search is unconditional)'
    for r in cond['roles']:
        bad = set(r) - set(ROLE_OLD) - set(H.ROLE_DEFAULTS)
        if bad or set(ROLE_OLD) - set(r):
            return None, f'role keys: not executor fields {sorted(bad)} / missing {sorted(set(ROLE_OLD) - set(r))}'
    try:
        spec = H.spec_of(fam, cond)
    except (KeyError, TypeError, ValueError) as e:
        return None, f'executor spec_of: {e!r}'
    key = H.strategy_key(spec)
    if key != row['strategy_key']:
        return None, f'executor strategy_key {key} != search key {row["strategy_key"]} (fields lost or changed)'
    why = H.executor_refusal(spec)
    if why is not None:
        return None, f'executor refusal {why}'
    if bool(H.is_extended(spec)) != bool(row['extended']):
        return None, f'extended flag {row["extended"]} != executor {H.is_extended(spec)}'
    try:
        bank = H.bank_doc([dict(family=fam, strategy=spec, lb95_veh_s=0.0)], CELL)
    except ValueError as e:
        return None, f'executor bank_doc: {e}'
    back = bank['clean5c']['templates'][0]['offer_condition']
    if _canon_json(back) != _canon_json(cond):
        return None, 'bank_doc round trip changes the offer_condition'
    got, refused = H.a75_bank_plans(bank, CELL)
    if refused or len(got) != 1 or got[0]['key'] != key or _canon_json(got[0]['spec']) != _canon_json(spec):
        return None, f'a75_bank_plans: refused {refused} / spec differs'
    if any(r.get('lc_target') is not None for r in spec['roles']):
        return None, 'a lane-change role (the registered search is hold-only)'
    return dict(family=fam, name=fam, key=key, draw_rank=i+1, stratum='wm_top', form=form_family(spec),
                n_roles=len(spec['roles']), extended=bool(H.is_extended(spec)), trigger=spec['trigger'],
                zone=spec['zone'], zone_d=spec.get('zone_d'), hold_s=sorted({r['hold_s'] for r in spec['roles']}),
                strategy=spec, strategy_sha256=sha_bytes(_canon_json(spec).encode()),
                wm=dict(score=row['score'], gain_vs_base=row['gain_vs_base'],
                        gain_vs_base_lcb=row['gain_vs_base_lcb'], feasible=row['feasible']),
                tokens=sorted(list(x) for x in form_tokens(spec))), None


def map_rows(rows):
    """MAPPING_RULE over the whole file -> (candidates, refusals [(rank, family, reason)])."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= N_MAX:
        return [], [(0, None, f'{len(rows) if isinstance(rows, list) else type(rows).__name__} rows (need 1..{N_MAX})')]
    cands, bad = [], []
    for i, row in enumerate(rows):
        c, why = map_row(i, row)
        if why:
            bad.append((i+1, row.get('family') if isinstance(row, dict) else None, why))
        else:
            cands.append(c)
    if len({c['family'] for c in cands}) != len(cands) or len({c['key'] for c in cands}) != len(cands):
        bad.append((0, None, 'duplicate family or strategy key'))
    return cands, bad


def candidates_doc(top_bytes, top_sha):
    """Pure: the sealed candidate document of a pinned top file (raises SystemExit on any mapping refusal)."""
    rows = json.loads(top_bytes.decode('utf-8'))
    cands, bad = map_rows(rows)
    if bad:
        raise SystemExit(json.dumps(dict(MAPPING_REFUSED=bad), default=str))
    cov = covered_tokens()
    for c in cands:
        c['uncovered'] = sorted(list(t) for t in form_tokens(c['strategy']) - cov)
        c['mech'] = bool(c['uncovered'])
    return D2.seal_doc(dict(
        schema='coopmwm.a80.candidates.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL, writer=WRITER,
        source=dict(file=TOP_NAME, sha256=top_sha, rows=len(rows)), mapping_rule=MAPPING_RULE, form_rule=FORM_RULE,
        mech_rule=MECH_RULE, executor=dict(module='tools/coopmwm/a75_holdforms.py', a75_sha256=A75_SHA256),
        covered_tokens=sorted(list(t) for t in cov), n_candidates=len(cands),
        candidates_by_form=dict(sorted(Counter(c['form'] for c in cands).items())),
        n_extended=sum(c['extended'] for c in cands), mech_plans=[c['family'] for c in cands if c['mech']],
        world_model='plans and their order come from the WM search file; no WM number is used by any SUMO rule',
        STOP=not cands, candidates=cands))


# ------------------------------------------------------------------------------------------------ receipt invariants

ONSET_ENDS = ('hold_end', 'at_drop', 'left')
GAP_ENDS = ('hold_end', 'gap_arm_end', 'at_drop', 'left')
EVENT_ENDS = ('timer', 'clear', 'at_drop', 'left', 'hold_end', 'gap_arm_end')


def plan_checks(r, spec, horizon=HORIZON):
    """Pure. r = a75 receipts of ONE A75_PLAN episode of ``spec``. -> (errors, exercised tokens, stats).
    Checks the executor's own records against the plan: member count per activation, release reasons and timed
    lengths per role, event start kind / onset second / gap trigger window / per-session event count / issued seconds
    (pulse count or every second) / timer length; open sessions only near the episode end."""
    H = _H()
    s = H.canonical(spec)
    roles, ext = s['roles'], H.is_extended(s)
    errs, ex = [], set()
    if r.get('refused') not in (None, {}, []):
        errs.append(f'refused {r.get("refused")}')
    acts = list(r.get('acts') or [])
    if acts:
        ex |= {t for t in form_tokens(s) if t[0] in ('plain', 'ext_trigger', 'ext_zone')}
    for act in acts:
        mem = act['members']
        if len(mem) != len(roles):
            errs.append(f'act@{act["t"]}: {len(mem)} members for {len(roles)} roles')
            continue
        if ext:
            ex.add(('ext_roles', '+'.join(sorted({role_kind(x) for x in roles}))))
        for i, vid in enumerate(mem):
            role = roles[i]
            on, hold = int(role['onset_s']), int(role['hold_s'])
            rel = (act.get('release') or {}).get(vid)
            gap = role_kind(role) == 'gap'
            need = on + hold + (int(role.get('gap_arm_s') or 0) if gap else 0)
            if rel is None:
                if int(act['t']) + need < horizon - 1:
                    errs.append(f'{vid}@{act["t"]}: session never released')
                continue
            el, why = int(rel[0]), rel[1]
            if why not in (GAP_ENDS if gap else ONSET_ENDS):
                errs.append(f'{vid}@{act["t"]}: release {why}')
            if not gap and why == 'hold_end' and el != on + hold:
                errs.append(f'{vid}@{act["t"]}: role {i} held {el} s != onset + hold_s {on + hold}')
            if gap and why == 'gap_arm_end' and el < on + int(role['gap_arm_s']):
                errs.append(f'{vid}@{act["t"]}: gap_arm_end at {el} s < onset + gap_arm_s')
    if not ext:
        return errs, ex, dict(activations=len(acts), events=0)
    per = Counter()
    events = list(r.get('holdforms') or [])
    for e in events:
        own = [a for a in acts if e['vid'] in a['members'] and int(a['t']) <= int(e['fire_t'])]
        if not own:
            errs.append(f'{e["vid"]}@{e["fire_t"]}: event without an activation')
            continue
        act = max(own, key=lambda a: int(a['t']))
        i = act['members'].index(e['vid'])
        role = roles[i]
        k = role_kind(role)
        if e['start_kind'] != k:
            errs.append(f'{e["vid"]}@{e["fire_t"]}: start {e["start_kind"]} for a {k} role')
            continue
        if k == 'onset' and int(e['fire_t']) != int(act['t']) + int(role['onset_s']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: onset event not at activation + onset_s')
        if k == 'gap':
            per[(e['vid'], int(act['t']))] += 1
            if e.get('trigger_vid') is None or not (float(role['gap_dx_lo']) <= float(e['dx_m'])
                                                    <= float(role['gap_dx_hi'])):
                errs.append(f'{e["vid"]}@{e["fire_t"]}: trigger / dx {e.get("trigger_vid")} {e.get("dx_m")} outside')
            if int(e['fire_t']) >= int(act['t']) + int(role['onset_s']) + int(role['gap_arm_s']):
                errs.append(f'{e["vid"]}@{e["fire_t"]}: fired after the arm window')
        if int(e['issued_s']) > int(role['hold_s']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: issued {e["issued_s"]} > hold_s')
        if e.get('end_t') is None:
            if int(e['fire_t']) + int(role['hold_s']) < horizon - 1:
                errs.append(f'{e["vid"]}@{e["fire_t"]}: event never ended')
            continue
        d = int(e['end_t']) - int(e['fire_t'])
        why = e.get('end_why')
        if why not in EVENT_ENDS or d > int(role['hold_s']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: end {why} after {d} s')
        if why == 'clear' and role.get('gap_release') != 'clear':
            errs.append(f'{e["vid"]}@{e["fire_t"]}: clear end for a timer role')
        if why in ('timer', 'hold_end') and d != int(role['hold_s']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: {why} after {d} s != hold_s')
        want = sum(bool(H.pulse_on(role, h)) for h in range(d))
        if int(e['issued_s']) != want:
            errs.append(f'{e["vid"]}@{e["fire_t"]}: issued {e["issued_s"]} != {want} law seconds')
        ex.add(('role', k, role_letters(role)))
        if int(role['onset_s']) > 0:
            ex.add(('onset_delay', k))
        if k == 'gap':
            ex.add(('gap_drop', role.get('gap_drop_m') is not None))
            if why == 'clear' or (why == 'timer' and role.get('gap_release', 'timer') == 'timer'):
                ex.add(('gap_release', why))
    for (vid, t0), n in per.items():
        role = roles[next(a for a in acts if int(a['t']) == t0 and vid in a['members'])['members'].index(vid)]
        if n > int(role.get('gap_events', 1)):
            errs.append(f'{vid}@{t0}: {n} gap events > gap_events')
        if n > 1:
            ex.add(('gap_events_multi', True))
        elif n == 1 and int(role.get('gap_events', 1)) == 1:
            ex.add(('gap_events_multi', False))
    return errs, ex, dict(activations=len(acts), events=len(events),
                          gap_fired=sum(e['start_kind'] == 'gap' for e in events))


def mech_verdict(rows, safety, uncovered):
    """Pure. rows: {family: [per-world dict(errors, exercised)]}; safety: [(family, world, safety)];
    uncovered: {family: [token]} -> (verdict, failures, exercise table)."""
    fails = [f'{fam} {x["world_id"]}: {x["errors"][:3]}' for fam, xs in rows.items() for x in xs if x['errors']]
    for fam, w, sf in safety:
        if any(int(sf.get(k) or 0) for k in D2.SAFETY_ZERO):
            fails.append(f'{fam} {w}: collisions/teleports {[sf.get(k) for k in D2.SAFETY_ZERO]}')
    table = {}
    for fam, toks in uncovered.items():
        seen = set()
        for x in rows.get(fam, ()):
            seen |= {tuple(t) for t in x['exercised']}
        table[fam] = {'/'.join(map(str, t)): ('exercised' if tuple(t) in seen else NOT_EXERCISED) for t in toks}
    return ('PASS' if not fails else 'FAIL'), fails, table


def cap_check(used_stage, used_total, need, stage):
    cap = dict(verify=VERIFY_CAP, confirm=CONFIRM_CAP)[stage]
    if used_stage+need > cap or used_total+need > CAP_SIM_S:
        raise SystemExit(f'{stage}: {used_stage} + {need} > stage cap {cap} (or phase cap {CAP_SIM_S})')
    return True


# ================================================================================================ host side

_ROOT = None
#: the A75-D2 module names re-pointed by _point_d2 (tests restore them)
D2_NAMES = ('AMENDMENT', 'STAGE', 'PHASE', 'CAP_SIM_S', 'VERIFY_CAP', 'CONFIRM_CAP', 'MECH_CAP', 'STAGE_CAP', 'PREFIX',
            'STAGE_PREFIXES', 'BANK_DIR', 'CANDS_BANK', 'D2_BANK', 'MECH_FILE', 'ADMISSION_FILE', 'RESULTS_FILE',
            'LOG_FILE', 'DROP_MANIFEST', 'ARM', 'CONFIRM_WAVE', 'FORMS', 'AUTHORITY', 'DISCLOSURE', 'DROP_RULE',
            'ADMIT_RULE', 'BANK_RULE', 'PRIMARY_RULE', 'core', 'candidates', 'verify_waves')


def _point_d2():
    """Re-point the A75-D2 module names at this root's names (functions of D2 read them at call time)."""
    D2.AMENDMENT, D2.STAGE, D2.PHASE = AMENDMENT, STAGE, PHASE
    D2.CAP_SIM_S = CAP_SIM_S
    D2.VERIFY_CAP, D2.CONFIRM_CAP, D2.MECH_CAP = VERIFY_CAP, CONFIRM_CAP, 0
    D2.STAGE_CAP = dict(mech=0, verify=VERIFY_CAP, confirm=CONFIRM_CAP)
    D2.PREFIX = dict(D2.PREFIX, v=PREFIX['v'], c=PREFIX['c'], i='a80i_unused_', m='a80m_unused_')
    D2.STAGE_PREFIXES = dict(mech=(), verify=(PREFIX['v'],), confirm=(PREFIX['c'],))
    D2.BANK_DIR = 'a80_banks'
    D2.CANDS_BANK = f'A80_CANDS_{CELL}.json'
    D2.D2_BANK = f'A80_BANK_{CELL}.json'
    D2.MECH_FILE = 'A80_MECH.json'
    D2.ADMISSION_FILE = f'A80_ADMISSION_{CELL}.json'
    D2.RESULTS_FILE = f'A80_RESULTS_{CELL}.json'
    D2.LOG_FILE = 'A80_LOG.jsonl'
    D2.DROP_MANIFEST = 'A80_DROPPED_RAW.jsonl'
    D2.ARM = ARM
    D2.CONFIRM_WAVE = f'A80C_{ARM}'
    D2.FORMS = FORMS
    D2.AUTHORITY, D2.DISCLOSURE, D2.DROP_RULE = AUTHORITY, DISCLOSURE, DROP_RULE
    D2.ADMIT_RULE, D2.BANK_RULE, D2.PRIMARY_RULE = ADMIT_RULE, BANK_RULE, PRIMARY_RULE
    D2.core = core
    D2.candidates = candidates
    D2.verify_waves = verify_waves


def core(which):
    """The a42 driver code (as A75-D2 uses it) with this root's phase / caps / prefixes / bank dir / stop file."""
    N, A50 = D2._mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42=STOP_FILE,
        HOST_BASE=os.environ.get(HOST_BASE_ENV, '/root/a78s2b'),
        VERIFY_SETTING=D2.SETTINGS[which], CONFIRM_SETTING=D2.SETTINGS[which],
        BANK_DIR=D2.BANK_DIR, IDENT_WAVE='A80_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A80', JOB_PREFIX='a80',
        VERIFY_TAG='a80v', CONFIRM_TAG='a80c', IDENT_TAG='a80i',
        VERIFY_WAVE_PREFIX='A80V_', CONFIRM_WAVE_PREFIX='A80C_',
        RESULTS_FILE=D2.RESULTS_FILE, LOG_FILE=D2.LOG_FILE, SCHEMA_PREFIX='coopmwm.a80',
        RUNNER_RANGES_ATTR=D2.RANGES[which],
        RESULT_LABEL='amendment 80; POST-HOC exploratory direct SUMO test of the A79 S2b WM search top plans on '
                     'MAPPO60C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a80_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (D2.SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = D2.check_d2
    return m


def top_env(env=None):
    env = os.environ if env is None else env
    p, s = env.get(TOP_ENV), env.get(TOP_SHA_ENV)
    if not p or not s:
        raise SystemExit(f'{TOP_ENV} and {TOP_SHA_ENV} are both required (the registered file and its sha256)')
    return Path(p), check_sha(s)


def pin_top(root, env=None):
    """Zero SUMO, write-once: MAPPING_RULE, the byte copy of the top file and the sealed candidate document."""
    N = _N()
    import a37_celltailor as C37
    root = Path(root)
    path, want = top_env(env)
    raw = path.read_bytes()
    root.mkdir(parents=True, exist_ok=True)
    if sha_bytes(raw) != want:
        raise N.Blocked(f'{path}: sha256 {sha_bytes(raw)} != registered {want}')
    if (root/CANDS_DOC).is_file():
        raise N.Blocked(f'{CANDS_DOC} exists (pin once)')
    try:
        doc = candidates_doc(raw, want)
    except SystemExit as e:
        N.publish(root/REFUSED_DOC, N.seal(dict(schema='coopmwm.a80.mapping_refused.v1', amendment=AMENDMENT,
                                                 source=dict(file=str(path), sha256=want), rule=MAPPING_RULE,
                                                 detail=str(e), consequence='STOP: nothing runs'), 'receipt_sha256'))
        raise
    p = C37.write_once(root/TOP_COPY, raw.decode('utf-8'))
    if N.file_sha(p) != want:
        raise N.Blocked(f'{TOP_COPY}: copy is not byte-identical')
    C37.write_once(root/CANDS_DOC, json.dumps(doc, indent=1, sort_keys=True)+'\n')
    return dict(top_sha256=want, doc_sha256=doc['doc_sha256'], n=doc['n_candidates'], mech_plans=doc['mech_plans'],
                by_form=doc['candidates_by_form'], n_extended=doc['n_extended'])


def candidates(root=None):
    """The pinned candidates of this root (doc seal + recomputation from the pinned byte copy)."""
    N = _N()
    root = Path(root or _ROOT)
    p, q = root/CANDS_DOC, root/TOP_COPY
    if not p.is_file() or not q.is_file():
        raise N.Blocked(f'{CANDS_DOC} / {TOP_COPY}: missing (run pin-top)')
    doc = json.loads(p.read_text(encoding='utf-8'))
    raw = q.read_bytes()
    if D2.seal_doc(doc)['doc_sha256'] != doc.get('doc_sha256') or doc['source']['sha256'] != sha_bytes(raw):
        raise N.Blocked(f'{CANDS_DOC}: seal / top-file sha mismatch')
    if doc != candidates_doc(raw, doc['source']['sha256']):
        raise N.Blocked(f'{CANDS_DOC}: not the mapping of the pinned file (edited / other code?)')
    if doc['STOP'] or doc['cell'] != CELL or not 1 <= doc['n_candidates'] <= N_MAX:
        raise N.Blocked(f'{CANDS_DOC}: STOP / wrong cell / size')
    return doc['candidates']


def verify_waves(D, panel, root, a66_root, cands):
    N = _N()
    waves = [(f'A80V_{c["family"]}', 'episode',
              D2._plan_specs(D, panel, root, a66_root, D2.CANDS_BANK, 'A75_PLAN', c['family'],
                             f'{PREFIX["v"]}{c["key"]}'))
             for c in cands]
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def mech_waves(D, panel, root, a66_root, cands):
    """The first MECH_WORLDS specs of the verification wave of every candidate with an uncovered token."""
    return [(n, k, s[:MECH_WORLDS]) for (n, k, s), c in zip(verify_waves(D, panel, root, a66_root, cands), cands)
            if c['mech']]


def mech_compare(D, panel, root, a66_root, cands):
    N = _N()
    H = _H()
    A67 = D2._mods()[2]
    root = Path(root)
    rows, safety, uncovered = {}, [], {c['family']: c['uncovered'] for c in cands if c['mech']}
    by = {c['family']: c for c in cands}
    for (wave, _, specs) in mech_waves(D, panel, root, a66_root, cands):
        fam = wave[len('A80V_'):]
        have, _ = D2._load(root, wave, specs)
        s = by[fam]['strategy']
        out = []
        for w, rec in have.items():
            r = rec['a75']
            errs, ex, st = plan_checks(r, s)
            if rec.get('executor_sha256_loaded') != A67.MAPPO60C_SHA256:
                errs.append('executor is not MAPPO60C')
            p = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
            if st['activations']:
                if not p.is_file():
                    errs.append('vehicle log missing (zone check impossible)')
                else:
                    errs += [x for x in D2.check_zone(r, H.zone_bounds(s), D2.vlog_pos(D2.read_vlog(p)))
                             if x != 'no activation']
            safety.append((fam, w, rec['safety']))
            out.append(dict(world_id=w, errors=errs, exercised=sorted(list(t) for t in ex), stats=st,
                            holdform_summary=r.get('holdform_summary'), hard_brakes=rec['safety'].get('hard_brakes')))
        rows[fam] = out
    verdict, fails, table = mech_verdict(rows, safety, uncovered)
    return N.seal(dict(schema='coopmwm.a80.mech.v1', amendment=AMENDMENT, stage=STAGE, rule=MECH_RULE,
                       a75_sha256=A75_SHA256, worlds=[r['world_id'] for r in panel['worlds'][:MECH_WORLDS]],
                       mech_plans=sorted(uncovered), verdict=verdict, failures=fails, exercise=table, rows=rows,
                       note=('no candidate carries a token outside the A75-D2 mechanics fixture: nothing to run'
                             if not uncovered else 'mechanics episodes = the first 2 verification episodes'),
                       consequence='PASS: verify / admit / confirm may run; FAIL: stop and report'), 'receipt_sha256')


def admit(D, D68, panel, root, a68_root, a66_root, ref68_reports, cands):
    """A75-D2 admit (unchanged rule / bank) after the receipt invariants on every verification record."""
    N = _N()
    root = Path(root)
    viol, exercised = [], {}
    for (name, _, specs), c in zip(verify_waves(D, panel, root, a66_root, cands), cands):
        have, _ = D2._load(root, name, specs)
        ex = set()
        for w, rec in have.items():
            errs, e, _ = plan_checks(rec['a75'], c['strategy'])
            ex |= e
            viol += [f'{c["family"]} {w}: {x}' for x in errs[:3]]
        exercised[c['family']] = sorted('/'.join(map(str, t)) for t in ex)
    if viol:
        raise N.Blocked(f'receipt invariants violated (STOP, nothing admitted): {viol[:10]}')
    v = D2.admit(D, D68, panel, root, a68_root, a66_root, ref68_reports, cands)
    extra = dict(receipt_invariants='PASS on every verification record', exercised_tokens=exercised,
                 wm=[dict(family=c['family'], draw_rank=c['draw_rank'], form=c['form'], **c['wm']) for c in cands])
    p = root/'A80_ADMISSION_EXTRA.json'
    N.publish(p, N.seal(dict(schema='coopmwm.a80.admission_extra.v1', admission_receipt_sha256=v['receipt_sha256'],
                             **extra), 'receipt_sha256'))
    return v


def mech_passed(root):
    return D2.mech_passed(root)


COMMAND_STAGE = {'mech': 'v', 'mech-compare': 'v', 'verify': 'v', 'admit': 'v', 'confirm': 'c', 'analyse': 'c',
                 'prepare': 'v'}
CAP_STAGE = {'mech': 'verify', 'verify': 'verify', 'confirm': 'confirm'}


def main(argv=None):
    global _ROOT
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('check-top', 'pin-top', 'prepare', 'dry', 'mech', 'mech-compare', 'verify',
                                       'admit', 'confirm', 'analyse', 'status', 'drop-raw'))
    p.add_argument('--final-root')
    p.add_argument('--top')
    p.add_argument('--a68-root')
    p.add_argument('--a70-root')
    p.add_argument('--a66-root')
    p.add_argument('--ref68-reports')
    p.add_argument('--ref70-reports')
    p.add_argument('--stage', choices=('v', 'c'), default='v')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    _point_d2()
    if a.command == 'check-top':
        raw = Path(a.top).read_bytes()
        doc = candidates_doc(raw, sha_bytes(raw))
        print(json.dumps(dict(sha256=sha_bytes(raw), n=doc['n_candidates'], by_form=doc['candidates_by_form'],
                              n_extended=doc['n_extended'], mech_plans=doc['mech_plans'],
                              uncovered={c['family']: c['uncovered'] for c in doc['candidates'] if c['mech']}),
                         indent=1))
        return
    N = _N()
    root = Path(a.final_root)
    _ROOT = root
    for other in (a.a68_root, a.a70_root, a.a66_root, a.ref68_reports, a.ref70_reports):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A80 final root is a new root')
    if a.command == 'pin-top':
        print(json.dumps(pin_top(root)))
        return
    which = COMMAND_STAGE.get(a.command, a.stage)
    D, panel, Dref = D2.setup(which, root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=D2.SETTINGS[which]),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/D2.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(st):
        if st == 'v':
            return verify_waves(D, panel, root, a.a66_root, candidates())
        return D2.confirm_waves(D, panel, root, a.a66_root)

    if a.command == 'prepare':
        print(json.dumps(D2.prepare(D, root, 'v')))
    elif a.command == 'dry':
        try:
            ws = waves_of(which)
        except N.Blocked as e:
            print(json.dumps(dict(stage=which, panel_sha256=panel['panel_sha256'], charged=D.charged(root),
                                  cap=CAP_SIM_S, note=str(e))))
            return
        out = dict(stage=which, panel_sha256=panel['panel_sha256'], n_waves=len(ws),
                   waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[0]['world']['seed'], s[-1]['world']['seed'])
                          for n, _, s in ws],
                   sim_s=sum(len(s)*HORIZON for _, _, s in ws), charged=D.charged(root), cap=CAP_SIM_S,
                   a75_sha256=_H().a75_digest())
        if which == 'v':
            mw = mech_waves(D, panel, root, a.a66_root, candidates())
            out['mech'] = dict(n_waves=len(mw), sim_s=sum(len(s)*HORIZON for _, _, s in mw),
                               plans=[n for n, _, _ in mw])
            have, src = D2.a68_ref(Dref, panel, a.a68_root, a.a66_root)
            worlds = [r['world_id'] for r in panel['worlds']]
            reps = D2.pinned_reports(a.ref68_reports, have, worlds, D2.ref_pins()['a68v_mappo60c'])
            out['reference'] = dict(wave='A68V_MAPPO60C', receipt_sha256=src['receipt_sha256'], n=len(have),
                                    reports_ok=all(x['ok'] for x in reps.values()))
        print(json.dumps(out, indent=1, default=str))
    elif a.command in CAP_STAGE:
        import fcntl
        if _H().a75_digest() != A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        if a.command in ('verify', 'confirm') and not mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        if a.command == 'mech' and (root/D2.MECH_FILE).is_file():
            raise N.Blocked(f'{D2.MECH_FILE} exists: mechanics ran')
        st = CAP_STAGE[a.command]
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = (mech_waves(D, panel, root, a.a66_root, candidates()) if a.command == 'mech'
                     else waves_of(which))
            pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in waves)
            cap_check(D2.stage_charged(D, root, st), D.charged(root), pending*HORIZON, st)
            drop = a.drop_raw and a.command != 'mech'
            log(dict(start=a.command, n_waves=len(waves), episodes=sum(len(w[2]) for w in waves), pending=pending,
                     workers=a.workers, chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop))

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    D2.drop_raw(root, waves, procs=a.drop_procs, log=log)
            if drop:
                D2.drop_raw(root, waves, procs=a.drop_procs, log=log)
            res = D2.execute(D, root, waves, workers=a.workers, chunk=a.chunk, log=hook, panel=panel) if waves \
                else dict(status='PASS', note='nothing to run')
            log(dict(done=a.command, **res))
    elif a.command == 'mech-compare':
        if (root/D2.MECH_FILE).is_file():
            raise N.Blocked(f'{D2.MECH_FILE} exists (one shot)')
        v = mech_compare(D, panel, root, a.a66_root, candidates())
        N.publish(root/D2.MECH_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'][:20],
                              exercise=v['exercise'])))
        if v['verdict'] != 'PASS':
            sys.exit(5)
    elif a.command == 'drop-raw':
        print(json.dumps(D2.drop_raw(root, waves_of(which), procs=a.drop_procs, log=log)))
    elif a.command == 'admit':
        if not mech_passed(root):
            raise N.Blocked('mechanics is not PASS')
        v = admit(D, Dref, panel, root, a.a68_root, a.a66_root, a.ref68_reports, candidates())
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n_tested=v['n_tested'], n_admitted=v['n_admitted'],
                              n_delay_pass=v['n_delay_pass'], per_form=v['per_form'], bank=v['bank'],
                              duplicates=v['duplicates'], consequence=v['consequence'])))
    elif a.command == 'analyse':
        v = D2.analyse(D, Dref, panel, root, a.a70_root, a.a66_root, a.ref70_reports)
        N.publish(root/D2.RESULTS_FILE, v)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, stage=which, mech=mech_passed(root),
                   admission=D2.admission(root) is not None)
        try:
            ws = waves_of(which)
            done = [N.published(root, n, s, D2.check_d2)[2] for n, _, s in ws]
            out.update(waves=len(ws), episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
