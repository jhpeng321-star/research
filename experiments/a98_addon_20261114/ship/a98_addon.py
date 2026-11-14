"""A98 (AMENDMENT 98; POST-HOC, exploratory; D2800_p60): two add-on arms on the A94/97 panel worlds 988700-988891
(panel_sha256 d85999dd..., byte copy of the A94 panel file), run in a NEW root after the A94/97 panel finished.

  BANK_S4    A75_BANK on the S4 admission's own bank file, rows unchanged (sha a428cd1d, 3 plans)
  BANK_S4S5  A75_BANK on the two-search bank (S5 + S4 rows, identical keys deduplicated; sha 40b16ed9, 6 plans; rebuilt
             in this tree with build_bank_a94.py in two-search mode)
Both on MAPPO60C / U0 base, exactly the A94 BANK_SEARCHED executor path (A81 bank arm builder, a75 d1af7e0f).

THIN WRAPPER over the registered A81 driver a81_bigconfirm.py (c755a551, unchanged) and the A94 driver a94_bank.py
(c8b05145, unchanged; only its pure helpers, setting registration and runner check are used - its point() is NOT
called). Mechanics: rule checks only, 2 episodes per arm on the A70 worlds 996300/996301 (cap 4,800). Confirm: 2 x 192 x
1,200 = 460,800. Phase cap 465,600.
Analysis (zero SUMO, after A94Q_DONE and this root's confirm): the four A94 arms' wave receipts and cached per-vehicle
reports are COPIED from /root/a94/root into <root>/ref94 (seal-verified; /root/a94 is only read), then paired contrasts
(a42 D._contrast, bootstrap 10,000 seed 20261104, diff = treatment - reference, five-group rule reported):
BANK_S4 - BANK_HAND, BANK_S4S5 - BANK_HAND, BANK_S4S5 - BANK_SEARCHED, BANK_S4 - BANK_SEARCHED, each new arm -
DUO_HOLD_60C and - MAPPO60C; all secondary / descriptive (the A94/97 primary is unchanged). Per plan: activations,
offers (fire / member_busy), overlap block, CAVs engaged at once.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault('A94_HAND_SHA256', 'none')       # a94_bank import only (its arm set is not used here)

import a81_bigconfirm as B  # noqa: E402
import a94_bank as A94  # noqa: E402

AMENDMENT = '98'
CELL = B.CELL
PHASE = 'addon_a98'
HORIZON = B.HORIZON
IDENT_N, CONFIRM_N = 2, 192
BLOCK, SEEDS_C, RESERVED_C = A94.BLOCK, A94.SEEDS_C, A94.RESERVED_C
ARMS = ('BANK_S4', 'BANK_S4S5')
MECH_EPISODES = len(ARMS)*IDENT_N
MECH_CAP, CONFIRM_CAP = MECH_EPISODES*HORIZON, len(ARMS)*CONFIRM_N*HORIZON
CAP_SIM_S = MECH_CAP+CONFIRM_CAP
assert (MECH_CAP, CONFIRM_CAP, CAP_SIM_S) == (4_800, 460_800, 465_600)
SETTINGS, RANGES = A94.SETTINGS, A94.RANGES               # the A94 setting / runner block (same worlds)
BANK_DIR = 'a98_banks'
BANK_FILE_OF = dict(BANK_S4=f'A98_BANK_S4_{CELL}.json', BANK_S4S5=f'A98_BANK_S4S5_{CELL}.json')
S4_SRC = os.environ.get('A98_S4_SRC', '/root/a85adm/root/a80_banks/A80_BANK_D2800_p60.json')
S4S5_SRC = os.environ.get('A98_S4S5_SRC', str(HERE/'a98'/f'A94_BANK_SEARCHED_{CELL}.json'))
PINS = dict(BANK_S4='a428cd1d320196578d55c84eea4b9b391f5e5c03928692a90dff0ac96667d7a8',
            BANK_S4S5='40b16ed95add6fcd63385dabcbf7e8a03fe8dbf20d0418e58e5aa8feee30ee04')
A94_ROOT = os.environ.get('A98_A94_ROOT', '/root/a94/root')
A94_ARMS = ('BANK_SEARCHED', 'BANK_HAND', 'DUO_HOLD_60C', 'MAPPO60C')
REF_DIR = 'ref94'
PREFIX = dict(i='a98i_', c='a98c_')
TAG = dict(BANK_S4='bank_s4', BANK_S4S5='bank_s4s5')
AUTHORITY = ('MAIN AMENDMENT 98 (A98, 2026-11-14 03:04; POST-HOC, exploratory): add-on arms BANK_S4 and BANK_S4S5 on the '
             'A94/97 panel worlds 988700-988891 (D2800_p60), paired with the A94/97 arms; phase addon_a98, caps mech '
             '4,800 + confirm 460,800')
DISCLOSURE = ('POST-HOC and exploratory; arms added after screening counts were known; uncorrected; same worlds as A94/97 '
              '(paired, not independent, never pooled with other panels); S4 plans are the same family as the '
              'hand-trimmed lane-2 arm of A89/A91; run after the A94/97 panel on the same host')
MECH_RULE = ('PASS iff each add-on arm on 996300 / 996301: executor MAPPO60C, a75 digest d1af7e0f, per plan no receipt '
             'invariant violation (a80_s2b.plan_checks), every activation inside that plan\'s zone +-1 m, no CAV in two '
             'sessions at once, no refusal, collisions = teleports = 0 (a plan that never activates is reported)')
PRIMARY_RULE = 'none in A98: every contrast is secondary / descriptive; the A94/97 primary is unchanged'
CONTRASTS = tuple((a, r, 'secondary') for a, r in (
    ('BANK_S4', 'BANK_HAND'), ('BANK_S4S5', 'BANK_HAND'), ('BANK_S4S5', 'BANK_SEARCHED'), ('BANK_S4', 'BANK_SEARCHED'),
    ('BANK_S4', 'DUO_HOLD_60C'), ('BANK_S4S5', 'DUO_HOLD_60C'), ('BANK_S4', 'MAPPO60C'), ('BANK_S4S5', 'MAPPO60C')))

_ORIG = {}


def point():
    if _ORIG:
        return
    names = dict(AMENDMENT=AMENDMENT, PHASE=PHASE, IDENT_N=IDENT_N, CONFIRM_N=CONFIRM_N, ARMS=ARMS,
                 MECH_EPISODES=MECH_EPISODES, MECH_CAP=MECH_CAP, CONFIRM_CAP=CONFIRM_CAP, CAP_SIM_S=CAP_SIM_S,
                 BLOCK=BLOCK, SEEDS_C=SEEDS_C, RESERVED_C=RESERVED_C, SETTINGS=SETTINGS, RANGES=RANGES,
                 PANEL_C='A94_PANEL_C.json', PANEL_C_PIN='A94_PANEL_C.sha256', BANK_DIR=BANK_DIR, PREFIX=PREFIX,
                 TAG=TAG, STAGE_CAP=dict(mech=MECH_CAP, confirm=CONFIRM_CAP), MECH_FILE='A98_MECH.json',
                 RESULTS_FILE=f'A98_RESULTS_{CELL}.json', LOG_FILE='A98_LOG.jsonl',
                 DROP_MANIFEST='A98_DROPPED_RAW.jsonl', AUTHORITY=AUTHORITY, DISCLOSURE=DISCLOSURE,
                 MECH_RULE=MECH_RULE, PRIMARY_RULE=PRIMARY_RULE,
                 DROP_RULE=B.DROP_RULE.replace('a81c_*', 'a98c_*'),
                 core=core, register_c_setting=A94.register_c_setting, _runner_check=A94._runner_check,
                 prepare=prepare, waves=waves, mech_compare=mech_compare, analyse=analyse)
    for k, v in names.items():
        _ORIG[k] = getattr(B, k)
        setattr(B, k, v)


def core(which):
    N, A50 = B._mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A98',
        HOST_BASE=os.environ.get('A98_HOST_BASE', '/root/a98'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A98_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A98', JOB_PREFIX='a98',
        VERIFY_TAG='a98v', CONFIRM_TAG=f'a98{which}', IDENT_TAG='a98i',
        VERIFY_WAVE_PREFIX='A98V_', CONFIRM_WAVE_PREFIX='A98C_',
        RESULTS_FILE=B.RESULTS_FILE, LOG_FILE=B.LOG_FILE, SCHEMA_PREFIX='coopmwm.a98',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 98; POST-HOC exploratory add-on arms on the A94/97 panel worlds',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a98_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = B._D2().check_d2
    return m


# ------------------------------------------------------------------------------------------------ banks / arms

def src_of(arm):
    return Path(S4_SRC if arm == 'BANK_S4' else S4S5_SRC)


def prepare(D, root, a75d2_root):
    """Write-once byte copies of the two registered bank files (pins fixed in code)."""
    N = B._mods()[0]
    from tools.coopmwm import a75_holdforms as H
    out = {}
    for arm in ARMS:
        src, pin, name = src_of(arm), PINS[arm], BANK_FILE_OF[arm]
        if N.file_sha(src) != pin:
            raise N.Blocked(f'{src}: not {pin[:8]}')
        q = D.C37.write_once(D.bank_path(root, name), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != pin or q.read_bytes() != src.read_bytes():
            raise N.Blocked(f'{q}: not the pinned bytes')
        doc = json.loads(q.read_text(encoding='utf-8'))
        c = H.make_core('A75_BANK', CELL, doc)
        if sorted(p['family'] for p in c.plans) != sorted(r['family'] for r in doc['rows']):
            raise N.Blocked(f'{name}: executor plans {[p["family"] for p in c.plans]}')
        out[arm] = dict(file=name, file_sha256=pin, source=str(src),
                        executor_priority=[(p['family'], p['lb95'], type(k).__name__) for p, k in zip(c.plans, c.cores)])
    return out


def bank_doc(D, root, arm):
    N = B._mods()[0]
    p = D.bank_path(root, BANK_FILE_OF[arm])
    if N.file_sha(p) != PINS[arm]:
        raise N.Blocked(f'{p}: not {PINS[arm][:8]}')
    return json.loads(p.read_text(encoding='utf-8'))


def arm_specs(D, arm, panel, root, a66_root, tag):
    """= the A94 BANK_SEARCHED construction (A81 bank arm builder, A75_BANK) on this arm's bank file."""
    _, _, A67, A68 = B._mods()
    D2 = B._D2()
    exe = A68.executor(a66_root)
    entry = A68.bank_entry(D, root, BANK_FILE_OF[arm], PINS[arm])
    base = A67.with_exec(D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family='_', tag=tag), exe, tag)
    return D2.a75_from_a37(base, 'A75_BANK', entry, family=None, tag=tag)


def waves(D, panel, root, a66_root, which):
    N = B._mods()[0]
    n = IDENT_N if which == 'i' else CONFIRM_N
    out = []
    for arm in ARMS:
        specs = arm_specs(D, arm, panel, root, a66_root, f'{PREFIX[which]}{TAG[arm]}')[:n]
        if len(specs) != n:
            raise N.Blocked(f'{arm}: {len(specs)} specs (want {n})')
        out.append((f'A98{which.upper()}_{arm}', 'episode', specs))
    if len({s['job_id'] for _, _, ss in out for s in ss}) != len(ARMS)*n:
        raise N.Blocked('duplicate job ids')
    return out


# ------------------------------------------------------------------------------------------------ mechanics

def rule_rows(root, have, arm, fams, plans, zones, worlds, fails, exe):
    """(= a94_bank._rule_rows, records passed in) per-plan rule checks of one bank arm's mechanics records."""
    N = B._mods()[0]
    D2 = B._D2()
    out = []
    for w in worlds:
        rec = have.get(w)
        if rec is None:
            fails.append(f'{arm} {w}: missing')
            continue
        r = rec.get('a75') or {}
        errs, per = [], {}
        log = Path(root)/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
        pos = None
        for fam in fams:
            rf = A94.split_family(r, fam)
            e, ex, st = A94._a80().plan_checks(rf, plans[fam]['spec'])
            errs += [f'{fam}: {x}' for x in e]
            per[fam] = dict(activations=st['activations'], events=st.get('events'),
                            exercised=sorted('/'.join(map(str, t)) for t in ex))
            if rf['acts']:
                if pos is None:
                    if not log.is_file():
                        errs.append('vehicle log missing (zone check impossible)')
                        continue
                    pos = D2.vlog_pos(D2.read_vlog(log))
                errs += [f'{fam}: {x}' for x in D2.check_zone(dict(rf, activations=len(rf['acts'])), zones[fam], pos)
                         if x != 'no activation']
        if r.get('refused') not in (None, {}, []):
            errs.append(f'refused {r.get("refused")}')
        if rec.get('executor_sha256_loaded') != exe:
            errs.append('executor is not MAPPO60C')
        if rec.get('a75_sha256_loaded') != D2.A75_SHA256:
            errs.append('a75 digest is not d1af7e0f')
        if any(int(rec['safety'].get(k) or 0) for k in D2.SAFETY_ZERO):
            errs.append('collisions / teleports')
        ov = A94.overlap_stats(r)
        if ov['cav_in_two_sessions']:
            errs.append('a CAV in two sessions at once')
        unknown = sorted({a.get('family', a.get('strategy')) for a in r.get('acts') or []} - set(fams))
        if unknown:
            errs.append(f'acts of unknown families {unknown}')
        out.append(dict(world_id=w, errors=errs[:8], per_family=per, overlap=ov, engaged=A94.engaged_hist(r)))
        if errs:
            fails.append(f'{arm} {w}: {errs[:3]}')
    return dict(families=list(fams), rows=out,
                activated={f: any(x['per_family'].get(f, {}).get('activations') for x in out)
                           or 'NOT ACTIVATED on 996300/996301' for f in fams})


def mech_compare(root, a70_root, ref70_reports, a75d2_root, a66_root):
    N, _, A67, _ = B._mods()
    from tools.coopmwm import a75_holdforms as H
    root = Path(root)
    D, panel = B.setup('i', root)
    ws = {w: s for w, _, s in waves(D, panel, root, a66_root, 'i')}
    worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
    fails, checks = [], {}
    for arm in ARMS:
        have, _ = B._load(root, f'A98I_{arm}', ws[f'A98I_{arm}'])
        doc = bank_doc(D, root, arm)
        plans = {p['family']: p for p in H.a75_bank_plans(doc, CELL)[0]}
        zones = {f: H.zone_bounds(p['spec']) for f, p in plans.items()}
        checks[arm] = rule_rows(root, have, arm, [r['family'] for r in doc['rows']], plans, zones, worlds, fails,
                                A67.MAPPO60C_SHA256)
        checks[arm]['zones'] = {f: list(z) for f, z in zones.items()}
    return N.seal(dict(schema='coopmwm.a98.mech.v1', amendment=AMENDMENT, rule=MECH_RULE,
                       panel_sha256=panel['panel_sha256'], worlds=worlds, verdict='PASS' if not fails else 'FAIL',
                       failures=fails, identity=[], rule_checks=checks,
                       consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ analysis

def copy_a94(a94_root, dst, worlds, panel_sha256):
    """Zero SUMO, read-only on the A94 root: copy the four A94 confirm wave receipts (all chunks) + the cached
    per-vehicle reports into dst (write-once dir); verify seals, worlds, job-id prefix, panel. -> {arm: {world: rec}}"""
    N = B._mods()[0]
    a94_root, dst = Path(a94_root), Path(dst)
    ch = (a94_root.parent/'chain_confirm.out')
    if not ch.is_file() or 'A94_CONFIRM_RUN_DONE' not in ch.read_text(encoding='utf-8'):
        raise N.Blocked('A94 confirm is not done')
    pin = (a94_root/'A94_PANEL_C.sha256').read_text(encoding='utf-8').split()
    if pin[3] != panel_sha256:
        raise N.Blocked('the A94 root panel is not this panel')
    dst.mkdir(parents=True, exist_ok=False)
    have, src = {}, {}
    for arm in A94_ARMS:
        files = sorted(a94_root.glob(f'WAVE_A94C_{arm}__*.json'))
        if not files:
            if arm == 'BANK_HAND':
                continue                       # A94 ran without a hand arm (hand = none)
            raise N.Blocked(f'no A94 wave for {arm}')
        recs, src[arm] = {}, []
        for f in files:
            shutil.copyfile(f, dst/f.name)
            v = N.verify(N.read(dst/f.name), 'receipt_sha256', f.name)
            src[arm].append(v['receipt_sha256'])
            for r in v['records']:
                if not r['job_id'].startswith('a94c_') or r['world_id'] in recs:
                    raise N.Blocked(f'{f.name}: unexpected record {r["job_id"]}')
                recs[r['world_id']] = r
                d = r.get('work_dir', r['job_id'])
                (dst/'evaluation'/d).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(a94_root/'evaluation'/d/'vehicles_report.json', dst/'evaluation'/d/'vehicles_report.json')
        if sorted(recs) != sorted(worlds):
            raise N.Blocked(f'A94 {arm}: worlds differ from this panel ({len(recs)} records)')
        have[arm] = recs
    return have, src


def analyse_core(D, have, reports, worlds, bootstrap, *, fams, meta):
    import numpy as np
    N = B._mods()[0]
    D2 = B._D2()
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = [D._contrast(have, reports, worlds, a, r, bootstrap, k) for a, r, k in CONTRASTS if a in have and r in have]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    for a in [x for x in have if x.startswith('BANK_')]:
        by_fam, offers, tot, eng, nact, per_world = Counter(), Counter(), Counter(), Counter(), Counter(), {}
        for w in worlds:
            r = have[a][w].get('a75') or {}
            for f, n in (r.get('activations_by_family') or {}).items():
                by_fam[f] += int(n)
            for k, n in (r.get('offers') or {}).items():
                offers[k] += int(n)
            ov = A94.overlap_stats(r)
            per_world[w] = ov['concurrent_s']
            for k in ('concurrent_s', 'acts_started_while_other_active', 'cav_in_two_sessions'):
                tot[k] += int(ov[k])
            for f, n in ov['active_s_by_family'].items():
                tot[f'active_s|{f}'] += int(n)
            for k, n in ov['seconds_by_n_plans_active'].items():
                nact[int(k)] += int(n)
            for k, n in A94.engaged_hist(r).items():
                eng[int(k)] += int(n)
        arms[a].update(families=list(fams.get(a, ())), activation_counts=D2.form_activations(have[a], worlds),
                       activations_by_family=dict(by_fam), offers_by_family_outcome=dict(offers),
                       offers_by_plan=A94.offers_by_plan(offers),
                       overlap=dict(tot, worlds_with_overlap=sum(1 for v in per_world.values() if v > 0),
                                    seconds_by_n_plans_active=dict(sorted(nact.items()))),
                       cavs_engaged_at_once=dict(seconds_by_k=dict(sorted(eng.items())), max_k=max(eng) if eng else 0))
    return N.seal(dict(schema='coopmwm.a98.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, n_worlds=len(worlds), bootstrap=bootstrap, rule=PRIMARY_RULE,
                       contrasts=rows, arms=arms, families={k: list(v) for k, v in fams.items()},
                       multiplicity='all contrasts secondary / descriptive, uncorrected', **meta), 'analysis_sha256')


def summary(v):
    return dict(analysis_sha256=v['analysis_sha256'],
                contrasts=[dict(contrast=f"{r['treatment']} - {r['reference']}", diff=r['total_minus_ref']['diff'],
                                ci95=r['total_minus_ref']['ci95'], groups_all_pass=(r.get('groups') or {}).get('all_pass'))
                           for r in v['contrasts']],
                banks={a: dict(concurrent_s=v['arms'][a]['overlap']['concurrent_s'],
                               activations_by_family=v['arms'][a].get('activations_by_family'))
                       for a in v['arms'] if 'overlap' in v['arms'][a]})


def analyse(D, panel, root, a66_root):
    N = B._mods()[0]
    D2 = B._D2()
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    have, reports, sources = {}, {}, {}
    for (wave, _, specs), arm in zip(waves(D, panel, root, a66_root, 'c'), ARMS):
        have[arm], s = B._load(root, wave, specs)
        sources[arm] = s.get('receipt_sha256')
        reports[arm] = D2._own_reports(root, have[arm], worlds)
    h94, s94 = copy_a94(A94_ROOT, root/REF_DIR, worlds, panel['panel_sha256'])
    for arm, recs in h94.items():
        have[arm] = recs
        reports[arm] = D2._own_reports(root/REF_DIR, recs, worlds)
        sources[f'a94:{arm}'] = s94[arm]
    fams = {arm: [r['family'] for r in bank_doc(D, root, arm)['rows']] for arm in ARMS}
    meta = dict(panel_sha256=panel['panel_sha256'], seeds=[SEEDS_C[0], SEEDS_C[-1]], root=str(root),
                a94_root=str(A94_ROOT), banks={a: dict(file=BANK_FILE_OF[a], file_sha256=PINS[a]) for a in ARMS},
                sources=sources, a94_arms_present=sorted(h94))
    return analyse_core(D, have, reports, worlds, bootstrap, fams=fams, meta=meta)


# ------------------------------------------------------------------------------------------------ CLI

def dry(root, a, which):
    D, panel = B.setup(which, root)
    if panel is None:
        raise SystemExit('panel missing')
    ws = waves(D, panel, root, a.a66_root, which)
    by = {n: s for n, _, s in ws}
    W = which.upper()
    print(json.dumps(dict(
        stage=which, setting=SETTINGS[which], panel_sha256=panel['panel_sha256'], arms=list(ARMS),
        contrasts=[list(c) for c in CONTRASTS],
        waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[-1]['job_id'], s[0]['world']['seed'], s[-1]['world']['seed'])
               for n, _, s in ws],
        episodes=sum(len(s) for _, _, s in ws), sim_s=sum(len(s)*HORIZON for _, _, s in ws),
        stage_cap=MECH_CAP if which == 'i' else CONFIRM_CAP, charged=D.charged(root), cap=CAP_SIM_S,
        a75_sha256=__import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest(),
        families={arm: [r['family'] for r in bank_doc(D, root, arm)['rows']] for arm in ARMS},
        spec_diff_S4S5_vs_S4=sorted({k for x, y in zip(by[f'A98{W}_BANK_S4S5'], by[f'A98{W}_BANK_S4'])
                                     for k in B.spec_diff(x, y)})), indent=1, default=str))


def _args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('command')
    for k in ('--final-root', '--a66-root', '--a70-root', '--ref70-reports', '--a75d2-root'):
        p.add_argument(k)
    p.add_argument('--stage', choices=('i', 'c'), default='c')
    return p.parse_args(argv)


def main(argv=None):
    point()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'dry':
        a = _args(argv)
        root = Path(a.final_root)
        if a.stage == 'c':
            A94.register_c_setting(root)
        return dry(root, a, a.stage)
    if argv and argv[0] == 'analyse':
        N = B._mods()[0]
        a = _args(argv)
        root = Path(a.final_root)
        D, panel = B.setup('c', root)
        if panel is None:
            raise N.Blocked('panel missing')
        v = analyse(D, panel, root, a.a66_root)
        N.publish(root/B.RESULTS_FILE, v)
        print(json.dumps(summary(v), default=str))
        return
    if argv and argv[0] == 'panel-c':
        raise SystemExit('A98 uses the byte-copied A94 panel (no panel-c)')
    return B.main(argv)


if __name__ == '__main__':
    main()
