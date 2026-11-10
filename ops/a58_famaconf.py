"""AMENDMENT 58 (S-FAMA-CONF2; registered before any outcome): the frozen A42 admitted bank on the second 60 % panel.

FAMA = the admitted A42 BIG_BANK (four plans; bank file A42_BIG_BANK_D2800_p60.json sha256 7e7dc76b..., a byte copy
in <root>/a42_banks/), runner arm CELL_BANK, the amendment-37 executor unchanged, on frozen MAPPO U0 (d9042cc5) - i.e.
exactly the A42 confirmation BIG_BANK spec, built by the A42 driver code (ops/a42_bigp60.py, private module copy with
the A58 names as A50/A52/A56) on the A50 confirmation panel a50c_D2800_p60 (worlds 996800-996895).

Reused comparators (no rerun): A50 U0 and DUO_HOLD, A52 MAPPO60 per-world records (analysis: ops/a58_analyse.py).

Identity (registered): DUO_HOLD on the first already-run A50 world (996800) through this code path must reproduce the
stored A50 record exactly (J_veh_s diff 0, identical safety counters, same world); FAILED -> stop, FAMA is not run.

Illustration (registered; not evidence): U0 and BIG_BANK on ONE A42 panel-1 world, the lower median of the stored
(BIG_BANK - U0) total-delay differences over the 96 A42 confirmation worlds (rule computed in code from the sealed
A42 receipts); the rerun must reproduce the stored A42 records (same comparison as the identity). The chain also
stops before FAMA if it does not (stricter than registered: it would mean the bank/executor does not reproduce A42
on this host).

Budget: ledger phase famaconf_a58 cap 125,000 sim-s: identity 1,200 + illustration 2,400 + FAMA 96 x 1,200 =
115,200 -> 118,800. No automatic replay (a failed dispatch raises and is reported). Own final root /root/a58/root,
job ids a58*, waves A58_IDENT_DUO_HOLD / A58X_{U0,BIG_BANK} / A58C_FAMA, stop file STOP_A58.

Commands (python ops/a58_famaconf.py <cmd> --final-root <root> [--ref-root <refs>]):
  prepare                       zero physics: check the bank byte copy (sha pin)
  dry                           spec summary + budget (zero physics)
  identity / identity-compare   1 episode; IDENTITY_A58.json (PASS / FAILED) vs <refs>/a50 WAVE_A50C_DUO_HOLD
  illustrate / illustrate-compare   2 episodes (A42 world by the median rule); ILLUSTRATION_A58.json vs <refs>/a42c
  confirm                       FAMA on the 96 worlds (refused unless identity PASS and illustration PASS)
  reports                       zero physics: vehicles_report.json for every published FAMA episode
  status / budget
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a35_transfer as T  # noqa: E402
import a50_fullp60 as A50  # noqa: E402   (only for its _clone helper)

AMENDMENT = '58'
CELL = 'D2800_p60'
CONF_SETTING = 'a50c_D2800_p60'         # panel 2 (A50 confirmation worlds 996800-996895)
ILL_SETTING = 'a42c_D2800_p60'          # panel 1 (A42 confirmation worlds 998600-998695), illustration only
BANK_FILE = f'A42_BIG_BANK_{CELL}.json'
BANK_SHA256 = '7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff'
A42_ADMITTED = ('rand_3a452b1348b01083', 'mut_2922c89d0b31d4c8', 'rand_bb1cb39d47f0a524', 'mut_d7251d07b4f749e0')
CAP_SIM_S = 125_000
IDENT_N, ILL_N, CONFIRM_N = 1, 2, 96
ARM = 'FAMA'
AUTHORITY = ('MAIN AMENDMENT 58 (owner 2026-11-11): S-FAMA-CONF2, the frozen admitted A42 BIG_BANK (7e7dc76b) on the '
             'A50 panel 996800-996895, paired with the stored A50 U0 / DUO_HOLD and A52 MAPPO60; one illustration '
             'world from panel 1; phase famaconf_a58 cap 125,000')
OVERRIDES = dict(
    A42_PHASE='famaconf_a58', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A58',
    HOST_BASE=os.environ.get('A58_HOST_BASE', '/root/a58'),
    VERIFY_SETTING=ILL_SETTING, CONFIRM_SETTING=CONF_SETTING,
    BANK_DIR='a42_banks', IDENT_WAVE='A58_IDENT_DUO_HOLD', AMENDMENT=AMENDMENT, LABEL='A58', JOB_PREFIX='a58',
    VERIFY_TAG='a58x', CONFIRM_TAG='a58c', IDENT_TAG='a58i', VERIFY_WAVE_PREFIX='A58X_', CONFIRM_WAVE_PREFIX='A58C_',
    CANDS_BANK_FILE='A58_UNUSED.json', BIG_BANK_FILE=BANK_FILE,
    ADMISSION_FILE='ADMISSION_A58_UNUSED.json', RESULTS_FILE='FAMACONF.json', LOG_FILE='A58_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a58', RUNNER_RANGES_ATTR='A50_RANGES',
    RESULT_LABEL='amendment 58; confirmation of the A42 bank on panel 2; never pooled; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a58_famaconf_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()
CONFIRM_WAVE = f'A58C_{ARM}'
ILL_WAVES = ('A58X_U0', 'A58X_BIG_BANK')
IDENTITY_FILE = 'IDENTITY_A58.json'
ILLUSTRATION_FILE = 'ILLUSTRATION_A58.json'


# ------------------------------------------------------------------------------------------------ bank / specs

def bank_entry(final_root, *, check_files=True):
    entry = D._bank_entry(final_root, BANK_FILE, check_files=check_files)
    if entry is None:
        return dict(path=f'{D.HOST_ROOT}/{D.BANK_DIR}/{BANK_FILE}', file_sha256=BANK_SHA256)
    if entry['file_sha256'] != BANK_SHA256:
        raise N.Blocked(f'{BANK_FILE}: sha {entry["file_sha256"][:12]} is not the admitted A42 bank {BANK_SHA256[:12]}')
    return entry


def prepare(final_root):
    from tools.coopmwm import a37_cellbank as C
    p = D.bank_path(final_root, BANK_FILE)
    entry = bank_entry(final_root)
    doc = json.loads(p.read_text(encoding='utf-8'))
    fams = [r.get('family') for r in doc.get('plans', doc.get('rows', []))]
    return dict(bank=entry, families=fams, a37_sha256=C.a37_digest())


def bank_specs(panel, final_root, tag, *, check_files=True):
    """The A42 confirmation BIG_BANK spec (runner arm CELL_BANK, a37 executor) with the byte-copied A42 bank."""
    return D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, bank_entry(final_root, check_files=check_files),
                        tag=tag)


def confirm_specs(panel, final_root, *, check_files=True):
    return bank_specs(panel, final_root, f'{D.CONFIRM_TAG}_{ARM.lower()}', check_files=check_files)


def ident_specs(panel, final_root, *, check_files=True):
    """DUO_HOLD (the A50 confirmation spec, T._arm_specs) on the first panel-2 world."""
    specs = T._arm_specs('DUO_HOLD', panel, final_root, check_files=check_files)[:IDENT_N]
    return D._retag(specs, f'{D.IDENT_TAG}_duo_hold')


# ------------------------------------------------------------------------------------------------ stored records

def stored_records(ref_dir, wave, expect_n):
    """{world_id: record} from sealed receipts WAVE_<wave>__*.json of an offloaded copy (seals verified)."""
    ref_dir = Path(ref_dir)
    files = sorted(ref_dir.glob(f'WAVE_{wave}__*.json'))
    if not files:
        raise N.Blocked(f'{ref_dir}: no WAVE_{wave} receipts')
    out = {}
    for p in files:
        v = N.verify(N.read(p), 'receipt_sha256', p.name)
        for r in v['records']:
            if r['world_id'] in out:
                raise N.Blocked(f'{wave}: duplicate {r["world_id"]}')
            out[r['world_id']] = r
    if len(out) != expect_n:
        raise N.Blocked(f'{wave}: {len(out)} records, expected {expect_n}')
    return out


def illustration_world(a42_ref):
    """Registered rule: the panel-1 world whose stored (BIG_BANK - U0) total-delay difference is the panel median
    (lower median of 96 = the 48th smallest, index 47)."""
    u0 = stored_records(a42_ref, 'A42C_U0', 96)
    bb = stored_records(a42_ref, 'A42C_BIG_BANK', 96)
    if set(u0) != set(bb):
        raise N.Blocked('A42 U0 / BIG_BANK worlds differ')
    diffs = sorted((bb[w]['J_veh_s']-u0[w]['J_veh_s'], w) for w in u0)
    d, w = diffs[(len(diffs)-1)//2]
    return dict(world_id=w, diff_veh_s=d, rank=(len(diffs)-1)//2+1, n=len(diffs),
                u0_J=u0[w]['J_veh_s'], big_bank_J=bb[w]['J_veh_s'])


def ill_specs(panel, final_root, world_id, *, check_files=True):
    u0 = [s for s in T._arm_specs('U0', panel, final_root, check_files=check_files) if s['world_id'] == world_id]
    bb = [s for s in bank_specs(panel, final_root, f'{D.VERIFY_TAG}_big_bank', check_files=check_files)
          if s['world_id'] == world_id]
    if len(u0) != 1 or len(bb) != 1:
        raise N.Blocked(f'{world_id}: not exactly one U0 / BIG_BANK spec on {ILL_SETTING}')
    return [(ILL_WAVES[0], 'episode', [dict(u0[0], job_id=f'{D.VERIFY_TAG}_u0_0000')]),
            (ILL_WAVES[1], 'episode', [dict(bb[0], job_id=f'{D.VERIFY_TAG}_big_bank_0000')])]


def _compare(got, ref, arm):
    return dict(arm=arm, world_id=got['world_id'], j_a58=got['J_veh_s'], j_ref=ref['J_veh_s'],
                abs_diff=abs(got['J_veh_s']-ref['J_veh_s']), safety_a58=got['safety'], safety_ref=ref['safety'],
                safety_equal=got['safety'] == ref['safety'], same_world=got['world'] == ref['world'],
                activations=[(got.get('a37') or {}).get('activations'), (ref.get('a37') or {}).get('activations')])


def _ok(rows, n):
    return len(rows) == n and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world'] for x in rows)


def identity_compare(final_root, panel, ref_root):
    specs = ident_specs(panel, final_root)
    got, _, cov = N.published(final_root, D.IDENT_WAVE, specs, D.C37.check_a37)
    if cov != len(specs):
        raise N.Blocked(f'{D.IDENT_WAVE}: {cov}/{len(specs)} published')
    ref = stored_records(Path(ref_root)/'a50', 'A50C_DUO_HOLD', 96)
    rows = [_compare(r, ref[r['world_id']], 'DUO_HOLD') for r in got]
    return N.seal(dict(schema='coopmwm.a58.identity.v1', amendment=AMENDMENT,
                       rule=('DUO_HOLD on the first A50 confirmation world through the A58 path reproduces the stored '
                             'A50 record exactly: J_veh_s difference 0, identical safety counters, same world'),
                       verdict='PASS' if _ok(rows, IDENT_N) else 'FAILED', worlds=rows,
                       consequence='PASS: FAMA may run; FAILED: stop and report, FAMA is not run'), 'receipt_sha256')


def illustration_compare(final_root, panel, ref_root):
    a42 = Path(ref_root)/'a42c'
    pick = illustration_world(a42)
    rows = []
    for (name, _, specs), arm in zip(ill_specs(panel, final_root, pick['world_id']), ('U0', 'BIG_BANK')):
        got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
        if cov != len(specs):
            raise N.Blocked(f'{name}: {cov}/{len(specs)} published')
        ref = stored_records(a42, f'A42C_{arm}', 96)
        rows += [_compare(r, ref[r['world_id']], arm) for r in got]
    return N.seal(dict(schema='coopmwm.a58.illustration.v1', amendment=AMENDMENT, world=pick,
                       rule=('illustration world = lower median of stored A42 (BIG_BANK - U0) total delay over the 96 '
                             'panel-1 worlds; the rerun reproduces the stored A42 records exactly (J diff 0, safety)'),
                       verdict='PASS' if _ok(rows, ILL_N) else 'FAILED', worlds=rows,
                       work_dirs={n: s[0]['job_id'] for n, _, s in ill_specs(panel, final_root, pick['world_id'])},
                       consequence='PASS: figure from these logs; FAILED: stop (no figure, FAMA not run)'),
                  'receipt_sha256')


def passed(final_root, name):
    p = Path(final_root)/name
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan():
    planned = (IDENT_N+ILL_N+CONFIRM_N)*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, identity=IDENT_N*N.HORIZON_S,
                illustration=ILL_N*N.HORIZON_S, confirmation=CONFIRM_N*N.HORIZON_S, planned=planned,
                margin=CAP_SIM_S-planned)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('prepare', 'dry', 'identity', 'identity-compare', 'illustrate',
                                       'illustrate-compare', 'confirm', 'reports', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--ref-root')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=97)
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    setting = ILL_SETTING if a.command.startswith('illustrate') else CONF_SETTING
    panel = D.setup(setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=setting), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def ill_waves():
        if not a.ref_root:
            p.error('--ref-root required')
        return ill_specs(panel, root, illustration_world(Path(a.ref_root)/'a42c')['world_id'])
    if a.command == 'prepare':
        print(json.dumps(prepare(root)))
    elif a.command == 'dry':
        if a.ref_root is None:
            p.error('--ref-root required')
        ident = ident_specs(panel, root)
        conf = confirm_specs(panel, root)
        pick = illustration_world(Path(a.ref_root)/'a42c')
        print(json.dumps(dict(
            budget=budget_plan(), charged=D.charged(root), panel_sha256=panel['panel_sha256'],
            ident=[(s['job_id'], s['world_id'], s['arm']) for s in ident],
            confirm=dict(n=len(conf), first=conf[0]['job_id'], last=conf[-1]['job_id'], arm=conf[0]['arm'],
                         seeds=[conf[0]['world']['seed'], conf[-1]['world']['seed']],
                         cell=conf[0]['world']['cell_key'], bank=conf[0]['checkpoint']['a37_bank'],
                         a37_sha256=conf[0]['checkpoint']['a37_sha256']),
            illustration_world=pick), indent=1, default=str))
    elif a.command in ('identity', 'illustrate', 'confirm'):
        import fcntl
        if a.command == 'confirm' and not (passed(root, IDENTITY_FILE) and passed(root, ILLUSTRATION_FILE)):
            raise N.Blocked(f'{IDENTITY_FILE} / {ILLUSTRATION_FILE} not PASS: FAMA is not run')
        if a.command == 'illustrate' and not passed(root, IDENTITY_FILE):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = [(D.IDENT_WAVE, 'episode', ident_specs(panel, root))]
            elif a.command == 'illustrate':
                waves = ill_waves()
            else:
                waves = [(CONFIRM_WAVE, 'episode', confirm_specs(panel, root))]
            need = sum(len(s) for _, _, s in waves)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay (A58 registration): a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel, a.ref_root)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['world_id'], x['abs_diff'], x['safety_equal'])
                                                            for x in v['worlds']])))
    elif a.command == 'illustrate-compare':
        v = illustration_compare(root, panel, a.ref_root)
        N.publish(root/ILLUSTRATION_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], world=v['world']['world_id'],
                              worlds=[(x['arm'], x['abs_diff'], x['safety_equal']) for x in v['worlds']])))
    elif a.command == 'reports':     # zero physics: vehicles_report.json next to every published FAMA episode log
        specs = confirm_specs(panel, root)
        recs, _, cov = N.published(root, CONFIRM_WAVE, specs, D.C37.check_a37)
        for r in recs:
            rep = N.episode_log_report(root, r)
            if rep is None or not rep['ok']:
                raise N.Blocked(f'{r["job_id"]}: report failed')
        print(json.dumps(dict(reports=len(recs), covered=cov)))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=passed(root, IDENTITY_FILE),
                   illustration_pass=passed(root, ILLUSTRATION_FILE))
        specs = confirm_specs(panel, root, check_files=False)
        out[CONFIRM_WAVE] = f'{N.published(root, CONFIRM_WAVE, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
