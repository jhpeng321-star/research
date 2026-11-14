"""CPU-only tests for the amendment-73 driver (fake runner; no simulator, no process pool): budget arithmetic, seed
blocks, the registered selection, specs of every stage at D2800_p20 on U0, pins, caps, identity / lc-report / admission /
analysis on fake records."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a73_lcside20 as D  # noqa: E402
import nm_finaleval as N  # noqa: E402

MIRRORS = [os.environ.get('A37_MIRROR', ''), '/root/a37ct/root', 'E:/agent-work/a32/stage/nm_finaleval_p20c_bank']
BIND = ('BOUND_U0.json', 'ORACLE_BINDING.json', 'PRERUN_PLAN.json', 'prerun_plant/CONFIG.json',
        'oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json', 'oracle_banks/LEARNED_TRIGGER_MODEL.json',
        'a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json')
SEARCHES = [os.environ.get('A73_SEARCH', ''),
            'E:/research-dense/runs/active/C4-V7FT-20261017/a73_search_p20/search_lc_p20/STRATEGY_SEARCH_D2800_p20.json']


def mirror():
    for m in MIRRORS:
        if m and all((Path(m)/f).is_file() for f in BIND):
            return Path(m)
    return None


@pytest.fixture
def root(tmp_path, monkeypatch):
    src = mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    if not all(p.is_file() for p in D.PANEL_FILE.values()) or not D.CANDS_FILE.is_file():
        pytest.skip('A73 panels / selection not built on this host')
    out = tmp_path/'root'
    for f in BIND:
        (out/f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src/f, out/f)
    for k in list(vars(N)):                                  # one setting per process: restore after each test
        if k.isupper():
            monkeypatch.setattr(N, k, deepcopy(getattr(N, k)) if isinstance(getattr(N, k), (dict, list)) else getattr(N, k))
    monkeypatch.setattr(N, 'SETTING', 'p60')
    pin = dict(amendment='73', candidates_doc_sha256=N.read(D.CANDS_FILE)['doc_sha256'],
               **{f'panel_{w}': N.read(D.PANEL_FILE[w])['panel_sha256'] for w in 'mvc'})
    (tmp_path/'PINS.json').write_text(json.dumps(pin), encoding='utf-8')
    monkeypatch.setattr(D, 'PINS_FILE', tmp_path/'PINS.json')
    return out


class FakeInner:
    """Fake physics: U0-like records (+ the arm's receipts); ``gain`` scales J of the named plan families."""
    physics = False
    bank_family = 'bank'

    def __init__(self, plan, lc=None, drift=(), gain=None):
        self.calls, self.plan, self.lc, self.drift, self.gain = [], plan, dict(lc or {}), set(drift), dict(gain or {})

    def plant_view(self, cfg):
        return deepcopy(self.plan['plant']['view'])

    def runner_status(self):
        return dict(ok=True, learned=True, native=True)

    def run(self, items, cfg, workers):
        self.calls.append(('run', len(items)))
        out = []
        for kind, spec, work in items:
            Path(work).mkdir(parents=True, exist_ok=True)
            ck, arm = spec['checkpoint'], spec['arm']
            rec = N.fake_episode(dict(spec, arm=arm if arm == 'LEARNED_TRIGGER' else 'U0'), work)
            rec.update({k: deepcopy(spec[k]) for k in N.IDENTITY})
            rec['executor_sha256_loaded'] = ck['executor_sha256']
            seed = spec['world']['seed']
            fam = ck.get('a44_family') or ck.get('a37_family') or self.bank_family
            ex = dict(activations=2+seed % 3, activations_by_family={fam: 2}, issued=200+seed % 7,
                      acts=[dict(t=400, family=fam, members=['main.1', 'main.2'], release={})], offers={},
                      plans=[dict(family=fam)])
            rec['J_veh_s'] *= self.gain.get(fam if arm != 'A44_BANK' else 'A44_BANK', 1.0)
            if arm == 'DUO_HOLD':
                rec['a32_sha256_loaded'] = ck['a32_sha256']
                rec['a32'] = dict(a32_arm=arm, activations=1)
            elif arm == 'A37_PLAN':
                rec['a37_sha256_loaded'] = ck['a37_sha256']
                rec['a37'] = dict(ex, a37_arm='A37_PLAN', executor='cell_bank_always')
            elif arm in ('A44_PLAN', 'A44_BANK'):
                if seed in self.drift:
                    ex['issued'] += 1
                rec['a44_sha256_loaded'] = ck['a44_sha256']
                rec['a44'] = dict(ex, a44_arm=arm, executor='cell_bank_always_lc', lc_members=self.lc.get(seed, []),
                                  lc_summary=dict(members=1, requested=1, completed=1, gave_up=0, never_requested=0))
            out.append(rec)
        return out


def run(root, waves, panel, **kw):
    M = D.mech()
    real = N.Runtime
    N.Runtime = lambda runner: FakeInner(N.prerun_plan(root), **kw)
    try:
        r = D.runtime(root)
    finally:
        N.Runtime = real
    return M.execute(root, waves, workers=2, chunk=96, log=lambda x: None, panel=panel, runtime=r), r


def lcm(vid, first, completed=None, gave_up=None):
    return dict(vid=vid, start=400, sumo_from=1, sumo_target=2, onset_s=20, first_request=first, requests=5,
                completed=completed, gave_up=gave_up, released=[140, 'hold_end'])


def test_budget_and_blocks():
    assert (D.MECH_CAP, D.VERIFY_CAP, D.CONFIRM_CAP, D.CAP_SIM_S) == (24_000, 201_600, 460_800, 686_400)
    assert D.HORIZON == N.HORIZON_S and D.BOOTSTRAP == dict(reps=10000, level=.95, seed=20261104,
                                                            method='paired_percentile')
    assert [D.block(w)[:2] for w in 'mvc'] == [(992408, 992415), (992524, 992547), (996400, 996495)]
    assert D.block('c')[2:] == (996400, 996499)
    for lo, hi, _, _ in D.SEEDS.values():
        assert not any(lo <= b1 and hi >= b0 for b0, b1 in D.FORBIDDEN)
        assert not (lo <= 992407 and hi >= 992400) and not (lo <= 992523 and hi >= 992500)   # A44 / A68 worlds used
        assert not (lo <= 996395 and hi >= 996200)                                         # A68 / A70 panels


def test_selection_is_reproducible_and_all_lane_change(tmp_path):
    from tools.coopmwm import a44_lcvariant as V
    doc = json.loads(D.CANDS_FILE.read_text(encoding='utf-8'))
    assert doc['cell'] == 'D2800_p20' and doc['amendment'] == '73' and doc['k'] == 6 and not doc['STOP']
    assert all(V.is_lc_plan(c['strategy']) and V.executor_refusal(c['strategy']) is None for c in doc['candidates'])
    assert doc['role_counts'] == {str(n): sum(c['n_roles'] == n for c in doc['candidates'])
                                  for n in sorted({c['n_roles'] for c in doc['candidates']})}
    scores = [c['score'] for c in doc['candidates']]
    assert scores == sorted(scores, reverse=True)
    search = next((Path(s) for s in SEARCHES if s and Path(s).is_file()), None)
    if search is None:
        pytest.skip('search file not on this host')
    again = D.select(search, tmp_path/'C.json')
    assert again['doc_sha256'] == doc['doc_sha256'] and again == doc
    with pytest.raises(SystemExit, match='exists'):
        D.select(search, tmp_path/'C.json')


def test_candidates_need_the_pin(root, monkeypatch, tmp_path):
    assert len(D.candidates()) == len(D.candidates(need_pin=False)) <= 6
    monkeypatch.setattr(D, 'PINS_FILE', tmp_path/'none.json')
    with pytest.raises(N.Blocked, match='registered selection'):
        D.candidates()
    panel = D.setup('m')
    with pytest.raises(N.Blocked, match='not pinned'):
        D.require_pins(panel, 'm')


def test_mechanics_stage_on_fake_records(root):
    panel = D.setup('m')
    D.require_pins(panel, 'm')
    M = D.mech()
    assert panel['cell']['cell_key'] == 'D2800_p20' and len(panel['worlds']) == 8
    M.prepare(root)
    (n37, _, s37), (n44, _, s44) = D.ident_waves(panel, root)
    assert len(s37) == len(s44) == 2 and [s['world']['seed'] for s in s44] == [992408, 992409]
    c37, c44 = s37[0]['checkpoint'], s44[0]['checkpoint']
    assert (s37[0]['arm'], s44[0]['arm'], c37['kind'], c44['kind']) == ('A37_PLAN', 'A44_PLAN', 'a37', 'a44')
    assert c37['a37_bank'] == c44['a44_bank'] and c44['a44_bank']['path'] == '/root/a73/root/a73_banks/A73_HOLD.json'
    assert c37['executor_sha256'] == c44['executor_sha256'] == N.FROZEN_MAPPO_SHA and 'bank_cell_key' not in c44
    assert s44[0]['world']['cell_key'] == 'D2800_p20' and s44[0]['job_id'] == 'a73i_a44_0000'
    out, _ = run(root, D.ident_waves(panel, root), panel)
    assert out['status'] == 'PASS' and M.charged(root) == 4*1200
    v = D.identity_compare(panel, root)
    assert v['verdict'] == 'PASS' and v['non_vacuous'] and len(v['worlds']) == 2
    lc = {992408: [lcm('main.1', 420, completed=425)], 992409: [lcm('main.2', 420, gave_up=450)]}
    ((name, _, specs),) = D.lcfix_waves(panel, root)
    assert name == 'A73M_LCFIX' and len(specs) == 8 and specs[0]['checkpoint']['a44_family'] == 'a44_fix_lc'
    out, _ = run(root, D.lcfix_waves(panel, root), panel, lc=lc)
    assert M.charged(root) == M.charged(root, D.STAGE_PREFIX['m']) == 12*1200 <= D.MECH_CAP
    r = M.lc_report(root, panel, D.LC_N)
    assert r['verdict'] == 'PASS' and r['totals']['commanded'] == 2 and r['totals']['completed'] == 1
    assert 'AMENDMENT 73' in r['authority']


def test_identity_detects_a_difference(root):
    panel = D.setup('m')
    D.mech().prepare(root)
    run(root, D.ident_waves(panel, root), panel, drift={992409})
    v = D.identity_compare(panel, root)
    assert v['verdict'] == 'FAILED' and v['worlds'][1]['differences'] == ['executor.issued']


def test_cap_and_job_prefix(root):
    D.setup('m')
    real = N.Runtime
    N.Runtime = lambda runner: FakeInner(N.prerun_plan(root))
    try:
        rt = D.runtime(root)
    finally:
        N.Runtime = real
    with pytest.raises(N.Blocked, match='cap'):
        rt.reserve(None, [dict(id=f'a73c_x_{i}', sim_s=1200) for i in range(573)])
    with pytest.raises(N.Blocked, match='a73'):
        rt.reserve(None, [dict(id='a69v_x', sim_s=1200)])


def test_verify_admit_on_fake_records(root, monkeypatch):
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))  # fake logs
    panel = D.setup('v')
    cands = D.candidates()
    D.prepare(root, cands)
    waves = D.verify_waves(panel, root, cands)
    assert [w for w, _, _ in waves] == ['A73V_U0']+[f'A73V_{c["family"]}' for c in cands]
    assert all(len(s) == 24 and s[0]['world']['seed'] == 992524 and s[-1]['world']['seed'] == 992547
               and s[0]['world']['cell_key'] == 'D2800_p20' for _, _, s in waves)
    u0, c0 = waves[0][2][0], waves[1][2][0]
    assert u0['arm'] == 'U0' and u0['checkpoint']['kind'] == 'u0' and u0['job_id'] == 'a73v_u0_0000'
    assert c0['arm'] == 'A44_PLAN' and c0['checkpoint']['a44_family'] == cands[0]['family']
    assert u0['checkpoint']['executor_sha256'] == c0['checkpoint']['executor_sha256'] == N.FROZEN_MAPPO_SHA
    assert sum(len(s) for _, _, s in waves)*1200 <= D.VERIFY_CAP
    good = cands[0]['family']
    out, _ = run(root, waves, panel, gain={good: 0.9})
    assert out['status'] == 'PASS' and D.mech().charged(root) == (len(cands)+1)*24*1200
    v = D.admit(panel, root, cands)
    assert v['admitted'] == [good] and v['reference'] == 'U0' and v['bank']['families'] == [good]
    rows = {x['family']: x for x in v['verdicts']}
    assert rows[good]['ci95'][1] < 0 and rows[good]['ADMITTED'] and rows[good]['lane_changes']['completed'] == 24
    assert all(not x['ADMITTED'] and not x['delay_test_pass'] for f, x in rows.items() if f != good)
    with pytest.raises(N.Blocked, match='one shot'):
        D.admit(panel, root, cands)


def test_confirm_and_analyse_on_fake_records(root, monkeypatch):
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a44_lcvariant as V
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))  # fake logs
    panel = D.setup('c')
    cands = D.candidates()
    with pytest.raises(N.Blocked, match='no A73 admission'):
        D.confirm_waves(panel, root)
    M = D.mech()
    plans = [dict(family=cands[0]['family'], strategy=cands[0]['strategy'], lb95_veh_s=100.0)]
    p = M.C37.write_once(root/D.BANK_DIR/D.LC_BANK, M.C37.bank_text(V.bank_doc(plans, D.CELL)))
    N.publish(root/D.ADMISSION_FILE, N.seal(dict(admitted=[cands[0]['family']],
                                                 bank=dict(file_sha256=N.file_sha(p))), 'receipt_sha256'))
    waves = D.confirm_waves(panel, root)
    assert [w for w, _, _ in waves] == ['A73C_U0', 'A73C_DUO_HOLD', 'A73C_LEARNED_TRIGGER', 'A73C_LC_BANK_20']
    assert [s[0]['arm'] for _, _, s in waves] == ['U0', 'DUO_HOLD', 'LEARNED_TRIGGER', 'A44_BANK']
    assert all(len(s) == 96 and (s[0]['world']['seed'], s[-1]['world']['seed']) == (996400, 996495)
               and s[0]['job_id'].startswith('a73c_') and 'bank_cell_key' not in s[0]['checkpoint']
               and s[0]['checkpoint']['executor_sha256'] == N.FROZEN_MAPPO_SHA for _, _, s in waves)
    duo, lt = waves[1][2][0]['checkpoint'], waves[2][2][0]['checkpoint']
    assert duo['kind'] == 'a32' and duo['duo_bank']['file_sha256'].startswith('39f384a2')
    assert lt['learned_trigger_sha256'] and C.a37_digest().startswith('0a902437')
    assert sum(len(s) for _, _, s in waves)*1200 == D.CONFIRM_CAP
    monkeypatch.setattr(FakeInner, 'bank_family', cands[0]['family'])
    out, _ = run(root, waves, panel, gain={'A44_BANK': 0.9})
    assert out['status'] == 'PASS'
    v = D.analyse(panel, root)
    rows = {(r['treatment'], r['reference']): r for r in v['contrasts']}
    assert list(rows)[:3] == [('LC_BANK_20', 'U0'), ('LC_BANK_20', 'DUO_HOLD'), ('LC_BANK_20', 'LEARNED_TRIGGER')]
    assert rows[('LC_BANK_20', 'U0')]['total_minus_ref']['ci95'][1] < 0 and v['bootstrap'] == D.BOOTSTRAP
    assert v['n_worlds'] == 96 and v['arms']['LC_BANK_20']['lane_changes']['completed'] == 96
