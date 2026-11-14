"""CPU-only tests for the amendment-37 driver (fake runner; no simulator, no process pool): pre-outcome candidate
files, spec construction on the mirrored bank-root bindings, the celltailor_a37 ledger caps, verify -> admit ->
eval -> analyse end to end on fake records, CELL_BANK := U0 when nothing is admitted, one-retry rule."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a37_celltailor as D  # noqa: E402
import nm_finaleval as N  # noqa: E402

MIRRORS = [os.environ.get('A37_MIRROR', ''), '/root/a37ct/root', '/root/autodl-fs/nm_finaleval_p20c_bank',
           'E:/agent-work/a32/stage/nm_finaleval_p20c_bank']
BIND = ('BOUND_U0.json', 'ORACLE_BINDING.json', 'PRERUN_PLAN.json', 'prerun_plant/CONFIG.json',
        'oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json', 'oracle_banks/LEARNED_TRIGGER_MODEL.json',
        'a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json')


def mirror():
    for m in MIRRORS:
        if m and all((Path(m)/f).is_file() for f in BIND):
            return Path(m)
    return None


@pytest.fixture
def root(tmp_path):
    src = mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    out = tmp_path/'root'
    for f in BIND:
        (out/f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src/f, out/f)
    D.prepare(out)
    return out


def use(monkeypatch, name):
    for k, v in dict(N.SETTINGS[name], SETTING=name, **N.SETTINGS_PINS.get(name, {})).items():
        monkeypatch.setattr(N, k, v)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return N.load_panel(N.PANEL_FILE)


class FakeInner:
    """Fake physics: U0-like records, J scaled per family/arm, receipts of each arm's executor."""
    physics = False

    def __init__(self, plan, factors=None, fail=()):
        self.calls, self.plan, self.factors, self.fail = [], plan, dict(factors or {}), set(fail)

    def plant_view(self, cfg):
        return deepcopy(self.plan['plant']['view'])

    def runner_status(self):
        return dict(ok=True, learned=True, native=True)

    def run(self, items, cfg, workers):
        self.calls.append(('run', len(items)))
        out = []
        for kind, spec, work in items:
            if spec['job_id'] in self.fail:
                self.fail.discard(spec['job_id'])
                raise RuntimeError(f'simulated environment failure {spec["job_id"]}')
            Path(work).mkdir(parents=True, exist_ok=True)
            ck = spec['checkpoint']
            if spec['arm'] == 'NATIVE':
                rec = N.fake_episode(spec, work)
            else:
                rec = N.fake_episode(dict(spec, arm='U0'), work)
                rec.update({k: deepcopy(spec[k]) for k in N.IDENTITY})
                rec['executor_sha256_loaded'] = ck['executor_sha256']
            rec['J_veh_s'] *= self.factors.get(ck.get('a37_family') or spec['arm'], 1.0)
            if spec['arm'] in D.A37_ARMS:
                rec['a37_sha256_loaded'] = ck['a37_sha256']
                rec['a37'] = dict(a37_arm=spec['arm'], activations=2,
                                  activations_by_family={ck.get('a37_family') or 'bank': 2})
            if spec['arm'] == 'DUO_HOLD':
                rec['a32_sha256_loaded'] = ck['a32_sha256']
                rec['a32'] = dict(a32_arm='DUO_HOLD', activations=1)
            if spec['arm'] == 'LEARNED_TRIGGER':
                rec['learned_trigger_sha256_loaded'] = ck['learned_trigger_sha256']
                rec['learned_trigger'] = dict(executor='learned_trigger', fires=1)
            out.append(rec)
        return out


def runtime(root, **kw):
    return D.A37Runtime(FakeInner(N.prerun_plan(root), **kw), root)


def log_to(lst):
    return lst.append


def test_candidate_files_are_sealed_pre_outcome(monkeypatch, tmp_path):
    for cell in D.CELLS:
        cands = D.candidates(cell)
        assert len(cands) == 9 and len({c['family'] for c in cands}) == 9
    doc = json.loads((D.CANDS_DIR/'CANDIDATES_D3200_p20.json').read_text(encoding='utf-8'))
    doc['candidates'][0]['strategy']['roles'][0]['v_w'] = 30.0
    (tmp_path/'CANDIDATES_D3200_p20.json').write_text(json.dumps(doc), encoding='utf-8')
    monkeypatch.setattr(D, 'CANDS_DIR', tmp_path)
    with pytest.raises(N.Blocked):
        D.candidates('D3200_p20')


def test_verify_specs_are_the_plant_specs_with_the_a37_arm(root, monkeypatch):
    panel = use(monkeypatch, 'a37v_D3200_p20')
    assert [r['world']['seed'] for r in panel['worlds']] == list(range(991700, 991724))
    waves = D.verify_waves('D3200_p20', panel, root)
    assert len(waves) == 10 and waves[0][0] == 'A37V_D3200_p20_U0'
    ids = [s['job_id'] for _, _, specs in waves for s in specs]
    assert len(ids) == 240 == len(set(ids)) and all(i.startswith('a37v_d3200_p20_') for i in ids)
    for name, _, specs in waves[1:]:
        ck = specs[0]['checkpoint']
        assert specs[0]['arm'] == 'A37_PLAN' and ck['kind'] == 'a37' and ck['a37_family'] == name.split('_p20_')[1]
        assert ck['a37_bank']['path'].startswith(D.HOST_ROOT) and 'bank_cell_key' not in ck
        assert 'learned_trigger_model' not in ck and ck['cell_key'] == 'D3200_p20'
        assert ck['executor_sha256'] == waves[0][2][0]['checkpoint']['executor_sha256']   # same U0 executor


def test_eval_specs_frozen_arms_transfer_and_cell_bank_needs_admission(root, monkeypatch):
    panel = use(monkeypatch, 'a35_D2400_p20')
    lt = D.eval_specs('LEARNED_TRIGGER', 'D2400_p20', panel, root)
    duo = D.eval_specs('DUO_HOLD', 'D2400_p20', panel, root)
    assert lt[0]['checkpoint']['bank_cell_key'] == 'D2800_p20' == duo[0]['checkpoint']['bank_cell_key']
    assert lt[0]['job_id'] == 'a37e_d2400_p20_learned_trigger_0000' and len(lt) == 96
    with pytest.raises(N.Blocked, match='ADMISSION'):
        D.eval_specs('CELL_BANK', 'D2400_p20', panel, root)
    ids = [s['job_id'] for s in D.ident_specs(root)]
    assert len(ids) == 8 == len(set(ids))


def test_ledger_caps_total_and_verification(root, tmp_path):
    rt = runtime(root)
    rt.verify = True
    rt.check_capacity(600_000)
    with pytest.raises(N.Blocked, match='verification cap'):
        rt.check_capacity(600_001)
    rt.reserve(None, [dict(id='a37v_x_0000', sim_s=599_000)])
    with pytest.raises(N.Blocked, match='verification cap'):
        rt.reserve(None, [dict(id='a37v_x_0001', sim_s=1_200)])
    with pytest.raises(N.Blocked, match='never mixes'):
        rt.reserve(None, [dict(id='a37v_x_0002', sim_s=1), dict(id='a37e_y_0000', sim_s=1)])
    rt.verify = False
    rt.reserve(None, [dict(id='a37e_y_0000', sim_s=1_301_000)])
    with pytest.raises(N.Blocked, match='celltailor_a37 cap'):
        rt.reserve(None, [dict(id='a37e_y_0001', sim_s=1)])
    rows = N.fe.ledger_rows(root)
    assert {r['phase'] for r in rows} == {D.A37_PHASE} and D.charged(root) == 1_900_000


def _verify_and_admit(root, monkeypatch, cell, factors, fail=()):
    panel = use(monkeypatch, f'a37v_{cell}')
    waves = D.verify_waves(cell, panel, root)
    logs = []
    out = D.execute_with_retry(root, waves, workers=4, chunk=48, log=log_to(logs), verify=True,
                               runtime=runtime(root, factors=factors, fail=fail))
    assert out['status'] == 'PASS'
    return D.admit(root, cell, panel), logs


def test_missing_group_is_not_allowed(root, monkeypatch):
    """The fake vehicle log has no origin:L0 vehicles: a group that is absent fails ALLOWED (registered rule)."""
    fams = [c['family'] for c in D.candidates('D2400_p20')]
    adm, _ = _verify_and_admit(root, monkeypatch, 'D2400_p20', {fams[0]: .9})
    v = {x['family']: x for x in adm['verdicts']}[fams[0]]
    assert v['ci95'][1] < 0 and v['allowed'] is False and not v['PASS'] and adm['admitted'] == []


def test_end_to_end_verify_admit_eval_analyse(root, monkeypatch):
    # the fake vehicle log has no origin:L0 vehicles; register the other four groups for this fixture only
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))
    fams = [c['family'] for c in D.candidates('D3200_p20')]
    adm, _ = _verify_and_admit(root, monkeypatch, 'D3200_p20', {fams[2]: .9, fams[5]: .95, fams[0]: 1.1})
    assert adm['admitted'] == [fams[2], fams[5]]                  # priority = verified lb95 desc
    assert D.charged(root, 'a37v_') == 240*1200
    v = {x['family']: x for x in adm['verdicts']}
    assert v[fams[2]]['PASS'] and not v[fams[0]]['PASS'] and v[fams[0]]['mean'] > 0
    with pytest.raises(N.Blocked, match='one shot'):
        D.admit(root, 'D3200_p20', use(monkeypatch, 'a37v_D3200_p20'))
    bank = json.loads((root/'a37_banks'/'A37_CELL_BANK_D3200_p20.json').read_text(encoding='utf-8'))
    assert [r['family'] for r in bank['rows']] == [fams[2], fams[5]]
    panel = use(monkeypatch, 'a35_D3200_p20')
    waves = D.stage_waves('eval', 'a35_D3200_p20', panel, root)
    assert [w[0] for w in waves] == [D.eval_wave(a, 'D3200_p20') for a in D.EVAL_ARMS]
    out = D.execute(root, waves, workers=4, chunk=96, log=lambda x: None,
                    runtime=runtime(root, factors=dict(CELL_BANK=.97, LEARNED_TRIGGER=.99)))
    assert out['status'] == 'PASS' and D.charged(root) == 240*1200+4*96*1200
    res = D.analyse(root, 'D3200_p20', panel)
    prim = res['contrasts'][0]
    assert (prim['treatment'], prim['reference'], prim['kind']) == ('CELL_BANK', 'LEARNED_TRIGGER', 'registered')
    assert res['primary']['supported'] is True and prim['total_minus_ref']['ci95'][1] < 0
    assert [(c['treatment'], c['reference']) for c in res['contrasts']] == list(D.CONTRASTS)+list(D.CONTEXT)
    assert res['bootstrap']['seed'] == 20261104 and 'groups' in prim and 'tail_cvar90' in prim


def test_nothing_admitted_means_cell_bank_is_u0(root, monkeypatch):
    adm, _ = _verify_and_admit(root, monkeypatch, 'D2400_p20', {})
    assert adm['admitted'] == [] and adm['cell_bank_is_u0'] and adm['cell_bank'] is None
    assert not (root/'a37_banks'/'A37_CELL_BANK_D2400_p20.json').exists()
    panel = use(monkeypatch, 'a35_D2400_p20')
    waves = D.stage_waves('eval', 'a35_D2400_p20', panel, root)
    assert [w[0] for w in waves] == [D.eval_wave(a, 'D2400_p20') for a in D.EVAL_ARMS[:3]]   # nothing to run
    D.execute(root, waves, workers=4, chunk=96, log=lambda x: None, runtime=runtime(root))
    res = D.analyse(root, 'D2400_p20', panel)
    assert res['primary']['cell_bank_is_u0'] and res['arms']['CELL_BANK']['is_u0']
    cu = next(c for c in res['contrasts'] if (c['treatment'], c['reference']) == ('CELL_BANK', 'U0'))
    assert cu['total_minus_ref']['diff'] == 0


def test_failed_episode_is_retried_once_then_stops(root, monkeypatch):
    panel = use(monkeypatch, 'a37v_D2400_p20')
    waves = D.verify_waves('D2400_p20', panel, root)[:1]
    logs = []
    out = D.execute_with_retry(root, waves, workers=4, chunk=24, log=log_to(logs), verify=True,
                               runtime=runtime(root, fail={'a37v_d2400_p20_u0_0003'}))
    assert out['status'] == 'PASS' and any('replay_authorized' in x for x in logs)
    assert len(list(root.glob('REPLAY_AUTHORIZED_*.json'))) == 1
    waves2 = D.verify_waves('D2400_p20', panel, root)[1:2]
    first = waves2[0][2][0]['job_id']
    with pytest.raises(RuntimeError):            # fails in the first run AND in its one authorised retry
        D.execute_with_retry(root, waves2, workers=4, chunk=24, log=log_to(logs), verify=True,
                             runtime=_twice(root, first))


def _twice(root, job):
    rt = runtime(root, fail={job})
    inner = rt.inner
    orig = inner.run

    def run(items, cfg, workers):
        for _, spec, _ in items:
            if spec['job_id'].split('#')[0] == job:
                inner.fail.add(spec['job_id'])
        return orig(items, cfg, workers)
    inner.run = run
    return rt


def test_native_plan_budget_rule(root, monkeypatch):
    panel = use(monkeypatch, 'a33')
    N.publish(root/'IDENTITY_GATE.json', dict(per_panel=dict(a33='PASS')), replace=True)
    assert D.native_plan(root, panel) == dict(need_u0=False, n_worlds=192, left_sim_s=1_900_000)
    N.publish(root/'IDENTITY_GATE.json', dict(per_panel=dict(a33='FAILED')), replace=True)
    rt = runtime(root)
    rt.reserve(None, [dict(id='a37e_x', sim_s=1_900_000-150_000)])
    plan = D.native_plan(root, panel)
    assert plan['need_u0'] and plan['n_worlds'] == 150_000//2400
    specs = D.native_specs(panel, root, 'NATIVE')
    assert specs[0]['checkpoint'] == dict(kind='native') and specs[0]['job_id'] == 'a37n_native_0000'
