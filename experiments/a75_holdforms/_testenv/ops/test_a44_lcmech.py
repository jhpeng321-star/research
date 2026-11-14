"""CPU-only tests for the amendment-44 mechanics driver (fake runner; no simulator, no process pool): fixture file,
seed-argument rule, specs of both executors on the same hold bank, the lcexec_a44_mech cap, identity-compare and
lc-report on fake records, physics refused until MAIN pins the panel."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a44_lcmech as D  # noqa: E402
import nm_finaleval as N  # noqa: E402

SEEDS = '992400-992407'
MIRRORS = [os.environ.get('A37_MIRROR', ''), '/root/a37ct/root', 'E:/agent-work/a32/stage/nm_finaleval_p20c_bank']
BIND = ('BOUND_U0.json', 'ORACLE_BINDING.json', 'PRERUN_PLAN.json', 'prerun_plant/CONFIG.json',
        'oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json', 'oracle_banks/LEARNED_TRIGGER_MODEL.json',
        'a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json')


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
    if not D.PANEL_FILE.is_file():
        pytest.skip('A44 block panel not built on this host')
    out = tmp_path/'root'
    for f in BIND:
        (out/f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src/f, out/f)
    for k in list(vars(N)):                                  # one setting per process: restore after each test
        if k.isupper():
            monkeypatch.setattr(N, k, deepcopy(getattr(N, k)) if isinstance(getattr(N, k), (dict, list)) else getattr(N, k))
    monkeypatch.setattr(N, 'SETTING', 'p60')
    D.prepare(out)
    return out


class FakeInner:
    """Fake physics: U0-like records; executor receipts per arm (lc receipts from a scripted table)."""
    physics = False

    def __init__(self, plan, lc=None, unsafe=(), drift=()):
        self.calls, self.plan, self.lc, self.unsafe, self.drift = [], plan, dict(lc or {}), set(unsafe), set(drift)

    def plant_view(self, cfg):
        return deepcopy(self.plan['plant']['view'])

    def runner_status(self):
        return dict(ok=True, learned=True, native=True)

    def run(self, items, cfg, workers):
        self.calls.append(('run', len(items)))
        out = []
        for kind, spec, work in items:
            Path(work).mkdir(parents=True, exist_ok=True)
            ck = spec['checkpoint']
            rec = N.fake_episode(dict(spec, arm='U0'), work)
            rec.update({k: deepcopy(spec[k]) for k in N.IDENTITY})
            rec['executor_sha256_loaded'] = ck['executor_sha256']
            seed = spec['world']['seed']
            ex = dict(activations=2+seed % 3, activations_by_family={ck.get('a44_family') or ck.get('a37_family'): 2},
                      issued=200+seed % 7, acts=[dict(t=400, members=['main.1', 'main.2'], release={})], offers={},
                      plans=[dict(family=ck.get('a44_family') or ck.get('a37_family'))])
            if seed in self.unsafe:
                rec['safety'] = dict(rec['safety'], collisions=1)
            if spec['arm'] == 'A37_PLAN':
                rec['a37_sha256_loaded'] = ck['a37_sha256']
                rec['a37'] = dict(ex, a37_arm='A37_PLAN', executor='cell_bank_always')
            else:
                if seed in self.drift:
                    ex['issued'] += 1
                lc = self.lc.get(seed, [])
                rec['a44_sha256_loaded'] = ck['a44_sha256']
                rec['a44'] = dict(ex, a44_arm=spec['arm'], executor='cell_bank_always_lc', lc_members=lc)
            out.append(rec)
        return out


def runtime(root, **kw):
    return D.A44Runtime(FakeInner(N.prerun_plan(root), **kw), root)


def lcm(vid, first, completed=None, gave_up=None):
    return dict(vid=vid, start=400, sumo_from=1, sumo_target=2, onset_s=20, first_request=first, requests=5,
                completed=completed, gave_up=gave_up, released=[140, 'hold_end'])


def test_fixture_file_is_the_registered_hand_made_fixture(tmp_path, monkeypatch):
    plans = D.fixtures()
    assert plans['lc']['roles'][1]['lc_target'] == 2 and plans['hold']['roles'][1]['lc_target'] is None
    assert [dict(r, lc_target=None) for r in plans['lc']['roles']] == plans['hold']['roles']
    doc = json.loads(D.FIXTURE_FILE.read_text(encoding='utf-8'))
    doc['plans']['lc']['roles'][1]['onset_s'] = 40
    (tmp_path/'F.json').write_text(json.dumps(doc), encoding='utf-8')
    monkeypatch.setattr(D, 'FIXTURE_FILE', tmp_path/'F.json')
    with pytest.raises(N.Blocked):
        D.fixtures()


def test_seed_argument_rule():
    assert D.parse_seeds(SEEDS) == (992400, 992407)
    for bad in ('992400-992408', '992400', '991700-991707', '964000-964001', '992499-992500'):
        with pytest.raises(N.Blocked):
            D.parse_seeds(bad)


def test_budget_arithmetic():
    assert D.IDENTITY_SIM_S == 4_800 and D.LC_SIM_S == 9_600 and D.RESERVE_SIM_S == 9_600
    assert D.IDENTITY_SIM_S+D.LC_SIM_S+D.RESERVE_SIM_S == D.A44_CAP_SIM_S == 24_000


def test_specs_both_executors_same_hold_bank(root):
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a44_lcvariant as V
    panel = D.setup(SEEDS)
    assert [w['world']['seed'] for w in panel['worlds']] == list(range(992400, 992408))
    assert panel['cell']['cell_key'] == 'D2800_p60'
    (n37, _, s37), (n44, _, s44) = D.waves('identity', panel, root)
    assert len(s37) == len(s44) == 2 and [s['world'] for s in s37] == [s['world'] for s in s44]
    c37, c44 = s37[0]['checkpoint'], s44[0]['checkpoint']
    assert s37[0]['arm'] == 'A37_PLAN' and c37['kind'] == 'a37' and c37['a37_sha256'] == C.a37_digest() == D.C37.C.a37_digest() \
        if hasattr(D.C37, 'C') else c37['a37_sha256'] == C.a37_digest()
    assert c37['a37_sha256'].startswith('0a902437')
    assert s44[0]['arm'] == 'A44_PLAN' and c44['kind'] == 'a44' and c44['a44_sha256'] == V.a44_digest()
    assert c37['a37_bank'] == c44['a44_bank'] and c37['a37_family'] == c44['a44_family'] == 'a44_fix_hold'
    assert c37['executor_sha256'] == c44['executor_sha256'] and c37['path'] == c44['path']      # same U0 plant
    ((name, _, lc),) = D.waves('lcfix', panel, root)
    assert name == 'A44M_LCFIX' and len(lc) == 8 and lc[0]['checkpoint']['a44_family'] == 'a44_fix_lc'
    assert lc[0]['checkpoint']['a44_bank']['path'].endswith('A44_LCFIX.json')
    assert len(D.waves('lcfix', panel, root, n_lc=3)[0][2]) == 3
    with pytest.raises(N.Blocked):
        D.waves('lcfix', panel, root, n_lc=9)


def test_physics_refused_until_pinned(root):
    panel = D.setup(SEEDS)
    with pytest.raises(N.Blocked, match='pinned'):
        D.require_pin(panel)


def test_identity_and_lc_report_on_fake_records(root):
    panel = D.setup(SEEDS)
    logs = []
    out = D.execute_with_retry(root, D.waves('identity', panel, root), workers=2, chunk=8, log=logs.append,
                               runtime=runtime(root))
    assert out['status'] == 'PASS' and D.charged(root) == 4*1200
    v = D.identity_compare(root, panel)
    assert v['verdict'] == 'PASS' and v['non_vacuous'] and len(v['worlds']) == 2
    lc = {992400: [lcm('main.1', 420, completed=425)], 992401: [lcm('main.2', 420, gave_up=450)],
          992402: [lcm('main.3', None, gave_up=420)], 992403: [lcm('main.4', 420)]}
    out = D.execute_with_retry(root, D.waves('lcfix', panel, root), workers=2, chunk=8, log=logs.append,
                               runtime=runtime(root, lc=lc))
    assert out['status'] == 'PASS' and D.charged(root) == 12*1200
    r = D.lc_report(root, panel)
    assert r['verdict'] == 'PASS'
    assert r['totals'] == dict(members=4, commanded=3, completed=1, gave_up=1, ended_uncompleted=1, never_requested=1,
                               collisions=0, teleports=0)


def test_identity_detects_a_difference_and_report_flags_collisions(root):
    panel = D.setup(SEEDS)
    D.execute_with_retry(root, D.waves('identity', panel, root), workers=2, chunk=8, log=lambda x: None,
                         runtime=runtime(root, drift={992401}))
    v = D.identity_compare(root, panel)
    assert v['verdict'] == 'FAILED' and v['worlds'][1]['differences'] == ['executor.issued']
    D.execute_with_retry(root, D.waves('lcfix', panel, root, n_lc=2), workers=2, chunk=8, log=lambda x: None,
                         runtime=runtime(root, lc={992400: [lcm('main.1', 420, completed=430)]}, unsafe={992401}))
    assert D.lc_report(root, panel, 2)['verdict'] == 'REVIEW'


def test_cap_blocks_over_budget(root):
    rt = runtime(root)
    with pytest.raises(N.Blocked, match='cap'):
        rt.reserve(None, [dict(id=f'a44m_x_{i}', sim_s=1200) for i in range(21)])
    with pytest.raises(N.Blocked):
        rt.reserve(None, [dict(id='a37v_x', sim_s=1200)])
