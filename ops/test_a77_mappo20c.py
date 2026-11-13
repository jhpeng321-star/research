"""CPU-only tests for the amendment-77 evaluation driver (fake runner; no simulator): names / budget / arms, the fresh
panel and runner block, the MAPPO20C binding rule (iteration-84 receipt of the registered training PLAN), the specs
(U0 confirmation specs with only the executor replaced), ledger cap / job prefix, the stored identity references and,
on the Linux host, identity -> confirm -> analyse end to end on fake records."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D42  # noqa: E402
import a52_mappo60 as A52  # noqa: E402
import a77_mappo20c as A77  # noqa: E402
import nm_finaleval as N  # noqa: E402
import test_a37_celltailor as T37  # noqa: E402
import test_a42_bigp60 as T42  # noqa: E402

D = A77.D
META = dict(format='c4_dreamer/3', trainer='mappo', iteration=84, lane_control='native', lane_head=False)
A50_REFS = [os.environ.get('A77_A50_REFS', ''), '/root/a66/refs/a50/root', 'E:/agent-work/a66-run/refs/a50/root']


def _runner_has_block():
    from tools.coopmwm import nm_eval_runner as R
    return getattr(R, A77.RUNNER_ATTR, None) is not None


needs_block = pytest.mark.skipif(not _runner_has_block(), reason='runner copy without A77_RANGES (patch_tables.py)')


@pytest.fixture(autouse=True)
def _restore_n(monkeypatch):
    for k in list(vars(N)):                                  # one setting per process: restore after each test
        if k.isupper():
            v = getattr(N, k)
            monkeypatch.setattr(N, k, deepcopy(v) if isinstance(v, (dict, list)) else v)
    monkeypatch.setattr(N, 'SETTING', 'p60')


def test_names_budget_arms_and_reused_code():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, D.STOP_A42, D.JOB_PREFIX, D.A42Runtime.phase, D.CELL) == \
        ('mappo20c_a77', 480_000, 'STOP_A77', 'a77', 'mappo20c_a77', 'D2800_p20')
    assert D.SETTINGS == (A77.SETTING,) and D.RUNNER_RANGES_ATTR == 'A77_RANGES'
    assert (D42.A42_PHASE, D42.JOB_PREFIX) == ('bigp60_a42', 'a42')                         # the original untouched
    b = A77.budget_plan()
    assert (b['identity'], b['confirmation'], b['planned'], b['margin']) == (4_800, 345_600, 350_400, 129_600)
    assert A77.budget_plan(('U0', 'MAPPO20C'))['planned'] == 235_200
    for name in ('execute', '_contrast', 'ident_specs', 'confirm_specs'):
        assert getattr(D, name).__code__.co_code == getattr(D42, name).__code__.co_code
    assert A77.parse_arms(None) == ('U0', 'MAPPO_CT', 'MAPPO20C') and A77.parse_arms('U0,MAPPO20C') == ('U0', 'MAPPO20C')
    for bad in ('MAPPO20C,U0', 'U0,MAPPO_CT', 'U0,MAPPO_CT,MAPPO20C,DUO_HOLD'):
        with pytest.raises(N.Blocked):
            A77.parse_arms(bad)
    assert A77.CONTRASTS[0] == ('MAPPO20C', 'U0', 'primary')
    assert A77.MAPPO_CT_SHA256.startswith('069a4f85') and A77.PINS['IDENT_U0'].startswith('d9042cc5')
    assert len(A77.PANEL_SHA256) == 64 and len(A77.TRAIN_PLAN_SHA256) == 64 and A77.PINS['MAPPO20C'] is None
    assert A77.BOOTSTRAP == dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')


def test_panel_file_is_the_fresh_block():
    panel = N.verify(N.read(A77.PANEL_FILE), 'panel_sha256', A77.PANEL_FILE.name)
    assert panel['panel_sha256'] == A77.PANEL_SHA256
    seeds = [w['world']['seed'] for w in panel['worlds']]
    assert seeds == list(range(997600, 997696)) and {w['world']['cell_key'] for w in panel['worlds']} == {'D2800_p20'}
    assert panel['seeds_reserved_unused'] == [997696, 997699]
    assert dict(panel['contract']['bootstrap']) == A77.BOOTSTRAP
    lo, hi = A77.SEEDS
    for b0, b1 in A77.FORBIDDEN + ((996400, 996499), (996300, 996399), (993000, 993299), (999500, 999699)):
        assert hi < b0 or lo > b1


@needs_block
def test_runner_block_and_setting():
    assert A77.runner_block() == (997600, 997699)
    A77.register_setting()
    s = N.SETTINGS[A77.SETTING]
    assert s['N_WORLDS'] == 96 and s['EVAL_RANGE'] == (997600, 997699) and s['SEEDS_RESERVED'] == (997696, 997699)
    assert s['CELL'] == N.SETTINGS['p20c']['CELL'] and N.SETTINGS_PINS[A77.SETTING]['PANEL_SHA256'] == A77.PANEL_SHA256


def _mirror_root(path):
    src = T37.mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    for f in T37.BIND:
        (path/f).parent.mkdir(parents=True, exist_ok=True)
        (path/f).write_bytes((src/f).read_bytes())
    return path


def _train_files(tmp, body, *, plan_sha=None, iteration=84):
    """A fake training PLAN whose sha is pinned as the registered one, and its BOUND_00084 receipt."""
    plan = tmp/'PLAN.json'
    plan.write_bytes(b'{"fake": "plan"}')
    sha = hashlib.sha256(body).hexdigest()
    rec = tmp/f'BOUND_{iteration:05d}.json'
    rec.write_text(json.dumps(dict(sha256=sha, iteration=iteration, lane_control='native',
                                   plan_sha256=plan_sha or hashlib.sha256(plan.read_bytes()).hexdigest())))
    return plan, rec, sha


@pytest.fixture
def root(tmp_path, monkeypatch):
    if not _runner_has_block():
        pytest.skip('runner copy without A77_RANGES')
    out = _mirror_root(tmp_path/'a77root')
    monkeypatch.setattr(N, 'export_meta', lambda p: dict(META))
    for which, body in (('MAPPO_CT', b'ct'), ('IDENT_U0', b'u0')):
        p = tmp_path/f'{which}.pt'
        p.write_bytes(body)
        monkeypatch.setitem(A77.PINS, which, hashlib.sha256(body).hexdigest())
        A77.bind(out, which, p)
    body = b'mappo20c'
    pol = tmp_path/'policy_00084.pt'
    pol.write_bytes(body)
    plan, rec, sha = _train_files(tmp_path, body)
    monkeypatch.setattr(A77, 'TRAIN_PLAN_SHA256', hashlib.sha256(plan.read_bytes()).hexdigest())
    A77.bind(out, 'MAPPO20C', pol, sha=sha, receipt=rec, plan=plan)
    A77.register_setting()
    return out


def test_mappo20c_bind_needs_the_registered_training_receipt(tmp_path, monkeypatch):
    out = _mirror_root(tmp_path/'r')
    monkeypatch.setattr(N, 'export_meta', lambda q: dict(META))
    body = b'final'
    pol = tmp_path/'policy_00084.pt'
    pol.write_bytes(body)
    plan, rec, sha = _train_files(tmp_path, body)
    with pytest.raises(N.Blocked, match='--sha'):
        A77.bind(out, 'MAPPO20C', pol)
    with pytest.raises(N.Blocked, match='registered A77 PLAN'):              # real pin 57f51364 != this fake plan
        A77.bind(out, 'MAPPO20C', pol, sha=sha, receipt=rec, plan=plan)
    monkeypatch.setattr(A77, 'TRAIN_PLAN_SHA256', hashlib.sha256(plan.read_bytes()).hexdigest())
    _, rec42, _ = _train_files(tmp_path, body, iteration=42)
    with pytest.raises(N.Blocked, match='iteration-84'):
        A77.bind(out, 'MAPPO20C', pol, sha=sha, receipt=rec42, plan=plan)
    with pytest.raises(N.Blocked, match='sha / lane'):
        A77.bind(out, 'MAPPO20C', pol, sha='0'*64, receipt=rec, plan=plan)
    monkeypatch.setattr(N, 'export_meta', lambda q: dict(META, lane_head=True))
    with pytest.raises(N.Blocked, match='speed-only'):
        A77.bind(out, 'MAPPO20C', pol, sha=sha, receipt=rec, plan=plan)
    monkeypatch.setattr(N, 'export_meta', lambda q: dict(META))
    v = A77.bind(out, 'MAPPO20C', pol, sha=sha, receipt=rec, plan=plan)
    assert v['training']['iteration'] == 84 and A77.read_bound(out, 'MAPPO20C')['sha256'] == sha
    with pytest.raises(N.Blocked, match='never rebound'):
        A77.bind(out, 'MAPPO20C', pol, sha=sha, receipt=rec, plan=plan)
    other = tmp_path/'x.pt'
    other.write_bytes(b'x')
    with pytest.raises(N.Blocked, match='not the registered'):
        A77.bind(out, 'MAPPO_CT', other)                                     # real pin 069a4f85 != this file


def test_specs_replace_only_the_executor(root, monkeypatch):
    panel = T37.use(monkeypatch, A77.SETTING)
    assert panel['panel_sha256'] == A77.PANEL_SHA256
    base = A77.confirm_specs('U0', panel, root)
    assert len(base) == 96 and [s['world']['seed'] for s in base] == list(range(997600, 997696))
    assert [s['job_id'] for s in base] == [f'a77c_u0_{i:04}' for i in range(96)]
    assert {s['world']['cell_key'] for s in base} == {'D2800_p20'} and {s['arm'] for s in base} == {'U0'}
    for arm in ('MAPPO_CT', 'MAPPO20C'):
        new = A77.confirm_specs(arm, panel, root)
        exe = A77.read_bound(root, arm)
        assert [s['job_id'] for s in new] == [f'a77c_{arm.lower()}_{i:04}' for i in range(96)]
        for b, s in zip(base, new):
            assert {k for k in s if s[k] != b[k]} == {'job_id', 'checkpoint'} and s['arm'] == 'U0'
            diff = {k for k in s['checkpoint'] if s['checkpoint'][k] != b['checkpoint'].get(k)}
            assert diff <= set(A52.EXECUTOR_FIELDS) and s['checkpoint']['executor_sha256'] == exe['sha256']
    ident = A77.ident_specs(root)
    assert [s['job_id'] for s in ident] == ['a77i_p20c_u0_0000', 'a77i_p20c_u0_0001', 'a77i_a35d3200_u0_0000',
                                            'a77i_a35d3200_u0_0001']
    assert [s['world']['seed'] for s in ident] == [993100, 993101, 998100, 998101]
    assert all(s['checkpoint']['executor_sha256'] == A77.PINS['IDENT_U0'] for s in ident)


def test_ledger_cap_and_prefix(root):
    rt = D.A42Runtime(T42.Fake(N.prerun_plan(root)), root)
    with pytest.raises(N.Blocked, match='mappo20c_a77 cap'):
        rt.check_capacity(480_001)
    with pytest.raises(N.Blocked, match='job ids are a77'):
        rt.reserve(None, [dict(id='a66c_u0_0000', sim_s=1)])


def test_stored_identity_references_cover_the_identity_worlds():
    ref = next((Path(p) for p in A50_REFS if p and (Path(p)/'BOUND_U0.json').is_file()), None)
    if ref is None:
        pytest.skip('no read-only copy of the A50 root on this host')
    recs = A52._a50_records(ref, A77.A50_IDENT_WAVE)
    assert sorted(recs) == ['wz3to2_993100', 'wz3to2_993101', 'wz3to2_998100', 'wz3to2_998101']
    assert all(r['safety']['collisions'] == 0 for r in recs.values())


class Fake77(T42.Fake):
    """Fake physics; MAPPO20C's executor scales J by ``factor``."""

    def __init__(self, plan, factor=1.0, **kw):
        super().__init__(plan, **kw)
        self.f = factor

    def run(self, items, cfg, workers):
        out = super().run(items, cfg, workers)
        for (kind, spec, work), rec in zip(items, out):
            if spec['job_id'].startswith('a77c_mappo20c'):
                rec['J_veh_s'] *= self.f
        return out


@pytest.mark.skipif(sys.platform == 'win32', reason='end-to-end vehicle-log reports run on the Linux host')
def test_end_to_end_identity_confirm_analyse(root, tmp_path, monkeypatch):
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))
    # a fake stored A50 identity wave (same fake physics as the host path)
    a50 = _mirror_root(tmp_path/'a50root')
    T37.use(monkeypatch, A77.SETTING)
    D.execute(root, [(D.IDENT_WAVE, 'episode', A77.ident_specs(root))], workers=4, chunk=4, log=lambda x: None,
              runtime=D.A42Runtime(Fake77(N.prerun_plan(root)), root))
    got, _, _ = N.published(root, D.IDENT_WAVE, A77.ident_specs(root), D.C37.check_a37)
    N.publish(a50/'WAVE_A50_IDENT_U0.json', N.seal(dict(records=got), 'receipt_sha256'))
    v = A77.identity_compare(root, a50)
    assert v['verdict'] == 'PASS' and len(v['worlds']) == 4
    N.publish(root/A77.IDENTITY_FILE, v)
    assert A77.identity_passed(root)
    panel = T37.use(monkeypatch, A77.SETTING)
    waves = [(A77.confirm_wave(a), 'episode', A77.confirm_specs(a, panel, root)) for a in A77.ARMS]
    out = D.execute(root, waves, workers=4, chunk=400, log=lambda x: None,
                    runtime=D.A42Runtime(Fake77(N.prerun_plan(root), factor=.97), root))
    assert out['status'] == 'PASS' and D.charged(root) == (4+3*96)*1200
    assert {r['phase'] for r in N.fe.ledger_rows(root)} == {'mappo20c_a77'}
    res = A77.analyse(root, panel)
    rows = {(c['treatment'], c['reference']): c for c in res['contrasts']}
    assert list(rows) == [('MAPPO20C', 'U0'), ('MAPPO20C', 'MAPPO_CT'), ('MAPPO_CT', 'U0')]
    m = rows[('MAPPO20C', 'U0')]['total_minus_ref']
    assert m['diff'] < 0 and m['ci95'][1] < 0 and res['reading']['per_contrast']['MAPPO20C-U0']['saves']
    assert 'groups' in rows[('MAPPO20C', 'U0')] and 'hard_brakes_minus_ref' in rows[('MAPPO20C', 'U0')]
