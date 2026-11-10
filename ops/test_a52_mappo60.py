"""CPU-only tests for the amendment-52 driver (fake runner; no simulator): MAPPO60 = the A50 U0 arm with only the
executor file replaced (a parameter: sealed binding path + sha256), own ledger phase / cap / job ids, the identity
compare against the A50 identity records, and the analysis (A50 contrast code) end to end on fake records."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a42_bigp60 as D42  # noqa: E402
import a50_fullp60 as A50  # noqa: E402
import a52_mappo60 as A52  # noqa: E402
import nm_finaleval as N  # noqa: E402
import test_a37_celltailor as T37  # noqa: E402
import test_a42_bigp60 as T42  # noqa: E402

D, D50 = A52.D, A50.D
META = dict(format='c4_dreamer/3', trainer='mappo', iteration=48, lane_control='native', lane_head=False)


def test_names_budget_and_reused_code():
    assert (D.A42_PHASE, D.A42_CAP_SIM_S, D.STOP_A42, D.JOB_PREFIX, D.A42Runtime.phase) == \
        ('mappo60_a52', 130_000, 'STOP_A52', 'a52', 'mappo60_a52')
    assert (D50.A42_PHASE, D50.A42Runtime.phase, D50.JOB_PREFIX) == ('bigp60_a50', 'bigp60_a50', 'a50')   # untouched
    assert D.SETTINGS == D50.SETTINGS == ('a50v_D2800_p60', 'a50c_D2800_p60') and D.IDENT_PANELS == D50.IDENT_PANELS
    b = A52.budget_plan()
    assert b['planned'] == 120_000 and b['margin'] == 10_000 and b['cap_sim_s'] == 130_000
    for name in ('execute', '_contrast', 'ident_specs', 'confirm_specs'):
        assert getattr(D, name).__code__.co_code == getattr(D42, name).__code__.co_code
    assert A52.MAPPO60_SHA256.startswith('7f0c8869') and A52.MAPPO60_SHA256.endswith('364443')
    assert A52.U0_SHA256.startswith('d9042cc5')
    assert A52.CONTRASTS == (('MAPPO60', 'U0'), ('DUO_HOLD', 'MAPPO60'), ('BIG_BANK', 'MAPPO60'),
                             ('LEARNED_TRIGGER', 'MAPPO60'))


def _mirror_root(path):
    src = T37.mirror()
    if src is None:
        pytest.skip('no mirrored bank-root bindings on this host')
    for f in T37.BIND:
        (path/f).parent.mkdir(parents=True, exist_ok=True)
        (path/f).write_bytes((src/f).read_bytes())
    return path


def _fake_policy(tmp, which, monkeypatch, body):
    p = tmp/f'{which}.pt'
    p.write_bytes(body)
    monkeypatch.setitem(A52.PINS, which, hashlib.sha256(body).hexdigest())
    return p


@pytest.fixture
def root(tmp_path, monkeypatch):
    out = _mirror_root(tmp_path/'a52root')
    monkeypatch.setattr(N, 'export_meta', lambda p: dict(META))
    m = _fake_policy(tmp_path, 'MAPPO60', monkeypatch, b'mappo60')
    u = _fake_policy(tmp_path, 'IDENT_U0', monkeypatch, b'u0')
    A52.bind(out, 'MAPPO60', m)
    A52.bind(out, 'IDENT_U0', u)
    return out


def test_bind_refuses_wrong_sha_lane_head_and_rebinding(tmp_path, monkeypatch):
    out = _mirror_root(tmp_path/'r')
    p = tmp_path/'x.pt'
    p.write_bytes(b'x')
    monkeypatch.setattr(N, 'export_meta', lambda q: dict(META))
    with pytest.raises(N.Blocked, match='not the registered'):
        A52.bind(out, 'MAPPO60', p)                                       # real pin 7f0c8869 != this file
    monkeypatch.setitem(A52.PINS, 'MAPPO60', hashlib.sha256(b'x').hexdigest())
    monkeypatch.setattr(N, 'export_meta', lambda q: dict(META, lane_head=True))
    with pytest.raises(N.Blocked, match='speed-only'):
        A52.bind(out, 'MAPPO60', p)
    monkeypatch.setattr(N, 'export_meta', lambda q: dict(META))
    v = A52.bind(out, 'MAPPO60', p)
    assert v['kind'] == 'a52_MAPPO60' and A52.read_bound(out, 'MAPPO60')['binding_sha256'] == v['binding_sha256']
    with pytest.raises(N.Blocked, match='never rebound'):
        A52.bind(out, 'MAPPO60', p)
    with pytest.raises(N.Blocked, match='which'):
        A52.binding_file('MAPPO_SPEED')


def test_with_executor_replaces_only_the_executor(root, monkeypatch):
    panel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    base = D.confirm_specs('U0', panel, root)
    # given the U0 binding itself the new path is a no-op on the episode descriptor
    same = A52.with_executor(base, N.mappo_binding(root, 'U0'))
    assert same == base
    new = A52.confirm_specs(panel, root)
    assert len(new) == 96 and [s['world']['seed'] for s in new] == list(range(996800, 996896))
    assert [s['job_id'] for s in new] == [f'a52c_mappo60_{i:04}' for i in range(96)]
    exe = A52.read_bound(root, 'MAPPO60')
    for b, s in zip(base, new):
        assert {k for k in s if s[k] != b[k]} == {'job_id', 'checkpoint'} and s['arm'] == 'U0'
        diff = {k for k in s['checkpoint'] if s['checkpoint'][k] != b['checkpoint'].get(k)}
        assert diff <= set(A52.EXECUTOR_FIELDS) and set(s['checkpoint']) == set(b['checkpoint'])
        assert s['checkpoint']['executor_sha256'] == exe['sha256'] and s['checkpoint']['path'] == b['checkpoint']['path']
    with pytest.raises(N.Blocked, match='only U0'):
        A52.with_executor([dict(base[0], arm='DUO_HOLD')], exe)
    ident = A52.ident_specs(root)
    assert [s['job_id'] for s in ident] == ['a52i_p20c_u0_0000', 'a52i_p20c_u0_0001', 'a52i_a35d3200_u0_0000',
                                            'a52i_a35d3200_u0_0001']
    assert all(s['checkpoint']['executor_sha256'] == A52.PINS['IDENT_U0'] for s in ident)


def test_ledger_cap_and_prefix(root):
    rt = D.A42Runtime(T42.Fake(N.prerun_plan(root)), root)
    with pytest.raises(N.Blocked, match='mappo60_a52 cap'):
        rt.check_capacity(130_001)
    with pytest.raises(N.Blocked, match='job ids are a52'):
        rt.reserve(None, [dict(id='a50c_u0_0000', sim_s=1)])


class Fake52(T42.Fake):
    """Fake physics; MAPPO60's executor scales J by ``mappo_factor``."""

    def __init__(self, plan, mappo_factor=1.0, **kw):
        super().__init__(plan, **kw)
        self.mf = mappo_factor

    def run(self, items, cfg, workers):
        out = super().run(items, cfg, workers)
        for (kind, spec, work), rec in zip(items, out):
            if spec['checkpoint'].get('executor_sha256') == A52.PINS['MAPPO60']:
                rec['J_veh_s'] *= self.mf
        return out


@pytest.mark.skipif(sys.platform == 'win32', reason='host-root remap needs POSIX paths (run on the Linux host)')
def test_end_to_end_identity_confirm_analyse(root, tmp_path, monkeypatch):
    monkeypatch.setattr(N, 'CO_PRIMARY_GROUPS', tuple(g for g in N.CO_PRIMARY_GROUPS if g != 'origin:L0'))
    # a fake finished A50 root: identity + verify/admit + the four confirmation arms
    a50 = _mirror_root(tmp_path/'a50root')
    cands = T42.fixture_cands()
    monkeypatch.setattr(D50, 'candidates', lambda *a, **k: cands)
    D50.prepare(a50)
    T42.use(monkeypatch, D50.VERIFY_SETTING)
    rt = lambda r, **kw: D50.A42Runtime(T42.Fake(N.prerun_plan(r), **kw), r)  # noqa: E731
    D50.execute(a50, [(D50.IDENT_WAVE, 'episode', D50.ident_specs(a50))], workers=4, chunk=4, log=lambda x: None,
                runtime=rt(a50))
    panel = T42.use(monkeypatch, D50.VERIFY_SETTING)
    D50.execute(a50, D50.verify_waves(panel, a50), workers=4, chunk=200, log=lambda x: None,
                runtime=rt(a50, factors={cands[1]['family']: .9}))
    D50.admit(a50, panel)
    cpanel = T42.use(monkeypatch, D50.CONFIRM_SETTING)
    D50.execute(a50, D50.stage_waves('confirm', D50.CONFIRM_SETTING, cpanel, a50), workers=4, chunk=400,
                log=lambda x: None, runtime=rt(a50, factors=dict(CELL_BANK=.97)))
    N.publish(a50/'LAUNCHER.json', dict(argv=['x', '--final-root', str(a50)]))
    copy = a50              # the analysis reads it as an offloaded copy whose host root is its own path
    # A52: identity -> compare -> confirm -> analyse
    T42.use(monkeypatch, D.VERIFY_SETTING)
    with pytest.raises(N.Blocked):
        A52.identity_compare(root, copy)                                  # nothing published yet
    D.execute(root, [(D.IDENT_WAVE, 'episode', A52.ident_specs(root))], workers=4, chunk=4, log=lambda x: None,
              runtime=D.A42Runtime(Fake52(N.prerun_plan(root)), root))
    v = A52.identity_compare(root, copy)
    assert v['verdict'] == 'PASS' and len(v['worlds']) == 4 and all(x['abs_diff'] == 0.0 for x in v['worlds'])
    N.publish(root/A52.IDENTITY_FILE, v)
    assert A52.identity_passed(root)
    cpanel = T42.use(monkeypatch, D.CONFIRM_SETTING)
    out = D.execute(root, [(A52.CONFIRM_WAVE, 'episode', A52.confirm_specs(cpanel, root))], workers=4, chunk=97,
                    log=lambda x: None, runtime=D.A42Runtime(Fake52(N.prerun_plan(root), mappo_factor=.99), root))
    assert out['status'] == 'PASS' and D.charged(root) == 100*1200
    assert {r['phase'] for r in N.fe.ledger_rows(root)} == {'mappo60_a52'}
    res = A52.analyse(root, copy, cpanel, host_root=str(copy))
    rows = {(c['treatment'], c['reference']): c for c in res['contrasts']}
    assert list(rows) == list(A52.CONTRASTS)
    m = rows[('MAPPO60', 'U0')]['total_minus_ref']
    assert m['diff'] < 0 and m['ci95'][1] < 0                            # fake: MAPPO60 = 0.99 x U0
    assert rows[('BIG_BANK', 'MAPPO60')]['total_minus_ref']['diff'] < 0   # fake: CELL_BANK = 0.97 x U0
    assert set(res['arms']) == {'MAPPO60', 'U0', 'DUO_HOLD', 'LEARNED_TRIGGER', 'BIG_BANK'}
    assert res['reading']['ci_upper_below_zero'].keys() == {'DUO_HOLD-MAPPO60', 'BIG_BANK-MAPPO60'}
    assert 'hard_brakes_minus_ref' in rows[('MAPPO60', 'U0')] and 'groups' in rows[('MAPPO60', 'U0')]
    assert res['analysis_sha256'] and not N.os.environ.get(N.HOST_FINAL_ROOT_ENV)
