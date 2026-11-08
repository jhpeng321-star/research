"""ORACLE-PLAN arm (N6 amendment 25): vendored executor integrity, digest pin, bank plan rebuild and refusals, the
privileged-bin gate, one global session + cooldown, deterministic priority, HDVs never commanded, determinism, and
that the hook never reads job / B / R outputs (CPU only, no SUMO)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.coopmwm import oracle_plan as O

UNION = Path('E:/research-dense/runs/active/C4-V7FT-20261017/wmverify_r3/GUIDANCE_C4SIT_priv_UNION_r3.json')
#: oracle_digest() prefix at c4/oracle-arm; any change to an executed function / constant moves it (re-registration)
PINNED = '7058d2ac7d58'


def role(sumo_lane, v_w=20.0, hold_s=120, onset_s=0, rank=0, lc=None):
    return dict(dmax=1.0, hold_s=hold_s, lc_target_sumo=lc, lc_target_wm=None if lc is None else 2 - lc, onset_s=onset_s,
                phase=0, rank=rank, slot=0, sumo_lane=sumo_lane, v_w=v_w, wm_lane=2 - sumo_lane)


def cond(roles, bins, trigger='wall', cooldown=120, mode='privileged', gate=None):
    return dict(cooldown_s=cooldown, edge='up', gate=gate, phases=None, roles=roles, slots=1, spread_m=200.0,
                start_gate=None, trigger=trigger, zone='full', zone_s=[-1724.56, -324.56],
                situation=dict(bins=bins, mode=mode, rule=O.SITUATION_RULE))


def bank(entries, cell='D2800_p20'):
    """entries: [(family, lb95, cond, key or None -> correct key)]"""
    rows, tpls = [], []
    for fam, lb, c, key in entries:
        k = O.strategy_key(O.spec_of(fam, c)) if key is None else key
        rows.append(dict(arm='B', regime=cell, family=fam, lb95_veh_s=lb, plan_source='wm' if fam.startswith('mut_')
                         else 'rule', template_sha256='t'+fam, situational=dict(bins=c['situation']['bins'])))
        tpls.append(dict(cell=cell, family=fam, template=dict(source=dict(strategy_key=k)), offer_condition=c))
    return dict(rows=rows, clean5c=dict(templates=tpls))


FREE = [dict(cong='free', lane_diff='L2notfaster')]
CONG = [dict(cong='congested', lane_diff='L2notfaster')]


def wm_rows(speed=25.0, slow_drop=False, with_cavs=True, hdv_lane2=True):
    """One CAV per SUMO lane at s ~ -1000 m (zone), HDVs ahead; slow_drop: HDVs near the drop at 5 m/s."""
    rows = []
    if with_cavs:
        for i, wm_lane in enumerate((0, 1, 2)):
            rows.append(dict(vid=f'c{wm_lane}', s_m=-1000.0 + 10 * i, v_mps=speed, lane=wm_lane, is_cav=True))
    for k, s in enumerate((-500.0, -300.0, -100.0, 0.0)):
        for wm_lane in (0, 1):
            v = 5.0 if slow_drop and s >= -600 else speed
            rows.append(dict(vid=f'h{wm_lane}_{k}', s_m=s, v_mps=v, lane=wm_lane, is_cav=False))
    if hdv_lane2:
        rows.append(dict(vid='h2', s_m=-950.0, v_mps=speed, lane=2, is_cav=False))
    return rows


def commands(rows):
    return {r['vid']: 0.1 for r in rows if r['is_cav']}


def run(core, seq):
    out = []
    for t, rows in seq:
        out.append(core.step(rows, commands(rows), t))
    return out


DUO = cond([role(1), role(2)], FREE)


# ------------------------------------------------------------------ integrity
def test_vendored_sources_equal_their_origin():
    assert all(O.vendored_check().values()) and len(O.VENDORED) == 16


def test_digest_pinned_and_sensitive(monkeypatch):
    d = O.oracle_digest()
    assert d.startswith(PINNED)
    monkeypatch.setattr(O, 'CONG_MPS', 14.0)
    assert O.oracle_digest() != d


def test_module_imports_and_no_rng_or_clock():
    tree = ast.parse(Path(O.__file__).read_text(encoding='utf-8'))
    mods = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= {'__future__', 'hashlib', 'inspect', 'itertools', 'json', 'collections', 'sys'}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree)
                                                                          if isinstance(n, ast.Attribute)}
    assert not names & {'random', 'time', 'torch', 'np', 'numpy', 'track2_backend', 'decision_log', 'EpisodePolicy',
                        'fullpol_learn', 'checkpoint', 'weights', 'samples', 'events', 'policy'}


# ------------------------------------------------------------------ bank rebuild / refusals / priority
def test_bank_plans_priority_and_refusals():
    lane = cond([role(2), role(1, lc=0)], FREE, trigger='lanes')
    doc = bank([('mut_b', 300.0, DUO, None), ('mut_a', 300.0, cond([role(1), role(2, v_w=19.0)], FREE), None),
                ('rule_c', 500.0, cond([role(2)], FREE), None), ('bad_key', 900.0, DUO, 'ffffffffffffffff'),
                ('lane_plan', 800.0, lane, None), ('own_sense', 700.0, cond([role(2)], FREE, mode='own_sensing'), None)])
    plans, refused = O.bank_plans(doc, 'D2800_p20')
    assert [p['family'] for p in plans][0] == 'rule_c'
    tie = [p for p in plans if p['lb95'] == 300.0]
    assert [p['key'] for p in tie] == sorted(p['key'] for p in tie)          # deterministic tie-break (key)
    assert refused == dict(strategy_key_mismatch=['bad_key'], non_hold_plan=['lane_plan'],
                           bins_not_privileged=['own_sense'])
    assert O.bank_plans(doc, 'D3200_p20') == ([], {})


@pytest.mark.skipif(not UNION.exists(), reason='union bank not on this host')
def test_union_bank_rebuilds_every_template_key():
    doc = json.loads(UNION.read_text(encoding='utf-8'))
    plans, refused = O.bank_plans(doc, 'D2800_p20')
    assert not refused and {p['family'] for p in plans} == {r['family'] for r in doc['rows'] if r['regime'] == 'D2800_p20'}
    plans3, refused3 = O.bank_plans(doc, 'D3200_p20')
    assert not refused3 and [p['family'] for p in plans3] == ['g2_gap1']


# ------------------------------------------------------------------ gate, session, cooldown
def test_activates_only_in_admitted_privileged_bin():
    core = O.OracleCore(bank([('duo', 100.0, DUO, None)]), 'D2800_p20')
    run(core, [(300, wm_rows(slow_drop=True))])                   # congested: feasible group, bin not admitted
    assert core.acts == [] and any('congested' in k[1] for k in core.offers)
    accel, lanes = core.step(wm_rows(), commands(wm_rows()), 301)  # free / L2notfaster: admitted
    assert len(core.acts) == 1 and core.acts[0]['family'] == 'duo' and core.acts[0]['holder'] == 'c0'
    assert set(accel) == {'c0', 'c1'} and all(a == pytest.approx(-1.0) for a in accel.values())   # 20 - 25 -> -dmax
    assert core.acts[0]['bin'] == FREE[0]


def test_no_feasible_group_no_activation():
    core = O.OracleCore(bank([('duo', 100.0, DUO, None)]), 'D2800_p20')
    rows = [r for r in wm_rows() if r['vid'] != 'c1']               # no CAV in SUMO lane 1 -> no wall
    run(core, [(300 + k, rows) for k in range(5)])
    assert core.acts == [] and not core.offers


def test_one_session_cooldown_and_priority():
    doc = bank([('low', 10.0, DUO, None), ('high', 50.0, cond([role(1, v_w=21.0), role(2, v_w=21.0)], FREE), None)])
    core = O.OracleCore(doc, 'D2800_p20')
    for t in range(300, 560):
        rows = wm_rows()
        core.step(rows, commands(rows), t)
        assert sum(1 for c in core.cores if c.active) <= 1             # never two sessions
    ts = [a['t'] for a in core.acts]
    assert core.acts[0]['family'] == 'high'                        # priority: lb95 descending
    assert ts == [300, 420, 540]                                    # cooldown 120 s, one session at a time


def test_never_commands_hdvs_and_deterministic():
    doc = bank([('duo', 100.0, DUO, None), ('solo', 90.0, cond([role(2)], FREE), None)])
    seq = [(t, wm_rows(speed=25.0 if t % 7 else 24.0, slow_drop=(t // 50) % 2 == 1)) for t in range(300, 700)]
    a1, a2 = O.OracleCore(doc, 'D2800_p20'), O.OracleCore(doc, 'D2800_p20')
    out1, out2 = run(a1, seq), run(a2, seq)
    assert out1 == out2 and a1.receipts() == a2.receipts()
    hdv = {r['vid'] for _, rows in seq for r in rows if not r['is_cav']}
    assert all(not (set(a) | set(ln)) & hdv for a, ln in out1)


def test_before_warmup_and_empty_bank():
    core = O.OracleCore(bank([('duo', 100.0, DUO, None)]), 'D2800_p20')
    assert core.step(wm_rows(), commands(wm_rows()), 299) == ({}, {}) and core.acts == []
    empty = O.OracleCore(bank([]), 'D2800_p20')
    assert empty.step(wm_rows(), commands(wm_rows()), 400) == ({}, {})


# ------------------------------------------------------------------ hook: never reads job / B / R outputs
class Poison:
    """job / prefixes guard: any read fails the test (B/R checkpoints, decision logs, weights live in job)."""

    def __getattr__(self, name):
        raise AssertionError(f'ORACLE read job/prefixes attribute {name!r}')

    def __getitem__(self, key):
        raise AssertionError(f'ORACLE read job/prefixes key {key!r}')

    def get(self, *a):
        raise AssertionError('ORACLE read job/prefixes via get')

    def __contains__(self, key):
        raise AssertionError('ORACLE probed job/prefixes')

    def __iter__(self):
        raise AssertionError('ORACLE iterated job/prefixes')


class Controller:
    __slots__ = ('apply', 'g5_lane_targets')          # any other attribute access raises AttributeError

    def __init__(self):
        self.g5_lane_targets = {}

        def base(rows, commands, now):
            return {vid: dict(requested_raw=a) for vid, a in commands.items()}
        self.apply = base


def test_hook_never_reads_job_or_br_outputs_and_writes_cav_only():
    hook = O.make_hook(bank([('duo', 100.0, DUO, None)]), 'D2800_p20')
    c = Controller()
    hook(c, Poison(), Poison())
    rows = wm_rows()
    out = c.apply(rows, commands(rows), 300.0)
    assert out['c0']['oracle_plan'] == 'duo' and out['c0']['requested_raw'] == pytest.approx(-1.0)
    assert out['c2'] == dict(requested_raw=0.1)                      # non-member CAV: U0 command untouched
    assert set(c.g5_lane_targets) == {'c0', 'c1'} and c.g5_lane_targets['c0'] == 0
    r = hook.receipts()
    assert r['activations'] == 1 and r['activations_by_family'] == {'duo': 1} and r['refused_counts'] == {}


def test_hook_refuses_hdv_request():
    hook = O.make_hook(bank([('duo', 100.0, DUO, None)]), 'D2800_p20')
    c = Controller()
    hook(c, Poison(), Poison())
    rows = [dict(r, is_cav=False) if r['vid'] == 'c1' else r for r in wm_rows()]
    cmds = dict(commands(wm_rows()))                                 # a command for a now-HDV row
    hook.core.step = lambda rows, commands, t: ({'c1': 0.0}, {})
    with pytest.raises(RuntimeError, match='HDVs never commanded'):
        c.apply(rows, cmds, 300.0)


# ------------------------------------------------------------------ runner wiring
def test_runner_oracle_hook_guards(tmp_path, monkeypatch):
    from tools.coopmwm import nm_eval_runner as R
    doc = bank([('duo', 100.0, DUO, None)])
    p = tmp_path/'bank.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    ck = dict(cell_key='D2800_p20', oracle_bank=dict(path=str(p), file_sha256=R._sha(p)), oracle_sha256=O.oracle_digest())
    hook, loaded = R.oracle_hook(ck, dict(cell_key='D2800_p20'))
    assert loaded == O.oracle_digest() and hook.core.plans[0]['family'] == 'duo'
    with pytest.raises(ValueError, match='cell_key'):
        R.oracle_hook(ck, dict(cell_key='D3200_p20'))
    with pytest.raises(ValueError, match='cell_key'):
        R.oracle_hook(ck, dict(cell=3))
    with pytest.raises(ValueError, match='oracle_sha256'):
        R.oracle_hook(dict(ck, oracle_sha256='0' * 64), dict(cell_key='D2800_p20'))
    with pytest.raises(ValueError, match='bank changed'):
        R.oracle_hook(dict(ck, oracle_bank=dict(path=str(p), file_sha256='0' * 64)), dict(cell_key='D2800_p20'))
