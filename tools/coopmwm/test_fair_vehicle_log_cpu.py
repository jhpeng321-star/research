"""clean5b-F per-vehicle logger (N6 port) + A-labelled groups + the fairness reward on REAL logger output. No SUMO.

A synthetic env drives the ported N6 ``VehicleLog`` exactly as the plant's ``logged_tick`` does (pre-step ground truth,
seconds 300..1199); the per-vehicle table must reproduce ``vehicle_log_report``'s delays; groups are labelled by the
A arm (origin SUMO lane + class, by vehicle id), unpaired vehicles are reported; the fairness reward in
ArmRuntime.train consumes the logger-produced groups and fails closed without them.
"""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import fair_vehicle_log as fv, fullpol_fair as ff, track2_backend as tb
from .test_fullpol_ev_cpu import scene
from .test_fullpol_fair_cpu import fair_cfg
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)

# (id, cav, WM lane, s at first second, speed m/s, first second, depart delay)
FLEET = [('h_l0', False, 2, -900., 20., 300, 0.), ('h_l0b', False, 2, -940., 22., 300, 0.),
         ('c_l1', True, 1, -800., 25., 300, 0.), ('h_l2', False, 0, -700., 30., 300, 0.),
         ('h_late', False, 1, -1500., 18., 400, 7.), ('h_down', False, 1, 120., 28., 300, 0.)]


class _Frame:
    def __init__(self, rows):
        n = len(rows)
        self.present = np.ones(n, bool)
        self.vid_index = np.arange(n)
        self.is_cav = np.array([r[1] for r in rows])
        self.s = np.array([r[3] for r in rows], float)
        self.v = np.array([r[4] for r in rows], float)
        self.a = np.array([r[5] for r in rows], float)
        self.lane = np.array([r[2] for r in rows], int)
        self.lane_carry = self.lane.copy()
        self.junction = np.zeros(n, bool)


class FakeEnv:
    """Minimal env surface VehicleLog.tick reads (frame arrays, builder ids, read-only TraCI getters)."""

    def __init__(self, fleet, slow=0.):
        self.fleet, self.slow, self.now = fleet, slow, 300.
        self.api = SimpleNamespace(vehicle=SimpleNamespace(
            getRoute=lambda vid: ['up_0' if vid.endswith('l0') or 'l0' in vid else 'up_1'],
            getDepartDelay=lambda vid: next(f[6] for f in fleet if f[0] == vid)))

    def step_to(self, t):
        self.now = float(t)
        rows = []
        for vid, cav, lane, s0, v0, t_first, _ in self.fleet:
            if t < t_first:
                continue
            v = max(0., v0-self.slow+2.*np.sin(t/17.+len(vid)))
            rows.append((vid, cav, lane, s0+v0*(t-t_first), v, 0.3*np.cos(t/11.)))
        self.frame = _Frame(rows)
        self.builder = SimpleNamespace(vehicle_ids=[r[0] for r in rows])


def run_log(tmp_path, name, fleet=FLEET, slow=0., t1=1200):
    work = tmp_path/name
    log = fv.VehicleLog(work)
    env = FakeEnv(fleet, slow)
    for t in range(300, t1):
        env.step_to(t)
        log.tick(env)
    receipt = log.close()
    return receipt, fv.per_vehicle(log.path), log.path


def test_port_table_matches_the_n6_report_and_origins(tmp_path):
    receipt, table, path = run_log(tmp_path, 'a')
    report = fv.vehicle_log_report(path)
    assert receipt['ticks'] == 900 and receipt['vehicles'] == len(FLEET) and report['ok'], report['problems']
    cav = [d for c, _, d in table.values() if c]
    hdv = [d for c, _, d in table.values() if not c]
    assert np.mean(cav) == pytest.approx(report['delay_by_class']['cav_mean_s'])
    assert np.mean(hdv) == pytest.approx(report['delay_by_class']['hdv_mean_s'])
    assert {vid: o for vid, (_, o, _) in table.items()} == dict(
        h_l0='L0', h_l0b='L0', c_l1='L1', h_l2='L2', h_late='L1', h_down='D')
    # depart delay counts only for vehicles first logged after t0 (N6 definition)
    late = table['h_late'][2]
    assert late > 7.0 and table['h_l0'][2] > 0.


def test_groups_use_a_labels_by_id_and_report_unpaired(tmp_path):
    _, a, _ = run_log(tmp_path, 'a')
    fleet_b = [f if f[0] != 'h_l2' else ('h_l2', False, 1, -700., 30., 300, 0.) for f in FLEET]   # B: lane differs
    fleet_b = fleet_b+[('x_new', False, 0, -600., 10., 300, 0.)]                                   # not in A
    _, b, _ = run_log(tmp_path, 'b', fleet_b, slow=8.)
    groups, unpaired = fv.groups(b, fv.labels(a))
    assert unpaired == dict(n=1, cav=0, hdv=1, delay_s=pytest.approx(b['x_new'][2]))
    assert set(groups) == {'class:cav', 'class:hdv', 'origin:L0', 'origin:L1', 'origin:L2'}   # D is not a group
    assert groups['origin:L2'] == dict(n=1, mean_s=pytest.approx(b['h_l2'][2]))                 # A's label wins
    assert groups['class:hdv']['n'] == 5 and groups['origin:L1']['n'] == 2


def test_update_step_writes_a_then_attaches_b_and_fails_closed(tmp_path, cfg):
    on = fair_cfg(dict(cfg, run_root=str(tmp_path/'root')))
    ra, ta, _ = run_log(tmp_path, 'a')
    rb, tb_, _ = run_log(tmp_path, 'b', slow=8.)
    ea = [dict(id='A1', world=dict(seed=989220), vehicle_log=ra, vehicle_table=ta)]
    eb = [dict(id='B1', world=dict(seed=989220), vehicle_log=rb, vehicle_table=tb_),
          dict(id='Bp', practice=dict(source='bank'), world=dict(seed=5))]
    with pytest.raises(ValueError, match='A labels'):
        fv.update_step(on, 'B', 1, [dict(eb[0])])                 # B before A: fail closed
    out_a = fv.update_step(on, 'A', 1, ea)
    assert out_a['worlds'] == ['989220'] and fv.a_groups_path(on['run_root'], 1).is_file()
    out_b = fv.update_step(on, 'B', 1, eb)
    assert out_b == dict(arm='B', episodes=1, unpaired=0, practice_skipped=1)
    assert set(eb[0]['vehicle_groups']) == set(fv.REGISTERED_GROUPS) and 'vehicle_groups' not in eb[1]
    for broken in (dict(eb[0], vehicle_table={}), dict(eb[0], vehicle_log=dict(rb, ticks=899)),
                   dict(eb[0], world=dict(seed=1))):
        with pytest.raises(ValueError, match='fail closed'):
            fv.update_step(on, 'B', 1, [{k: v for k, v in broken.items() if k != 'vehicle_groups'}])
    off = dict(on, clean5_ev=dict(on['clean5_ev'], fair=dict(on['clean5_ev']['fair'], reward=False)))
    assert fv.update_step(off, 'A', 2, ea) is None                # reward OFF: no logger step


def test_fairness_reward_reads_real_logger_output_and_fails_closed(cfg, monkeypatch, tmp_path):
    """E2E: logger tables -> A groups file + B vehicle_groups -> ArmRuntime.train fairness receipt = the excess
    harm computed from the same per-vehicle logs; without the logger output train refuses (fail closed)."""
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    from .test_track2_backend_cpu import fake_load
    fake_load(monkeypatch)
    on = fair_cfg(dict(cfg, run_root=str(tmp_path/'root')))
    ra, ta, _ = run_log(tmp_path, 'a')
    rb, tb_, _ = run_log(tmp_path, 'b', slow=14.)                 # B much slower: some groups beyond +10 s
    fv.update_step(on, 'A', 1, [dict(id='A1', world=dict(seed=989220), vehicle_log=ra, vehicle_table=ta)])
    runtime = tb.build_arm_runtime(on, 'B')['runtime']
    runtime.safety_checked = True
    p = runtime.policy
    c = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    vehicles = [dict(r, v_mps=2.) for r in scene()]
    p.observe(vehicles, 300, native_commands={r['vid']: 0. for r in vehicles if r['is_cav']})
    c.apply(vehicles, {r['vid']: 0. for r in vehicles if r['is_cav']}, 300)
    s = c.events[-1]['sample']
    reg = on['track2_reward']
    rows = []
    for vid in s['context']['members']:
        nb = {v: -1. for v in s['track2']['reward_neighbourhoods'][vid]}
        rows.append(dict(episode_id='B1', time_s=300, cav_id=vid, source='sumo', own_reward=-1., neighbour_rewards=nb,
                         reward=-1.+reg['weight']*(sum(nb.values())/len(nb) if nb else 0.), weight=reg['weight'],
                         weight_provenance=reg['provenance']))
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    rewards = dict(schema=tb.REWARD_SCHEMA, arm='B', update=1, rows=rows)
    bare = dict(id='B1', arm='B', steps=[dict(time_s=300, sample=s)], world=dict(seed=989220), vehicle_log=rb,
                vehicle_table=tb_)
    with pytest.raises(ValueError, match='fail closed'):
        runtime.train([dict(bare)], rewards=rewards)              # logger groups not attached -> refused
    episode = dict(bare)
    fv.update_step(on, 'B', 1, [episode])
    rep = runtime.train([episode], rewards=rewards)
    a_groups = json.loads(fv.a_groups_path(on['run_root'], 1).read_text())['989220']
    expected, _ = ff.excess_harm(episode['vehicle_groups'], a_groups)
    assert expected > 0 and rep['fullpol_fair']['total_harm_veh_s'] == pytest.approx(expected)


def test_logged_tick_feeds_the_pre_step_truth(monkeypatch, tmp_path):
    cf = pytest.importorskip('tools.coopfree.backend')
    seen = []
    monkeypatch.setattr(cf, 'tick', lambda b, controller=None, trace=None: seen.append(('step', b.env.now)) or [])
    log = SimpleNamespace(tick=lambda env: seen.append(('log', env.now)))
    with fv.logged_tick(log):
        cf.tick(SimpleNamespace(env=SimpleNamespace(now=301.)), None, None)
    assert seen == [('log', 301.), ('step', 301.)] and cf.tick is not None
