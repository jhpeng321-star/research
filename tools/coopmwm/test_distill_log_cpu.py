"""CPU tests (zero SUMO) for the DISTILL teacher logger: off path unchanged, on path read-only, schema/labels."""
import math
import types

import numpy as np
import pytest

from tools.coopmwm import distill_log as D


class Frame:
    def __init__(self, ids):
        self.vid_index = np.array(ids)


class FakeEnv:
    """K=4 slots. Script: per second a dict slot -> (vid, s, v, lane, cav)."""

    def __init__(self, script):
        self.script, self.i, self.calls = script, 0, []
        self._load()

    def _load(self):
        rows = self.script[self.i]
        self.now = 300.0 + self.i
        self.rows = rows
        self.frame = Frame([sum(map(ord, rows[k][0])) if k in rows else -1 for k in range(4)])

    def slot_vehicle_ids(self):
        return [self.rows[k][0] if k in self.rows else None for k in range(4)]

    def state(self):
        import torch
        g = lambda f, d, dt: torch.tensor([[f(self.rows[k]) if k in self.rows else d for k in range(4)]], dtype=dt)  # noqa: E731
        return dict(s_m=g(lambda r: r[1], 0., torch.float32), v_mps=g(lambda r: r[2], 0., torch.float32),
                    a_mps2=g(lambda r: 0.1 * r[2], 0., torch.float32), lane=g(lambda r: r[3], -1, torch.long),
                    present=g(lambda r: True, False, torch.bool), is_cav=g(lambda r: r[4], False, torch.bool),
                    length_m=g(lambda r: 5., 5., torch.float32), junction=g(lambda r: r[3] < 0, False, torch.bool))

    def step(self, accel_by_slot=None, *, lanes=None, lane_actors=None):
        self.calls.append((np.array(accel_by_slot, dtype=float).tolist(),
                           None if lanes is None else np.array(lanes).tolist()))
        self.i += 1
        self._load()
        return None, {}


def plant_tick(b, controller=None, trace=None):
    """Stand-in for coopfree.backend.tick: CAV slots get a command (slot 3 native), lanes from the controller."""
    env = b.env
    cav = [k for k, r in env.rows.items() if r[4]]
    cmd = np.full((1, 4), np.nan)
    for k in cav:
        if k != 3:
            cmd[0, k] = -0.5 if env.rows[k][0] == 'c1' else 0.25
    lanes = None
    if controller is not None and controller.get(int(env.now)):
        lanes = np.full((1, 4), -1)
        for k, tgt in controller[int(env.now)].items():
            lanes[0, k] = tgt
    env.step(cmd, lanes=lanes)
    return [dict(vid=env.rows[k][0], mappo_safe=0.9) for k in cav if k != 3]


class Core:
    def __init__(self):
        self.model = dict(families=['famA', 'famB'])
        act = dict(t=301, members=['c2', 'c1'], prob=0.8)
        self.acts = [act]
        self.plans = [dict(family='famB')]
        self.cores = [types.SimpleNamespace(active={})]
        self._act = act
        self.scores = []

    def owners(self):
        return {v: 'famB' for v in self.cores[0].active}


class Hook:
    def __init__(self):
        self.core = Core()


def script():
    # slot: (vid, s, v, lane, cav)
    base = {0: ('c1', 10., 20., 1, True), 1: ('h1', 30., 21., 1, False), 2: ('c2', 50., 19., 2, True),
            3: ('c3', 70., 18., 0, True)}
    s3 = dict(base)
    s3[2] = ('c2', 70., 19., 1, True)                       # c2: lane 2 -> 1 during second 301
    s3[0] = ('c9', 0., 15., 0, True)                         # slot 0 reused by a new CAV (c1 left during 301)
    return [base, dict(base), s3, dict(s3)]


def run(logger, controller=None, steps=3):
    cf = types.SimpleNamespace(tick=plant_tick)
    env = FakeEnv(script())
    b = types.SimpleNamespace(env=env)
    out = []
    hook = getattr(logger, 'hook', None)
    with D.distill_tick(logger, cf=cf):
        for _ in range(steps):
            if hook is not None and int(env.now) == 301:
                hook.core.cores[0].active = {'c2': dict(act=hook.core._act), 'c1': dict(act=hook.core._act)}
                hook.core.scores.append([301, 'famB', 0.8, True])
            out.append(cf.tick(b, controller, None))
    assert cf.tick is plant_tick
    return out, env.calls


def test_off_path_patches_nothing():
    from tools.coopmwm import nm_eval_runner as R
    cf = types.SimpleNamespace(tick=plant_tick)
    with D.distill_tick(None, cf=cf) as v:
        assert v is None and cf.tick is plant_tick
    assert R.distill_logger(dict(arm='LEARNED_TRIGGER'), {}, ('hook', 'sha')) is None
    from tools.coopfree import backend as cfb
    original = cfb.tick
    with R.distill_tick(None) as v:
        assert v is None and cfb.tick is original
    assert cfb.tick is original


def test_distill_refused_off_learned_arm():
    from tools.coopmwm import nm_eval_runner as R
    with pytest.raises(ValueError):
        R.distill_logger(dict(arm='U0', distill_log=dict(path='x.npz')), {}, None)


def test_on_path_is_read_only(tmp_path):
    ctl = {301: {2: 1}}
    ref = run(None, ctl)
    log = D.DistillLogger(tmp_path/'s.npz', hook=Hook(), meta=dict(seed=999990))
    got = run(log, ctl)
    assert repr(got) == repr(ref)                            # same audits, same env.step arguments, same order


def test_schema_and_labels(tmp_path):
    log = D.DistillLogger(tmp_path/'s.npz', hook=Hook(), meta=dict(seed=999990))
    run(log, {301: {2: 1}})
    rec = log.close()
    s = D.load_shard(tmp_path/'s.npz')
    assert rec['steps'] == 3 and s['meta']['schema'] == D.SCHEMA and int(s['K']) == 4
    assert list(s['t']) == [300, 301, 302] and s['off'][-1] == 12 and len(s['cav_step']) == 9
    names = list(s['vid_names'])
    vid_of = lambda k: names[s['vid'][s['cav_row'][k]]]  # noqa: E731
    rows = {(int(s['t'][s['cav_step'][k]]), vid_of(k)): k for k in range(len(s['cav_step']))}
    # applied accel = exactly what env.step received; slot 3 native
    assert s['accel_applied'][rows[(300, 'c1')]] == np.float32(-0.5)
    assert math.isnan(s['accel_applied'][rows[(300, 'c3')]]) and s['ctrl_src'][rows[(300, 'c3')]] == 0
    assert s['accel_u0'][rows[(300, 'c2')]] == np.float32(0.9)
    # lane: c2 2 -> 1 over second 301 (label -1, commanded -1); unchanged elsewhere; last step invalid
    k = rows[(301, 'c2')]
    assert s['lane_label'][k] == -1 and s['lane_valid'][k] and s['lane_cmd'][k] == -1
    assert s['lane_label'][rows[(300, 'c2')]] == 0 and s['lane_valid'][rows[(300, 'c2')]]
    assert not s['lane_valid'][rows[(301, 'c1')]]          # c1 left at 302
    assert not s['lane_valid'][rows[(302, 'c2')]]          # last logged step
    # plan membership at 301
    k = rows[(301, 'c1')]
    assert s['plan_active'][k] and s['plan_family'][k] == 1 and s['plan_role'][k] == 1 and s['plan_onset'][k]
    assert s['ctrl_src'][k] == 2 and abs(s['trigger_prob'][k] - 0.8) < 1e-6
    assert not s['plan_active'][rows[(300, 'c1')]]
    assert list(s['score_t']) == [301] and s['score_fire'][0]
    # teacher memory: c2 at 301 sees its own 300 command (0.25); slot 0 reuse at 302 does not inherit c1's memory
    off = s['off']
    row_c2_301 = off[1] + list(s['slot'][off[1]:off[2]]).index(2)
    assert s['cmd_accel_mps2'][row_c2_301] == np.float32(0.25)
    row_c9 = off[2] + list(s['slot'][off[2]:off[3]]).index(0)
    assert s['cmd_accel_mps2'][row_c9] == 0 and s['dwell_age_s'][row_c9] == 3.0
    row_c2_302 = off[2] + list(s['slot'][off[2]:off[3]]).index(2)
    assert s['lane_intent'][row_c2_302] == -1 and s['dwell_age_s'][row_c2_302] == 0.0
    # a_next: measured next-step accel of the same vehicle
    assert abs(s['a_next'][rows[(300, 'c1')]] - 2.0) < 1e-5
    # actor-input reconstruction: [1,K] state dict
    steps = list(D.actor_states(s, torch_tensors=False))
    i, t, st = steps[1]
    assert t == 301 and st['s_m'].shape == (1, 4) and st['present'].all() and st['lane'][0, 2] == 2
    assert set(st) >= {'s_m', 'v_mps', 'a_mps2', 'lane', 'present', 'is_cav', 'length_m', 'cmd_accel_mps2',
                       'lane_intent', 'dwell_age_s'}


def test_seed_guards():
    assert D.check_seeds([999990, 999991]) == [999990, 999991]
    for bad in ([993100], [989650], [964100], [962001], [977500], [991226]):
        with pytest.raises(ValueError):
            D.check_seeds(bad)
    ms = types.SimpleNamespace(EXTRA_TRAIN_RANGES=((989000, 989099),))
    D.install_distill_range(ms)
    D.install_distill_range(ms)
    assert ms.EXTRA_TRAIN_RANGES == ((989000, 989099), (999000, 999999))
    with pytest.raises(ValueError):
        D.guard_path('/x/sealed/y.npz')


def test_launcher_seed_rules():
    from ops import distill_collect as C
    assert C.parse_seeds('999000-999002') == [999000, 999001, 999002]
    for bad in ('993100-993101', '989600', '964000-964001', '962000', '977000', '991226', '998999-999000'):
        with pytest.raises(SystemExit):
            C.parse_seeds(bad)
    with pytest.raises(SystemExit):
        C.guard_path('/home/ubuntu/x_holdout/y')
    # overlay paths map back into this repository
    for dest in C.OVERLAY:
        src = next(r + dest[len(k):] for k, r in C.REPO_OF.items() if dest.startswith(k))
        assert (__import__('pathlib').Path(__file__).resolve().parents[2]/src).is_file(), src
