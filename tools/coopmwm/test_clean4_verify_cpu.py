"""CPU tests for the clean4 pre-launch re-verification pipeline (no SUMO, no GPU, no ssh).

A deterministic per-second plant stands in for the SUMO branch workers (real sv.verify,
verify_perseed.confirm_all8 and bank_statistics run on its tapes); a fake accel_plant episode drives
the real Capture hook; geometry/runtime adapters are fakes (the production ones are N1 stubs)."""
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.coopmwm import clean4_verify as cv
from tools.coopmwm import group_lane_bank as glb
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm import verify_perseed as vp
from tools.coopmwm.exact_grade import BudgetBlocked
from tools.coopmwm.protocol import digest
from tools.coopnative.storage import canonical

WORLD = dict(seed=989200, cell=4, repeat=0, penetration=0.4, demand=2800, map='wz3to2', cell_key='D2800_p40')
SPEC = Path(__file__).resolve().parents[2]/'assets'/'research'/'newmap_candidates'/'wz3to2'/'MAP_SPEC.json'
OFFSET = cv.X_B-75.44              # fake WM frame: s = 75.44 at the drop
WM_TO_NM = {2: 0, 1: 1, 0: 2}      # fake WM lane -> NEWMAP2 lane (2 = closing lane)
NM_TO_WM = {v: k for k, v in WM_TO_NM.items()}


class FakeGeom:
    name, x_drop = 'fake', cv.X_B

    def locate(self, row):
        return ('up', WM_TO_NM[row['lane']], float(row['s_m'])+OFFSET)


GEOM = FakeGeom()


def row(vid, x, nm_lane, v=25.0, cav=True):
    return dict(vid=vid, s_m=x-OFFSET, lane=NM_TO_WM[nm_lane], v_mps=v, is_cav=cav, length_m=5.0)


# ------------------------------------------------------------------ ledger

def test_ledger_registration_precharge_cap_and_duplicates(tmp_path):
    with pytest.raises(ValueError):
        cv.charge(tmp_path, 'x', 10)                      # never before the registration
    with pytest.raises(ValueError):
        cv.init(tmp_path, cap=400001)                     # hard phase cap (MAIN: 400,000)
    reg = cv.init(tmp_path, cap=5000)
    assert reg['phase'] == 'clean4_prelaunch_verify' and reg['cap_sim_s'] == 5000 and reg['rules']['selection']
    assert cv.init(tmp_path, cap=5000)['registration_sha256'] == reg['registration_sha256']
    with pytest.raises(ValueError):
        cv.init(tmp_path, cap=6000)                       # never rewritten
    cv.charge(tmp_path, 'a', 3000, stage='collect')
    with pytest.raises(BudgetBlocked):
        cv.charge(tmp_path, 'a', 1)                       # duplicate: never replayed
    with pytest.raises(BudgetBlocked):
        cv.charge(tmp_path, 'b', 2001)                    # over cap
    with pytest.raises(ValueError):
        cv.charge(tmp_path, 'c', 0)
    cv.charge(tmp_path, 'b', 2000)
    assert cv.spent(tmp_path) == 5000 and cv.remaining(tmp_path) == 0
    assert {r['phase'] for r in cv.ledger_rows(tmp_path)} == {cv.PHASE}
    p = tmp_path/'REGISTRATION.json'
    p.write_text(p.read_text().replace('"cap_sim_s":5000', '"cap_sim_s":80000'))
    with pytest.raises(ValueError):
        cv.charge(tmp_path, 'd', 1)                       # tampered registration refuses


def test_init_refuses_root_with_prior_charges(tmp_path):
    (tmp_path/'CHARGES.jsonl').write_text('{"id":"x","sim_s":1}\n')
    with pytest.raises(ValueError):
        cv.init(tmp_path)


def test_concurrent_charges_never_exceed_cap(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    cv.init(tmp_path, cap=5000)

    def one(i):
        try:
            cv.charge(tmp_path, f'c{i}', 300)
            return True
        except BudgetBlocked:
            return False
    with ThreadPoolExecutor(8) as pool:
        ok = sum(pool.map(one, range(24)))
    assert ok == 16 and cv.spent(tmp_path) == 4800


# ------------------------------------------------------------------ selection

def test_n3_rule_pair_and_wall_geometry():
    rows = [row('a1', 1500, 0), row('a2', 1560, 0), row('b1', 1530, 1), row('b2', 1700, 1), row('c1', 1600, 2),
            row('far', 2400, 0), row('hdv', 1590, 2, cav=False)]
    m = cv.n3_select(rows, 470, geom=GEOM)
    assert m['rank'] == 0 and m['kind'] == 'pair+wall'
    assert m['pair'] == ['a2', 'b1']                      # min dx (tie 30 m) -> most downstream closing CAV
    assert m['wall'] == ['a2', 'b1', 'c1']                # min spread 70 m
    assert 'far' not in {c['id'] for c in m['zone']} and 'hdv' not in {c['id'] for c in m['zone']}
    assert cv.n3_select([row('a1', 1500, 0), row('b1', 1650, 1)], 450, geom=GEOM) is None   # dx 150 > 100
    w = cv.n3_select([row('a1', 1500, 0), row('b1', 1650, 1), row('c1', 1690, 2)], 820, geom=GEOM)
    assert w['rank'] == 2 and w['pair'] is None and w['wall'] == ['a1', 'b1', 'c1']


def test_window_keeper_prefers_rank_then_earliest():
    k = cv.WindowKeeper()
    assert k.window_of(450) == 0 and k.window_of(455) is None and k.window_of(900) == 1 and k.window_of(600) is None
    assert k.wants(0, 1)
    k.keep(0, 450, dict(rank=1), 's450')
    assert not k.wants(0, 1) and not k.wants(0, 2) and k.wants(0, 0)
    assert k.keep(0, 470, dict(rank=0), 's470') == [450]
    assert not k.wants(0, 0)
    assert k.final()[0][0] == 470


class Guarded(dict):
    """An observation row that fails the test if any outcome/hidden field is read or the row is iterated."""
    FORBIDDEN = {'team_delay_vs', 'J_veh_s', 'collisions', 'teleports', 'outcome', 'driver_type'}

    def __getitem__(self, key):
        assert key not in self.FORBIDDEN, f'selector read {key}'
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        assert key not in self.FORBIDDEN, f'selector read {key}'
        return dict.get(self, key, default)

    def __iter__(self):
        raise AssertionError('row iterated')

    def keys(self):
        raise AssertionError('row keys read')

    def items(self):
        raise AssertionError('row items read')

    def values(self):
        raise AssertionError('row values read')


def rows_at(t):
    out = [row('a1', 1500, 0), row('b1', 1530, 1), row('h1', 1510, 1, cav=False)]
    if 470 <= t < 600:
        out.append(row('c1', 1600, 2))
    if 800 <= t < 820:
        out = [row('a1', 2500, 0), row('b1', 2520, 1)]
    elif t >= 820:
        out = [row('a1', 1500, 0), row('b1', 1650, 1), row('c1', 1690, 2)]
    return [Guarded(r, team_delay_vs=123.4, J_veh_s=9.9, collisions=1, outcome='x', driver_type='aggr') for r in out]


class FakeRT:
    base = 'frozen_u0'

    def __init__(self, root):
        self.root = root

    def job(self, **kw):
        return cv.job_fields(runtime='rt', checkpoint='ck', **kw)

    def episode(self, job, controller_hook=None):
        assert any(r['id'] == 'collect_'+job['id'] and r['sim_s'] == 1200 for r in cv.ledger_rows(self.root)), \
            'collection must be charged before physics'
        c = SimpleNamespace(events=[], generator='gen')
        prefixes = {}
        controller_hook(c, job, prefixes)
        for t in range(300, job['stop']):
            prefixes[str(t)] = t
            c.events.append(dict(time_s=t, sample=dict(context=dict(members=['a1']), action=[[0.1]]),
                                 commands={'a1': 0.25, 'b1': 0.25, 'c1': 0.25}, rows=rows_at(t)))
            controller_hook.after_step(t, dict(totals=dict(team_delay_vs=1e9)))
        return dict(actual_sim_s=job['stop']-300)


def test_collect_selects_earliest_best_moment_per_window_and_never_reads_outcomes(tmp_path):
    cv.init(tmp_path)
    captured = []

    def capture(controller, prefixes, job, t, sample, generator):
        captured.append(t)
        return dict(t0=t, prefix_n=len(prefixes))
    leads = [dict(id='nm2_D2800_p40_s989200_t450', seed=989200, cell_key='D2800_p40', t0=450, window=0, arm='B')]
    saved = cv.collect(tmp_path, WORLD, 'B', {'w': 1}, geom=GEOM, rt=FakeRT(tmp_path), leads=leads,
                       capture_fn=capture, snapshot_sha=digest)
    assert captured == [450, 470, 820]                    # pair at 450, pair+wall at 470 supersedes, wall at 820
    assert [Path(d).name for d in saved] == ['ctx_t470', 'ctx_t820']
    c0 = json.loads((Path(saved[0])/'CONTEXT.json').read_text())
    c1 = json.loads((Path(saved[1])/'CONTEXT.json').read_text())
    assert c0['moment']['kind'] == 'pair+wall' and c0['geometry']['pair'] == ['a1', 'b1']
    assert c0['geometry']['wall'] == ['a1', 'b1', 'c1'] and c0['lead'] == leads[0]['id']
    assert c1['moment']['kind'] == 'wall' and c1['lead'] is None and c1['window'] == 1
    prof = c0['policy_profile']
    assert len(prof['a1']) == cv.PROFILE_S and prof['a1'][0] == pytest.approx(0.35) and prof['b1'][0] == 0.25
    assert c0['cav_track']['c1'][0][1] == NM_TO_WM[2] and c0['cav_track']['c1'][130] is None   # left the rows at 600
    assert c0['snapshot_sha256'] == digest(torch.load(Path(saved[0])/'SNAPSHOT.pt', weights_only=False))
    assert c0['state_id'] == 'c4v_B_D2800_p40_s989200_t470'
    rec = json.loads((tmp_path/'collect'/'c4v_B_D2800_p40_s989200'/'COLLECTION_RECEIPT.json').read_text())
    assert rec['charged_sim_s'] == 1200 and rec['marks'] == 22
    assert [r['id'] for r in cv.ledger_rows(tmp_path)] == ['collect_c4v_B_D2800_p40_s989200']
    with pytest.raises(BudgetBlocked):                    # never replayed
        cv.collect(tmp_path, WORLD, 'B', {'w': 1}, geom=GEOM, rt=FakeRT(tmp_path), capture_fn=capture,
                   snapshot_sha=digest)


def test_collect_stubs_refuse_before_any_charge(tmp_path):
    cv.init(tmp_path)
    with pytest.raises(NotImplementedError):
        cv.collect(tmp_path, WORLD, 'B', {}, rt=FakeRT(tmp_path))            # geometry stub (N1 pending)
    with pytest.raises(NotImplementedError):
        cv.collect(tmp_path, WORLD, 'B', {}, geom=GEOM)                      # runtime stub (N1/N4 pending)
    assert cv.ledger_rows(tmp_path) == [] and not (tmp_path/'collect').exists()


# ------------------------------------------------------------------ plans

def ctx_for(t0=450, arm='B', zone=None, profile_s=cv.PROFILE_S):
    ident = f'c4v_{arm}_D2800_p40_s989200'
    zone = zone or [dict(id='cA', lane=0, x=1500.0, v=25.0), dict(id='cB', lane=1, x=1520.0, v=25.0),
                    dict(id='cC', lane=2, x=1560.0, v=25.0), dict(id='cD', lane=1, x=1700.0, v=25.0)]
    zone = [dict(c, s_m=c['x']-OFFSET, edge='up') for c in zone]
    ids = [c['id'] for c in zone]
    pair = [c['id'] for c in zone if c['lane'] == 0][:1]+[c['id'] for c in zone if c['lane'] == 1][:1]
    wall = [next(c['id'] for c in zone if c['lane'] == ln) for ln in (0, 1, 2)
            if any(c['lane'] == ln for c in zone)]
    return dict(state_id=f'{ident}_t{t0}', world=WORLD, arm=arm, t0=t0, window=cv.window_index(t0),
                base='frozen_u0', moment=dict(kind='pair+wall'), members=list(dict.fromkeys(pair+wall)),
                geometry=dict(pair=pair if len(pair) == 2 else None, wall=wall if len(wall) == 3 else None, zone=zone),
                policy_profile={v: [0.1]*profile_s for v in ids}, cav_track={v: [[0., 1, 25.]]*profile_s for v in ids},
                lead='nm2_D2800_p40_s989200_t450', selection=cv.SELECTION_RULE)


def test_provisional_builder_law_release_and_anchor():
    ctx = ctx_for(zone=[dict(id='cA', lane=0, x=2600.0, v=30.0), dict(id='cB', lane=1, x=2590.0, v=30.0)])
    var = dict(name='pair_g', family='closing_pair', members=['cA', 'cB'],
               params=dict(v_w=22.0, hold_s=120, dmax=1.0, amax=1.0, release_x=cv.X_B))
    p = cv.provisional_builder(ctx, ['cA', 'cB'], var)
    assert p['provisional'] and p['builder'] == cv.PROVISIONAL and p['duration_s'] == 120
    ra, rb_ = p['release_s']['cA'], p['release_s']['cB']
    assert 0 < ra <= rb_ < 10                             # both reach the drop within seconds
    assert all(p['profile'][0][k] == -1.0 for k in range(ra))                       # clipped at -dmax
    assert all(p['perturbation_profile'][0][k] == 0.0 for k in range(ra, 120))      # zero after release
    assert all(x == 0.1 for x in p['policy_profile'][1])
    # anchor (pair_t): the closing-lane CAV reaching X_B-150 releases BOTH members
    ctx = ctx_for(zone=[dict(id='cA', lane=0, x=2500.0, v=20.0), dict(id='cB', lane=1, x=2000.0, v=20.0)])
    var = dict(name='pair_t', family='closing_pair', members=['cA', 'cB'],
               params=dict(v_w=22.0, hold_s=150, dmax=1.0, amax=1.0, release_x=cv.X_B,
                           anchor='cA', anchor_release_x=cv.X_B-150.0))
    p = cv.provisional_builder(ctx, ['cA', 'cB'], var)
    assert p['release_s']['cA'] == p['release_s']['cB'] == 3
    assert p['profile'][1][:4] == [1.0, 1.0, 0.0, 0.1]    # <= amax, holds v_w, then policy reference
    # a commanded second without a reference is never imputed
    ctx['policy_profile']['cB'][5] = None
    assert cv.provisional_builder(ctx, ['cA', 'cB'], var) is None
    ctx['cav_track']['cB'] = [[0., 1, 20.]]*5+[None]*145   # ... unless the CAV left the rows
    assert cv.provisional_builder(ctx, ['cA', 'cB'], var) is not None


def test_plans_variants_solos_and_r_procedure():
    ctx = ctx_for()
    built = cv.plans(ctx)
    names = [built['meta'][g['id']]['variant'] for g in built['groups']]
    assert names == ['pair_g', 'pair_h', 'pair_t', 'wall_g', 'wall_h']
    fams = {g['family'] for g in built['groups']}
    assert fams == {'closing_pair', 'abreast_wall'}
    g = built['groups'][0]
    assert [s['members'] for s in built['solos'][g['id']]] == [['cA'], ['cB']]
    assert built['solos'][g['id']][0]['id'] == g['id']+'__solo_cA'
    assert built['meta'][g['id']]['max_member_dist_t0_m'] == pytest.approx(math.hypot(20, 3.2))
    pt = next(x for x in built['groups'] if x['variant']['name'] == 'pair_t')
    assert pt['variant']['params']['anchor'] == 'cA'
    r = cv.r_proposals(dict(ctx, arm='R'))
    assert r == cv.r_proposals(dict(ctx, arm='R'))        # seeded by the state id
    assert 1 <= len(r) <= cv.N_R
    for p in r:
        xs = [c['x'] for c in ctx['geometry']['zone'] if c['id'] in p['members']]
        assert 2 <= len(p['members']) <= 3 and max(xs)-min(xs) <= cv.R_SPREAD
        assert set(p['members']) not in ({'cA', 'cB'}, {'cA', 'cB', 'cC'}) and p['family'] == cv.R_FAMILY
        assert p['params']['v_w'] in cv.R_GRID['v_w'] and p['params']['hold_s'] in cv.R_GRID['hold_s']


# ------------------------------------------------------------------ verify (fake SUMO plant)

def fp(t, delay):
    def hexf(x):
        return float(x).hex()
    return dict(t_s=t, metrics=dict(team_delay_vs=hexf(delay), arrived=hexf(t), collisions=hexf(0.),
                                    teleports=hexf(0.), ttc_below_3_vs=hexf(0.), hard_brake=hexf(0.)))


def cum0(t):
    return 3.+0.5*t*t/100.


def seed_of(snap):
    salt = snap['rng'].get('replication_salt')
    if salt is None:
        return 1
    if '#seed' in salt:
        return int(salt.split('#seed', 1)[1].split('#', 1)[0])
    return 2


class Plant:
    """Per-second plant: gains[(plan_id, seed)] = delay reduction per controlled second; member solos
    gain 0 (so a gaining group is cooperative). Writes RESULT.json, observable trace and spec."""

    def __init__(self, work, gains):
        self.work, self.gains = Path(work), gains

    def admit(self, specs):
        return None

    def rate(self, spec, snap, t):
        p, k = spec['plan'], seed_of(snap)
        base = 2.0+0.4*math.sin(0.21*t+k)+0.05*math.sin(1.3*t+k)
        if p is None or t >= spec['window']['start_s']+int(p['duration_s']):
            return base
        g = 0. if '__solo_' in p['id'] else self.gains.get((p['id'], k), -0.3)
        return base-g+0.02*math.cos(0.7*t+len(p['id']))

    def run(self, spec, snap):
        from tools.coopmwm.accel_storage import write as awrite
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        bdir = self.work/'branches'/spec['verification_id']
        d = bdir/spec['branch']
        d.mkdir(parents=True, exist_ok=True)
        (bdir/f'{spec["branch"]}.spec.json').write_text(json.dumps(spec))
        obs = dict(time_s=t0, vehicles=[dict(lane='m1', s_m=10., v_mps=8., length_m=5.),
                                        dict(lane='m1', s_m=0., v_mps=10., length_m=5.)])
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.write(canonical(obs)+'\n')
        prefixes = {str(t): fp(t, cum0(t)) for t in range(300, t0+1)}
        cum = cum0(t0)
        for t in range(t0, t_end):
            cum += self.rate(spec, snap, t)
            if t+1 < t_end:
                prefixes[str(t+1)] = fp(t+1, cum)
        tape = dict(prefixes=prefixes, final=fp(t_end, cum), steps=[])
        awrite(d/'RESULT.json', tape)
        return dict(prefixes=prefixes, final=tape['final'], actual_sim_s=t_end-300, result_sha256=digest(tape))


def make_ctx_dir(root, t0=450, arm='B'):
    ctx = ctx_for(t0, arm)
    ident = ctx['state_id'].rsplit('_t', 1)[0]
    snap = dict(episode=dict(id=ident, world=WORLD, arm=arm, update=0, role='train', training_seed=20261101),
                policy=dict(state_dict_sha256='p'), sumo=dict(runtime='rt', checkpoint='ck', warm_history=True,
                                                             accel_bounds=[-3., 2.6], noise_seed=11),
                t0=t0, end_s=1200, prefixes={str(t): fp(t, cum0(t)) for t in range(300, t0+1)}, samples={},
                rng=dict(generator_state_sha256='g'), committed_plans=[])
    ctx['snapshot_sha256'] = sv.snapshot_sha(snap)
    d = Path(root)/'collect'/ident/f'ctx_t{t0}'
    d.mkdir(parents=True)
    torch.save(snap, d/'SNAPSHOT.pt')
    (d/'CONTEXT.json').write_text(json.dumps(ctx))
    frame = [cv.observable(row('cA', 1500, 0)), cv.observable(row('cB', 1520, 1))]
    (d/'FRAMES.json').write_text(json.dumps(cv.frames_record(WORLD, t0, frame, frame)))
    return d, ctx


def factory(gains):
    return lambda vroot, workers: Plant(vroot, gains)


def pair_g_id(ctx):
    return 'c4v_pair_g_cA_cB'


def test_stage_gating_seed1_fail_spends_no_confirmation(tmp_path):
    cv.init(tmp_path)
    d, ctx = make_ctx_dir(tmp_path)
    res = cv.run_context(tmp_path, d, seeds=3, workers=1, runner_factory=factory({}))
    assert res['status'] == 'screened' and res['passing'] == 0 and 'native' not in res
    ids = [r['id'] for r in cv.ledger_rows(tmp_path)]
    assert len(ids) == 1 and ids[0].startswith('scr_') and not any(i.startswith('nat_') for i in ids)
    screen = [json.loads(x) for x in (tmp_path/'results'/'screen.jsonl').read_text().splitlines()]
    assert len(screen) == 5 and not any(r['passed'] for r in screen)
    assert all('advantage LB95 <= 0' in r['reason'] for r in screen)
    # ONE screen verification: policy + 5 groups + 12 member solos, precharged in full
    rec = json.loads(next((tmp_path/'screen'/ctx['state_id']/'verifications').glob('*.json')).read_text())
    assert rec['k'] == 17 and cv.spent(tmp_path) == rec['charged_sim_s'] == 18*(450+150+120)
    assert not (tmp_path/'results'/'native.jsonl').exists() and not (tmp_path/'minebank').exists()
    assert cv.run_context(tmp_path, d, runner_factory=factory({}))['status'] == 'skipped_claimed'
    # every lost group (W2 rule: own-seed synergy < 0) -> one negative-frame row in the W2 frame format
    out = cv.extract(tmp_path)
    neg = [json.loads(x) for x in (tmp_path/'CLEAN4_NEGATIVE_FRAMES.jsonl').read_text().splitlines()]
    assert out['negative_frames'] == len(neg) == 5 and out['frames_missing'] == 0
    for n in neg:
        assert n['kind'] == 'lost' and n['basis'] == 'synergy' and n['synergy_veh_s']['1'] < 0
        assert set(n['frames']) == {'449', '450'} and n['world_seed'] == 989200 and n['arm'] == 'B'
        assert all(set(r) == set(cv.FRAME_KEYS) for rows in n['frames'].values() for r in rows)
        assert n['frames_sha256'] == digest(n['frames']) and n['source'] == cv.PHASE and n['members']


def test_pass_confirms_natively_and_extracts_bound_rows(tmp_path):
    cv.init(tmp_path)
    d, ctx = make_ctx_dir(tmp_path)
    original = rb._confirm_charge
    gains = {(pair_g_id(ctx), k): 0.6 for k in (1, 2, 3)}
    res = cv.run_context(tmp_path, d, seeds=3, workers=1, runner_factory=factory(gains))
    assert rb._confirm_charge is original                 # scoped probe_native binding restored
    assert res['passing'] == 1 and res['native'] == 'confirmed' and res['admitted'] is True
    rows = cv.ledger_rows(tmp_path)
    nat = [r for r in rows if r['id'].startswith('nat_')]
    kinds = {r['stage'] for r in nat}
    assert {'first', 'confirm_seed2_all8', 'confirm_seed3_all8', 'confirm_solo_all8_s1'} <= kinds
    assert sum(r['sim_s'] for r in rows) == cv.spent(tmp_path) <= 80000
    bank = rb.bank_root(cv.bank_dir(tmp_path), 'B')
    assert (bank/'snapshots'/(ctx['state_id']+'.pt')).exists() and (bank/'decisions'/(ctx['state_id']+'.json')).exists()
    rec = next(iter(vp.records(cv.bank_dir(tmp_path), 'B').values()))
    assert sorted({e['seed_index'] for e in rec['per_seed_v1']['examples']}) == [1, 2, 3]

    out = cv.extract(tmp_path, range_m=200.0, cell='D2800_p40')
    assert out['mined_B'] == 3 and out['mined_R'] == 0
    moves = [json.loads(x) for x in (tmp_path/'CLEAN4_MOVES_B.jsonl').read_text().splitlines()]
    assert sorted(m['seed'] for m in moves) == [1, 2, 3]
    for m in moves:
        assert m['schema'] == 'coopmwm-mined-move-v1' and m['map'] == 'wz3to2' and m['family'] == 'closing_pair'
        assert m['source'] == 'clean4_prelaunch_verify' and m['lead'] == ctx['lead'] and m['variant'] == 'pair_g'
        assert m['run_root'] == str(cv.bank_dir(tmp_path)) and m['situation_id'] == ctx['state_id']
        assert m['n_reps'] == 3 and m['duration_s'] == 120 and m['members'] == ['cA', 'cB']
        assert all(k in m for k in glb.MINED_BINDING_FIELDS)
        assert m['synergy_lower95'][str(m['seed'])] > 0 and m['reachable'] is True
        with pytest.raises(ValueError, match='TRAIN world seed'):
            glb.binding(m)                                # this process is not bound to wz3to2
    from tools.coopmwm import map_descriptor as md
    md._reset_for_tests()
    md.bind(md.load(SPEC))
    try:
        for m in moves:                                   # nm-train 7a0f7721: map-aware TRAIN predicate
            b = glb.binding(m)
            assert b['seed'] == m['seed'] and b['n_reps'] == 3 and b['world'] == WORLD
    finally:
        md._reset_for_tests()
    negs = [json.loads(x) for x in (tmp_path/'CLEAN4_NEGATIVES.jsonl').read_text().splitlines()]
    assert sorted(n['variant'] for n in negs if n['stage'] == 'screen') == ['pair_h', 'pair_t', 'wall_g', 'wall_h']
    pot = json.loads((tmp_path/'POTENTIAL_INPUTS.json').read_text())
    c = pot['chosen']
    assert c['screened_moments'] == 1 and c['robust_moments'] == 1 and len(c['b_examples']) == 3
    assert all(e['reachable'] and e['robust'] for e in c['b_examples'])
    assert {s['path'].rsplit('\\', 1)[-1].rsplit('/', 1)[-1] for s in pot['sources']} >= {'CLEAN4_MOVES_B.jsonl'}
    from tools.coopmwm import clean4_potential as gate
    v = gate.verdict(cell=dict(cell_key='D2800_p40'), screened_moments=c['screened_moments'],
                     robust_moments=c['robust_moments'], b_examples=c['b_examples'], sources=pot['sources'],
                     range_m=pot['range_m'])
    assert v['checks']['robust_share']['passed'] and not v['checks']['b_examples']['passed']   # 3 < 8


def test_confirmation_refused_when_cap_cannot_cover_the_stage(tmp_path):
    cv.init(tmp_path, cap=18*720+5000)
    d, ctx = make_ctx_dir(tmp_path)
    original = rb._confirm_charge
    gains = {(pair_g_id(ctx), k): 0.6 for k in (1, 2, 3)}
    res = cv.run_context(tmp_path, d, seeds=3, workers=1, runner_factory=factory(gains))
    assert res['passing'] == 1 and res['native'] == 'refused_budget'
    assert not any(r['id'].startswith('nat_') for r in cv.ledger_rows(tmp_path)) and rb._confirm_charge is original
    nat = json.loads((tmp_path/'results'/'native.jsonl').read_text().splitlines()[0])
    assert nat['confirm_quote_sim_s'] > cv.remaining(tmp_path)


def test_named_slate_and_quote(tmp_path):
    cv.init(tmp_path)
    d, ctx = make_ctx_dir(tmp_path)
    groups = cv.plans(ctx, [v for v in cv.variants_for(ctx) if v['name'] == 'pair_g'])['groups']
    q = cv.quote_moment(ctx, groups, seeds=3)
    assert q['n_groups'] == 1 and q['n_solos'] == 2 and q['screen_sim_s'] == 4*(450+120+120)   # policy + group + 2 solos
    assert q['confirm_one_max_sim_s'] == q['confirm_all_sim_s'] == 17370
    res = cv.run_context(tmp_path, d, workers=1, runner_factory=factory({}), names=('pair_g',))
    assert res['screened'] == 1 and cv.spent(tmp_path) == q['screen_sim_s']


def test_screen_refused_without_charge_or_claim(tmp_path):
    cv.init(tmp_path, cap=1000)
    d, ctx = make_ctx_dir(tmp_path)
    res = cv.run_context(tmp_path, d, workers=1, runner_factory=factory({}))
    assert res['status'] == 'refused_screen' and cv.ledger_rows(tmp_path) == []
    assert not (tmp_path/'claims'/f'verify_{ctx["state_id"]}.claim').exists()


def test_forbidden_salt_refuses_confirmation(tmp_path, monkeypatch):
    cv.init(tmp_path)
    d, ctx = make_ctx_dir(tmp_path)
    monkeypatch.setattr(cv, 'generator_seed', lambda salt: 989650)
    gains = {(pair_g_id(ctx), k): 0.6 for k in (1, 2, 3)}
    res = cv.run_context(tmp_path, d, seeds=3, workers=1, runner_factory=factory(gains))
    assert res['native'] == 'refused_salt' and not any(r['id'].startswith('nat_') for r in cv.ledger_rows(tmp_path))


def test_verify_argument_guards(tmp_path):
    cv.init(tmp_path)
    d, _ = make_ctx_dir(tmp_path)
    with pytest.raises(ValueError):
        cv.verify(tmp_path, d, [], seeds=1, runner_factory=factory({}))
    with pytest.raises(ValueError):
        cv.verify(tmp_path, d, [], tail_s=60, runner_factory=factory({}))
    ctx = json.loads((d/'CONTEXT.json').read_text())
    (d/'CONTEXT.json').write_text(json.dumps(dict(ctx, snapshot_sha256='0'*64)))
    with pytest.raises(ValueError):
        cv.verify(tmp_path, d, [], runner_factory=factory({}))


# ------------------------------------------------------------------ extract rule + potential math

def ex(**kw):
    e = dict(admitted=True, cooperative=True, safety_ok=True,
             synergy=dict(lower95_one_sided_veh_s=3.0, pooled_total_veh_s=9.0),
             tape=dict(branch=dict(window=dict(collisions=0., teleports=0.))))
    e.update(kw)
    return e


def test_mined_ok_admission_rule():
    plan = dict(family='abreast_wall', members=['a', 'b', 'c'])
    assert cv.mined_ok(ex(), plan)
    assert not cv.mined_ok(ex(admitted=False), plan)
    assert not cv.mined_ok(ex(cooperative='not_cooperative'), plan)
    assert not cv.mined_ok(ex(safety_ok=False), plan)
    assert not cv.mined_ok(ex(synergy=dict(lower95_one_sided_veh_s=-0.1)), plan)
    assert not cv.mined_ok(ex(tape=dict(branch=dict(window=dict(collisions=1.)))), plan)
    assert not cv.mined_ok(ex(tape=dict(branch=dict(window=dict(teleports=1.)))), plan)
    assert not cv.mined_ok(ex(), dict(family='abreast_wall', members=['a']))
    assert not cv.mined_ok(ex(), dict(family='ease_off', members=['a', 'b']))
    assert cv.mined_ok(ex(), dict(family='R_random', members=['a', 'b']))


def test_potential_inputs_math():
    C = 'D3200_p60'
    contexts = [dict(arm='B', state_id='m1', cell_key=C, screen_status='rejected', lead='n1'),
                dict(arm='B', state_id='m2', cell_key=C, screen_status='accepted', lead=None),
                dict(arm='B', state_id='m3', cell_key=C, screen_status='no_candidates'),
                dict(arm='B', state_id='m4', cell_key=C, screen_status='refused'),
                dict(arm='B', state_id='m5', cell_key=C, screen_status='refused'),
                dict(arm='B', state_id='m5', cell_key=C, screen_status='accepted'),    # later screen wins
                dict(arm='R', state_id='r1', cell_key=C, screen_status='accepted')]
    examples = ([dict(arm='B', situation_id='m1', plan_sha256='p'*64, seed=s, ok=True, n_reps=3) for s in (1, 2, 3)]
                + [dict(arm='B', situation_id='m2', plan_sha256='q'*64, seed=s, ok=s != 2, n_reps=3) for s in (1, 2, 3)]
                + [dict(arm='B', situation_id='m5', plan_sha256='z'*64, seed=s, ok=True, n_reps=3) for s in (1, 2)])
    def mk(sid, p, s, dist):
        return dict(arm='B', cell=C, situation_id=sid, plan_sha256=p*64, seed=s, verification_id='v_'+sid,
                    max_member_dist_t0_m=dist)
    mined = [mk('m1', 'p', 1, 150.), mk('m1', 'p', 2, 150.), mk('m1', 'p', 3, 150.), mk('m2', 'q', 1, 250.),
             mk('m2', 'q', 3, None), dict(arm='R', cell=C, situation_id='r1', plan_sha256='r'*64, seed=1,
                                         verification_id='v_r1', max_member_dist_t0_m=10.)]
    pot = cv.potential_inputs(contexts, mined, examples, range_m=200., cell=C)
    c = pot['chosen']
    assert c['screened_moments'] == 4                     # m1, m2, m3 (no candidates), m5; m4 refused
    assert c['refused_or_unscreened'] == 1 and c['screened_with_lead'] == 1
    assert c['robust_moments'] == 1                       # only m1 passes on EVERY verified seed (m5 lacks seed 3)
    assert [e['reachable'] for e in c['b_examples']] == [True, True, True, False, False]
    assert [e['robust'] for e in c['b_examples']] == [True, True, True, False, False]
    assert c['r_screened_moments'] == 1 and c['r_examples'] == 1
    assert len({e['id'] for e in c['b_examples']}) == 5


# ------------------------------------------------------------------ seeds / salts

def test_seed_ranges_and_salts_never_forbidden():
    for s in (962000, 962999, 964000, 964199, 977500, 989600, 989699):
        assert cv.seed_forbidden(s)
    for s in (961999, 963000, 964200, 978000, 989599, 989700):
        assert not cv.seed_forbidden(s)
    assert cv.check_world(WORLD) == 'map_scenarios.guard'
    assert cv.check_world(cv.make_world(989007, 3200, 0.6)) == 'map_scenarios.guard'   # PLANGEN2 extra TRAIN
    for bad in (dict(WORLD, seed=989600), dict(WORLD, seed=964100), dict(WORLD, seed=986000),
                dict(WORLD, seed=989150), dict(WORLD, repeat=3),
                dict(WORLD, map='merge'), dict(WORLD, cell_key='D2400_p40'), dict(WORLD, demand=2700)):
        with pytest.raises(ValueError):
            cv.check_world(bad)
    for seed in range(cv.TRAIN_RANGE[0], cv.TRAIN_RANGE[1]+1):
        for arm in ('B', 'R'):
            assert not cv.seed_forbidden(cv.noise_seed(dict(WORLD, seed=seed), arm))
    ctx = ctx_for()
    plans = cv.plans(ctx)['groups'][:2]
    vid, salts = cv.check_salts('c4v_B_D2800_p40_s989200', 450, plans, 4)
    assert sorted(salts) == [2, 3, 4, 5, 6] and salts[2] == vid
    assert all(not cv.seed_forbidden(cv.generator_seed(s)) for s in salts.values())
    assert cv.generator_seed(vid) == int(digest(dict(replication=vid))[:15], 16)


def test_cli_init_and_status(tmp_path, capsys):
    cv.main(['init', '--root', str(tmp_path), '--cap', '40000'])
    cv.main(['status', '--root', str(tmp_path)])
    out = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert len(out[0]['registration_sha256']) == 64
    assert out[1]['cap_sim_s'] == 40000 and out[1]['spent_sim_s'] == 0 and out[1]['remaining_sim_s'] == 40000


# ------------------------------------------------------------------ N1 geometry, freeze rules, frozen moves

class FakeDesc:
    kind, name = 'lane_drop', 'wz3to2'
    bottleneck = dict(drop_s_m=75.44, drop_x_m=2696.0, closing_lane=2, adjacent_lane=1, through_lanes=[0, 1])


def test_descriptor_geometry_matches_n3_frame(monkeypatch):
    from tools.coopmwm import map_descriptor as md
    g = cv.DescriptorGeometry(FakeDesc())
    rows = [row('a1', 1500, 0), row('a2', 1560, 0), row('b1', 1530, 1), row('b2', 1700, 1), row('c1', 1600, 2)]
    assert cv.n3_select(rows, 470, geom=g) == cv.n3_select(rows, 470, geom=GEOM)
    assert g.locate(dict(s_m=80.0, lane=0))[0] == 'past_drop'
    desc = md.load(SPEC)                                  # the registered N1 wz3to2 descriptor (not bound)
    lane, s = desc.lane_and_s('up_0', 2196.0)             # SUMO closing lane 500 m before the drop
    assert cv.DescriptorGeometry(desc).locate(dict(s_m=s, lane=lane)) == ('up', 0, pytest.approx(2196.0))
    monkeypatch.delenv('COOPMWM_MAP_SPEC', raising=False)
    assert isinstance(cv.geometry(), cv.Wz3to2GeometryStub)
    with pytest.raises(RuntimeError):
        cv.production_env(WORLD)                          # COOPMWM_MAP_SPEC / COOPMWM_G5_RUNTIME required


PG = dict(release_x=2696.0, amax=1.0, v_w=22.0, hold_s=120, dmax=1.0)
LB = dict(target_lane=2, giveup_x=2396.0, release_x=2696.0, hold_s=300)


def src_row(arm, sid, cell, demand, pen, t0, family, variant, members, params, rep=1, r_kind=None, as_str=False,
            role=None, dur=10):
    mv = dict(map='wz3to2', variant=variant, family=family,
              members=[dict(id=m, lane=i, params=params, **({'role': role} if role else {}))
                       for i, m in enumerate(members)])
    r = dict(schema='coopmwm-mined-move-v1', arm=arm, situation_id=sid, seed=rep, t0=t0, family=family,
             variant=variant, members=members, plan_sha256=f'{sid}_{variant}_{rep}', verification_id=f'nm2v_{sid}',
             world=dict(map='wz3to2', cell=cell, demand=demand, penetration=pen, seed=989200, repeat=0, split='TRAIN'),
             move=json.dumps(mv) if as_str else mv, duration_s=dur, release_el={m: dur for m in members},
             profile=[[0.0]*dur for _ in members], policy_profile=[[0.0]*dur for _ in members],
             perturbation_profile=[[0.0]*dur for _ in members])
    if r_kind:
        r['r_kind'] = r_kind
    return r


def write_sources(d):
    d.mkdir()
    sid = 'nm2_D2800_p40_s989200_t470'
    b1 = [src_row('B', sid, 'D2800_p40', 2800, 0.4, 470, 'closing_pair', 'pair_g', ['a1', 'b1'], PG, k)
          for k in (1, 2, 3)]
    b1.append(src_row('B', 'nm2_D2800_p20_s989200_t450', 'D2800_p20', 2800, 0.2, 450, 'abreast_wall', 'wall_h',
                      ['a1', 'b1', 'c1'], dict(PG, v_w=15.0, hold_s=90, dmax=1.5)))
    b1 += [src_row('B', sid, 'D2800_p40', 2800, 0.4, 470, 'abreast_wall', 'wall_g', ['a1', 'b1', 'c1'], PG, k)
           for k in (1, 2)]
    b2 = [src_row('B', 'ck_a', 'D2800_p40', 2800, 0.4, 800, 'lane_balance', 'B_lane_balance', ['a1', 'b1', 'c1'],
                  LB, as_str=True, role='lane_to'),
          src_row('B', 'ck_b', 'D2800_p40', 2800, 0.4, 800, 'discharge', 'B_discharge', ['a1', 'b1'],
                  dict(tau=0.6, hold_s=180), as_str=True)]
    r1 = [src_row('R', 'pg2_r', 'D2800_p40', 2800, 0.4, 470, 'R_random', 'R0', ['b1', 'c1'], dict(PG, hold_s=90),
                  r_kind='hold'),
          src_row('R', 'pg2_r', 'D2800_p40', 2800, 0.4, 800, 'R_random', 'R4', ['a1', 'b1', 'c1'], LB,
                  r_kind='lane_balance', role='lane_to')]
    out = []
    for name, rows in (('PLANGEN_MOVES_B.jsonl', b1), ('COOPKINDS_MOVES_B.jsonl', b2),
                       ('PLANGEN2_MOVES_R.jsonl', r1), ('COOPKINDS_MOVES_R.jsonl', [])):
        (d/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        out.append(d/name)
    return out


def test_freeze_main_rules(tmp_path):
    from tools.coopmwm import map_descriptor as md
    cv.init(tmp_path)
    paths = write_sources(tmp_path/'src')
    ml = cv.freeze(tmp_path, paths, descriptor=md.load(SPEC))
    # option (a): B wall/pair moves only; every R row listed as zero-guidance; run moves first, then exclusions
    assert [(m['variant'], m['status']) for m in ml['moves']] == [
        ('pair_g', 'pending'), ('wall_g', 'pending'), ('wall_h', 'excluded_cell'),
        ('B_lane_balance', cv.EXCLUDED_F3), ('B_discharge', cv.EXCLUDED_F3), ('R0', cv.EXCLUDED_R_A),
        ('R4', cv.EXCLUDED_R_A)]
    assert ml['run_list'] == [m['move_id'] for m in ml['moves'][:2]] and ml['fairaudit']['cut'] is False
    assert [m['index'] for m in ml['moves']] == list(range(7)) and ml['moves'][5]['move_id'].startswith('R005_')
    assert ml['option_a']['fact'] == cv.OPTION_A_FACT and ml['moves'][5]['reason'] == cv.OPTION_A_FACT
    assert ml['r_procedure_fact'] == cv.R_PROCEDURE_FACT and ml['fairaudit']['rules'] == list(cv.FAIRAUDIT_RULES)
    assert ml['moves'][0]['seeds'] == [1, 2, 3] and ml['moves'][0]['n_rows'] == 3 and ml['moves'][1]['seeds'] == [1, 2]
    assert ml['moves'][0]['world'] == WORLD and ml['main_rules'] == list(cv.MAIN_RULES) and ml['n_executable'] == 2
    assert ml['moves'][0]['lane_drop_macro']['family'] == 'closing_pair'
    assert ml['moves'][1]['lane_drop_macro']['lanes'] == [0, 1, 2]
    assert [s['name'] for s in ml['sources']] == [p.name for p in paths]
    assert all(len(s['sha256']) == 64 for s in ml['sources'])
    fa = ml['fairaudit']
    assert fa['collections']['worlds'] == 1 and fa['budget_B'] == 400000-1200 and fa['budget_R'] == 0
    with pytest.raises(ValueError):
        cv.freeze(tmp_path, paths, descriptor=md.load(SPEC))   # frozen once
    other = tmp_path/'other'
    cv.init(other)
    cv.charge(other, 'x', 1)
    with pytest.raises(ValueError):
        cv.freeze(other, paths, descriptor=md.load(SPEC))      # only before any physics
    with pytest.raises(ValueError):
        cv.run_context(tmp_path, tmp_path/'none', runner_factory=factory({}))   # frozen: only moves run
    small = tmp_path/'small'                              # cap too small: BOTH arms cut (here to zero)
    cv.init(small, cap=12000)
    ml = cv.freeze(small, paths, descriptor=md.load(SPEC))
    assert ml['run_list'] == [] and ml['fairaudit']['cut'] is True
    cut = [m for m in ml['moves'] if m['status'] == cv.EXCLUDED_F4]
    assert sorted(m['variant'] for m in cut) == ['pair_g', 'wall_g']
    with pytest.raises(ValueError):
        cv.run_move(small, cut[0]['move_id'], runner_factory=factory({}))


def test_fair_cut_and_interleave():
    c = cv.fair_cut([10]*4, [10]*4, 1000)
    assert (c['k_B'], c['k_R'], c['cut']) == (4, 4, False)
    c = cv.fair_cut([10]*4, [100]*4, 400)                 # R limits f -> B is cut by the same fraction
    assert (c['fraction'], c['k_B'], c['k_R'], c['cut']) == (0.5, 2, 2, True)
    c = cv.fair_cut([10]*10, [10]*5, 105)                 # budgets 70 / 35
    assert (c['fraction'], c['k_B'], c['k_R'], c['budget_B'], c['budget_R']) == (0.7, 7, 3, 70, 35)
    assert cv.interleave([1, 2, 3], ['a']) == [1, 'a', 2, 3] and cv.interleave([], ['a', 'b']) == ['a', 'b']


def snap_capture(controller, prefixes, job, t, sample, generator):
    return dict(episode=dict(id=job['id'], world=job['world'], arm=job['arm'], update=0, role='train',
                             training_seed=20261101),
                policy=dict(state_dict_sha256='p'), sumo=dict(runtime='rt', checkpoint='ck', warm_history=True,
                                                             accel_bounds=[-3., 2.6], noise_seed=job['noise_seed']),
                t0=t, end_s=1200, prefixes={str(x): fp(x, cum0(x)) for x in range(300, t+1)}, samples={},
                rng=dict(generator_state_sha256='g'), committed_plans=[])


def test_frozen_moves_collect_run_extract(tmp_path, monkeypatch):
    from tools.coopmwm import lane_drop as ld, map_descriptor as md, moves as mvs
    desc = md.load(SPEC)
    monkeypatch.setattr(cv, '_descriptor', lambda: desc)
    cv.init(tmp_path)
    ml = cv.freeze(tmp_path, write_sources(tmp_path/'src'))
    v = {m['variant']: m['move_id'] for m in ml['moves']}
    pg, wg = v['pair_g'], v['wall_g']
    r = cv.run_move(tmp_path, pg, runner_factory=factory({}), workers=1)
    assert r['status'] == 'refused' and not (tmp_path/'move_status').exists()    # not collected: not started
    for bad in (v['wall_h'], v['B_lane_balance'], v['B_discharge'], v['R0'], v['R4'], 'unknown'):
        with pytest.raises(ValueError):
            cv.run_move(tmp_path, bad, runner_factory=factory({}))
    with pytest.raises(SystemExit):
        cv.main(['status', '--root', str(tmp_path), '--arm', 'B', '--done'])      # B run list not finished
    reqs = {arm: cv.move_requests(tmp_path, arm) for arm in ('B', 'R')}
    assert reqs['R'] == []                                # option (a): R moves are not run
    assert [(j['ident'], [(q['t0'], q['members']) for q in j['requests']]) for j in reqs['B']] == [
        ('c4v_B_D2800_p40_s989200', [(470, ['a1', 'b1', 'c1'])])]
    for arm, jobs in reqs.items():
        for j in jobs:
            cv.collect(tmp_path, j['world'], arm, {'w': 1}, geom=GEOM, rt=FakeRT(tmp_path), windows=(),
                       requests=j['requests'], frame_times=[600], capture_fn=snap_capture)
    obs = cv._jsonl(tmp_path/'OBS_TABLES.jsonl')
    assert sorted((o['arm'], o['kind'], o['time_s']) for o in obs) == [
        ('B', 'move', 469), ('B', 'move', 470), ('B', 'negative', 599), ('B', 'negative', 600)]
    for o in obs:
        assert o['rows'] and all('driver_type' not in row for row in o['rows']) and 'driver_type' in o['dropped_keys']
        assert any(row['is_cav'] is False for row in o['rows'])          # detected HDV rows kept
        assert o['lane_drop_features'] == 'not_available' and o['sample_context']['members'] == ['a1']
    assert [o['move_ids'] for o in obs if o['arm'] == 'B' and o['kind'] == 'move'][0] == [pg, wg]
    ctx = json.loads((cv.move_ctx_dir(tmp_path, ml['moves'][0])/'CONTEXT.json').read_text())
    assert ctx['kind'] == 'requested' and ctx['move_ids'] == [pg, wg]
    gains = {('c4v_pair_g_a1_b1', k): 0.6 for k in (1, 2, 3)}
    assert cv.next_move(tmp_path)[1]['move_id'] == pg
    out = cv.run_move(tmp_path, pg, runner_factory=factory(gains), workers=1)
    assert out['status'] == 'confirmed_admitted' and out['origin'] == cv.ORIGIN_N2
    assert cv.move_status(tmp_path, ml['moves'][0])['status'] == 'confirmed_admitted'
    with pytest.raises(ValueError, match='already started'):
        cv.run_move(tmp_path, pg, runner_factory=factory(gains), workers=1)
    assert all(r.get('arm') in ('B', 'R') for r in cv.ledger_rows(tmp_path))
    last = cv.run_next(tmp_path, runner_factory=factory(gains), workers=1)
    assert last['move_id'] == wg and last['status'] == 'screen_rejected'
    assert cv.run_next(tmp_path, runner_factory=factory(gains))['status'] == 'no_moves_left'
    assert cv.arm_progress(tmp_path, 'R')['done'] and cv.arm_progress(tmp_path, 'B')['done']
    cv.main(['status', '--root', str(tmp_path), '--arm', 'B', '--done'])          # no SystemExit: B finished
    assert cv.arm_spent(tmp_path, 'B') <= cv.arm_budget(tmp_path, 'B') and cv.arm_spent(tmp_path, 'R') == 0
    st = {m['move_id']: (cv.move_status(tmp_path, m) or {}).get('status') for m in ml['moves']}
    assert all(st[v[k]] is None for k in ('wall_h', 'B_lane_balance', 'B_discharge', 'R0', 'R4'))
    assert len(cv._jsonl(tmp_path/'results'/'moves.jsonl')) == 2
    verified = [r['plan'] for r in cv._jsonl(tmp_path/'results'/'plans.jsonl') if not r.get('skipped')]
    assert len(verified) == 2
    for plan in verified:                                 # every verified plan is N2-exact (zero legacy projection)
        assert isinstance(mvs.from_plan(plan), ld.LaneDropMove) and plan['origin'] == cv.ORIGIN_N2
    ctxs = cv.contexts(tmp_path, 'B')                     # metadata only: winner AND loser contexts
    assert [c['move_id'] for c in ctxs] == [pg, wg] and all(c['seed'] == 989200 for c in ctxs)
    by_plan = {r['move_id']: r['plan'] for r in cv._jsonl(tmp_path/'results'/'plans.jsonl') if r.get('move_id')}
    for c in ctxs:
        assert Path(c['snapshot_path']).exists() and json.loads(Path(c['receipt_path']).read_text())['episode']
        assert c['group_plan'] == by_plan[c['move_id']] and c['group_plan']['lane_drop_macro']
    cv.main(['contexts', '--root', str(tmp_path), '--arm', 'B', '--out', str(tmp_path/'ctx.json')])
    assert [c['move_id'] for c in json.loads((tmp_path/'ctx.json').read_text())] == [pg, wg]
    ex = cv.extract(tmp_path, cell='D2800_p40')
    assert ex['mined_B'] == 3 and ex['mined_R'] == 0 and ex['frames_missing'] == 0
    for m in (json.loads(x) for x in (tmp_path/'CLEAN4_MOVES_B.jsonl').read_text().splitlines()):
        assert m['move_id'] == pg and m['range_m'] == 100.0 and m['origin'] == cv.ORIGIN_N2 and m['situation_id'].endswith('_r470')
        assert m['lane_drop_macro']['family'] == 'closing_pair'
    neg = [json.loads(x) for x in (tmp_path/'CLEAN4_NEGATIVE_FRAMES.jsonl').read_text().splitlines()]
    assert sorted((n['kind'], n['arm'], str(n.get('move_id'))) for n in neg) == [
        ('lost', 'B', wg), ('requested', 'B', 'None')]
    assert all(set(n['frames']) == {str(n['t0']-1), str(n['t0'])} for n in neg)
    assert ex['moves']['by_status'] == {'confirmed_admitted': 1, 'screen_rejected': 1, 'excluded_cell': 1,
                                        cv.EXCLUDED_F3: 2, cv.EXCLUDED_R_A: 2}


def test_n2_builder_plans_bind_to_lane_drop_moves(tmp_path):
    """Every N2-built plan binds exactly (moves.from_plan -> LaneDropMove); the provisional plan does not."""
    from tools.coopmwm import lane_drop as ld, map_descriptor as md, moves as mvs
    desc = md.load(SPEC)
    rows = write_sources(tmp_path/'src')
    src = [json.loads(x) for x in rows[0].read_text().splitlines()]
    zone = [dict(id='a1', lane=0, x=2600.0, v=30.0), dict(id='b1', lane=1, x=2590.0, v=30.0),
            dict(id='c1', lane=2, x=2580.0, v=30.0)]
    ctx = ctx_for(zone=zone)
    for row in (src[0], src[4]):                          # pair_g (closing_pair), wall_g (abreast_wall)
        macro = ld.move_from_mined_row(row, desc).to_dict()
        var = dict(name=row['variant'], family=row['family'], members=macro['members'], params=PG,
                   lane_drop_macro=macro, descriptor=desc, move_id='m')
        plan = cv.n2_builder(ctx, macro['members'], var)
        move = mvs.from_plan(plan)
        assert isinstance(move, ld.LaneDropMove) and plan['lane_drop_macro'] == macro
        rel = dict(zip(plan['members'], plan['member_release_el']))
        assert plan['duration_s'] == max(rel.values()) and all(0 < el < 10 for el in rel.values())
        for i, v in enumerate(plan['members']):           # the N2 law while held: clip((22-v)/1, -1, 1) = -1 here
            assert all(plan['profile'][i][k] == -1.0 for k in range(rel[v]))
        assert plan['origin'] == cv.ORIGIN_N2 and plan['source']['origin'] == cv.ORIGIN_N2
    legacy = cv.provisional_builder(ctx, ['a1', 'b1'], dict(name='pair_g', family='closing_pair', members=['a1', 'b1'],
                                                            params=dict(PG)))
    with pytest.raises(ValueError, match="without lane_drop_macro"):   # nm-train 684fe30c fail-closed
        mvs.from_plan(legacy)                             # why frozen moves never use the provisional builder
    macro = ld.move_from_mined_row(src[0], desc).to_dict()
    var = dict(name='pair_g', family='closing_pair', members=macro['members'], params=PG, lane_drop_macro=macro,
               descriptor=desc)
    assert cv.n2_builder(ctx_for(zone=zone[:1]), macro['members'], var) is None      # member not upstream


def _law_tape(plan, t0, start, view, *, perturb=0.0, extra_s=5):
    """Fake group-branch tape: members follow the plan profile exactly (perturb adds a speed offset)."""
    s = {v: start[v][0] for v in plan['members']}
    v_ = {v: start[v][1]+perturb for v in plan['members']}
    steps = {}
    for k in range(plan['duration_s']+extra_s):
        steps[t0+k] = dict(time_s=t0+k, rows=[dict(vid=v, s_m=s[v], v_mps=v_[v], lane=1, is_cav=True, length_m=5.)
                                              for v in plan['members']],
                           requests={v: plan['profile'][i][k] for i, v in enumerate(plan['members'])
                                     if k < plan['duration_s']})
        for i, v in enumerate(plan['members']):
            a = plan['profile'][i][k] if k < plan['duration_s'] else 0.
            nv = max(0., v_[v]+a)
            s[v] += 0.5*(v_[v]+nv)
            v_[v] = nv
    return steps


def test_law_fidelity_pure():
    from tools.coopmwm import lane_drop as ld, map_descriptor as md
    desc = md.load(SPEC)
    view = ld.MapView.from_descriptor(desc)
    rows = write_sources_rows()
    macro = ld.move_from_mined_row(rows[0], desc).to_dict()
    zone = [dict(id='a1', lane=0, x=2600.0, v=30.0), dict(id='b1', lane=1, x=2590.0, v=30.0)]
    ctx = ctx_for(zone=zone)
    var = dict(name='pair_g', family='closing_pair', members=macro['members'], params=PG, lane_drop_macro=macro,
               descriptor=desc)
    plan = cv.n2_builder(ctx, macro['members'], var)
    start = {c['id']: (c['x']-OFFSET, c['v']) for c in zone}
    exact = cv.law_fidelity(plan, _law_tape(plan, 470, start, view), 470, view)
    assert exact['max_abs_delta_a'] == 0.0 and exact['n_member_seconds'] > 0
    assert all(p['release_diff_s'] == 0 for p in exact['per_member'].values())
    assert all(p['max_abs_request_minus_profile'] == 0.0 for p in exact['per_member'].values())
    off = cv.law_fidelity(plan, _law_tape(plan, 470, start, view, perturb=-5.0), 470, view)
    assert off['max_abs_delta_a'] > 0.5                   # slower actual speeds: the closed-loop law differs
    assert any(p['release_diff_s'] != 0 for p in off['per_member'].values())


def write_sources_rows():
    sid = 'nm2_D2800_p40_s989200_t470'
    return [src_row('B', sid, 'D2800_p40', 2800, 0.4, 470, 'closing_pair', 'pair_g', ['a1', 'b1'], PG, 1)]


def test_fidelity_report_reads_only_and_writes_outside(tmp_path, monkeypatch):
    from tools.coopmwm import lane_drop as ld, map_descriptor as md
    desc = md.load(SPEC)
    monkeypatch.setattr(cv, '_descriptor', lambda: desc)
    root = tmp_path/'run'
    cv.init(root)
    ml = cv.freeze(root, write_sources(tmp_path/'src'))
    pg = next(m['move_id'] for m in ml['moves'] if m['variant'] == 'pair_g')
    for j in cv.move_requests(root, 'B'):
        cv.collect(root, j['world'], 'B', {'w': 1}, geom=GEOM, rt=FakeRT(root), windows=(), requests=j['requests'],
                   capture_fn=snap_capture)
    ctx = json.loads((cv.move_ctx_dir(root, ml['moves'][0])/'CONTEXT.json').read_text())
    start = {c['id']: (c['s_m'], c['v']) for c in ctx['geometry']['zone']}
    view = ld.MapView.from_descriptor(desc)

    class TapePlant(Plant):
        def run(self, spec, snap):
            r = super().run(spec, snap)
            p = spec['plan']
            if p is not None and '__solo_' not in p['id']:
                from tools.coopmwm.accel_storage import read, write as awrite
                d = self.work/'branches'/spec['verification_id']/spec['branch']
                tape = read(d/'RESULT.json')
                tape['steps'] = list(_law_tape(p, spec['window']['start_s'], start, view).values())
                (d/'RESULT.json').unlink()
                (d/'RESULT.json.gz').unlink(missing_ok=True)
                awrite(d/'RESULT.json', tape)
                r['result_sha256'] = digest(tape)
            return r
    cv.run_move(root, pg, runner_factory=lambda vroot, w: TapePlant(vroot, {}), workers=1)
    before = sorted(str(x) for x in root.rglob('*'))
    with pytest.raises(ValueError):
        cv.fidelity(root, root/'FID.json')                # never writes under the run root
    rep = cv.fidelity(root, tmp_path/'FID.json')
    assert sorted(str(x) for x in root.rglob('*')) == before
    by = {r['move_id']: r for r in rep['moves']}
    assert by[pg]['fidelity'] == 'compared' and by[pg]['max_abs_delta_a'] == 0.0
    assert [r['fidelity'] for r in rep['moves'] if r['move_id'] != pg] == ['not_verified']
    assert json.loads((tmp_path/'FID.json').read_text())['summary']['compared'] == 1
