"""HARDEN-3 end-to-end CPU rehearsal of the B/R resume (u4,u5). NO SUMO, NO GPU, NO remote.

Runs the REAL orchestration in a temporary root laid out like the live root:
warm_runner.run_arm (B and R concurrently, R pipelined on B's random quota),
accel_runner.wave, accel_plant.episode, acceleration.Controller, plan_client.guide,
parallel_groups (B), quota_admission, capacity_broker.run (+ warm_broker.Remote.call
/warm_transport.retry), sumo_verify.install/verified_guide/verify (k=8 + policy, beta=1),
LocalBranchRunner + VerifyController replay/prefix/CRN checks, verification_grades reuse,
exact_runtime.enqueue/evidence, exact_grade.exact_update (PPO/AWR), search_learning,
checkpoint writing, append-only accounting and warm_health guards. On the GPU side
the REAL mppi_runtime.plan / mppi_search.search / overlap.recheck / centre_service.result /
query_errors.execute run with a fake world-model evaluator.

Fake leaves only:
- SUMO: ``FakeBackend`` + ``fake_tick`` (deterministic toy kinematics, schema-valid rows,
  totals, fingerprints, trace lines). Branches use the same fake plant via the real
  LocalBranchRunner (thread-local Controller injection instead of one process per branch).
- GPU worker spool: ``FakeGPU`` (latency, death, malformed result) behind the real RPC ABI
  (status/submit/forecast) through ``FakeRemote.raw`` (transient SSH errors injectable).
- WM evaluator: ``fake_evaluate`` (deterministic team-delay surrogate + acceleration trace).
- Learned proposer: REAL plan_proposer.ProposerNet tiny CPU checkpoint (save/load
  round trip) feeding ``proposal_seeds`` into the real search as ``extra_seeds`` (B only).

Platform shims (Windows CPU only; no-ops on Linux): fcntl.flock via msvcrt, a fake /proc
for owner identity receipts, os.nice/getpriority, SIGHUP/non-main-thread signal calls.
Process isolation of episodes/branches/grader is NOT rehearsed (threads are used).
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import types

import numpy as np
import torch

from .fcntl_compat import install as _install_fcntl

_install_fcntl()

BOUNDS = [-3., 1.5]
DEADLINE = 1793894340.0          # RESOURCE_EXTENSION_1651 (Nov 5 23:59 +08)
AMENDMENT = 'h3' + '0'*62
SLOTS = ('gpuA_primary', 'gpuA_second', 'gpuR_primary', 'gpuR_second')
TL = threading.local()
TICKS = [0]
EVENTS = []
_EV_LOCK = threading.Lock()


def event(kind, **kw):
    with _EV_LOCK:
        EVENTS.append(dict(kind=kind, unix=time.time(), **kw))


# ============================================================== fake SUMO plant

class FakeTraci:
    class vehicle:
        @staticmethod
        def getLength(vid):
            return 5.


class FakeBackend:
    """Deterministic toy road; state depends only on world seed and applied accelerations."""

    def __init__(self, path, world, reservation, checkpoint):
        self.path, self.world, self.allowed = Path(path), world, reservation
        rng = np.random.default_rng(int(world['seed']))
        cars = []
        s = -150.
        sizes = [2, 3, 4, 5, 6][int(world['seed']) % 5:] + [2, 3, 4, 5, 6][:int(world['seed']) % 5]
        for c, size in enumerate(sizes[:3]):
            for i in range(size):
                cars.append(dict(vid=f'cav{c}_{i}', s_m=s, v_mps=float(18+4*rng.random()), lane=int(i % 2), is_cav=True))
                s -= 35.
            cars.append(dict(vid=f'hdv{c}', s_m=s, v_mps=float(20+3*rng.random()), lane=int(c % 2), is_cav=False))
            s -= 190.
        cars.append(dict(vid='hdv_ramp', s_m=-60., v_mps=14., lane=2, is_cav=False))
        self.cars = cars
        self.target = {c['vid']: float(22+6*rng.random()) for c in cars}
        self.now, self.actual = 300., 300.
        self.totals = dict.fromkeys(('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'collisions',
                                     'teleports', 'ttc_below_3_vs', 'hard_brake'), 0.)
        self.traci = FakeTraci
        self.previous = None

    def rows(self):
        return [dict(vid=c['vid'], s_m=round(c['s_m'], 6), v_mps=round(c['v_mps'], 6), lane=c['lane'],
                     is_cav=c['is_cav'], length_m=5.) for c in sorted(self.cars, key=lambda c: c['vid'])]

    def fingerprint(self):
        state = json.dumps([[c['vid'], round(c['s_m'], 9), round(c['v_mps'], 9)] for c in self.cars])
        return dict(t_s=float(self.now), state_sha256=hashlib.sha256(state.encode()).hexdigest(),
                    metrics={k: float(v).hex() for k, v in self.totals.items()})

    def close(self):
        pass


def fake_tick(b, controller=None, trace=None):
    now, rows = float(b.now), b.rows()
    commands = {r['vid']: float(np.clip(.3*(b.target[r['vid']]-r['v_mps']), *BOUNDS)) for r in rows if r['is_cav']}
    desired = controller.apply(rows, commands, now)
    audit = []
    for c in b.cars:
        if c['is_cav']:
            raw = desired[c['vid']]['requested_raw'] if c['vid'] in desired else commands[c['vid']]
            a = float(np.clip(raw, *BOUNDS))
            if c['vid'] in desired:
                audit.append(dict(vid=c['vid'], clamped=bool(a != raw), mappo_safe=commands[c['vid']]))
        else:
            a = float(np.clip(.5*(b.target[c['vid']]-c['v_mps']), *BOUNDS))
        c['v_mps'] = float(max(0., c['v_mps']+a))
        c['s_m'] += c['v_mps']
        if c['s_m'] > 500.:
            c['s_m'] -= 1400.
    deficit = sum(max(0., 30.-c['v_mps'])/30. for c in b.cars)
    b.totals['road_delay_vs'] += deficit
    b.totals['team_delay_vs'] += deficit
    b.totals['hard_brake'] += sum(1. for x in audit if x['clamped'])
    if trace is not None:
        trace.write(json.dumps(dict(time_s=now, n=len(rows), actions=len(audit)))+'\n')
    b.now += 1.
    b.actual += 1.
    TICKS[0] += 1
    return audit


def _fake_leverprobe():
    """tools.leverprobe.backend is always the fake plant; selection is real when importable."""
    try:
        import tools.leverprobe.selection  # noqa: F401
        pkg = sys.modules['tools.leverprobe']
    except Exception:
        pkg = types.ModuleType('tools.leverprobe');pkg.__path__ = []
        sel = types.ModuleType('tools.leverprobe.selection')
        exec(_SELECTION_FEATURES, sel.__dict__)
        sys.modules['tools.leverprobe'] = pkg;sys.modules['tools.leverprobe.selection'] = sel
        pkg.selection = sel
    fake = types.ModuleType('tools.leverprobe.backend')
    fake.setup = lambda runtime, checkpoint: None
    fake.Backend = FakeBackend
    fake.guard = None
    sys.modules['tools.leverprobe.backend'] = fake;pkg.backend = fake


# Verbatim copy of c4/leverprobe tools/leverprobe/selection.py features() (observable only),
# used only when that package is absent from the local checkout.
_SELECTION_FEATURES = '''
import itertools
ZONE = (-100., 75.44)
def features(rows):
    zone = [r for r in rows if ZONE[0] <= r['s_m'] <= ZONE[1]
            and r['lane'] in (0, 1, 2)]
    cavs = [r for r in zone if r['is_cav']]
    upstream = [r for r in cavs if r['lane'] in (0, 1) and r['s_m'] < 0]
    pairs = [(a, b) if a['lane'] == 0 else (b, a)
             for a, b in itertools.combinations(upstream, 2) if a['lane'] != b['lane']]
    pairs.sort(key=lambda ab:(abs(ab[0]['s_m']-ab[1]['s_m']),
                               -min(r['s_m'] for r in ab), ab[0]['vid'], ab[1]['vid']))
    return dict(cav_count=len(cavs), density=len(zone)/(3*(ZONE[1]-ZONE[0])/1000),
                cav_share=len(cavs)/len(zone) if zone else 0.,
                ramp=any(r['lane'] == 2 for r in zone),
                queued=sum(r['v_mps'] < 5 for r in zone),
                mean_speed=sum(r['v_mps'] for r in zone)/len(zone) if zone else 0.,
                pair=[r['vid'] for r in pairs[0]] if pairs else [])
'''


# ============================================================== fake WM evaluator

def fake_evaluate(rt, record, actions, k, h, chunk_actions=None, trim=False, pad_actions=None, speed2=False,
                  memory=None, trace=None, transfers=False):
    """Deterministic surrogate team delay per action/future; never a SUMO or WM output."""
    seed = int(hashlib.sha256(json.dumps(record.get('state_id', ''), sort_keys=True).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    base = 1000.+100.*rng.random()
    common = rng.normal(0., 5., int(k))
    target = rng.normal(0., .4)
    out = {}
    for a in actions:
        if 'profile' in a and a['profile'] is not None:
            prof = np.asarray(a['profile'], float)
            cost = base+30.*float(np.mean(np.square(prof-target)))-8.*min(1., float(a.get('duration_s', 0))/300.)
        else:
            cost = base+(0. if a['id'] != 'MAPPO' else 3.)
        own = np.random.default_rng(int(hashlib.sha256(str(a['id']).encode()).hexdigest()[:8], 16)).normal(0, 1., int(k))
        out[a['id']] = [dict(team_delay=float(cost+common[i]+own[i])) for i in range(int(k))]
    if trace is not None:
        n = len(record['rows']);width = max(2, int(pad_actions or 2))
        for t in range(int(h)):
            trace.append(torch.full((width, n), .01*math.sin(t/30.)))
    return out


class FakeRT:
    class mp:
        class policy:
            accel_min, accel_max = BOUNDS


# ============================================================== fake GPU spool + transport

class FakeGPU:
    """One GPU worker: durable claim per key, inference latency, injectable faults."""

    def __init__(self, slot, cfg, proposer=None):
        self.slot, self.cfg, self.proposer = slot, cfg, proposer
        self.inbox, self.forecasts, self.claimed = queue.Queue(), {}, set()
        self.alive, self.failure, self.searches = True, None, 0
        self.lock = threading.Lock()
        self.owner = dict(pid=10000+SLOTS.index(slot), start_ticks='1', boot_id='harness')
        threading.Thread(target=self.loop, daemon=True, name='gpu-'+slot).start()

    def status(self):
        from .warm_rpc import GATES
        receipts = {g: dict(pass_=True, max_cost_error=0.) for g in GATES}
        receipts['SERVICE_OWNER.json'] = dict(self.owner);receipts['SPOOL_OWNER.json'] = dict(self.owner)
        if self.failure:
            receipts['SPOOL_FAILED.json'] = dict(error=self.failure)
        return dict(receipts=receipts, state='S' if self.alive else 'exited', ready=True,
                    alive=self.alive, failed=self.failure is not None)

    def submit(self, record):
        s = self.status()
        if not s['alive'] or s['failed'] or not s['ready']:
            raise ValueError('unqualified or dead GPU spool')
        key = record['state_id']
        with self.lock:
            if key not in self.claimed:
                self.claimed.add(key);self.inbox.put(record)
        return dict(key=key, owner=self.owner)

    def loop(self):
        while True:
            record = self.inbox.get()
            if not self.alive:
                continue
            begin = time.monotonic()
            die = self.cfg.get('die_slot') == self.slot and self.searches == self.cfg.get('die_after', 1)
            with _EV_LOCK:
                if self.cfg.get('die_on') == record.get('mode') == 'centre' and not self.cfg.get('_died'):
                    die = self.cfg['_died'] = True
            time.sleep(self.cfg['latency_s']*(.5 if record.get('mode') == 'centre' else 1.))
            if die:
                self.alive, self.failure = False, "RuntimeError('CUDA error: an illegal memory access was encountered')"
                event('gpu_worker_died', slot=self.slot, key=record['state_id'])
                continue
            try:
                result = self.answer(record)
            except Exception as error:   # mirrors plan_service: an uncaught fault kills the spool
                self.alive, self.failure = False, repr(error)
                event('gpu_worker_exception', slot=self.slot, error=repr(error), tb=traceback.format_exc())
                continue
            self.searches += 1
            with _EV_LOCK:
                malformed = self.cfg.get('malformed', 0) > 0 and record.get('mode') != 'centre'
                if malformed:
                    self.cfg['malformed'] -= 1
            if malformed:
                result = dict(accepted=True, plan=dict(id='broken'), note='malformed rehearsal response')
                event('gpu_malformed_response', slot=self.slot, key=record['state_id'])
            self.forecasts[record['state_id']] = dict(key=record['state_id'], result=result,
                                                      wall_s=time.monotonic()-begin, unix=time.time())

    def answer(self, r):
        from tools.coopnative.policy import encode
        from .committed import validate
        from .protocol import digest
        centre = r.get('mode') == 'centre'
        if centre:
            r = {k: v for k, v in r.items() if k != 'mode'}
        allowed = {'state_id', 'world_id', 'time_s', 'rows', 'group', 'commands', 'policy_residuals', 'history'}
        allowed |= {k for k in ('policy_state', 'search_learning', 'committed_plans') if k in r}
        if set(r) != allowed:
            raise ValueError('observable plan request whitelist')
        validate(r.get('committed_plans', []), r['group'])
        encode(r['rows'])
        if centre:
            from .centre_service import result
            return result(FakeRT, r, None)
        from .mppi_runtime import plan
        from .query_errors import execute
        binding = dict(speed2=True, speed2_transfers=False, batch_width_by_K={'4': 34, '8': 10},
                       label_list8=self.cfg.get('label_list8', True))
        out = execute(r, lambda: plan(FakeRT, r, None, binding))
        assert out.get('record_sha256') == digest(r)
        return out


def _memoize_interp():
    """Harness speed only: cache the pure knot interpolation matrices (identical values)."""
    from . import mppi_search as ms
    if hasattr(ms._interp_matrix, 'harness_cache'):
        return
    original, cache, lock = ms._interp_matrix, {}, threading.Lock()

    def cached(src, dst):
        key = (np.asarray(src, float).tobytes(), np.asarray(dst, float).tobytes())
        with lock:
            hit = cache.get(key)
        if hit is None:
            hit = original(src, dst)
            with lock:
                cache[key] = hit
        return hit.copy()
    cached.harness_cache = cache
    ms._interp_matrix = cached


def _proposer_search(proposer):
    """Wrap the real search: learned proposer top-8 seeds enter as extra_seeds (B only)."""
    from . import mppi_search as ms
    from . import plan_proposer as pp
    original = ms.search

    def search(record, group, policy_traj, bounds, evaluate, regime, **kw):
        if proposer is not None and kw.get('extra_seeds') is None:
            kw['extra_seeds'] = pp.proposal_seeds(record, group, policy_traj, 8, proposer, bounds, regime)
        return original(record, group, policy_traj, bounds, evaluate, regime, **kw)
    search.harness_original = original
    return search


def make_remote_class(gpus, cfg):
    from .warm_broker import Remote
    from .warm_transport import SSHConnectionError

    class FakeRemote(Remote):
        def raw(self, value):
            with _EV_LOCK:
                inject = cfg.get('transport_errors', 0) > 0 and value.get('action') == 'forecast'
                if inject:
                    cfg['transport_errors'] -= 1
            if inject:
                event('transport_error_injected', slot=self.endpoint['slot'])
                raise SSHConnectionError('Connection closed by 1.15.42.207 port 22')
            gpu = gpus[self.endpoint['slot']]
            action = value.get('action')
            if action == 'status' and set(value) == {'action'}:
                return json.loads(json.dumps(gpu.status()))
            if action == 'submit' and set(value) == {'action', 'record'}:
                return gpu.submit(json.loads(json.dumps(value['record'])))
            if action == 'forecast' and set(value) == {'action', 'key'}:
                f = gpu.forecasts.get(value['key'])
                return json.loads(json.dumps(dict(forecast=f, status=gpu.status())))
            raise ValueError('restricted warm RPC action/fields')
    return FakeRemote


# ============================================================== platform shims

def _proc_path_factory(fake_proc):
    real = Path

    def ProcPath(*parts):
        if parts and str(parts[0]).startswith('/proc'):
            rest = str(real(*map(str, parts))).replace('\\', '/')[len('/proc'):].lstrip('/')
            return real(fake_proc)/rest
        return real(*parts)
    return ProcPath


def _platform_shims(root):
    if os.name == 'posix':
        return
    fake_proc = root.parent/'fake_proc'
    (fake_proc/'sys/kernel/random').mkdir(parents=True, exist_ok=True)
    (fake_proc/'sys/kernel/random/boot_id').write_text('harness\n')
    d = fake_proc/str(os.getpid());d.mkdir(exist_ok=True)
    (d/'stat').write_text(' '.join([str(os.getpid()), '(python)', 'S']+['0']*18+['1']+['0']*5)+'\n')
    for name, value in (('nice', lambda inc: 0), ('getpriority', lambda which, who: 0), ('PRIO_PROCESS', 0)):
        if not hasattr(os, name):
            setattr(os, name, value)
    if not hasattr(signal, 'SIGHUP'):
        signal.SIGHUP = signal.SIGTERM
    original = signal.signal

    def guarded(sig, handler):
        if threading.current_thread() is not threading.main_thread():
            return None
        return original(sig, handler)
    signal.signal = guarded
    from . import capacity_broker
    capacity_broker.Path = _proc_path_factory(fake_proc)


class _PlantModule(types.ModuleType):
    """accel_plant.Controller assignment is thread-local (LocalBranchRunner patches it
    module-wide; in production each branch is its own process)."""

    @property
    def Controller(self):
        return getattr(TL, 'factory', None) or self.__dict__['_harness_dispatch']

    @Controller.setter
    def Controller(self, value):
        TL.factory = None if value is self.__dict__['_harness_dispatch'] else value


def _thread_local_controller():
    from . import accel_plant
    if isinstance(accel_plant, _PlantModule):
        return
    original = accel_plant.__dict__['Controller']

    def dispatch(*a, **kw):
        factory = getattr(TL, 'factory', None)
        return (factory or original)(*a, **kw)
    accel_plant.__dict__['_harness_dispatch'] = dispatch
    accel_plant.__dict__['Controller'] = dispatch
    accel_plant.__class__ = _PlantModule


# ============================================================== live-like root

def _seal(value, key):
    from . import exact_grade as xg
    return xg._seal(value, key)


def build_root(root, cfg):
    from .accel_learner import save
    from .acceleration import Policy
    from .accel_protocol import worlds
    from .storage import write
    from . import search_learning
    from .warm_protocol import SEED
    root.mkdir(parents=True)
    for d in ('requests', 'responses', 'centre_requests', 'centre_responses', 'jobs'):
        (root/d).mkdir()
    (root/'AUTHORITY.md').write_text('HARDEN-3 CPU rehearsal authority (temporary root)\n'+AMENDMENT+'\n')
    train = worlds('train')
    n = cfg['episodes_per_update']
    plans = {u: [train[(u*7+i*5) % len(train)] for i in range(n)] for u in (4, 5, 6)}
    for u in plans:
        seen = set();out = []
        for w in plans[u]+train:
            if w['seed'] not in seen and len(out) < n:
                seen.add(w['seed']);out.append(w)
        plans[u] = out
    torch.manual_seed(SEED)
    for arm in ('B', 'R'):
        policy = Policy();opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        rng = torch.Generator().manual_seed(SEED)
        save(root/f'{arm}_checkpoint_003.pt', policy, opt, rng, 3, 0)
    budget = dict(total=11_000_000, phases=dict(superseded=811640, train=3_000_000, checks=500_000, dev=500_000,
                                                sumo_verification=1_000_000, sumo_verification_k8=1_500_000,
                                                exact_plan_grading=500_000, discovery=138_200))
    budget['allocated'] = sum(budget['phases'].values());budget['unallocated'] = budget['total']-budget['allocated']
    budget.update(updates=dict(A=48, B=24, R=24))
    write(root/'RESOURCE_EXTENSION_1651.json', dict(amendment_sha256=AMENDMENT, gpu_deadline=DEADLINE,
                                                    budget=budget, harness=True))
    (root/'DNS_GUIDED_RELEASE_1651.json').write_text(json.dumps(dict(harness=True)))
    write(root/'WORKERS.json', dict(workers=[dict(slot=s) for s in SLOTS]))
    write(root/'WARM_REMOTE_BINDING.json', dict(endpoints=[dict(slot=s) for s in SLOTS], deadline_unix=DEADLINE,
                                                search_sha256='harness'))
    write(root/'BROKER_CONFIG.json', dict(amendment_sha256=AMENDMENT, deadline_unix=DEADLINE,
                                          source_root=str(Path(__file__).resolve().parents[2]),
                                          receipt_tag='_H3', decline_withdrawn=cfg.get('decline_withdrawn', True)))
    # Real smoke receipts already passed live; reproduced in the same sealed formats.
    for d in (root/'exact_grades', root/'sumo_verification'):
        d.mkdir(parents=True, exist_ok=True)
    smoke = _seal(dict(pass_=True, all_five_branches_graded=True, harness_fixture=True), 'smoke_sha256')
    (root/'sumo_verification'/'SMOKE_PASSED.json').write_text(json.dumps(smoke))
    (root/'exact_grades'/'SMOKE_PASSED.json').write_text(json.dumps(dict(harness_fixture=True)))
    learned = search_learning.empty('B', SEED, BOUNDS)
    progress = {arm: dict(running=[], beta=1., learned=learned,
                          exact_rng_state=np.random.default_rng(SEED).bit_generator.state, exact_count=12)
                for arm in ('B', 'R')}
    return dict(plans=plans, budget=budget, progress=progress)


def _check_worlds():
    from .accel_protocol import worlds
    return worlds('check')[:3]


def child_install(parent=False):
    """Installed by the rehearsal sitecustomize in EVERY child (episodes, branch workers)."""
    _install_fcntl()
    _fake_leverprobe()
    from tools.coopfree import backend as cf
    from . import accel_plant, sumo_verify
    cf.tick = fake_tick
    accel_plant.install_seed_scope = lambda: None           # registered-world guards: TRAIN worlds only
    accel_plant.guard = lambda world, role: None
    if os.name != 'posix' and not hasattr(os, 'nice'):
        os.nice = lambda inc: 0
    if os.environ.get('H3_LEGACY_REF'):
        _install_legacy(os.environ['H3_LEGACY_REF'])
    crash = os.environ.get('H3_BRANCH_CRASH_DIR')
    if crash and not parent and not hasattr(accel_plant.episode, 'harness'):
        # Branch workers run ``-m tools.coopmwm.sumo_verify`` (a separate __main__ module),
        # so inject at the shared accel_plant.episode used by LocalBranchRunner.
        original = accel_plant.episode

        def episode(job):
            if job.get('branch') and not str(job['id']).endswith('_policy'):
                for i in range(int(os.environ.get('H3_BRANCH_CRASH_N', '1'))):
                    try:
                        os.close(os.open(os.path.join(crash, f'crash_{i}'), os.O_CREAT | os.O_EXCL))
                    except FileExistsError:
                        continue
                    raise RuntimeError('rehearsal: SUMO branch worker crashed (TraCI connection lost)')
            return original(job)
        episode.harness = True
        accel_plant.episode = episode


def _child_environment(base, cfg):
    shim = base/'sitecustomize_dir';shim.mkdir(parents=True, exist_ok=True)
    (shim/'sitecustomize.py').write_text(
        'import os\nif os.environ.get("H3_REHEARSAL"):\n'
        '    from tools.coopmwm.rehearsal_cpu import child_install\n    child_install()\n')
    top = Path(__file__).resolve().parents[2]
    os.environ.update(H3_REHEARSAL='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      PYTHONPATH=os.pathsep.join([str(shim), str(top)]))
    if cfg.get('legacy_wait_all'):
        os.environ['H3_LEGACY_REF'] = cfg.get('legacy_ref', 'aadee535')
    if cfg.get('branch_crash'):
        d = base/'crash_markers';d.mkdir(exist_ok=True)
        os.environ.update(H3_BRANCH_CRASH_DIR=str(d), H3_BRANCH_CRASH_N=str(cfg['branch_crash']))


def install_runtime(root, cfg, layout):
    """Patch ONLY leaves/platform + an in-memory binding, like incident_recovery.install."""
    from . import warm_runner as wr, runner, exact_runtime
    from .warm_protocol import install_accounting
    from . import mppi_runtime, centre_service  # noqa: F401
    child_install(parent=True)
    mppi_runtime.evaluate = fake_evaluate
    _platform_shims(root)
    _memoize_interp()
    runner.admission = lambda b: None                       # Linux load/disk/meminfo guard
    exact_runtime.assert_live = lambda r: None              # /proc grader owner identity
    wr.budget = lambda: copy.deepcopy(layout['budget'])
    wr.worlds = lambda role: _check_worlds() if role == 'check' else __import__(
        'tools.coopmwm.accel_protocol', fromlist=['worlds']).worlds(role)
    wr.batch_worlds = lambda u: copy.deepcopy(layout['plans'][u])

    def verify(r, a):
        install_accounting(r, layout['budget'])
        k = cfg['k']
        b = dict(runtime='fake-runtime', checkpoint='fake-checkpoint', accel_bounds=BOUNDS,
                 gpu_deadline=DEADLINE, N_exec=cfg['N_exec'], workers=cfg['workers'], warm_history=False,
                 health_guards=True, safety_reference=dict(by_world={}, unmatched_zero_event_benchmark=True,
                                                           source='harness'),
                 budget=layout['budget'], awr=dict(coefficient=1.), resume_update=3,
                 resume_progress=copy.deepcopy(layout['progress'][a]), runtime_receipt_suffix='_H3',
                 quota_aware_admission=True, parallel_group_queries=a == 'B' and cfg.get('parallel_groups', True),
                 precharged_unattempted=[], stop_after_update=cfg['last_update'],
                 sumo_verify=dict(root=str(r/'sumo_verification'), run_root=str(r),
                                  phase='sumo_verification_k8' if k == 8 else 'sumo_verification',
                                  k=k, workers=cfg.get('verify_workers') or k+1, timeout_s=3600, guide_beta=1.))
        if cfg.get('b_replay'):
            b.update(b_replay=True, replay_beta=1., slots=cfg['slots'])
        if cfg.get('fanout_factor'):
            b['parallel_fanout_factor'] = cfg['fanout_factor']
        if cfg.get('r_concurrent') and a == 'R':
            b['r_concurrent_lag1_quota'] = True
        return b
    wr.verify = verify


def _seed_check_fixtures(root):
    from .storage import write
    records = []
    for w in _check_worlds():
        records.append(dict(id=f'mappo_check_{w["seed"]}', world=w, arm='mappo',
                            metrics={h: dict(team_delay_vs=1000., collisions=0., ttc_below_3_vs=0., hard_brake=0.,
                                             teleports=0., road_delay_vs=1000., source_wait_vs=0.)
                                     for h in ('600', '900')}))
    write(root/'MAPPO_CHECK.json', dict(records=records))
    write(root/'A_CHECK_RECORDS_004.json', [dict(r, id=r['id'].replace('mappo', 'A'), arm='A') for r in records])


# ============================================================== scenario run

def run_scenario(name, cfg, workdir):
    from . import capacity_broker
    EVENTS.clear()
    cfg = dict(DEFAULTS, **cfg)
    base = Path(workdir)/name
    if base.exists():
        shutil.rmtree(base)
    root = base/'run'
    _child_environment(base, cfg)
    layout = build_root(root, cfg)
    install_runtime(root, cfg, layout)
    _seed_check_fixtures(root)
    proposer = tiny_proposer(base)
    from . import mppi_search
    if not hasattr(mppi_search.search, 'harness_original'):
        mppi_search.search = _proposer_search(proposer)
    if cfg.get('legacy_wait_all'):
        _install_legacy(cfg.get('legacy_ref', 'aadee535'))   # parent broker; children via H3_LEGACY_REF
    if cfg.get('r_concurrent'):
        from .storage import write
        # Live root already holds B_RANDOM_QUOTA_003 from the completed B u3.
        write(root/'B_RANDOM_QUOTA_003.json', dict(sampling={'merge-dense': .95, 'ordinary': .95},
              tickets=[dict(regime='ordinary', duration_s=d, group_size=n, id=i)
                       for i, (d, n) in enumerate([(60, 2), (120, 3), (180, 2), (300, 4), (90, 2), (240, 3),
                                                   (60, 5), (150, 2), (120, 6), (30, 2)])]))
    gpus = {s: FakeGPU(s, cfg) for s in SLOTS[:cfg['slots']]}
    with (root/'WARM_REMOTE_BINDING.json').open('w') as f:
        json.dump(dict(endpoints=[dict(slot=s) for s in gpus], deadline_unix=DEADLINE, search_sha256='harness'), f)
    (root/'WORKERS.json').unlink();(root/'WORKERS.json').write_text(json.dumps(dict(workers=[dict(slot=s) for s in gpus])))
    capacity_broker.Remote = make_remote_class(gpus, cfg)
    from . import warm_runner as wr
    outcome = {}
    started = time.time()

    def arm_thread(arm):
        try:
            wr.run_arm(root, arm);outcome[arm] = 'complete'
        except BaseException as error:
            outcome[arm] = 'failed: '+repr(error)
            event('arm_failed', arm=arm, error=repr(error), tb=traceback.format_exc())

    def broker_thread():
        try:
            capacity_broker.run(root, root/'BROKER_CONFIG.json');outcome['broker'] = 'exited'
        except BaseException as error:
            outcome['broker'] = 'stopped: '+repr(error)[:400]
    threads = {a: threading.Thread(target=arm_thread, args=(a,), daemon=True, name='arm-'+a) for a in ('B', 'R')}
    bt = threading.Thread(target=broker_thread, daemon=True, name='broker');bt.start()
    collectors = {}
    if cfg.get('b_replay'):
        from . import replay_bank
        from .accel_protocol import worlds
        train = [w for w in worlds('train') if w not in sum(layout['plans'].values(), [])]
        done = threading.Event()

        def collector(arm, rounds):
            try:
                b = wr.verify(root, arm)
                st = replay_bank.collect(root, arm, b, train[(0 if arm == 'B' else 7):], lambda: done.is_set(),
                                         episodes_per_round=cfg['collect_episodes'],
                                         verify_workers=cfg.get('verify_workers'), rounds=rounds)
                outcome.setdefault('collector_'+arm, []).append(dict(rounds=rounds, statuses=st))
            except BaseException as error:
                outcome.setdefault('collector_'+arm, []).append('failed: '+repr(error))
                event('collector_failed', arm=arm, error=repr(error), tb=traceback.format_exc())
        warm = [threading.Thread(target=collector, args=(a, cfg['bank_warmup_rounds']), daemon=True) for a in ('B', 'R')]
        for t in warm:
            t.start()
        for t in warm:
            t.join()
        event('bank_warmup_done', bank=bank_state(root))
        collectors = {a: threading.Thread(target=collector, args=(a, None), daemon=True) for a in ('B', 'R')}
        for t in collectors.values():
            t.start()
        started = time.time()
    for t in threads.values():
        t.start()
    last, last_change, deadlock = None, time.time(), False
    while any(t.is_alive() for t in threads.values()):
        time.sleep(1.)
        snap = progress_signature(root)
        if snap != last:
            last, last_change = snap, time.time()
        elif time.time()-last_change > cfg['stall_s']:
            deadlock = True
            event('harness_stall', idle_s=time.time()-last_change, signature=snap)
            break
        if time.time()-started > cfg['timeout_s']:
            event('harness_timeout');break
    wall = time.time()-started
    if collectors:
        done.set()
        for t in collectors.values():
            t.join(600)
    state = diagnose(root, cfg, gpus)
    state['bank'] = bank_state(root)
    # End the broker deterministically (registered health-pause path, never a STOP).
    (root/'PAUSE_HARNESS_END.json').write_text(json.dumps(dict(harness=True)))
    bt.join(30)
    return dict(scenario=name, config={k: v for k, v in cfg.items()}, outcome=outcome, deadlock_or_idle=deadlock,
                wall_s=round(wall, 1), **state,
                events=[{k: v for k, v in e.items() if k != 'tb'} for e in EVENTS][-40:],
                failures=[e for e in EVENTS if e['kind'] in ('arm_failed', 'gpu_worker_exception')][:4])


def _install_legacy(ref):
    """Reproduce the pre-fix client wait-all + hash-ordered broker from git ``ref`` (read-only)."""
    import subprocess
    from . import parallel_groups, capacity_broker
    top = Path(__file__).resolve().parents[2]
    for name, target, attrs in (('parallel_groups', parallel_groups, ('query',)),
                                ('capacity_broker', capacity_broker, ('pending', 'capacity_declines'))):
        src = subprocess.run(['git', '-C', str(top), 'show', f'{ref}:tools/coopmwm/{name}.py'],
                             capture_output=True, text=True, check=True).stdout
        m = types.ModuleType(f'tools.coopmwm._legacy_{name}');m.__package__ = 'tools.coopmwm'
        exec(compile(src, f'{ref}:{name}.py', 'exec'), m.__dict__)
        for a in attrs:
            setattr(target, a, getattr(m, a))
    # The legacy broker signature had no 'withdrawn' argument.
    legacy = capacity_broker.capacity_declines
    capacity_broker.capacity_declines = lambda items, assigned, slots, blocking=(), drain=False, withdrawn=():         legacy(items, assigned, slots, blocking, drain)


def bank_state(root):
    out = {}
    for arm in ('B', 'R'):
        d = root/'replay_bank'/arm
        st = {}
        for p in (d/'sumo_verification'/'verifications').glob('*.json'):
            v = json.loads(p.read_text());st[v['status']] = st.get(v['status'], 0)+1
        out[arm] = dict(snapshots=sum(1 for _ in (d/'snapshots').glob('*.json')), verifications=st,
                        skipped=sum(1 for _ in (d/'skipped').glob('*.json')),
                        charged_check=[json.loads(p.read_text())['charged_sim_s'] ==
                                       (json.loads(p.read_text())['k']+1)*json.loads(p.read_text())['window']['end_s']
                                       for p in (d/'sumo_verification'/'verifications').glob('*.json')
                                       if json.loads(p.read_text())['charged_sim_s']].count(False) == 0)
    return out


def progress_signature(root):
    count = lambda pattern: sum(1 for _ in root.glob(pattern))
    ticks = 0
    for p in root.glob('jobs/*/observable.jsonl.gz'):
        try:
            ticks += p.stat().st_size
        except OSError:
            pass
    return (count('replay_bank/*/sumo_verification/branches/*/*.result.json'), count('replay_bank/*/snapshots/*.json'),
            count('sumo_verification/branches/*/*.result.json'), count('jobs/*/CLAIM.json'), count('responses/*.json'), count('centre_responses/*.json'), count('sumo_verification/verifications/*.json'),
            count('jobs/*/RESULT.json'), count('*_checkpoint_*.pt'), count('broker_completed/*.json'), ticks,
            count('sumo_verification/branches/*/*/RESULT.json'))


def diagnose(root, cfg, gpus):
    from .storage import ledger
    from .sumo_verify import quote
    stops = sorted(p.name for p in root.glob('*STOP*.json'))
    pauses = sorted(p.name for p in root.glob('PAUSE_*.json'))
    verifs = []
    for p in sorted((root/'sumo_verification'/'verifications').glob('*.json')):
        v = json.loads(p.read_text())
        verifs.append(dict(id=v['verification_id'], arm=v['episode']['arm'], update=v['episode']['update'],
                           status=v['status'], k=v['k'], t0=v['t0'], end=v['window']['end_s'],
                           charged=v['charged_sim_s'], quote_k=quote(v['t0'], v['k']),
                           matches_k_plus_1=v['charged_sim_s'] in (0, (v['k']+1)*v['window']['end_s'])))
    charges = ledger(root)
    phases = {}
    for c in charges:
        phases[c['phase']] = phases.get(c['phase'], 0)+c['sim_s']
    updates = {}
    for arm in ('B', 'R'):
        for u in (4, 5):
            p = root/f'{arm}_UPDATE_{u:03}.json'
            if p.exists():
                v = json.loads(p.read_text())
                updates[f'{arm}{u}'] = dict(executed=v['proposal_routing']['executed'],
                                            queries=v['proposal_routing']['queries'],
                                            accepted=v['proposal_routing']['accepted'],
                                            checkpoint=(root/f'{arm}_checkpoint_{u:03}.pt').exists(),
                                            graded=v['trust'].get('graded'), wall_s=round(v['wall_s'], 1),
                                            query_wait_wall_s=round(v['proposal_routing'].get('query_wait_wall_s', 0.), 1),
                                            eligible_windows=v['proposal_routing'].get('eligible_query_windows'),
                                            bank=v['trust'].get('bank'),
                                            imitation_steps=sum(m.get('exact_imitation_steps', 0) for m in v['ppo']),
                                            ppo_steps=len(v['ppo']),
                                            random_matching=v.get('random_matching'),
                                            verifications=sum(1 for x in verifs if x['arm'] == arm and x['update'] == u))
    req = sum(1 for _ in root.glob('requests/*.json'))
    resp = sum(1 for _ in root.glob('responses/*.json'))
    declines = {}
    for p in root.glob('responses/*.json'):
        v = json.loads(p.read_text())
        if not v.get('accepted'):
            declines[v.get('reason', '?')[:60]] = declines.get(v.get('reason', '?')[:60], 0)+1
    verify_charge = sum(c['sim_s'] for c in charges if c['phase'].startswith('sumo_verification'))
    return dict(stops=stops, pauses=pauses, updates=updates, plan_requests=req, plan_responses=resp,
                unanswered_requests=req-resp, response_declines=declines,
                withdrawn=sum(1 for _ in root.glob('request_withdrawals/*.json')),
                verifications=dict(n=len(verifs), by_status={s: sum(1 for v in verifs if v['status'] == s)
                                                             for s in {v['status'] for v in verifs}},
                                   all_charges_equal_k_plus_1_branches=all(v['matches_k_plus_1'] for v in verifs),
                                   all_within_quote=all(v['charged'] <= v['quote_k'] for v in verifs),
                                   charged_sim_s=sum(v['charged'] for v in verifs),
                                   ledger_verification_sim_s=verify_charge),
                charges_by_phase=phases, gpu=dict({s: dict(alive=g.alive, searches=g.searches, failure=g.failure)
                                                   for s, g in gpus.items()}),
                quarantine_requests=sum(1 for _ in root.glob('worker_restart_requests/*.json')),
                broker_events=sum(1 for _ in root.glob('broker_events/*.json')))


def tiny_proposer(base):
    from . import plan_proposer as pp
    torch.manual_seed(0)
    model = pp.ProposerNet()
    path = base/'proposer'/'tiny'
    pp.save_checkpoint(model, dict(harness='tiny untrained CPU checkpoint'), path)
    loaded, _ = pp.load_checkpoint(path)
    return loaded


DEFAULTS = dict(k=8, N_exec=10, slots=4, workers=12, episodes_per_update=6, last_update=5, latency_s=1.,
                stall_s=240., timeout_s=3600., transport_errors=0, malformed=0, branch_crash=0,
                die_slot=None, die_after=1, decline_withdrawn=True, parallel_groups=True,
                fanout_factor=2, r_concurrent=True,   # resume config (MAIN, harden-int)
                b_replay=True, collect_episodes=2, bank_warmup_rounds=1)   # MAIN B-REPLAY

SCENARIOS = dict(
    a_clean=dict(),
    b_slow_search=dict(latency_s=20.),
    c_worker_dies=dict(die_slot='gpuA_second', die_after=1),
    c2_worker_dies_on_centre=dict(die_on='centre'),
    d_transport_3=dict(transport_errors=3),
    e_malformed=dict(malformed=1),
    f_branch_crash=dict(branch_crash=1),
    g_legacy_wait_all=dict(legacy_wait_all=True, latency_s=20.),
    h_r_pipelined_legacy=dict(r_concurrent=False),
    i_unbounded_fanout=dict(fanout_factor=None),
)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('scenario', choices=sorted(SCENARIOS))
    p.add_argument('--workdir', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--set', action='append', default=[], help='override key=json')
    a = p.parse_args(argv)
    cfg = dict(SCENARIOS[a.scenario])
    for item in a.set:
        k, v = item.split('=', 1);cfg[k] = json.loads(v)
    work = a.workdir or tempfile.mkdtemp(prefix='h3rehearsal_')
    result = run_scenario(a.scenario, cfg, work)
    text = json.dumps(result, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    import multiprocessing
    for child in multiprocessing.active_children():   # own pool workers only
        child.terminate()
    sys.stdout.flush();os._exit(0)   # daemon fake GPU threads


if __name__ == '__main__':
    main()
