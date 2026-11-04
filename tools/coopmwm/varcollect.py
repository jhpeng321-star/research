"""Variety plan-collection path for arms B and R on cpuS (ON only via a staged config).

Owner direction 2026-11-04 (c4/variety-collect), on top of the reviewed c4/checking-cpus +
c4/plan-variety: a collection root that is SEPARATE from the ledger.

1. PINNED read-only policies: B_checkpoint_k / R_checkpoint_k are loaded from ANOTHER root
   (``policies_root``, the coop3 run root), sha256-checked against the staged config, with ONE
   k for both arms (``checkpoint_k_B``/``checkpoint_k_R`` must agree; a mismatch refuses and
   nothing runs). The collection root never stores or updates a policy. A coop3 V2V
   checkpoint loads through the SAME builder the live coop3 runtime uses (strict state-dict
   load; v2v_launch3.INTENT_OPTIONS), and its episodes/branch replays/confirmations run the
   V2V observation path exactly as coop3 training episodes do (t-1 visibility, partner
   intents from the policy head).
2. Split collection root from ledger: the new root (the argv; its NAME is a parameter) holds
   jobs/snapshots/bank/broker files only. EVERY sim-s charge goes to the ONE shared training
   ledger through a configurable charge route (charge_route.mode, MAIN 2026-11-04 16:2x ledger
   host move): 'rpc' (default) is the EXISTING cpuS->ledger-host RPC route
   (tools/coopmwm/cpus_rpc.py ``reserve``, the binding's additive ``collect`` id-prefixes/
   phases) and must answer a ``budget`` probe before anything runs; 'local' is allowed ONLY on
   the registered single ledger host itself (LEDGER_HOST.json at the ledger run root: host name
   + ledger path + sha256 of the AUTHORITY line) and charges the UNCHANGED accel_storage.reserve
   there - any other host refuses, so exactly one ledger ever exists. Window verifications in
   ``sumo_verification``, confirmations in ``bank_verification_ext``. No second CHARGES.jsonl
   is ever created under the collection root.
3. Arm B searches the WM through the broker: CONFIG ONLY - ``stage_broker`` writes the worker
   staging script, the startup-gate list and the single-slot broker config file into the
   collection root. Nothing here ssh'es, stages remotely or starts any process. The staged
   STAGE_WORKER.py is a FRESH-WORKER path only (fix round 2): it stages a NEW EMPTY worker
   root + a binding pinning this build's numerical and variety sources (sha256-checked against
   the staged overlay) and starts the fresh service (tools.coopmwm.warm_spool), which itself
   RUNS all 8 startup gates (hotadd.GATES; no adopted-gate set exists) before the slot is
   served; the broker admits the slot only after all 8 receipts pass. It refuses to start when
   the worker root already exists or carries any owner PID/lock - it never adopts and never
   signals another process. Arm R keeps the unscored random candidates with the SAME registered
   variety mix: B takes floor(variety_structured_share*population) structured slots inside its
   search (mppi_search), R draws each candidate as Bernoulli(share) structured/free
   (sample_unscored) - identical registered keys on the snapshot meta.
4. Episode ids are parameterised: ``<collect_id_prefix>_<arm>_k<k:03>_e<n:05d>_<seed>``; the
   historical hard-coded id appears nowhere in this path.
5. OFF/default byte-identity: no existing entry point imports this module; the only edits to
   shared modules are the RPC binding's additive ``collect`` key (absent = unchanged bytes) and
   ls_collection's additive ``episode_prefix`` (default = the historical id).
"""
import hashlib
import json
import re
import time
from pathlib import Path

WINDOW_PHASE = 'sumo_verification'        # window verification charges (ls_collection convention)
CONFIRM_PHASE = 'bank_verification_ext'   # confirmation/solo charges (MAIN 2026-11-02 22:32)
ROUTE_PHASES = (WINDOW_PHASE, CONFIRM_PHASE)
ROUTE_MODES = ('rpc', 'local')            # how charges reach the ONE shared training ledger
ARMS = ('B', 'R')
ROUTE_KEYS = ('q_host', 'q_control', 'q_python', 'q_pythonpath', 'run_root')
LEDGER_HOST_KEYS = ('host', 'ledger_path', 'authority_sha256')
BROKER_ENDPOINT_KEYS = ('slot', 'destination', 'port', 'identity_file', 'known_hosts')
OWNER_KEYS = ('boot_id', 'pid', 'start_ticks')

REGISTERED = dict(
    policies_root='', checkpoint_k=-1, checkpoint_k_B=None, checkpoint_k_R=None,
    checkpoint_sha256=None, charge_route=None, collect_id_prefix='', episode_phase='',
    runtime='', accel_bounds=None, gpu_deadline=0., worlds=None,
    slots=1, fanout_factor=2., demand_phase_end_s=1200., warm_history=False,
    workers=9, timeout_s=3600., drain_s=0., broker=None, variety=None, extra_seeds=0,
    gpu_host=None,
)


def split_config(cfg):
    """Validate a staged VARCOLLECT_CONFIG.json (a copy; raises ValueError on any refusal).
    Unknown keys refuse, mirroring variety.split_config / coop_search.split_config."""
    unknown = set(cfg)-set(REGISTERED)
    if unknown:
        raise ValueError(f'unregistered varcollect config keys {sorted(unknown)}')
    v = dict(REGISTERED, **cfg)
    if not isinstance(v['policies_root'], str) or not v['policies_root']:
        raise ValueError('policies_root must be a non-empty path string')
    k = v['checkpoint_k']
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError('checkpoint_k must be a non-negative int')
    for arm in ARMS:
        ka = v[f'checkpoint_k_{arm}']
        if ka is None:
            v[f'checkpoint_k_{arm}'] = k
        elif not isinstance(ka, int) or isinstance(ka, bool) or ka < 0:
            raise ValueError(f'checkpoint_k_{arm} must be a non-negative int')
    if v['checkpoint_k_B'] != v['checkpoint_k_R']:
        raise ValueError(f'mismatched pinned checkpoint k: B {v["checkpoint_k_B"]} != '
                         f'R {v["checkpoint_k_R"]} (the same k is required for both arms)')
    shas = v['checkpoint_sha256']
    if not isinstance(shas, dict) or set(shas) != set(ARMS) or \
            not all(isinstance(s, str) and re.fullmatch(r'[0-9a-f]{64}', s) for s in shas.values()):
        raise ValueError(f'checkpoint_sha256 must map {ARMS} to sha256 hex digests')
    route = v['charge_route']
    mode = route.get('mode', 'rpc') if isinstance(route, dict) else None
    if mode not in ROUTE_MODES:
        raise ValueError(f'charge_route.mode must be one of {ROUTE_MODES}')
    required = ROUTE_KEYS if mode == 'rpc' else ('run_root',)
    if any(not isinstance(route.get(key), str) or not route[key] for key in required):
        raise ValueError(f'charge_route must carry non-empty {required} for mode {mode!r}')
    unknown = set(route)-set(ROUTE_KEYS)-{'phases', 'mode'}
    if unknown:
        raise ValueError(f'unregistered charge_route keys {sorted(unknown)}')
    if not route_phases(v):
        raise ValueError('charge_route.phases must be a non-empty list of phase names')
    if not isinstance(v['collect_id_prefix'], str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,40}',
                                                                       v['collect_id_prefix']):
        raise ValueError('collect_id_prefix must match [a-z][a-z0-9_]{1,40}')
    if not isinstance(v['episode_phase'], str) or not v['episode_phase']:
        raise ValueError('episode_phase must name the route phase episodes are charged in')
    if not isinstance(v['runtime'], str) or not v['runtime']:
        raise ValueError('runtime must be a non-empty string')
    bounds = v['accel_bounds']
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or \
            not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in bounds) or \
            not float(bounds[0]) < float(bounds[1]):
        raise ValueError('accel_bounds must be an ordered (min, max) pair')
    if not isinstance(v['gpu_deadline'], (int, float)) or isinstance(v['gpu_deadline'], bool) or \
            not float(v['gpu_deadline']) > 0:
        raise ValueError('gpu_deadline must be a positive unix timestamp')
    worlds_ = v['worlds']
    if not isinstance(worlds_, list) or not worlds_ or \
            not all(isinstance(w, dict) and isinstance(w.get('seed'), int) for w in worlds_):
        raise ValueError('worlds must be a non-empty list of world dicts with int seeds')
    for key, lo in (('slots', 1), ('workers', 2)):
        if not isinstance(v[key], int) or isinstance(v[key], bool) or int(v[key]) < lo:
            raise ValueError(f'{key} must be an int >= {lo}')
    for key in ('fanout_factor', 'demand_phase_end_s', 'timeout_s', 'drain_s'):
        if not isinstance(v[key], (int, float)) or isinstance(v[key], bool) or not float(v[key]) >= 0:
            raise ValueError(f'{key} must be a non-negative number')
    if not isinstance(v['extra_seeds'], int) or isinstance(v['extra_seeds'], bool) or \
            v['extra_seeds'] < 0:
        raise ValueError('extra_seeds must be a non-negative int (CRN seeds 3..2+n for '
                         'cooperative plans, the live per_seed_v1 bank option; 0 = off)')
    if v['broker'] is not None:
        v['broker'] = validate_broker(v['broker'])
        selected_gpu_host(v['broker'], v['gpu_host'])
    elif v['gpu_host'] is not None and v['gpu_host'] not in GPU_HOSTS:
        raise ValueError(f'gpu_host must be one of {GPU_HOSTS}')
    if v['variety'] is not None:
        from .variety import split_config as variety_split
        probe = dict(v['variety'])
        resolved = variety_split(probe)   # registered variety keys only; raises on unknown keys
        v['variety'] = None if resolved is None else probe
    return v


def load_config(root):
    """The staged config of a collection root; refuses when the file is missing/unparsable."""
    path = Path(root)/'VARCOLLECT_CONFIG.json'
    if not path.is_file():
        raise ValueError(f'no VARCOLLECT_CONFIG.json staged in {path.parent}')
    return split_config(json.loads(path.read_text(encoding='utf-8')))


# ---------------------------------------------------------------- pinned policies

def route_phases(cfg):
    route = cfg.get('charge_route') or {}
    phases = route.get('phases')
    return tuple(str(p) for p in phases) if phases else ROUTE_PHASES


def phase_registered(cfg, phase):
    """Every charge of this path must be in the route's registered phases; else refuse."""
    if phase not in route_phases(cfg):
        raise ValueError(f'phase {phase} is not registered on the collection charge route '
                         f'{route_phases(cfg)}')
    return True


def route_prefixes(cfg):
    """The id prefixes the cpuQ RPC binding must register for this path's charges."""
    prefix = cfg['collect_id_prefix']
    return tuple(sorted({f'{prefix}_', f'sv_{prefix}_', f'confirm_replication_sv_{prefix}_',
                         f'confirm_extended_sv_{prefix}_', f'confirm_solo_sv_{prefix}_'}))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pinned_checkpoints(cfg):
    """{arm: {path, k, sha256}} of the pinned read-only checkpoints in ANOTHER root.

    Refuses (ValueError, before anything runs): a per-arm k mismatch, a missing checkpoint
    file, or a sha256 that differs from the staged config."""
    k = {arm: int(cfg[f'checkpoint_k_{arm}']) for arm in ARMS}
    if k['B'] != k['R']:
        raise ValueError(f'mismatched pinned checkpoint k: B {k["B"]} != R {k["R"]}')
    root = Path(cfg['policies_root'])
    out = {}
    for arm in ARMS:
        path = root/f'{arm}_checkpoint_{k[arm]:03d}.pt'
        if not path.is_file():
            raise ValueError(f'pinned checkpoint missing: {path}')
        sha = file_sha256(path)
        if sha != cfg['checkpoint_sha256'][arm]:
            raise ValueError(f'pinned {arm} checkpoint sha256 mismatch: {sha} != '
                             f'{cfg["checkpoint_sha256"][arm]} ({path})')
        out[arm] = dict(path=str(path), k=k[arm], sha256=sha)
    return out


BASE_ACTOR_INPUT = 33          # plain policy actor input width (29 features + 4 broadcast latent)


def pinned_policy(cfg, arm):
    """(policy, pin) - the PINNED checkpoint of ``arm`` loaded read-only with the SAME builder
    the live coop3 runtime uses. The payload's own update id must equal the pinned k (a
    swapped/renamed file refuses BEFORE any policy is built). A coop3 V2V checkpoint (intent
    head or widened actor input) loads through v2v_runtime.load_checkpoint with the registered
    launch3 intent options (v2v_launch3.INTENT_OPTIONS: v2v_dim=48 = K=4 partner slots x
    (6+H+1), horizon 5, intent head, detach_trunk) - the exact load path of the live runtime
    (v2v_runner.run_arm -> v2v_live.hooks['load_checkpoint'](path, config=cfg)); its
    state-dict load is STRICT (a missing or unexpected key refuses). A base-width checkpoint
    keeps the plain accel_learner.load bytes."""
    pin = pinned_checkpoints(cfg)[arm]
    import torch
    saved = torch.load(pin['path'], map_location='cpu', weights_only=False)
    if int(saved.get('update', -1)) != pin['k']:
        raise ValueError(f'pinned {arm} checkpoint payload update {saved.get("update")} != '
                         f'pinned k {pin["k"]}')
    state = saved.get('policy') or {}
    if 'intent_head.weight' in state or int(state['actor.0.weight'].shape[1]) != BASE_ACTOR_INPUT:
        from .v2v_launch3 import INTENT_OPTIONS
        from .v2v_runtime import load_checkpoint
        policy, _, _, _ = load_checkpoint(pin['path'], config=dict(INTENT_OPTIONS))
    else:
        from .accel_learner import load
        policy, _, _, _ = load(pin['path'])
    return policy, pin


def policy_v2v(policy):
    """True when the pinned policy is the live coop3 V2V class (intent head or widened input
    layer); a base-width plain policy stays on the legacy non-V2V bytes."""
    if getattr(policy, 'intent_horizon', 0):
        return True
    return int(policy.actor[0].weight.shape[1]) != BASE_ACTOR_INPUT


def v2v_job_keys(policy):
    """The coop3 training-wave V2V episode keys for the pinned policy - exactly the extras
    v2v_runner.run_arm's wave adds to every coop3 train episode (v2v=True and the registered
    v2v_dim=48). Empty for a plain policy: the job dict keeps its legacy bytes."""
    if not policy_v2v(policy):
        return {}
    from .v2v_launch3 import INTENT_OPTIONS
    return dict(v2v=True, v2v_dim=int(INTENT_OPTIONS['v2v_dim']))


# ---------------------------------------------------------------- charge route (ONE ledger)

def route_mode(cfg):
    """'rpc' (default) or 'local' - how charges reach the ONE shared training ledger."""
    route = cfg.get('charge_route') or {}
    return route.get('mode', 'rpc')


def ledger_host_binding(route):
    """The registered single-ledger binding at the ledger run root: LEDGER_HOST.json carrying
    the host name, the ledger path and the sha256 of the run root's AUTHORITY line. A missing,
    incomplete, tampered or relocated binding refuses (ValueError)."""
    root = Path(route['run_root'])
    path = root/'LEDGER_HOST.json'
    if not path.is_file():
        raise ValueError(f'no LEDGER_HOST.json registered at {path}')
    b = json.loads(path.read_text(encoding='utf-8'))
    unknown = set(b)-set(LEDGER_HOST_KEYS)
    if unknown:
        raise ValueError(f'unregistered LEDGER_HOST.json keys {sorted(unknown)}')
    if any(not isinstance(b.get(k), str) or not b.get(k) for k in LEDGER_HOST_KEYS):
        raise ValueError(f'LEDGER_HOST.json must carry non-empty {LEDGER_HOST_KEYS}')
    if not re.fullmatch(r'[0-9a-f]{64}', b['authority_sha256']):
        raise ValueError('LEDGER_HOST.json authority_sha256 must be a sha256 hex digest')
    authority = root/'AUTHORITY.md'
    if not authority.is_file():
        raise ValueError(f'no AUTHORITY.md at {authority}')
    line = next((l.strip() for l in authority.read_text(encoding='utf-8').splitlines() if l.strip()), '')
    if hashlib.sha256(line.encode('utf-8')).hexdigest() != b['authority_sha256']:
        raise ValueError(f'LEDGER_HOST.json authority_sha256 does not match the AUTHORITY line '
                         f'of {authority}')
    if Path(b['ledger_path']) != root:
        raise ValueError(f'LEDGER_HOST.json ledger_path {b["ledger_path"]} is not the registered '
                         f'ledger run root {root}')
    return b


def local_charges(cfg, binding):
    """The charge route of the ledger host ITSELF (mode 'local'): the UNCHANGED
    accel_storage.reserve at the registered ledger path, with the ledger run root's own
    registered caps (warm_protocol.install_accounting). Any OTHER host refuses here - charges
    must then go through the ledger host's RPC route - so there is never a second ledger."""
    import socket
    here = socket.gethostname()
    if here != binding['host']:
        raise ValueError(f"mode 'local' refused: this host {here!r} is not the registered single "
                         f"ledger host {binding['host']!r}; its charges must go through the RPC "
                         "charge route of the ledger host")
    from .warm_protocol import install_accounting
    root = Path(binding['ledger_path'])
    install_accounting(root)

    def charged(jobs, phase):
        phase_registered(cfg, phase)
        from . import accel_storage
        return accel_storage.reserve(root, jobs, phase)
    return charged


def route_charges(cfg):
    """Install the ONE-ledger charge route and refuse before anything runs. mode 'rpc'
    (default): the EXISTING cpuS->ledger-host RPC route (cpus_checking.install_remote_accounting),
    refused unless it answers a read-only ``budget`` probe (one attempt: an unavailable route
    refuses the whole path, promptly). mode 'local': ONLY on the registered single ledger host
    itself (LEDGER_HOST.json binding), charging the SAME unchanged reserve at the registered
    ledger path. Returns the routed charge callable."""
    if route_mode(cfg) == 'local':
        return local_charges(cfg, ledger_host_binding(cfg['charge_route']))
    from .cpus_checking import install_remote_accounting, rpc
    route = dict(cfg['charge_route'])
    try:
        budget = rpc(route, 'budget')
    except Exception as error:      # transport refused/timeout/unparsable: refuse the whole path
        raise ValueError(f'charge route unavailable: {error!r}') from error
    if not budget.get('ok'):
        raise ValueError(f'charge route unavailable: {budget}')
    reserve = install_remote_accounting(route)

    def charged(jobs, phase):
        phase_registered(cfg, phase)
        return reserve(Path(route['run_root']), jobs, phase)
    return charged


def refuse_local_ledger(root):
    """A second ledger must never appear under the collection root (a file OR a directory
    decoy named CHARGES.jsonl); the ONE shared ledger lives on the registered single ledger
    host (behind its RPC route, or directly through accel_storage.reserve in mode 'local' on
    that host)."""
    p = Path(root)/'CHARGES.jsonl'
    if p.exists() or p.is_symlink():
        raise ValueError(f'a second ledger appeared at {p}: collection charges must only go '
                         'through the registered charge route of the single ledger host')


# ---------------------------------------------------------------- parameterised ids

def job_id(cfg, arm, index, world):
    """Parameterised collection episode id (the historical hard-coded id appears nowhere)."""
    return (f'{cfg["collect_id_prefix"]}_{arm}_k{int(cfg["checkpoint_k_B"]):03d}'
            f'_e{int(index):05d}_{world["seed"]}')


# ---------------------------------------------------------------- arm B broker (CONFIG ONLY)

FRESH_SERVICE_MODULE = 'tools.coopmwm.warm_spool'   # fresh service: runs ALL 8 gates itself
# The GPU-side plan-variety sources: the fresh worker's binding must pin them alongside the
# numerical set, so the served search re-derives the registered variety decision groups.
VARIETY_GPU_SOURCES = ('pytools/tools/coopmwm/variety.py', 'pytools/tools/coopmwm/niche_collect.py')
GPU_HOSTS = ('gpuA', 'gpuR')
FRESH_GPU_SOURCES = tuple('pytools/tools/coopmwm/'+n+'.py' for n in
                          ('fresh_worker', 'warm_service', 'warm_spool', 'hotadd'))


def selected_gpu_host(broker, host=None):
    slot = broker['workers'][0]['endpoint']['slot']
    inferred = next((h for h in GPU_HOSTS if slot.startswith(h)), None)
    host = host or inferred
    if host not in GPU_HOSTS or (inferred is not None and inferred != host):
        raise ValueError('gpu_host must select gpuA or gpuR and agree with the broker slot')
    return host


def worker_environment(host):
    # Explicit deployment contracts, NEVER the cpuS collector's interpreter/torch.
    # A different installed GPU image must supply its own explicit worker_env pin.
    return dict(host=host, python_executable='/root/miniconda3/bin/python',
                python_version='3.12', torch_version='2.8.0+cu128', cuda_version='12.8',
                variables=dict(WMTEACH_BACKEND='FULL-v4', OMP_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))


def check_worker_environment(pin):
    if not isinstance(pin, dict) or set(pin) != set(worker_environment('gpuA')) or \
            any(not isinstance(pin[k], str) or not pin[k] for k in
                ('host', 'python_executable', 'python_version', 'torch_version')) or \
            pin['host'] not in GPU_HOSTS or \
            not re.fullmatch(r'3\.\d+', pin['python_version']) or \
            not re.fullmatch(r'\d+\.\d+\.\d+\+[a-z0-9]+', pin['torch_version']) or \
            (pin['cuda_version'] is not None and not isinstance(pin['cuda_version'], str)) or \
            pin['variables'] != worker_environment(pin['host'])['variables']:
        raise ValueError('worker_env must be a complete explicit host environment pin')
    return pin


def validate_broker(broker):
    """Exactly ONE new GPU worker slot, in the LS_ENDPOINTS.json shape ls_broker.run consumes.
    The optional ``binding`` is the operator-staged fresh-worker GPU binding snapshot (its
    hashes must pin the variety sources; an adopted-gate set refuses)."""
    if not isinstance(broker, dict):
        raise ValueError('broker must be a dict')
    unknown = set(broker)-{'amendment_sha256', 'gpu_deadline', 'workers', 'binding', 'worker_env'}
    if unknown:
        raise ValueError(f'unregistered broker keys {sorted(unknown)}')
    for key in ('amendment_sha256', 'gpu_deadline'):
        if key not in broker:
            raise ValueError(f'broker config lacks {key}')
    if not re.fullmatch(r'[0-9a-f]{64}', str(broker['amendment_sha256'])):
        raise ValueError('broker amendment_sha256 must be a sha256 hex digest')
    workers = broker['workers']
    if not isinstance(workers, list) or len(workers) != 1:
        raise ValueError(f'the variety collection uses exactly ONE new GPU slot, '
                         f'got {len(workers) if isinstance(workers, list) else "?"}')
    w = workers[0]
    if not isinstance(w, dict) or set(w) != {'endpoint', 'owner'}:
        raise ValueError('each broker worker carries exactly endpoint + owner')
    endpoint, owner = w['endpoint'], w['owner']
    if not isinstance(endpoint, dict) or any(not endpoint.get(key) for key in BROKER_ENDPOINT_KEYS):
        raise ValueError(f'broker endpoint needs non-empty {BROKER_ENDPOINT_KEYS}')
    if not isinstance(owner, dict) or any(owner.get(key) in (None, '') for key in OWNER_KEYS):
        raise ValueError(f'broker owner needs {OWNER_KEYS}')
    if broker.get('binding') is not None:
        check_fresh_binding(broker['binding'])
    if broker.get('worker_env') is not None:
        check_worker_environment(broker['worker_env'])
    return broker


def check_fresh_binding(binding):
    """A fresh-worker GPU binding must NEVER be an adopted-gate set (the fresh service runs
    ALL 8 hotadd.GATES itself) and its hashes must pin the variety sources alongside the
    numerical set (a binding without them cannot serve the registered variety search)."""
    if not isinstance(binding, dict):
        raise ValueError('broker binding must be a dict')
    if 'adopted_startup_gates' in binding:
        raise ValueError('a fresh worker RUNS all eight startup gates itself; a binding with '
                         'adopted_startup_gates is an adoption record and refuses')
    hashes = binding.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('broker binding must carry a non-empty hashes map')
    if not all(re.fullmatch(r'[0-9a-f]{64}', str(h)) for h in hashes.values()):
        raise ValueError('broker binding hashes must map source names to sha256 hex digests')
    missing = [n for n in VARIETY_GPU_SOURCES if n not in hashes]
    if missing:
        raise ValueError(f'broker binding must pin the variety sources {missing}')
    if 'qualification' in binding:
        from .fresh_worker import pending
        if binding['qualification'] != pending(hashes):
            raise ValueError('fresh qualification must be pending, never copied from prior receipts')
    return binding


def repo_root():
    """This build's source tree root (the layout the staged overlay is produced from)."""
    import tools.coopmwm
    return Path(tools.coopmwm.__file__).resolve().parents[2]


def fresh_binding(broker, gpu_host=None):
    """Source-pinned fresh worker with a PENDING qualification contract. All actual
    qualification evidence is computed by its own eight startup gates; passing receipts
    from another worker refuse, including in an explicitly supplied source binding.
    The immutable contract points to the completed worker-owned qualification receipt.
    GPU image pins are explicit and independent of the collector's CPU environment.
    """
    from .fresh_worker import pending
    host = selected_gpu_host(broker, gpu_host)
    supplied = check_fresh_binding(dict(broker['binding'])) if broker.get('binding') else {}
    env = check_worker_environment(broker.get(
        'worker_env', supplied.get('worker_env', worker_environment(host))))
    if env['host'] != host:
        raise ValueError('worker_env host differs from gpu_host')
    from .warm_service import NUMERICAL
    repo = repo_root()
    names = [n for n in NUMERICAL
             if n.endswith('.py') and n.startswith(('pytools/tools/coopmwm/',
                                                    'pytools/tools/coopnative/'))]
    hashes = {}
    for rel in (() if supplied else tuple(names)+VARIETY_GPU_SOURCES+FRESH_GPU_SOURCES):
        path = repo/rel.split('pytools/', 1)[1]
        if not path.is_file():
            raise ValueError(f'fresh-worker source missing from this build: {rel}')
        hashes[rel] = file_sha256(path)
    hashes = supplied.get('hashes', hashes)
    result = dict(supplied, amendment_sha256=broker['amendment_sha256'],
                gpu_deadline=float(broker['gpu_deadline']), deadline_unix=float(broker['gpu_deadline']),
                memory_mode='warm', startup_qualification=True, speed2=True, speed2_transfers=True,
                concurrent_plans=True, memory_fraction=.42 if host=='gpuA' else .34,
                batch_width_by_K={'4': 18, '8': 10} if host=='gpuA' else {'4': 4, '8': 4},
                niche_comparison=True, service=FRESH_SERVICE_MODULE, hashes=hashes,
                gpu_host=host, worker_env=env, qualification=pending(hashes))
    result.pop('batch_widths', None)
    result.pop('batch_width', None)
    return result


def broker_readiness(broker, receipts):
    """The single slot is admitted only when ALL 8 startup gates (hotadd.GATES, the fresh-worker
    startup set; a warm_service.ADOPTABLE subset is NOT enough) are present and pass_ true
    (the same rule shape as ls_collect_ops.collection_gpu_status)."""
    from .hotadd import GATES
    gates = {}
    for name in GATES:
        value = receipts.get(name)
        if isinstance(value, dict):
            gates[name] = bool(value.get('pass_'))
    ready = bool(broker['workers']) and all(gates.get(name) for name in GATES)
    return dict(ready=ready, gates=gates, required=list(GATES))


STAGE_WORKER_TEMPLATE = '''"""Variety collection GPU worker staging (c4/variety-collect; CONFIG artifact).

Staged by tools/coopmwm/varcollect.py stage_broker; nothing ran at staging time. The OPERATOR
copies this script to the GPU host and runs it there; this build never ssh'es or starts it.

FRESH-WORKER PATH ONLY (fix round 2). Starting this script stages a NEW EMPTY worker root plus
the registered binding (hashes pin this build's numerical and VARIETY sources; the binding is
checked against the --overlay byte for byte) and starts the FRESH service
(CONFIG['fresh']['service'] = tools.coopmwm.warm_spool), which itself RUNS ALL EIGHT startup
gates (hotadd.GATES: WARM_BATCH_SINGLE x2 + COMMITTED x2 + PARITY x2 + WARM_PARITY x2) before
any search is served - the binding carries no adopted_startup_gates, so nothing is ever
inherited. The broker admits the slot only after all 8 gate receipts pass (broker_readiness /
STARTUP_GATES.json rule; this script's --check prints the same verdict). REFUSALS (exit 3,
JSON on stdout): the worker root already exists, or any owner PID/lock file is present - a
fresh worker never shares a root, never adopts a running process and never sends any process
a termination request. There is no takeover mode in this script at all.

Usage (GPU host): python3 STAGE_WORKER.py <worker_root> --overlay <overlay_dir> [--check]
The --overlay directory has the worker-root layout: run artifacts at its top level
(INPUTS.json, FROZEN_COSTS.json, ...) and the package tree under pytools/; its entire content
is copied into the NEW worker root, then every binding hash is verified there.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG = {config}

OWNER_OR_LOCK = ('SPOOL_OWNER.json', 'SERVICE_OWNER.json', 'SPOOL_WORKER.claim', 'SERVICE.claim',
                 'SPOOL_FAILED.json', 'SPOOL_STOP.json', 'LS_GPU_ORIGINALS.json',
                 'FRESH_DISPATCH.json')


def refuse(reason, root):
    print(json.dumps(dict(refused=True, reason=reason, root=str(root),
                          slot=CONFIG['endpoint']['slot'])))
    return 3


def gate_receipts(root):
    """EVERY named gate accounted for: a missing file or an unparsable/failing receipt is
    False, so the slot stays unserved until all eight pass."""
    receipts = {{}}
    for name in CONFIG['startup_gates']:
        p = root/name
        passed = False
        if p.is_file():
            try:
                passed = json.loads(p.read_text()).get('pass_') is True
            except ValueError:
                passed = False
        receipts[name] = passed
    return receipts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--overlay', required=True)
    ap.add_argument('--check', action='store_true', help='print readiness only, start nothing')
    a = ap.parse_args()
    root = Path(a.root)
    if a.check:
        receipts = gate_receipts(root)
        ready = bool(receipts) and all(receipts.values())
        print(json.dumps(dict(root=str(root), overlay=a.overlay, ready=ready, receipts=receipts,
                              slot=CONFIG['endpoint']['slot'],
                              amendment_sha256=CONFIG['amendment_sha256'],
                              gpu_deadline=CONFIG['gpu_deadline'])))
        return 0 if ready else 2
    # ---- fresh start: refuse every pre-existing root / owner PID / lock (never adopt,
    # ---- never signal another process; there is no takeover mode in this script).
    if root.exists():
        owners = [n for n in OWNER_OR_LOCK if (root/n).exists()]
        reason = ('owner/lock present in the existing root: '+','.join(owners) if owners else
                  'worker root already exists; a fresh worker stages a NEW EMPTY root')
        return refuse(reason, root)
    fresh = CONFIG['fresh']
    binding = fresh['binding']
    if binding.get('adopted_startup_gates'):
        return refuse('a fresh worker RUNS all eight startup gates itself; an adopted-gate '
                      'binding refuses', root)
    overlay = Path(a.overlay)
    if not overlay.is_dir():
        return refuse('overlay directory missing: '+str(overlay), root)
    if (overlay/'GPU_BINDING.json').is_file():
        return refuse('overlay carries its own GPU_BINDING.json; the fresh binding is written '
                       'by this script from CONFIG only', root)
    inherited = [n for n in CONFIG['startup_gates']+list(OWNER_OR_LOCK)+
                 ['FRESH_STARTUP.json', 'FRESH_QUALIFICATION.json'] if (overlay/n).exists()]
    if inherited:
        return refuse('overlay carries prior worker evidence: '+','.join(inherited), root)
    missing = [n for n in binding['hashes'] if not (overlay/n).is_file()]
    if missing:
        return refuse('overlay lacks registered sources: '+','.join(sorted(missing)), root)
    bad = [n for n, h in binding['hashes'].items()
           if hashlib.sha256((overlay/n).read_bytes()).hexdigest() != h]
    if bad:
        return refuse('staged source sha256 mismatch: '+','.join(sorted(bad)), root)
    pin = binding['worker_env']
    env = dict(os.environ, **pin['variables'])
    probe_code = ('import json,sys,torch; print(json.dumps(dict('
                  'python_version=".".join(map(str,sys.version_info[:2])), '
                  'torch_version=str(torch.__version__), cuda_version=torch.version.cuda)))')
    try:
        probe = subprocess.run([pin['python_executable'], '-B', '-c', probe_code],
                               env=env, capture_output=True, text=True, timeout=60, check=True)
        actual = json.loads(probe.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return refuse('pinned worker environment unavailable: '+str(error), root)
    if any(actual.get(k) != pin[k] for k in actual) or set(actual) != {{
            'python_version', 'torch_version', 'cuda_version'}}:
        return refuse('pinned worker environment mismatch: '+str(actual), root)
    root.mkdir(parents=True)                 # exclusive: a raced second start refuses here too
    shutil.copytree(overlay, root, dirs_exist_ok=True)
    bad = [n for n, h in binding['hashes'].items()
           if hashlib.sha256((root/n).read_bytes()).hexdigest() != h]
    if bad:
        shutil.rmtree(root)
        return refuse('copied source sha256 mismatch: '+','.join(sorted(bad)), root)
    with (root/'GPU_BINDING.json').open('x') as f:
        json.dump(binding, f, sort_keys=True)
    env['PYTHONPATH'] = str(root/'pytools')
    # cwd = the worker root's own package tree: the freshly staged service module is the one
    # that runs; a repo checkout in the caller's cwd can never shadow it.
    extra = {{'start_new_session': True}} if os.name == 'posix' else {{}}
    with (root/'SPOOL.log').open('x') as log:
        proc = subprocess.Popen([pin['python_executable'], '-B', '-u', '-m', fresh['service'], str(root)],
                                cwd=str(root/'pytools'), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, **extra)
    print(json.dumps(dict(root=str(root), started=proc.pid, service=fresh['service'],
                          slot=CONFIG['endpoint']['slot'], gates=CONFIG['startup_gates'],
                          note='the fresh service writes its own owner receipt and RUNS all '
                               'eight startup gates itself; the broker admits the slot only '
                               'after every gate receipt passes (--check reports readiness)')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''


def stage_broker(cfg, root):
    """Write the arm-B broker artifacts into the collection root (CONFIG ONLY; no ssh, no
    start): broker/BROKER_CONFIG.json (the single-slot config; on cpuS it becomes the root's
    LS_ENDPOINTS.json), broker/STARTUP_GATES.json and broker/STAGE_WORKER.py."""
    broker = validate_broker(cfg['broker'])
    from .hotadd import GATES
    d = Path(root)/'broker'
    d.mkdir(parents=True, exist_ok=True)
    gates = dict(slot=broker['workers'][0]['endpoint']['slot'], required=list(GATES),
                 rule='ALL 8 startup gates (hotadd.GATES - the full fresh-worker startup set) '
                      'must be present with pass_ true before the slot is served '
                      '(ls_collect_ops.collection_gpu_status readiness rule shape); a '
                      'failed/missing gate keeps the broker from assigning this slot any search')
    config = dict(broker, shape='LS_ENDPOINTS.json (tools.coopmwm.ls_broker.run)',
                  startup_gates=list(GATES), path_shape_note='on cpuS install this file as '
                  '<collection_root>/LS_ENDPOINTS.json next to LS_RECOVERY_CONFIG.json',
                   fresh=dict(binding=fresh_binding(broker, cfg.get('gpu_host')),
                              service=FRESH_SERVICE_MODULE))
    _write_json(d/'BROKER_CONFIG.json', config)
    _write_json(d/'STARTUP_GATES.json', gates)
    script = STAGE_WORKER_TEMPLATE.format(config=repr(dict(
        amendment_sha256=broker['amendment_sha256'], gpu_deadline=broker['gpu_deadline'],
        endpoint=broker['workers'][0]['endpoint'],
        startup_gates=list(GATES), fresh=config['fresh'])))
    _write_text(d/'STAGE_WORKER.py', script)
    return dict(broker=d/'BROKER_CONFIG.json', gates=d/'STARTUP_GATES.json',
                worker_script=d/'STAGE_WORKER.py')


def _write_json(path, value):
    tmp = Path(str(path)+'.publishing')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    tmp.replace(path)      # durable replace; never overwrite silently


def _write_text(path, text):
    tmp = Path(str(path)+'.publishing')
    tmp.write_text(text)
    tmp.replace(path)


# ---------------------------------------------------------------- collector (one arm)

def search_config(cfg):
    """The request search_config for arm B: the registered it2_K2 boundary config plus the
    variety keys, so the GPU side re-derives the same additive decision groups (R never
    receives a search_config: it draws unscored random candidates locally)."""
    if not cfg.get('variety'):
        return None
    from .niche_collect import search_config as niche_search_config
    return niche_search_config(0, variety=cfg['variety'])


def round_jobs(cfg, arm, policy, pin, root, start, worlds_, episodes_per_round):
    """The episode jobs of one collection round (plain policy episodes; the collector guide
    snapshots windows and queues requests; no plan is ever executed). A V2V pinned policy
    runs the coop3 training-episode shape: the job carries the same v2v keys the live
    runtime's wave adds (v2v=True, v2v_dim=48), so accel_plant rebuilds the SAME policy class
    from the weights and wraps the controller with the V2V observation path (t-1 visibility,
    partner intents from the policy head)."""
    from .policy_wm import actor_state
    from .protocol import digest
    jobs = []
    for i in range(episodes_per_round):
        w = worlds_[(start+i) % len(worlds_)]
        ident = job_id(cfg, arm, start+i, w)
        collect = dict(slots=int(cfg['slots']), fanout_factor=float(cfg['fanout_factor']),
                       demand_phase_end_s=float(cfg['demand_phase_end_s']))
        if cfg.get('variety'):
            if arm == 'B':
                collect['search_config'] = search_config(cfg)
            collect['variety'] = dict(cfg['variety'])
        jobs.append(dict(id=ident, path=str(Path(root)/'jobs'/ident), world=w, role='train',
                         arm=arm, weights=policy.state_dict(), runtime=cfg['runtime'],
                         checkpoint=pin['sha256'], noise_seed=int(digest(dict(collect=ident))[:15], 16),
                         deterministic=False, sim_s=1200, queue_root=str(root),
                         accel_bounds=list(cfg['accel_bounds']), gpu_deadline=float(cfg['gpu_deadline']),
                         guided=False, beta=1., policy_state=actor_state(policy),
                         warm_history=bool(cfg['warm_history']), collector_policy_update=int(pin['k']),
                         b_replay=True, collect=collect, **v2v_job_keys(policy)))
    return jobs


def run_episodes(jobs):
    """Real episode execution (accel_plant; each job is plain policy, never replayed)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from .accel_plant import episode
    with ProcessPoolExecutor(max_workers=len(jobs),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        return list(pool.map(episode, jobs))


def collect(root, cfg, arm, stop, episodes_per_round=2, rounds=None, verify_workers=None,
            verify_per_round=None, episodes=None):
    """Continuous variety collection for ONE arm from the NEW root until ``stop()``.

    Every charge goes through the routed reserve (window verifications: ``sumo_verification``,
    confirmations: ``bank_verification_ext``, episodes: the registered ``episode_phase``);
    confirmations ALWAYS use the per-seed all8 verification (verify_mode 'all8_both_seeds',
    the live per_seed_v1 bank rule; ``extra_seeds`` config = CRN seeds 3..2+n for cooperative
    plans, 0 = off); ``episodes`` overrides the real runner (CPU tests inject a fake)."""
    from . import replay_bank as rb, sumo_verify as sv
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm}')
    root = Path(root)
    refuse_local_ledger(root)
    phase_registered(cfg, WINDOW_PHASE)          # every charge phase validated BEFORE anything runs
    phase_registered(cfg, CONFIRM_PHASE)
    phase_registered(cfg, cfg['episode_phase'])
    if arm == 'B' and cfg.get('broker') is None:
        raise ValueError('arm B searches the WM through the broker: stage a single-slot '
                         'broker config (stage_broker) first')
    reserve = route_charges(cfg)                       # refuses when the route is unavailable
    shared = Path(cfg['charge_route']['run_root'])
    policy, pin = pinned_policy(cfg, arm)              # refuses wrong k/sha BEFORE any charge
    v2v = policy_v2v(policy)                           # coop3 V2V pinned policy -> V2V observations
    if arm == 'B':
        stage_broker(cfg, root)                        # CONFIG ONLY artifacts (validated above)
    bank = rb.bank_root(root, arm)
    bank.mkdir(parents=True, exist_ok=True)
    for n in ('jobs', 'requests', 'responses', 'centre_requests', 'centre_responses',
              'quota_admissions'):
        (root/n).mkdir(parents=True, exist_ok=True)
    runner = sv.SubprocessBranchRunner(bank/'sumo_verification'/'branches',
                                       workers=int(cfg['workers']), timeout_s=float(cfg['timeout_s']))
    charge = sv.accel_charge(str(shared), WINDOW_PHASE)

    def runner_for(work):
        return sv.SubprocessBranchRunner(Path(work)/'branches', workers=2,
                                         timeout_s=float(cfg['timeout_s']))
    decided = set()
    n = 0
    statuses = []
    run = episodes or run_episodes
    from . import bank_v2v
    while not stop() and (rounds is None or n < rounds):
        start = sum(1 for _ in (root/'jobs').glob(f'{cfg["collect_id_prefix"]}_{arm}_*'))
        jobs = round_jobs(cfg, arm, policy, pin, root, start, cfg['worlds'], episodes_per_round)
        reserve(jobs, cfg['episode_phase'])            # the ONLY episode charge: the shared ledger
        refuse_local_ledger(root)
        run(jobs)
        with bank_v2v.enabled(v2v):    # branch subprocesses inherit C4NM_V2V (coop3 branch path)
            statuses += rb.verify_pending(root, arm, cfg['accel_bounds'], runner, charge, verify_workers,
                                          decided, verify_per_round)
            statuses += rb.confirm_pending(root, arm, runner_for, str(shared), CONFIRM_PHASE, workers=2,
                                           limit=verify_per_round, verify_mode='all8_both_seeds',
                                           extra_seeds=int(cfg['extra_seeds']))
        rb.bank_quality(root, arm)
        refuse_local_ledger(root)
        n += 1
        if 'budget_blocked' in statuses:
            break
    deadline = time.time()+float(cfg['drain_s'])
    while time.time() < deadline:
        with bank_v2v.enabled(v2v):
            statuses += rb.verify_pending(root, arm, cfg['accel_bounds'], runner, charge, verify_workers,
                                          decided, verify_per_round)
        refuse_local_ledger(root)
        time.sleep(2)
    return statuses


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='the NEW collection root (its name is a parameter)')
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--gpu-host', choices=GPU_HOSTS, help='host of the single fresh broker slot')
    a = ap.parse_args(argv)
    cfg = load_config(a.root)
    if a.gpu_host is not None:
        cfg = split_config(dict(cfg, gpu_host=a.gpu_host))
    statuses = collect(a.root, cfg, a.arm, lambda: (Path(a.root)/'MAIN_STOP.json').exists(),
                       episodes_per_round=a.episodes, rounds=a.rounds)
    print(json.dumps(dict(root=str(a.root), arm=a.arm, statuses=statuses), default=str))


if __name__ == '__main__':
    main()
