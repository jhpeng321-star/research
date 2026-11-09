"""DISTILL teacher-data collection launcher (DIST-LOG 2026-11-09). Stdlib only.

Subcommands
  stage   (run LOCALLY from the worktree, zero SUMO): copy the exact a28 LEARNED_TRIGGER plant source (gpuA
          src_p20c_b5268408_5ecc4358) + this branch's overlay (git HEAD blobs) + the a28 inputs (prerun_plant CONFIG
          and u0 residual, BOUND_U0, ORACLE_BINDING, bank, model; sha-verified) to a collection host.
  collect (run ON the host): N DISTILL seeds x 1200 s, one fresh interpreter per episode (nice 19), a registered
          worker count, memory guard (MemAvailable >= --mem-min-gb before every start), disk guard, per-episode
          receipt + sim-s ledger charged at start, no replay of claimed/interrupted seeds, STOP file.
  sync    (run ON the host): rsync complete shards + receipts to the training host (private IP, --dest variable).
  status  (run ON the host): counts / sim-s.

Never kills by image name: only the launcher's own child PIDs are signalled (on STOP / SIGTERM).
Seeds: DISTILL range 999000-999999 only; forbidden ranges (eval panels 989600-989699 / 993000-993999, 964000-964199,
962xxx, 977xxx) refused. Collection requires --registration-id (the integrator registers before any real SUMO).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

DISTILL_RANGE = (999000, 999999)
SMOKE_SEEDS = (999990, 999991)
FORBIDDEN = ((989600, 989699), (993000, 993999), (964000, 964199), (962000, 962999), (977000, 977999))
EPISODE_SIM_S = 1200
HOSTS = dict(cpuZ='172.17.16.11', cpuW='172.17.16.14')
STAGE_DIR = '/home/ubuntu/distill_stage'
PY = '/home/ubuntu/miniconda3/bin/python'
# a28 (amendment 28) LEARNED_TRIGGER plant on gpuA (ops/nm_finaleval_clean5a_registration_amend28.json)
GPUA = 'gpuA'
GPUA_SRC = '/root/autodl-fs/nm_finaleval_p20c_stage/src_p20c_b5268408_5ecc4358'
GPUA_INPUTS = dict(
    cfg='/root/autodl-fs/nm_finaleval_p20c_bank/prerun_plant/CONFIG.json',
    residual='/root/autodl-fs/nm_finaleval_p20c_bank/prerun_plant/A_checkpoint_000.pt',
    bound_u0='/root/autodl-fs/nm_finaleval_p20c_bank/BOUND_U0.json',
    oracle_binding='/root/autodl-fs/nm_finaleval_p20c_bank/ORACLE_BINDING.json',
    bank='/autodl-fs/data/nm_finaleval_p20c_bank/oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json',
    model='/autodl-fs/data/nm_finaleval_p20c_bank/oracle_banks/LEARNED_TRIGGER_MODEL.json')
#: registered shas (amendments 26-28); cfg / binding files are recorded as found
EXPECTED = dict(residual='5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734',
                bank='fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82',
                model='8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720')
LOCAL_NAMES = dict(cfg='CONFIG.json', residual='A_checkpoint_000.pt', bound_u0='BOUND_U0.json',
                   oracle_binding='ORACLE_BINDING.json', bank='GUIDANCE_R6_WMONLY_PRIORITY.json',
                   model='LEARNED_TRIGGER_MODEL.json')
#: host files that must equal gpuA's (frozen executor, map spec); the frozen runtime is hash-verified by leverprobe
HOST_PINS = {'/home/ubuntu/mappospeed/production2/policy_00169.pt':
             'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca',
             '/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/MAP_SPEC.json':
             '386400b78d27e50a376ee28a5d18d25a5baeb27fd4bb0436fb171e5ccafd4e0d'}
OVERLAY = ('pytools/tools/coopmwm/nm_eval_runner.py', 'pytools/tools/coopmwm/distill_log.py',
           'pytools/tools/coopmwm/test_distill_log_cpu.py', 'ops/distill_collect.py')
REPO_OF = {'pytools/tools/': 'tools/', 'ops/': 'ops/'}
ENV = dict(SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',
           COOPMWM_MAP_SPEC='/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/MAP_SPEC.json',
           COOPMWM_G5_RUNTIME='1', NM_SETTING='p20c')
BANNED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'driver_params')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def guard_path(p):
    low = str(p).lower()
    if any(t in low for t in BANNED):
        raise SystemExit(f'refusing banned path {p}')
    return p


def parse_seeds(text):
    lo, _, hi = str(text).partition('-')
    lo, hi = int(lo), int(hi or lo)
    seeds = list(range(lo, hi+1))
    bad = [s for s in seeds if any(a <= s <= b for a, b in FORBIDDEN) or not DISTILL_RANGE[0] <= s <= DISTILL_RANGE[1]]
    if bad or not seeds:
        raise SystemExit(f'seeds must lie in the DISTILL range {DISTILL_RANGE} (refused: {bad[:5]})')
    return seeds


# ---------------------------------------------------------------- stage (local)

def ssh(host, cmd, *, data=None, check=True):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, cmd], input=data, capture_output=True, timeout=1800)
    if check and r.returncode != 0:
        raise SystemExit(f'ssh {host} failed ({r.returncode}): {r.stderr.decode(errors="replace")[-800:]}')
    return r.stdout


def overlay_tar(repo):
    """Overlay = this worktree's committed HEAD blobs (LF), mapped into the deployed src layout."""
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--'] + [
        next(r + d[len(k):] for k, r in REPO_OF.items() if d.startswith(k)) for d in OVERLAY],
        capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f'overlay files have uncommitted changes:\n{dirty}')
    buf, sums = tempfile.SpooledTemporaryFile(), {}
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for dest in OVERLAY:
            src = next(r + dest[len(k):] for k, r in REPO_OF.items() if dest.startswith(k))
            blob = subprocess.run(['git', '-C', str(repo), 'show', f'HEAD:{src}'], capture_output=True,
                                  check=True).stdout
            info = tarfile.TarInfo(dest)
            info.size, info.mode, info.mtime = len(blob), 0o644, 0
            import io
            tar.addfile(info, io.BytesIO(blob))
            sums[dest] = sha_bytes(blob)
    buf.seek(0)
    return commit, buf.read(), sums


def cmd_stage(a):
    host = a.host
    repo = Path(__file__).resolve().parents[1]
    commit, otar, osums = overlay_tar(repo)
    c8 = commit[:8]
    stage = f'{STAGE_DIR}/{c8}'
    print(f'stage {host}: commit {commit}, overlay {len(osums)} files -> {stage}')
    exists = ssh(host, f'test -e {stage} && echo yes || echo no').decode().strip()
    if exists == 'yes':
        raise SystemExit(f'{host}:{stage} exists (stages are never overwritten)')
    # 1) inputs from gpuA, sha-verified
    files, shas = {}, {}
    for k, path in GPUA_INPUTS.items():
        data = ssh(GPUA, f'cat {path}')
        shas[k] = sha_bytes(data)
        if k in EXPECTED and shas[k] != EXPECTED[k]:
            raise SystemExit(f'gpuA {k} sha {shas[k]} != registered {EXPECTED[k]}')
        files[k] = data
    ob = json.loads(files['oracle_binding'])
    if ob['bank']['file_sha256'] != shas['bank'] or ob['learned_trigger_model']['file_sha256'] != shas['model']:
        raise SystemExit('ORACLE_BINDING does not bind the fetched bank/model')
    u0 = json.loads(files['bound_u0'])
    if u0['residual_sha256'] != shas['residual']:
        raise SystemExit('BOUND_U0 does not bind the fetched residual')
    # 2) the exact a28 source tree
    src_tgz = ssh(GPUA, f'tar -C {os.path.dirname(GPUA_SRC)} -czf - {os.path.basename(GPUA_SRC)}')
    # 3) push + build on the host
    ssh(host, f'mkdir -p {stage}/inputs {stage}/incoming')
    ssh(host, f'cat > {stage}/incoming/src.tgz', data=src_tgz)
    ssh(host, f'cat > {stage}/incoming/overlay.tar', data=otar)
    for k, data in files.items():
        ssh(host, f'cat > {stage}/inputs/{LOCAL_NAMES[k]}', data=data)
    src = f'{stage}/src'
    build = (f'set -e; cd {stage}; mkdir -p src_x; tar -xzf incoming/src.tgz -C src_x; '
             f'mv src_x/{os.path.basename(GPUA_SRC)} src; rmdir src_x; tar -xf incoming/overlay.tar -C src; '
             f'grep -c $(printf "\\r") src/pytools/tools/coopmwm/distill_log.py && exit 7 || true')
    ssh(host, build)
    stage_doc = dict(schema='distill_stage/1', created=now(), host=host, commit=commit, base_src=f'gpuA:{GPUA_SRC}',
                     src=src, overlay=osums, py=PY, env=dict(ENV, PYTHONPATH=':'.join(
                         [f'{src}/pytools', f'{src}/runtime_src', f'{src}/WM_PACKAGE',
                          '/home/ubuntu/e3final/runtime/src', '/home/ubuntu/e3final/runtime']),
                         PATH=f'{ENV["SUMO_HOME"]}/bin:/home/ubuntu/miniconda3/bin:/usr/bin:/bin',
                         DISTILL_COMMIT=commit),
                     dwell_s=3.0, sha256=shas, gpuA_inputs=GPUA_INPUTS, host_pins=HOST_PINS,
                     **{k: f'{stage}/inputs/{LOCAL_NAMES[k]}' for k in GPUA_INPUTS})
    ssh(host, f'cat > {stage}/STAGE.json', data=json.dumps(stage_doc, indent=1, sort_keys=True).encode())
    verify = ' && '.join([f'echo "{v}  {p}" | sha256sum -c --quiet' for p, v in HOST_PINS.items()] +
                         [f'echo "{shas[k]}  {stage}/inputs/{LOCAL_NAMES[k]}" | sha256sum -c --quiet' for k in shas] +
                         [f'echo "{v}  {src}/{p}" | sha256sum -c --quiet' for p, v in osums.items()])
    ssh(host, verify + ' && echo STAGE_OK')
    print(json.dumps(dict(status='staged', host=host, stage=f'{stage}/STAGE.json', commit=commit), indent=1))


# ---------------------------------------------------------------- collect (on the host)

def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    raise RuntimeError('MemAvailable missing')


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def append(path, row):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str), encoding='utf-8')
    os.replace(tmp, path)


def cmd_collect(a):
    root = Path(guard_path(a.root))
    stage = json.loads(Path(a.stage).read_text(encoding='utf-8'))
    seeds = parse_seeds(a.seeds)
    smoke = a.stop != EPISODE_SIM_S
    if smoke and (any(s not in SMOKE_SEEDS for s in seeds) or len(seeds) > 2 or not 300 < a.stop < 1200):
        raise SystemExit('a shortened (smoke) run: seeds 999990-999991 only, <= 2 episodes, 300 < stop < 1200')
    if not smoke and any(s in SMOKE_SEEDS for s in seeds):
        raise SystemExit('the smoke seeds are never used for collection')
    if not a.registration_id:
        raise SystemExit('--registration-id required (register before any real SUMO)')
    for d in ('claims', 'receipts', 'shards', 'work', 'logs'):
        (root/d).mkdir(parents=True, exist_ok=True)
    reg = dict(registration_id=a.registration_id, seeds=a.seeds, host=stage['host'], workers=a.workers,
               budget_sim_s=a.budget_sim_s, stop_s=a.stop, stage_commit=stage['commit'], mem_min_gb=a.mem_min_gb,
               disk_min_gb=a.disk_min_gb)
    rpath = root/'REGISTRATION.json'
    if rpath.exists():
        old = json.loads(rpath.read_text(encoding='utf-8'))
        if {k: old.get(k) for k in reg} != reg:
            raise SystemExit(f'{rpath} differs from these arguments (a root is bound to one registration)')
    else:
        write_json(rpath, dict(reg, created=now()))
    ledger = root/'LEDGER.jsonl'

    def charged():
        if not ledger.exists():
            return 0
        return sum(json.loads(ln).get('charged_sim_s', 0) for ln in ledger.read_text(encoding='utf-8').splitlines() if ln)
    todo = [s for s in seeds if not (root/'claims'/f'{s}.json').exists()]
    skipped = len(seeds) - len(todo)
    env = dict(os.environ, **stage['env'])
    running = {}           # pid -> (seed, Popen, t0)
    stop_requested = []

    def on_term(signum, frame):
        stop_requested.append(signum)
        (root/'STOP').write_text(f'signal {signum} {now()}\n')
    signal.signal(signal.SIGTERM, on_term)
    print(json.dumps(dict(status='collect', root=str(root), seeds=len(seeds), todo=len(todo), skipped_claimed=skipped,
                          workers=a.workers, charged_before=charged())), flush=True)

    def finish(pid, rc):
        seed, proc, t0 = running.pop(pid)
        out = root/'work'/str(seed)/'distill_record.json'
        shard = root/'shards'/f'distill_{seed}.npz'
        rec = None
        if rc == 0 and out.exists():
            try:
                rec = json.loads(out.read_text(encoding='utf-8'))
            except ValueError:
                rec = None
        ok = rec is not None and shard.exists() and (rec.get('distill') or {}).get('sha256') == sha_file(shard)
        receipt = dict(seed=seed, status='complete' if ok else 'failed', rc=rc, pid=pid, wall_s=round(time.time()-t0, 1),
                       charged_sim_s=a.stop, stop_s=a.stop, registration_id=a.registration_id, end=now(),
                       distill=(rec or {}).get('distill'), J_veh_s=(rec or {}).get('J_veh_s'),
                       safety=(rec or {}).get('safety'),
                       learned_trigger=dict((k, (rec or {}).get('learned_trigger', {}).get(k)) for k in
                                            ('activations', 'fires', 'scores', 'issued')),
                       log=f'logs/{seed}.log')
        write_json(root/'receipts'/f'{seed}.json', receipt)
        append(ledger, dict(kind='end', seed=seed, status=receipt['status'], rc=rc, charged_sim_s=0, t=now()))
        print(json.dumps(dict(seed=seed, status=receipt['status'], rc=rc, wall_s=receipt['wall_s'])), flush=True)

    queue = list(todo)
    while queue or running:
        for pid in list(running):
            rc = running[pid][1].poll()
            if rc is not None:
                finish(pid, rc)
        if (root/'STOP').exists() or stop_requested:
            if queue:
                print(json.dumps(dict(status='stopping', unstarted=len(queue))), flush=True)
            queue = []
            if stop_requested and running:
                for pid in list(running):               # own child PIDs only
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        started = False
        if queue and len(running) < a.workers:
            mem, disk = mem_available_gb(), disk_free_gb(root)
            if charged() + a.stop > a.budget_sim_s:
                print(json.dumps(dict(status='budget_reached', charged=charged(), budget=a.budget_sim_s)), flush=True)
                queue = []
            elif mem < a.mem_min_gb or disk < a.disk_min_gb:
                pass                                     # guard: wait for memory / disk
            else:
                seed = queue.pop(0)
                claim = root/'claims'/f'{seed}.json'
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # never replay a claimed seed
                os.write(fd, json.dumps(dict(seed=seed, host=stage['host'], t=now(), mem_gb=round(mem, 1),
                                             disk_gb=round(disk, 1))).encode())
                os.close(fd)
                append(ledger, dict(kind='start', seed=seed, charged_sim_s=a.stop, registration_id=a.registration_id,
                                    t=now()))
                work = root/'work'/str(seed)
                log = open(root/'logs'/f'{seed}.log', 'w')
                cmd = ['nice', '-n', '19', stage['py'], '-B', '-m', 'tools.coopmwm.distill_log', 'run-episode',
                       '--stage', a.stage, '--seed', str(seed), '--work', str(work),
                       '--shard', str(root/'shards'/f'distill_{seed}.npz'), '--out', str(work/'distill_record.json'),
                       '--stop', str(a.stop)]
                proc = subprocess.Popen(cmd, cwd=stage['src'], env=env, stdout=log, stderr=subprocess.STDOUT)
                log.close()
                running[proc.pid] = (seed, proc, time.time())
                started = True
        if not started:
            time.sleep(2 if running else 0.2)
    status = cmd_status(a, quiet=True)
    print(json.dumps(dict(status='done', **status)), flush=True)


def cmd_status(a, quiet=False):
    root = Path(a.root)
    rec = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'receipts').glob('*.json'))]
    claims = len(list((root/'claims').glob('*.json')))
    ledger = root/'LEDGER.jsonl'
    charged = sum(json.loads(ln).get('charged_sim_s', 0) for ln in ledger.read_text(encoding='utf-8').splitlines()
                  if ln) if ledger.exists() else 0
    out = dict(root=str(root), claims=claims, complete=sum(r['status'] == 'complete' for r in rec),
               failed=sum(r['status'] != 'complete' for r in rec), running_or_interrupted=claims-len(rec),
               charged_sim_s=charged, shard_bytes=sum((p.stat().st_size for p in (root/'shards').glob('*.npz')), 0))
    if not quiet:
        print(json.dumps(out, indent=1))
    return out


def cmd_sync(a):
    """rsync complete shards + their receipts (+ ledger/registration) to --dest over ssh (private IP)."""
    root = Path(a.root)
    if not a.dest:
        raise SystemExit('--dest user@<private-ip>:<gpuA training data dir> required (DISTILL_DEST)')
    files = ['REGISTRATION.json', 'LEDGER.jsonl']
    for p in sorted((root/'receipts').glob('*.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        if r['status'] == 'complete':
            files += [f'receipts/{p.name}', f'shards/distill_{r["seed"]}.npz']
    lst = root/'SYNC_FILES.txt'
    lst.write_text('\n'.join(files) + '\n', encoding='utf-8')
    cmd = ['rsync', '-a', '--files-from', str(lst), '-e', 'ssh -o BatchMode=yes', str(root) + '/',
           a.dest.rstrip('/') + f'/{a.label or Path(root).name}/']
    r = subprocess.run(cmd)
    append(root/'SYNC.jsonl', dict(t=now(), dest=a.dest, files=len(files), rc=r.returncode))
    print(json.dumps(dict(status='synced' if r.returncode == 0 else 'sync_failed', files=len(files), rc=r.returncode)))
    sys.exit(r.returncode)


def main(argv=None):
    p = argparse.ArgumentParser(prog='distill_collect')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage')
    s.add_argument('--host', required=True, choices=sorted(HOSTS))
    c = sub.add_parser('collect')
    c.add_argument('--stage', required=True, help='<STAGE_DIR>/<commit8>/STAGE.json on this host')
    c.add_argument('--root', required=True)
    c.add_argument('--seeds', required=True, help='LO-HI inside 999000-999999')
    c.add_argument('--workers', type=int, required=True)
    c.add_argument('--budget-sim-s', type=int, required=True)
    c.add_argument('--registration-id', required=True)
    c.add_argument('--mem-min-gb', type=float, default=30.0)
    c.add_argument('--disk-min-gb', type=float, default=12.0)
    c.add_argument('--stop', type=int, default=EPISODE_SIM_S, help='1200; smoke: 300<stop<1200 with seeds 999990-1')
    st = sub.add_parser('status')
    st.add_argument('--root', required=True)
    y = sub.add_parser('sync')
    y.add_argument('--root', required=True)
    y.add_argument('--dest', default=os.environ.get('DISTILL_DEST'))
    y.add_argument('--label', default=None)
    a = p.parse_args(argv)
    dict(stage=cmd_stage, collect=cmd_collect, status=cmd_status, sync=cmd_sync)[a.cmd](a)


if __name__ == '__main__':
    main()
