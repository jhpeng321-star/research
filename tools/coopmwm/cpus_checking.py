"""cpuS plan checking (MAIN 2026-11-04 ~11:20): exact all8 SUMO verification of the plan bank on cpuS,
training stays on cpuQ. GLUE ONLY - the bank logic is the reviewed code, unchanged:
verify_perseed.confirm_all8 (c4/extra-seeds dd7c7b42 extra_seeds option), bank_builder.quoted_jobs_all8 /
all8_charge / extras_charge, slim_tapes.slim_record (c4/slim-tapes 3a0e19c5), sumo_verify.verify /
SubprocessBranchRunner, on top of the running c4/v2v-resume a7ae6d9a.

ARCHITECTURE (documented choice): coordinator + branch workers run on cpuS against a read-only
REPLICA of the needed warm1 bank inputs at the IDENTICAL absolute path (/home/ubuntu/coopmwm20261102_warm1
on cpuS), so every sealed record/receipt carries the same paths it would have had on cpuQ. Nothing that
owns state is replicated:
  * ledger: tools.coopmwm.accel_storage.reserve (the single write point of every charge hook) and
    storage.ledger are replaced in the cpuS worker by RPCs to cpuQ (tools/coopmwm/cpus_rpc.py), where
    the UNCHANGED reserve writes cpuQ CHARGES.jsonl under its ACCOUNTING.claim lock after the delegate
    cap check under the delegate lock(s). The replica's CHARGES.jsonl is a DIRECTORY, so any unpatched
    local ledger read/write fails loudly instead of forking a second ledger.
  * health: confirm_all8(health=...) runs warm_health.assert_running on cpuQ's real run root (RPC).
  * slimming: confirm_all8(slim_rejected_tapes=False) then slim_tapes.slim_record(bank, sealed,
    archive_dir=<cpuS archive>) - the reviewed migration variant: SLIM.json written, full rejected tapes
    MOVED (sha-verified) to the cpuS archive instead of deleted, i.e. they exist only on cpuS.
  * landing: after sealing, the decision's files (record + its verifications' claims/receipts/alerts +
    branch dirs minus scenarios/ and minus the slimmed full tapes) are pushed to cpuQ with rsync only
    if none exists there, sha256-verified on cpuQ, record LAST (readers see tapes before the record),
    and only while cpuQ keeps >= push_min_free_after_bytes free after the push (registered 9 GiB
    backfill guard); otherwise the sealed decision waits on cpuS (its charges are already on cpuQ).
Scheduling: B and R alternate (started counts differ by <= 1; R stops when B's queue is empty - matched
pace), newest policy version first after the registered tier-0 no_solos decisions; a new decision starts
only if min(delegate cap, phase cap minus phase_reserve_sim_s kept for launch gates) headroom covers
(running+1) x worst_case_sim_s.
STOP: <control>/STOP.json (in-flight decisions finish, pushes continue). Only own child PIDs are waited on.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SHIM = 'libmkl_intel_shim.so'


# ---------------------------------------------------------------- RPC client (cpuS -> cpuQ)

class RpcError(RuntimeError):
    """Transport failure (ssh/parse): the remote outcome is unknown."""


def rpc(cfg, op, payload=None, timeout=600):
    remote = (f"cd /tmp && env PYTHONPATH={cfg['q_pythonpath']} CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "
              f"MKL_NUM_THREADS=1 nice -n 19 {cfg['q_python']} -B {cfg['q_control']}/cpus_rpc.py "
              f"{cfg['q_control']} {op}")
    try:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', cfg['q_host'], remote],
                           input=json.dumps(payload or {}), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RpcError(f'rpc {op} timeout') from error
    if p.returncode != 0:
        raise RpcError(f'rpc {op} exit {p.returncode}: {p.stderr[-500:]}')
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RpcError(f'rpc {op} unparsable output: {p.stdout[-300:]!r} {p.stderr[-300:]!r}') from error


def rpc_retry(cfg, op, payload=None, tries=5, wait=10):
    """Bounded retries for READ-ONLY ops only (never for reserve)."""
    for i in range(tries):
        try:
            return rpc(cfg, op, payload)
        except RpcError:
            if i == tries-1:
                raise
            time.sleep(wait*(i+1))


def remote_reserve(cfg):
    """Drop-in for accel_storage.reserve(root, jobs, phase): same ValueError contract; rows cached."""
    import socket

    def reserve(root, jobs, phase):
        assert Path(root) == Path(cfg['run_root']), (root, cfg['run_root'])
        jobs = [dict(id=j['id'], sim_s=int(j.get('sim_s', 1200))) for j in jobs]
        try:
            v = rpc(cfg, 'reserve', dict(jobs=jobs, phase=phase, host=socket.gethostname()))
        except RpcError as error:
            # Unknown outcome: decide from the authoritative ledger (read-only retries), never re-charge.
            ids = {j['id'] for j in jobs}
            rows = [r for r in rpc_retry(cfg, 'ledger')['rows'] if r['id'] in ids]
            if len(rows) == len(ids):
                reserve.rows.update({r['id']: r for r in rows});return None
            if rows:
                raise RuntimeError(f'partial remote reservation after transport error: {sorted(r["id"] for r in rows)}')
            raise ValueError(f'remote reservation transport failure (nothing charged): {error}') from error
        if not v.get('ok'):
            raise ValueError(v.get('error', 'remote reservation refused'))
        reserve.rows.update({r['id']: r for r in v['rows']})
        return None
    reserve.rows = {}
    return reserve


def remote_ledger(cfg):
    def ledger(root):
        assert Path(root) == Path(cfg['run_root']), (root, cfg['run_root'])
        return rpc_retry(cfg, 'ledger')['rows']
    return ledger


def remote_health(cfg, arm):
    def health():
        v = rpc_retry(cfg, 'health', dict(arm=arm))
        if not v.get('ok'):
            from tools.coopmwm.warm_health import HealthPause
            if v.get('type') == 'HealthPause':
                raise HealthPause(v.get('error'))
            raise ValueError(f'cpuQ health: {v.get("type")}: {v.get("error")}')
    return health


def install_remote_accounting(cfg):
    """Every charge path ends in accel_storage.reserve (sv.accel_charge, bank_builder.extras_charge and
    the worker precharge all import it at call time) -> route it to cpuQ; ledger reads likewise."""
    from tools.coopmwm import accel_storage, storage
    reserve, ledger = remote_reserve(cfg), remote_ledger(cfg)
    accel_storage.reserve = reserve
    accel_storage.ledger = ledger
    storage.ledger = ledger
    return reserve


# ---------------------------------------------------------------- landing (push to cpuQ)

def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def push_plan(bank, record):
    """(files, record_path): every file of this decision's verifications that must land on cpuQ.

    Per charged verification (kind dir == kind name): claims/<v>.json, verifications/<v>.json,
    alerts/BUDGET_BLOCKED_<v>.json and branches/<v>/** EXCEPT scenarios/ (generated SUMO inputs,
    kept on cpuS) - slimmed full tapes are already absent (moved to the cpuS archive). The record
    itself is returned separately and is pushed LAST."""
    bank = Path(bank);files = []
    for c in record['charges']:
        kd, v = bank/c['kind'], c['verification_id']
        for p in (kd/'claims'/f'{v}.json', kd/'verifications'/f'{v}.json', kd/'alerts'/f'BUDGET_BLOCKED_{v}.json'):
            if p.is_file():
                files.append(p)
        bd = kd/'branches'/v
        if bd.is_dir():
            for p in sorted(bd.rglob('*')):
                if p.is_file() and 'scenarios' not in p.relative_to(bd).parts:
                    files.append(p)
    assert len(set(files)) == len(files)
    return files, bank/'confirmations_all8'/(record['verification_id']+'.json')


def push_decision(cfg, arm, vid, log):
    """Land one sealed decision on cpuQ (no overwrite, sha256-verified, record last). Returns a receipt."""
    from tools.coopmwm import replay_bank as rb
    from tools.coopmwm.exact_grade import verify_sealed
    bank = Path(rb.bank_root(Path(cfg['run_root']), arm))
    rec_path = bank/'confirmations_all8'/(vid+'.json')
    record = json.loads(rec_path.read_text());verify_sealed(record, 'confirmation_sha256')
    files, rec_path = push_plan(bank, record)
    rows = [dict(path=str(p), sha256=_sha(p), size=p.stat().st_size) for p in files]
    rec_row = dict(path=str(rec_path), sha256=_sha(rec_path), size=rec_path.stat().st_size)
    total = sum(r['size'] for r in rows)+rec_row['size']
    free = rpc_retry(cfg, 'disk')['free']
    if free-total < int(cfg['push_min_free_after_bytes']):
        return dict(status='held_disk', cpuq_free=free, bytes=total)
    present = rpc_retry(cfg, 'absent', dict(paths=[r['path'] for r in rows]+[rec_row['path']]))['present']
    if present:
        return dict(status='refused_existing', present=present[:20], n_present=len(present))
    lst = Path(cfg['control'])/'push_lists'/f'{arm}_{vid}.txt'
    lst.parent.mkdir(exist_ok=True)

    def send(paths, name):
        lst.with_suffix(f'.{name}.txt').write_text(''.join(str(p).lstrip('/')+'\n' for p in paths))
        cmd = ['rsync', '-a', '--ignore-existing', '--files-from', str(lst.with_suffix(f'.{name}.txt')),
               '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=20', '/', f'{cfg["q_host"]}:/']
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            raise RpcError(f'rsync {name} exit {p.returncode}: {p.stderr[-500:]}')
    started = time.time()
    send(files, 'files')
    bad = rpc_retry(cfg, 'verify', dict(files=rows))['bad']
    if bad:
        return dict(status='verify_failed_files', bad=bad[:20], n_bad=len(bad))
    send([rec_path], 'record')
    bad = rpc_retry(cfg, 'verify', dict(files=[rec_row]))['bad']
    if bad:
        return dict(status='verify_failed_record', bad=bad)
    receipt = dict(status='pushed', arm=arm, verification_id=vid, files=len(rows)+1, bytes=total,
                   record_sha256=rec_row['sha256'], confirmation_sha256=record['confirmation_sha256'],
                   cpuq_free_before=free, wall_s=time.time()-started,
                   manifest_sha256=hashlib.sha256(json.dumps(rows+[rec_row], sort_keys=True).encode()).hexdigest())
    rpc_retry(cfg, 'note', dict(name='PUSHED', row=receipt))
    return receipt


# ---------------------------------------------------------------- worker: one decision

def worker(control, arm, vid, workers):
    control = Path(control);workers = int(workers)
    cfg = json.loads((control/'BINDING.json').read_text())
    assert os.environ.get('LD_PRELOAD', '').endswith(SHIM), 'cpuS bit-parity shim missing (env.sh not sourced)'
    root = Path(cfg['run_root']);phase = cfg['phase'];extra = int(cfg['extra_seeds'])
    att = control/'attempts';att.mkdir(exist_ok=True)
    claim, attempt = att/f'{arm}_{vid}.claim.json', att/f'{arm}_{vid}.json'
    if claim.exists() or attempt.exists():
        raise SystemExit('already attempted; never replay '+vid)
    import torch
    torch.set_num_threads(1)
    os.nice(max(0, 19-os.nice(0)))
    from tools.coopmwm import replay_bank as rb, sumo_verify as sv, verify_perseed as vp, slim_tapes
    from tools.coopmwm.bank_builder import quoted_jobs_all8, all8_charge, extras_charge
    reserve = install_remote_accounting(cfg)
    health = remote_health(cfg, arm)
    bank = Path(rb.bank_root(root, arm))
    r = next(x for x in rb.bank_decisions(root, arm, sources=()) if x['verification_id'] == vid)
    by_vid = {}
    for p in (bank/'decisions').glob('*.json'):
        v = json.loads(p.read_text());by_vid[v['verification_id']] = v
    snapshot = rb._decision_snapshot(root, arm, r, by_vid)
    candidates = vp.first_seed_candidates(r, r['_branches'])
    jobs = quoted_jobs_all8(snapshot, candidates, vid, sv.build_branches, rb.replicate)
    need = sum(j['sim_s'] for j in jobs)
    rec_q = str(bank/'confirmations_all8'/(vid+'.json'))
    if rpc_retry(cfg, 'absent', dict(paths=[rec_q]))['present'] or Path(rec_q).exists():
        with attempt.open('x') as f:
            json.dump(dict(unix=time.time(), arm=arm, verification_id=vid, status='already_sealed', new_sim_s=0), f)
        return
    health()
    with claim.open('x') as f:
        json.dump(dict(unix=time.time(), pid=os.getpid(), arm=arm, verification_id=vid, jobs=jobs,
                       precharged_sim_s=need, verify_mode=vp.ALL8, extra_seeds=extra,
                       slim='slim_record(archive_dir) after seal'), f)
    try:
        reserve(root, jobs, phase)
    except ValueError as error:
        with attempt.open('x') as f:
            json.dump(dict(unix=time.time(), arm=arm, verification_id=vid, status='budget_wait', new_sim_s=0,
                           required_sim_s=need, error=str(error)), f)
        return
    entries = dict(reserve.rows)
    original = rb._confirm_charge
    hook = all8_charge(original, entries)
    if extra:
        hook = extras_charge(hook, snapshot, vid, extra)
    rb._confirm_charge = hook
    slim = None
    try:
        receipt = {k: v for k, v in r.items() if not k.startswith('_')}

        def runner_for(work):
            return sv.SubprocessBranchRunner(Path(work)/'branches', workers=workers,
                                             timeout_s=cfg['branch_timeout_s'])
        out = vp.confirm_all8(root, arm, receipt, snapshot, runner_for, str(root), phase, workers=workers,
                              version=r['_policy_version'], candidates=candidates, branches_dir=r['_branches'],
                              health=health, slim_rejected_tapes=False, extra_seeds=extra)
        slim = slim_tapes.slim_record(bank, out, archive_dir=Path(cfg['archive_dir']))
        ex = out['per_seed_v1']['examples']
        result = dict(unix=time.time(), arm=arm, verification_id=vid, status='complete', verify_mode=vp.ALL8,
                      policy_version=r['_policy_version'], admitted=out['admitted'], winners=out['winners'],
                      admitted_examples=sum(e['admitted'] for e in ex),
                      admitted_examples_seed12=sum(e['admitted'] and e['seed_index'] in (1, 2) for e in ex),
                      cooperative_examples=sum(e.get('cooperative') is True for e in ex),
                      cooperative_examples_seed12=sum(e.get('cooperative') is True and e['seed_index'] in (1, 2)
                                                      for e in ex),
                      extra_seed_examples=sum(e['seed_index'] >= 3 for e in ex),
                      coop_status=[e.get('cooperative_status') for e in ex if e['admitted']],
                      record_sha256=out['confirmation_sha256'], precharged_sim_s=need,
                      extra_group_precharged_sim_s=(sum(j['sim_s'] for j in
                                                        getattr(hook, 'extras_state', {}).get('entries', {}).values())
                                                    if extra else 0),
                      charged_sim_s=out['charged_sim_s'], actual_sim_s=out['actual_sim_s'],
                      slim={k: v for k, v in (slim or {}).items() if k != 'rows'})
    except Exception as error:
        result = dict(unix=time.time(), arm=arm, verification_id=vid, status='failed', verify_mode=vp.ALL8,
                      error=repr(error), traceback=traceback.format_exc(), precharged_sim_s=need, replay=False)
    finally:
        rb._confirm_charge = original
    with attempt.open('x') as f:
        json.dump(result, f, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != 'traceback'}, default=str), flush=True)


# ---------------------------------------------------------------- coordinator

def log(control, **kw):
    with (Path(control)/'EVENTS.jsonl').open('a') as f:
        f.write(json.dumps(dict(unix=time.time(), **kw), default=str)+'\n')


def _attempts(control):
    out = {}
    for p in (Path(control)/'attempts').glob('*.json'):
        if not p.name.endswith('.claim.json'):
            try:
                v = json.loads(p.read_text());out[(v['arm'], v['verification_id'])] = v
            except (ValueError, KeyError):
                pass
    return out


def _max_running(control, default):
    """Live concurrency knob: <control>/CONCURRENCY.json {"max_running": n} (else the BINDING value)."""
    try:
        return int(json.loads((Path(control)/'CONCURRENCY.json').read_text())['max_running'])
    except (OSError, ValueError, KeyError):
        return default


def coordinator(control):
    control = Path(control)
    cfg = json.loads((control/'BINDING.json').read_text())
    assert os.environ.get('LD_PRELOAD', '').endswith(SHIM), 'cpuS bit-parity shim missing (env.sh not sourced)'
    q = json.loads((control/'QUEUE.json').read_text())['queues']
    att = control/'attempts';att.mkdir(exist_ok=True)
    queues = {a: [x for x in q[a] if not (att/f'{a}_{x["verification_id"]}.claim.json').exists()] for a in 'BR'}
    log(control, event='start', B=len(queues['B']), R=len(queues['R']), pid=os.getpid())
    W, maxrun, worst = int(cfg['branch_workers']), int(cfg['max_running']), int(cfg['worst_case_sim_s'])
    started = {a: sum(1 for p in att.glob(f'{a}_*.claim.json')) for a in 'BR'}
    py = sys.executable     # workers run as -m (never a script dir on sys.path[0]); cwd = control/logs
    (control/'logs').mkdir(exist_ok=True)
    running, pushed, stop_reason, last_status, budget = {}, set(), None, 0, None
    for p in (control/'pushed').glob('*.json') if (control/'pushed').exists() else []:
        pushed.add(p.stem)
    (control/'pushed').mkdir(exist_ok=True)
    held_until, budget_t = 0, 0
    while True:
        for pid, (p, arm, vid, t) in list(running.items()):
            if p.poll() is not None:
                log(control, event='worker_exit', pid=pid, arm=arm, verification_id=vid, rc=p.returncode,
                    wall_s=time.time()-t)
                del running[pid]
        # land sealed decisions (at most one per loop pass; disk-held -> retry after push_retry_s)
        unpushed = [(k, v) for k, v in sorted(_attempts(control).items(), key=lambda kv: kv[1].get('unix', 0))
                    if v.get('status') == 'complete' and f'{k[0]}_{k[1]}' not in pushed
                    and not (control/'push_failed'/f'{k[0]}_{k[1]}.json').exists()]
        pushing_off = (control/'STOP_PUSH.json').exists()
        if unpushed and not pushing_off and time.time() >= held_until:
            (arm, vid), v = unpushed[0];key = f'{arm}_{vid}'
            try:
                rc = push_decision(cfg, arm, vid, log)
            except Exception as error:
                rc = dict(status='push_error', error=repr(error))
            if rc['status'] == 'pushed':
                (control/'pushed'/f'{key}.json').write_text(json.dumps(rc));pushed.add(key)
                log(control, event='pushed', arm=arm, verification_id=vid, bytes=rc['bytes'], files=rc['files'])
            elif rc['status'] == 'held_disk':
                held_until = time.time()+int(cfg.get('push_retry_s', 300))
                log(control, event='push_held_disk', arm=arm, verification_id=vid, **rc)
            else:     # never retried automatically: an operator reconciles (no overwrite on cpuQ)
                (control/'push_failed').mkdir(exist_ok=True)
                (control/'push_failed'/f'{key}.json').write_text(json.dumps(rc, default=str))
                log(control, event='push_failed', arm=arm, verification_id=vid, **rc)
        stop = (control/'STOP.json').exists()
        if (stop or stop_reason) and not running and (not unpushed or pushing_off):
            break
        if not stop and not stop_reason:
            arm = 'B' if started['B'] <= started['R'] else 'R'
            if not queues[arm]:
                if not running:
                    stop_reason = f'queue_done ({arm} queue empty; matched pace)'
                    log(control, event='queue_done', arm=arm, started=started)
            elif len(running) < _max_running(control, maxrun) and os.getloadavg()[0] <= cfg['max_load'] and \
                    abs(started['B']-started['R']) <= 1:
                if time.time()-budget_t > 20:
                    try:
                        budget, budget_t = rpc_retry(cfg, 'budget'), time.time()
                    except RpcError as error:
                        budget = None;log(control, event='budget_rpc_error', error=repr(error))
                if budget is not None:
                    # phase_reserve_sim_s: bank_verification_ext kept free for registered launch gates
                    head = min(budget['delegate_cap']-budget['all8_used'],
                               budget['phase_cap']-budget['phase_used']-int(cfg.get('phase_reserve_sim_s', 0)),
                               budget['total_cap']-budget['total_used'])
                    if head < (len(running)+1)*worst:
                        if not running:
                            stop_reason = f'budget headroom {head} < worst case {worst}'
                            log(control, event='budget_stop', headroom=head, budget=budget)
                    else:
                        item = queues[arm].pop(0);vid = item['verification_id']
                        lf = (control/'logs'/f'{arm}_{vid}.log').open('x')
                        p = subprocess.Popen([py, '-B', '-u', '-m', 'tools.coopmwm.cpus_checking', 'worker',
                                              str(control), arm, vid, str(W)],
                                             stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                             cwd=str(control/'logs'))
                        running[p.pid] = (p, arm, vid, time.time());started[arm] += 1;budget_t = 0
                        log(control, event='start_decision', pid=p.pid, arm=arm, verification_id=vid,
                            policy_version=item.get('policy_version'), tier=item.get('tier'), headroom=head,
                            running=len(running), started=dict(started))
                        time.sleep(3)
                        continue
        if time.time()-last_status > 60:
            last_status = time.time()
            a = _attempts(control)
            s = dict(unix=time.time(), running=len(running), running_pids=sorted(running), started=started,
                     queued={k: len(v) for k, v in queues.items()}, load1=os.getloadavg()[0],
                     cpus_free=shutil.disk_usage('/home/ubuntu').free, budget=budget, stop_reason=stop_reason,
                     complete={k: sum(v['status'] == 'complete' and v['arm'] == k for v in a.values()) for k in 'BR'},
                     failed=sum(v['status'] == 'failed' for v in a.values()),
                     pushed=len(pushed),
                     admitted_examples={k: sum(v.get('admitted_examples', 0) for v in a.values() if v['arm'] == k)
                                        for k in 'BR'},
                     cooperative_examples={k: sum(v.get('cooperative_examples', 0) for v in a.values()
                                                  if v['arm'] == k) for k in 'BR'},
                     extra_seed_examples={k: sum(v.get('extra_seed_examples', 0) for v in a.values()
                                                 if v['arm'] == k) for k in 'BR'},
                     charged_sim_s=sum(v.get('charged_sim_s', 0) or 0 for v in a.values()))
            tmp = control/'STATUS.json.tmp'
            tmp.write_text(json.dumps(s, indent=1, default=str));tmp.replace(control/'STATUS.json')
        time.sleep(5)
    log(control, event='exit', stop=(control/'STOP.json').exists(), reason=stop_reason, pushed=len(pushed))


if __name__ == '__main__':
    if sys.argv[1] == 'worker':
        worker(*sys.argv[2:6])
    elif sys.argv[1] == 'coordinator':
        coordinator(sys.argv[2])
    elif sys.argv[1] == 'push':
        cfg = json.loads((Path(sys.argv[2])/'BINDING.json').read_text())
        print(json.dumps(push_decision(cfg, sys.argv[3], sys.argv[4], None), default=str))
