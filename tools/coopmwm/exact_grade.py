"""Exact SUMO grading of every executed world-model plan (owner lever 5, MAIN 2026-11-02 02:24).

grade = J_without - J_with over [t0, t_end], t_end = min(t0 + D + 120, episode end), where
J = increase of the cumulative source-inclusive team delay (``team_delay_vs``: on-road speed
deficit + source waiting, the comparator J) read from per-second accel_plant fingerprints.
Positive grade = the plan helped. Units: veh-s.

- with-plan: the REAL training episode (its RESULT.json fingerprints); no new physics.
- without-plan: exact rerun of the same episode (same world, policy checkpoint, noise seed,
  runtime) from t=300 to t0 by replaying the recorded one-second samples (the accel_plant
  replay path used by exact_plan); every fingerprint 300..t0 must equal the original
  (PrefixMismatch -> STOP receipt, never imputed). From t0 the controller samples the
  policy live with NO plan and NO guide: the group (and everyone else) follow the policy's
  own actions. The torch generator is fast-forwarded through the original's exact draw
  schedule (policy AR(1) innovations + guide query draw + R ticket draw + beta draw) and
  the AR noise memory is restored from the recorded t0 sample, so the branch reuses the
  original exploration stream (common random numbers). Verified bitwise at t0: the live
  own action must equal the recorded own action. After t0 the original's guide/beta draws
  are consumed at the same seconds (never acted on), so the stream stays aligned while
  the CAV membership is unchanged; ``aligned_until_s`` records the first divergence.
  Registered fallback (noise_mode='mean') if the t0 check fails: plan members use the
  policy mean action, everyone else keeps sampling; recorded in the receipt.
- SUMO itself is behind an injectable runner (AccelPlantGradeRunner = real executor;
  CPU tests use a fake deterministic plant).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from .branch_eval import totals_at
from .protocol import canonical, digest

SCHEMA = 'coopmwm/exact_grade/v1'
EXPERIMENT = 'S-COOPMWM-EXACT-GRADE'
WARMUP_S = 300
EPISODE_END_S = 1200
TAIL_S = 120
METRIC = 'team_delay_vs'
NOISE_MODES = ('crn', 'mean')
MAX_ATTEMPTS = 2           # orphaned (dead worker) attempts only; physics/executor errors never retry
T_FLOOR_VEH_S = 1.
WEIGHT_CLIP = 20.          # same registered clip as the AWR amendment
HARM_MIN_PLANS = 30        # owner health guard 3b
BOOT_SEED = 20261101
N_BOOT = 10000
STATUSES = ('graded', 'stopped', 'failed', 'overlap_unresolved')
N_FREEZE_TEMPERATURE = 20  # MAIN 2026-11-02: uniform weights before, then T frozen from the first 20 grades


class PrefixMismatch(ValueError):
    """The rerun did not reproduce the original episode bit-for-bit before the plan start."""


class BudgetBlocked(RuntimeError):
    """The accounting layer refused the charge; the grade is not run (and not imputed)."""


def sha_value(value):
    return digest(value)


def _seal(record, key):
    body = {k: v for k, v in record.items() if k != key}
    return dict(body, **{key: digest(body)})


def verify_sealed(record, key):
    body = {k: v for k, v in record.items() if k != key}
    if record.get(key) != digest(body):
        raise ValueError(f'{key} mismatch (edited after sealing?)')
    return True


# ---------------------------------------------------------------- job specs

def grade_id(episode_id, t0):
    return f'xg_{episode_id}_t{int(t0):04d}'


def executed_plan_steps(result):
    """(index, step) for every executed plan START in an original training episode."""
    out = []
    for i, s in enumerate(result['steps']):
        b = s['sample'].get('behaviour') or {}
        if b.get('executed') and b.get('plan_start'):
            if not (s.get('proposal') or {}).get('plan'):
                raise ValueError(f'executed plan at t={s["time_s"]} has no recorded plan')
            out.append((i, s))
    return out


def episode_end_s(result):
    return int(float(result['final']['t_s']))


def window_end(t0, duration_s, end_s, tail_s=TAIL_S):
    return min(int(t0) + int(duration_s) + tail_s, int(end_s))


def train_noise_seed(world_seed, update, seed):
    """Same formula as accel_runner.wave for training waves ident=f'warm_u{u:03}'."""
    return int(digest(dict(seed=seed, world=world_seed, wave=f'warm_u{int(update):03}'))[:15], 16)


def job_specs(result, meta, noise_mode='crn'):
    """One sealed spec per executed plan in an original RESULT.

    meta: run_root, result_path, result_sha256 (optional; else digest of result), update,
    arm ('B'|'R'), checkpoint_path, checkpoint_sha256, checkpoint_update (= update-1),
    noise_seed, runtime, sumo_checkpoint, beta, warm_history, accel_bounds, role.
    """
    if meta['arm'] not in ('B', 'R'):
        raise ValueError('only arms B and R execute plans')
    if noise_mode not in NOISE_MODES:
        raise ValueError('unregistered noise mode')
    end = episode_end_s(result)
    starts = executed_plan_steps(result)
    result_sha = meta.get('result_sha256') or digest(result)
    specs = []
    for _, s in starts:
        t0 = int(s['time_s']);plan = s['proposal']['plan'];b = s['sample']['behaviour']
        D = int(plan['duration_s'])
        if D <= 0:
            raise ValueError('executed plan duration missing')
        t_end = window_end(t0, D, end)
        later = [int(x['time_s']) for _, x in starts if t0 < int(x['time_s']) < t_end]
        others = other_plans(result, t0, D, t_end)
        clash = sorted({v for o in others for v in o['members']} & set(plan['members']))
        spec = dict(
            schema=SCHEMA, experiment=EXPERIMENT, grade_id=grade_id(result['id'], t0),
            episode=dict(id=result['id'], world=result['world'], arm=meta['arm'], update=int(meta['update']),
                         run_root=str(meta.get('run_root', '')), result_path=str(meta.get('result_path', '')),
                         result_sha256=result_sha, end_s=end, role=meta.get('role', 'train'),
                         training_seed=meta.get('training_seed')),
            policy=dict(checkpoint_path=str(meta['checkpoint_path']), checkpoint_sha256=meta['checkpoint_sha256'],
                        checkpoint_update=int(meta.get('checkpoint_update', int(meta['update'])-1))),
            rng=dict(noise_seed=int(meta['noise_seed']), generator='torch.Generator(cpu).manual_seed(noise_seed)',
                     prefix='recorded one-second samples replayed 300..t0-1; generator fast-forwarded through the '
                            'original draw schedule', noise_mode=noise_mode),
            sumo=dict(runtime=meta.get('runtime'), checkpoint=meta.get('sumo_checkpoint'),
                      warm_history=bool(meta.get('warm_history', False)), accel_bounds=meta.get('accel_bounds'),
                      beta=meta.get('beta', 1.)),
            plan=dict(id=plan['id'], members=list(plan['members']), duration_s=D, start_s=t0,
                      option=b.get('option', plan.get('family')), regime=b.get('regime'),
                      expected_saving=s['proposal'].get('expected_saving'),
                      plan_sha256=digest(plan)),
            window=dict(start_s=t0, end_s=t_end, nominal_end_s=t0+D+TAIL_S, tail_s=TAIL_S,
                        truncated=t_end < t0+D+TAIL_S, metric=METRIC),
            overlapping_plan_starts=later,
            replayed_plans=others,
            overlap_unresolved=bool(clash), overlap_clash_members=clash,
            sim_s=0 if clash else budget_sim_s_window(t_end),
        )
        specs.append(_seal(spec, 'spec_sha256'))
    return specs


def other_plans(result, t0, duration_s, t_end):
    """Every OTHER executed plan active in [t0, t_end) of the original episode (MAIN overlap rule):
    replayed in the without-plan branch exactly as recorded (per-second absolute accelerations)."""
    out = {}
    for s in result['steps']:
        t = int(s['time_s']);b = s['sample'].get('behaviour') or {}
        if not (t0 <= t < t_end) or t0 <= t < t0+duration_s or 'absolute_accels' not in s['sample']:
            continue
        pid = b.get('plan_id');members = list(b.get('proposal_members') or s['sample']['absolute_accels'])
        o = out.setdefault((pid, tuple(members)), dict(id=pid, members=members, first_s=t, last_s=t, seconds=0))
        o['last_s'] = t;o['seconds'] += 1
    return sorted(out.values(), key=lambda o: o['first_s'])


def budget_sim_s_window(t_end):
    # Charged like every accel_plant job: simulated seconds from t=0 to the stop (warmup included).
    return int(t_end)


def budget_sim_s(spec):
    return budget_sim_s_window(spec['window']['end_s'])


def quote(plans, end_s=EPISODE_END_S, tail_s=TAIL_S):
    """plans: iterable of (start_s, duration_s). Returns total charged sim-s (one branch each)."""
    return int(sum(window_end(t, d, end_s, tail_s) for t, d in plans))


# ---------------------------------------------------------------- RNG schedule

def guide_draws(step, arm):
    """Generator calls the original guide + controller issued AFTER policy.sample at this step.

    plan_client.guide: one torch.rand per eligible query (B and R reach it whenever
    behaviour.eligible_query is recorded); arm R draws torch.randint for the matched random
    plan (recorded proposal has random_match and accepted). acceleration.Controller: one
    torch.rand (beta) for every accepted proposal.
    """
    calls = []
    b = step['sample'].get('behaviour') or {}
    if arm in ('B', 'R') and b.get('eligible_query'):
        calls.append('rand')
    p = step.get('proposal')
    if p is not None and p.get('accepted') and p.get('random_match') is not None:
        calls.append('randint')
    if p is not None and p.get('accepted'):
        calls.append('rand')
    return calls


def issue(generator, call):
    if call == 'rand':
        torch.rand((), generator=generator)
    elif call == 'randint':
        torch.randint(2**31, (), generator=generator)
    else:
        raise ValueError(call)


def policy_draws(generator, n_members):
    # acceleration.Policy.sample: common innovation (1,1) then member innovations (n,1).
    torch.randn((1, 1), generator=generator)
    torch.randn((n_members, 1), generator=generator)


def fast_forward(generator, steps, arm):
    for s in steps:
        if s['sample'].get('deterministic'):
            raise ValueError('training episode steps must be stochastic')
        policy_draws(generator, len(s['sample']['context']['members']))
        for c in guide_draws(s, arm):
            issue(generator, c)


# ---------------------------------------------------------------- without-plan controller

def grade_controller_class():
    from .acceleration import Controller, requests

    class GradeController(Controller):
        """Replays the recorded prefix, then the graded plan's members follow the policy (no guide);
        every OTHER plan active after t0 is replayed from its recorded per-second accelerations."""

        def __init__(self, policy, generator, original_steps, t0, plan_members, arm, noise_mode='crn',
                     duration_s=0):
            super().__init__(policy, generator, deterministic=False, guide=None, beta=0., replay=None, branch=None)
            self.by_time = {int(s['time_s']): s for s in original_steps}
            self.t0, self.plan_members, self.arm, self.mode = int(t0), list(plan_members), arm, noise_mode
            self.duration = int(duration_s)
            missing = [t for t in range(WARMUP_S, self.t0+1) if t not in self.by_time]
            if missing:
                raise ValueError(f'original tape lacks seconds {missing[:3]}')
            fast_forward(generator, [self.by_time[t] for t in range(WARMUP_S, self.t0)], arm)
            self.audit = dict(noise_mode_requested=noise_mode, noise_mode_used=None, crn_verified_t0=None,
                              crn_failure=None, aligned_until_s=None, live_steps=0,
                              replayed_prefix_steps=0, without_plan_of=None, replayed_plans={},
                              replay_skipped_member_seconds=0)
            self._aligned = True

        def apply(self, rows, commands, now):
            t = int(now)
            if t < self.t0:
                self.replay = {str(t): self.by_time[t]['sample']}
                try:
                    desired = super().apply(rows, commands, now)
                finally:
                    self.replay = None
                self.audit['replayed_prefix_steps'] += 1
                return desired
            if t == self.t0:
                rec = self.by_time[t]['sample']
                self.memory = dict(zip(rec['context']['members'], [float(x[0]) for x in rec['previous_noise']]))
                self.audit['without_plan_of'] = (self.by_time[t].get('proposal') or {}).get('plan', {}).get('id')
            desired = super().apply(rows, commands, now)
            event = self.events[-1];sample = event['sample']
            if self.plan is not None or sample.get('absolute_accels'):
                raise ValueError('without-plan branch executed a plan')
            if t == self.t0:
                rec = self.by_time[t]['sample']
                ok = (sample['context']['members'] == rec['context']['members']
                      and sample['own_action'] == rec['own_action'])
                self.audit['crn_verified_t0'] = bool(ok)
                if not ok and self.mode == 'crn':
                    self.mode = 'mean'
                    self.audit['crn_failure'] = 'live own action at t0 differs from the recorded own action'
                self.audit['noise_mode_used'] = self.mode
            orig = self.by_time.get(t)
            changed = False
            if self.mode == 'mean':
                with torch.no_grad():
                    mean, _ = self.policy(sample['context'])
                for i, v in enumerate(sample['context']['members']):
                    if v in self.plan_members:
                        sample['action'][i] = [float(mean[i, 0])]
                changed = True
            replayed = None
            if orig is not None and not (self.t0 <= t < self.t0+self.duration) and 'absolute_accels' in orig['sample']:
                accels = dict(orig['sample']['absolute_accels'])
                if set(accels) & set(self.plan_members):
                    raise ValueError('other-plan replay touches the graded plan members (overlap_unresolved)')
                sample['absolute_accels'] = accels
                members = sample['context']['members']
                for v, raw in accels.items():
                    if v in members:
                        sample['action'][members.index(v)] = [raw-commands[v]]
                replayed = (orig['sample'].get('behaviour') or {}).get('plan_id')
                key = str(replayed);self.audit['replayed_plans'][key] = self.audit['replayed_plans'].get(key, 0)+1
                self.audit['replay_skipped_member_seconds'] += sum(v not in commands for v in accels)
                changed = True
            if changed:
                desired = requests(sample, commands)
                event['requests'] = desired
            if self._aligned and (orig is None or orig['sample']['context']['members'] != sample['context']['members']):
                self._aligned = False
                self.audit['aligned_until_s'] = t
            if orig is not None:
                for c in guide_draws(orig, self.arm):
                    issue(self.generator, c)
            sample['behaviour'] = dict(sample.get('behaviour') or {}, route='plan_replay' if replayed else 'policy',
                                       executed=False, replayed_plan_id=replayed, exact_grade_branch='without_plan')
            self.audit['live_steps'] += 1
            return desired

        def finish(self, stop):
            if self._aligned:
                self.audit['aligned_until_s'] = int(stop)
            return self.audit

    return GradeController


# ---------------------------------------------------------------- grading

def check_prefix(branch, original, t0, warmup_s=WARMUP_S):
    for t in range(warmup_s, int(t0)+1):
        mine, ref = branch['prefixes'].get(str(t)), original['prefixes'].get(str(t))
        if mine is None or ref is None or canonical(mine) != canonical(ref):
            raise PrefixMismatch(f'fingerprint differs from the original episode at t={t}')
    if canonical(totals_at(branch, t0)) != canonical(totals_at(original, t0)):
        raise PrefixMismatch('cumulative totals differ at the plan start')
    return True


def window_totals(result, start, end):
    a, b = totals_at(result, start), totals_at(result, end)
    return {k: b[k]-a[k] for k in a}


def grade(spec, original, runner):
    """Run one without-plan branch and return the graded receipt body (unsealed).

    Raises PrefixMismatch (caller writes a STOP receipt)."""
    verify_sealed(spec, 'spec_sha256')
    if spec.get('overlap_unresolved'):
        raise ValueError('overlap_unresolved specs are never run')
    if original['id'] != spec['episode']['id']:
        raise ValueError('original episode identity mismatch')
    t0, t_end = spec['window']['start_s'], spec['window']['end_s']
    if t_end > episode_end_s(original):
        raise ValueError('window exceeds the original episode')
    step = next((s for s in original['steps'] if int(s['time_s']) == t0), None)
    b = (step or {}).get('sample', {}).get('behaviour') or {}
    if not (b.get('executed') and b.get('plan_start')) or step['proposal']['plan']['id'] != spec['plan']['id']:
        raise ValueError('spec plan start not found in the original tape')
    branch = runner.run(spec, original)
    check_prefix(branch, original, t0)
    audit = branch.get('audit') or {}
    if 'replayed_plans' in audit:
        expected = {}
        for o in spec.get('replayed_plans', []):
            o_end = min(o['last_s'], t_end-1)
            n = sum(1 for s in original['steps'] if o['first_s'] <= int(s['time_s']) <= o_end
                    and 'absolute_accels' in s['sample'] and not (t0 <= int(s['time_s']) < t0+spec['plan']['duration_s']))
            expected[str(o['id'])] = expected.get(str(o['id']), 0)+n
        if audit['replayed_plans'] != {k: v for k, v in expected.items() if v}:
            raise ValueError('other-plan replay does not match the recorded plans')
    for s in branch.get('steps', []):
        if int(s['time_s']) >= t0 and (s['sample'].get('behaviour') or {}).get('executed'):
            raise ValueError('without-plan branch recorded an executed plan step')
    w_with, w_without = window_totals(original, t0, t_end), window_totals(branch, t0, t_end)
    J_with, J_without = w_with[METRIC], w_without[METRIC]
    return dict(status='graded', grade_veh_s=J_without-J_with, J_with_veh_s=J_with, J_without_veh_s=J_without,
                window_with=w_with, window_without=w_without,
                noise=branch.get('audit', {}), prefix=dict(checked_from_s=WARMUP_S, checked_to_s=t0, pass_=True),
                actual_sim_s=branch.get('actual_sim_s'))


def receipt(spec, body, attempt, started, **extra):
    r = dict(schema=SCHEMA, grade_id=spec['grade_id'], spec_sha256=spec['spec_sha256'],
             episode_id=spec['episode']['id'], world_seed=spec['episode']['world']['seed'],
             result_path=spec['episode']['result_path'], training_seed=spec['episode'].get('training_seed'),
             arm=spec['episode']['arm'], update=spec['episode']['update'], plan=spec['plan'],
             window=spec['window'], overlapping_plan_starts=spec['overlapping_plan_starts'],
             replayed_plans=spec.get('replayed_plans', []), overlap_clash_members=spec.get('overlap_clash_members', []),
             charged_sim_s=spec['sim_s'], attempt=attempt, started_unix=started, finished_unix=time.time(),
             host=socket.gethostname(), pid=os.getpid(), sign='grade = J_without - J_with; positive = plan helped',
             **body, **extra)
    if r['status'] not in STATUSES:
        raise ValueError('unknown status')
    return _seal(r, 'receipt_sha256')


# ---------------------------------------------------------------- durable queue

def _write_new(path, value):
    """Durable no-overwrite JSON write: tmp + fsync + hard link (fails if the target exists)."""
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _proc_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        return None


def default_alive(claim):
    """Never signals a process. Foreign host or no /proc -> assume alive (never steal)."""
    if claim.get('host') != socket.gethostname() or not Path('/proc').exists():
        return True
    return _proc_start(claim['pid']) is not None and _proc_start(claim['pid']) == claim.get('proc_start')


def local_charge(root):
    """Append-only local ledger (tests / dry runs). Production: accel_charge."""
    def charge(spec, attempt):
        with (Path(root)/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            f.write(canonical(dict(id=f'{spec["grade_id"]}_a{attempt}', phase='exact_grade',
                                   sim_s=spec['sim_s']))+'\n');f.flush();os.fsync(f.fileno())
    return charge


def accel_charge(run_root, phase='exact_grade'):
    """Charge into the live run ledger under the registered exact_grade phase (Linux, fcntl)."""
    def charge(spec, attempt):
        from .warm_protocol import install_accounting
        from .accel_storage import reserve
        install_accounting()
        try:
            reserve(Path(run_root), [dict(id=f'{spec["grade_id"]}_a{attempt}', sim_s=spec['sim_s'])], phase)
        except ValueError as e:
            raise BudgetBlocked(str(e)) from e
    return charge


class GradeQueue:
    """queue/<id>.json spec | claims/<id>.json live claim | attempts/<id>.a<n>.json history |
    receipts/<id>.json final (graded|stopped|failed). Receipts are never overwritten."""

    def __init__(self, root, charge=None, alive=default_alive, max_attempts=MAX_ATTEMPTS):
        self.root = Path(root)
        for d in ('queue', 'claims', 'attempts', 'receipts'):
            (self.root/d).mkdir(parents=True, exist_ok=True)
        self.charge = charge or local_charge(self.root)
        self.alive, self.max_attempts = alive, max_attempts

    def enqueue(self, spec):
        verify_sealed(spec, 'spec_sha256')
        p = self.root/'queue'/f'{spec["grade_id"]}.json'
        if p.exists():
            if _read(p)['spec_sha256'] != spec['spec_sha256']:
                raise ValueError('a different spec is already queued under '+spec['grade_id'])
            return False
        _write_new(p, dict(spec, enqueued_unix=time.time()))
        return True

    def enqueue_episode(self, result, meta, noise_mode='crn'):
        return [s['grade_id'] for s in job_specs(result, meta, noise_mode) if self.enqueue(s)]

    def spec(self, gid):
        s = _read(self.root/'queue'/f'{gid}.json');s.pop('enqueued_unix', None)
        verify_sealed(s, 'spec_sha256')
        return s

    def receipt(self, gid):
        p = self.root/'receipts'/f'{gid}.json'
        return _read(p) if p.exists() else None

    def attempts(self, gid):
        return sorted(self.root.glob(f'attempts/{gid}.a*.json'))

    def status(self, gid):
        r = self.receipt(gid)
        if r is not None:
            return r['status']
        c = self.root/'claims'/f'{gid}.json'
        if c.exists():
            return 'running' if self.alive(_read(c)) else 'orphaned'
        return 'pending'

    def ids(self):
        return sorted(p.stem for p in (self.root/'queue').glob('*.json'))

    def recover(self):
        """Move dead-worker claims to attempts/ (preserved, charged); the grade becomes pending again
        until MAX_ATTEMPTS is reached, then a 'failed' receipt is written. Never kills anything."""
        moved = []
        for c in sorted((self.root/'claims').glob('*.json')):
            claim = _read(c);gid = c.stem
            if self.receipt(gid) is not None or self.alive(claim):
                continue
            target = self.root/'attempts'/f'{gid}.a{claim["attempt"]}.json'
            _write_new(target, dict(claim, outcome='orphaned', recovered_unix=time.time()))
            os.unlink(c);moved.append(gid)
            if len(self.attempts(gid)) >= self.max_attempts:
                spec = self.spec(gid)
                _write_new(self.root/'receipts'/f'{gid}.json', receipt(spec, dict(
                    status='failed', grade_veh_s=None, reason='worker died on every attempt; not imputed'),
                    claim['attempt'], claim['started_unix']))
        return moved

    def pending(self):
        return [g for g in self.ids() if self.status(g) == 'pending']

    def claim(self, gid):
        attempt = len(self.attempts(gid))+1
        if attempt > self.max_attempts:
            raise ValueError('attempts exhausted for '+gid)
        claim = dict(grade_id=gid, attempt=attempt, pid=os.getpid(), host=socket.gethostname(),
                     proc_start=_proc_start(os.getpid()), started_unix=time.time())
        fd = os.open(self.root/'claims'/f'{gid}.json', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(canonical(claim)+'\n');f.flush();os.fsync(f.fileno())
        return claim

    def run_one(self, gid, runner, load_original):
        if self.receipt(gid) is not None:
            return self.receipt(gid)              # idempotent: graded/stopped/failed are final
        spec = self.spec(gid)
        if spec.get('overlap_unresolved'):     # MAIN overlap rule: no physics, no charge, never imputed
            r = receipt(spec, dict(status='overlap_unresolved', grade_veh_s=None,
                                   reason='another executed plan in the window shares members with this plan'),
                        0, time.time())
            _write_new(self.root/'receipts'/f'{gid}.json', r)
            return r
        try:
            claim = self.claim(gid)
        except FileExistsError:
            return None                           # another owned worker holds it
        cpath = self.root/'claims'/f'{gid}.json'
        try:
            self.charge(spec, claim['attempt'])   # durable charge BEFORE any physics
        except BudgetBlocked as e:
            _write_new(self.root/'attempts'/f'{gid}.a{claim["attempt"]}.json',
                       dict(claim, outcome='budget_blocked', error=str(e), charged=False))
            os.unlink(cpath)
            if not (self.root/'GRADER_BLOCKED.json').exists():
                _write_new(self.root/'GRADER_BLOCKED.json', dict(grade_id=gid, error=str(e), unix=time.time()))
            raise
        try:
            body = grade(spec, load_original(spec), runner)
        except PrefixMismatch as e:
            body = dict(status='stopped', grade_veh_s=None, reason='PrefixMismatch: '+str(e),
                        prefix=dict(checked_from_s=WARMUP_S, checked_to_s=spec['window']['start_s'], pass_=False))
        except Exception as e:  # physics/executor error: a STOP for this grade, never retried or imputed
            body = dict(status='failed', grade_veh_s=None, reason=repr(e))
        r = receipt(spec, body, claim['attempt'], claim['started_unix'])
        _write_new(self.root/'receipts'/f'{gid}.json', r)
        _write_new(self.root/'attempts'/f'{gid}.a{claim["attempt"]}.json', dict(claim, outcome=r['status']))
        os.unlink(cpath)
        return r

    def drain(self, runner, load_original, limit=None):
        done = []
        for gid in self.pending()[:limit]:
            r = self.run_one(gid, runner, load_original)
            if r is not None:
                done.append(r)
        return done

    def frozen_temperature(self, arm, training_seed=None):
        """Durable MAIN temperature: uniform until the 20th exact grade of this arm/seed, then frozen
        (first write wins; TEMPERATURE_<arm>_<seed>.json is never rewritten)."""
        path = self.root/f'TEMPERATURE_{arm}_{training_seed}.json'
        if path.exists():
            return temperature([], arm, training_seed, frozen=_read(path))
        t = temperature(self.receipts(), arm, training_seed)
        if t['temperature_veh_s'] is not None:
            try:
                _write_new(path, t)
            except FileExistsError:
                return temperature([], arm, training_seed, frozen=_read(path))
        return t

    def receipts(self):
        out = []
        for p in sorted((self.root/'receipts').glob('*.json')):
            r = _read(p);verify_sealed(r, 'receipt_sha256');out.append(r)
        return out


def supervise(root, workers=4, poll_s=10., python=None, once=False, extra_args=()):
    """cpuQ supervisor: bounded subprocess workers (nice 19), owned PIDs logged; never kills.

    Each worker: python -m tools.coopmwm.exact_grade run-one --root R --grade-id G.
    Resume-safe: on (re)start it recovers dead claims and only launches pending grades."""
    import subprocess
    q = GradeQueue(root, charge=lambda s, a: None)
    running = {}
    log = Path(root)/'WORKERS.jsonl'
    def nice():
        os.nice(max(0, 19-os.nice(0)))
    while True:
        if (Path(root)/'GRADER_BLOCKED.json').exists() or (Path(root)/'GRADER_STOP.json').exists():
            break
        for gid, p in list(running.items()):
            if p.poll() is not None:
                running.pop(gid)
        q.recover()
        for gid in q.pending():
            if len(running) >= workers:
                break
            if gid in running:
                continue
            p = subprocess.Popen([python or sys.executable, '-m', 'tools.coopmwm.exact_grade', 'run-one',
                                  '--root', str(root), '--grade-id', gid, *extra_args],
                                 preexec_fn=nice if os.name == 'posix' else None)
            running[gid] = p
            with log.open('a', encoding='utf-8') as f:
                f.write(canonical(dict(pid=p.pid, grade_id=gid, host=socket.gethostname(), unix=time.time()))+'\n')
        if once and not running:
            break
        time.sleep(poll_s)
    for p in running.values():
        p.wait()


# ---------------------------------------------------------------- real executor adapter

@contextmanager
def grade_controller(original_steps, spec):
    """Scoped swap of accel_plant.Controller in an isolated worker process."""
    from . import accel_plant
    cls = grade_controller_class()
    holder = {}
    def factory(policy, generator, deterministic=False, guide=None, beta=1., replay=None, branch=None):
        if guide is not None or replay is not None:
            raise ValueError('grade episodes must not construct a live guide or replay tape')
        holder['controller'] = cls(policy, generator, original_steps, spec['window']['start_s'],
                                   spec['plan']['members'], spec['episode']['arm'], spec['rng']['noise_mode'],
                                   spec['plan']['duration_s'])
        return holder['controller']
    original = accel_plant.Controller
    accel_plant.Controller = factory
    try:
        yield holder
    finally:
        accel_plant.Controller = original


class AccelPlantGradeRunner:
    """Adapter onto tools.coopmwm.accel_plant.episode (cpuQ / GPU-host CPUs). Not run on Windows."""

    def __init__(self, work_root, path_map=None):
        self.work_root, self.path_map = Path(work_root), dict(path_map or {})

    def run(self, spec, original):
        from .accel_learner import load
        from .protocol import sha
        from .accel_plant import episode
        pol = spec['policy']
        ckpt = remap(pol['checkpoint_path'], self.path_map)
        if sha(ckpt) != pol['checkpoint_sha256']:
            raise ValueError('rollout checkpoint SHA mismatch')
        policy, _, _, payload = load(Path(ckpt))
        if int(payload['update']) != pol['checkpoint_update']:
            raise ValueError('rollout checkpoint update mismatch')
        t0, t_end, ep = spec['window']['start_s'], spec['window']['end_s'], spec['episode']
        n = 1+sum(1 for _ in self.work_root.glob(f'jobs/{spec["grade_id"]}/attempt*'))
        job = dict(id=f'{spec["grade_id"]}_a{n}', path=str(self.work_root/'jobs'/spec['grade_id']/f'attempt{n}'),
                   world=ep['world'], role=ep['role'], arm=ep['arm'], weights=policy.state_dict(),
                   runtime=spec['sumo']['runtime'], checkpoint=spec['sumo']['checkpoint'],
                   noise_seed=spec['rng']['noise_seed'], deterministic=True, guided=False, health_guards=False,
                   sim_s=t_end, stop=t_end, beta=spec['sumo']['beta'], warm_history=spec['sumo']['warm_history'],
                   accel_bounds=spec['sumo']['accel_bounds'],
                   branch=dict(time_s=t0, takeover_s=t0, revert=[],
                               prefixes={str(t): original['prefixes'][str(t)] for t in range(WARMUP_S, t0+1)}))
        with grade_controller(original['steps'], spec) as holder:
            try:
                result = episode(job)
            except ValueError as e:
                if 'prefix mismatch' in str(e):
                    raise PrefixMismatch(str(e)) from e
                raise
        result['audit'] = holder['controller'].finish(t_end)
        return result


def remap(path, path_map):
    """Host path translation (identity stays the sealed SHA): {'/gpu/root': '/cpuq/mirror'}."""
    path = str(path)
    for src, dst in sorted((path_map or {}).items(), key=lambda kv: -len(kv[0])):
        if path.startswith(src):
            return dst+path[len(src):]
    return path


def load_original_from_disk(spec, path_map=None):
    from .accel_storage import read
    result = read(Path(remap(spec['episode']['result_path'], path_map)))
    if digest(result) != spec['episode']['result_sha256']:
        raise ValueError('original RESULT digest mismatch')
    return result


# ---------------------------------------------------------------- learning / trust (pure)

def graded(receipts):
    return [r for r in receipts if r['status'] == 'graded']


def _same(r, arm, training_seed):
    return r['arm'] == arm and (training_seed is None or r.get('training_seed') == training_seed)


def temperature(receipts, arm, training_seed=None, frozen=None, n_freeze=N_FREEZE_TEMPERATURE, floor=T_FLOOR_VEH_S):
    """MAIN rule: before n_freeze exact grades exist in this arm/seed -> uniform weights (T=None);
    at the n_freeze-th grade T = max(floor, RMS of those FIRST n_freeze grades, both signs, ordered by
    receipt completion time) is frozen for the rest of the run. Pass the persisted record as ``frozen``."""
    if frozen is not None:
        verify_sealed(frozen, 'temperature_sha256')
        if frozen['arm'] != arm or frozen.get('training_seed') != training_seed:
            raise ValueError('frozen temperature belongs to another arm/seed')
        return frozen
    g = sorted((r for r in graded(receipts) if _same(r, arm, training_seed)),
               key=lambda r: (r['finished_unix'], r['grade_id']))
    if len(g) < n_freeze:
        return dict(rule='uniform_before_freeze', arm=arm, training_seed=training_seed, n=len(g),
                    n_freeze=n_freeze, temperature_veh_s=None)
    first = [float(r['grade_veh_s']) for r in g[:n_freeze]]
    return _seal(dict(rule='frozen_rms_first_exact_grades', arm=arm, training_seed=training_seed, n_freeze=n_freeze,
                      floor_veh_s=floor, grade_ids=[r['grade_id'] for r in g[:n_freeze]], grades_veh_s=first,
                      temperature_veh_s=max(floor, float(np.sqrt(np.mean(np.square(first)))))),
                 'temperature_sha256')


def weight(grade_veh_s, temperature_veh_s, clip=WEIGHT_CLIP):
    """0 for non-positive grades; 1 (uniform) before the freeze; else min(clip, exp(g/T))."""
    if not math.isfinite(grade_veh_s):
        raise ValueError('nonfinite exact grade')
    if grade_veh_s <= 0:
        return 0.
    if temperature_veh_s is None:
        return 1.
    if temperature_veh_s <= 0 or clip < 1:
        raise ValueError('invalid registered temperature/clip')
    z = grade_veh_s/temperature_veh_s
    return float(clip) if z >= math.log(clip) else math.exp(z)


def imitation_weights(receipts, arm, temperature_record, training_seed=None, clip=WEIGHT_CLIP):
    """Buffer of ALL exact-positive graded plans of this arm/seed; every update imitates the whole buffer.
    Stopped / failed / overlap_unresolved / ungraded / non-positive plans are never imitated."""
    T = temperature_record['temperature_veh_s'];out = {}
    counts = dict(graded=0, positive=0, nonpositive=0, stopped=0, failed=0, overlap_unresolved=0, other_arm=0)
    for r in receipts:
        verify_sealed(r, 'receipt_sha256')
        if not _same(r, arm, training_seed):
            counts['other_arm'] += 1;continue
        if r['status'] != 'graded':
            counts[r['status']] += 1;continue
        counts['graded'] += 1
        if r['grade_veh_s'] <= 0:
            counts['nonpositive'] += 1;continue
        counts['positive'] += 1
        out[r['grade_id']] = dict(weight=weight(r['grade_veh_s'], T, clip), grade_veh_s=r['grade_veh_s'],
                                  episode_id=r['episode_id'], result_path=r.get('result_path'),
                                  plan_id=r['plan']['id'], start_s=r['plan']['start_s'],
                                  duration_s=r['plan']['duration_s'], receipt_sha256=r['receipt_sha256'])
    return dict(weights=out, counts=counts, temperature=temperature_record, clip=clip, arm=arm,
                training_seed=training_seed, rule='whole exact-positive buffer every update')


def imitation_pairs(original, entry):
    """(sample, weight) for every committed plan step of a weighted plan, from the original tape."""
    t0, D, pid = entry['start_s'], entry['duration_s'], entry['plan_id']
    pairs = []
    for s in original['steps']:
        t = int(s['time_s'])
        if t0 <= t < t0+D:
            b = s['sample'].get('behaviour') or {}
            if not b.get('executed') or b.get('plan_id') != pid:
                raise ValueError('incomplete committed plan tape')
            pairs.append((s['sample'], entry['weight']))
    if len(pairs) != D:
        raise ValueError('committed plan extends past the recorded tape')
    return pairs


def _bootstrap_mean(values, worlds, n_boot, seed):
    ws = sorted(set(worlds));idx = {w: i for i, w in enumerate(ws)}
    x = np.zeros((len(ws), 2))
    for v, w in zip(values, worlds):
        x[idx[w], 0] += v;x[idx[w], 1] += 1
    mean = float(x[:, 0].sum()/x[:, 1].sum())
    if len(ws) < 2:
        return mean, None
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ws), np.full(len(ws), 1/len(ws)), size=n_boot)
    d = counts@x;m = d[:, 0]/d[:, 1]
    return mean, list(map(float, np.quantile(m, [.025, .975])))


def trust_evidence(receipts, previous, n_boot=N_BOOT, seed=BOOT_SEED, arm=None, training_seed=None):
    """Trust (beta) from exact grades only: world-cluster bootstrap of the mean paired grade
    (veh-s, positive = plans help), same decay rule as exploration.trust, pooled and per bucket.
    Also evaluates owner health guard 3b (>= 30 plans and CI upper bound < 0 -> pause)."""
    from .exploration import BUCKETS, bucket_beta, decay
    prior = previous if isinstance(previous, dict) else dict(beta=float(previous))
    if arm is not None:
        receipts = [r for r in receipts if _same(r, arm, training_seed)]
    g = graded(receipts)
    out = dict(beta=prior['beta'], buckets={}, estimand='exact paired SUMO grade J_without-J_with (veh-s)',
               graded=len(g), stopped=sum(r['status'] == 'stopped' for r in receipts),
               failed=sum(r['status'] == 'failed' for r in receipts),
               overlap_unresolved=sum(r['status'] == 'overlap_unresolved' for r in receipts), worlds=len({r['world_seed'] for r in g}),
               harm_guard=dict(min_plans=HARM_MIN_PLANS, pause=False, ci95=None))
    if len(g) < 2 or out['worlds'] < 2:
        out.update(mean=None, ci95=None, reason='insufficient exact grades in two independent worlds')
        out['buckets'] = {k: dict(beta=bucket_beta(prior, *k.split('|')), ci95=None) for k in BUCKETS}
        return out
    mean, ci = _bootstrap_mean([r['grade_veh_s'] for r in g], [r['world_seed'] for r in g], n_boot, seed)
    out.update(mean=mean, ci95=ci, beta=decay(prior['beta'], mean, ci) if ci else prior['beta'])
    out['harm_guard'].update(ci95=ci, pause=bool(len(g) >= HARM_MIN_PLANS and ci is not None and ci[1] < 0))
    for k in BUCKETS:
        option, traffic = k.split('|');old = bucket_beta(prior, option, traffic)
        sub = [r for r in g if (r['plan'].get('option') or 'MPPI') == option and (r['plan'].get('regime') or 'ordinary') == traffic]
        if len({r['world_seed'] for r in sub}) < 2:
            out['buckets'][k] = dict(beta=old, previous=old, ci95=None, n=len(sub));continue
        m, c = _bootstrap_mean([r['grade_veh_s'] for r in sub], [r['world_seed'] for r in sub], n_boot, seed)
        out['buckets'][k] = dict(beta=decay(old, m, c), previous=old, mean=m, ci95=c, n=len(sub))
    return out


def lag_report(queue, update, now=None):
    """Per-update grading status and lag (s) from enqueue to receipt."""
    now = now or time.time()
    rows = dict(pending=0, running=0, orphaned=0, graded=0, stopped=0, failed=0, overlap_unresolved=0)
    lags = []
    for gid in queue.ids():
        spec = _read(queue.root/'queue'/f'{gid}.json')
        if spec['episode']['update'] != update:
            continue
        st = queue.status(gid);rows[st] += 1
        r = queue.receipt(gid)
        lags.append((r['finished_unix'] if r else now)-spec['enqueued_unix'])
    return dict(update=update, **rows, max_lag_s=max(lags) if lags else None,
                mean_lag_s=float(np.mean(lags)) if lags else None)


def exact_update(policy, optimizer, episodes, generator, pairs, coefficient=1., epochs=4):
    """MAIN 02:01 update with lever-5 plan grades: PPO on own steps only (no plan ratios), critic on
    all current-batch steps, imitation ONLY from exact-positive graded plans (pairs may come from
    earlier episodes; they enter neither the actor nor the critic loss)."""
    from .accel_learner import advantages, own_step, ratio, imitation
    records = []
    for ep in episodes:
        records.extend((s['sample'], a, r, 0., False) for s, (a, r) in zip(ep['steps'], advantages(ep['steps'])))
    records.extend((s, 0., 0., float(w), True) for s, w in pairs if w > 0)
    adv = torch.tensor([x[1] for x in records], dtype=torch.float32)
    own = torch.tensor([own_step(x[0]) and not x[4] for x in records])
    if own.any():
        adv = (adv-adv[own].mean())/adv[own].std(unbiased=False).clamp_min(1e-8)
    metrics = []
    for _ in range(epochs):
        for idx in torch.randperm(len(records), generator=generator).split(128):
            batch = [records[int(i)] for i in idx]
            crit = [j for j, x in enumerate(batch) if not x[4]]
            own_j = [j for j in crit if own_step(batch[j][0])]
            zero = sum(p.sum() for p in policy.parameters())*0
            if own_j:
                logps = torch.stack([policy.score(batch[j][0])[0] for j in own_j])
                ratios = torch.stack([ratio(lp, batch[j][0]) for lp, j in zip(logps, own_j)])
                a = adv[idx[own_j]]
                actor = -torch.minimum(ratios*a, ratios.clamp(.8, 1.2)*a).mean()
                delta = logps-torch.tensor([batch[j][0]['logp'] for j in own_j])
                kl = float((delta.exp()-1-delta).mean().detach())
            else:
                actor, kl = zero, 0.
            if crit:
                values = torch.stack([policy(batch[j][0]['context'])[1] for j in crit])
                returns = torch.tensor([batch[j][2] for j in crit]);old = torch.tensor([batch[j][0]['value'] for j in crit])
                clipped = old+(values-old).clamp(-.2, .2)
                vloss = torch.maximum((values-returns).square(), (clipped-returns).square()).mean()
            else:
                vloss = zero
            aux = torch.stack([imitation(policy, x[0], x[3]) if x[4] else zero for x in batch]).mean()*coefficient
            loss = actor+.5*vloss+aux
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite acceleration PPO')
            optimizer.zero_grad();loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True);optimizer.step()
            metrics.append(dict(loss=float(loss.detach()), kl=kl, gradient_norm=float(norm), imitation=float(aux.detach()),
                                own_policy_steps=len(own_j), critic_steps=len(crit),
                                exact_imitation_steps=sum(x[4] for x in batch),
                                loss_rule='MAIN0201 own PPO + all-step critic; lever5 exact-positive plan AWR'))
            if kl > .045:
                return metrics
    return metrics


# ---------------------------------------------------------------- smoke

def smoke(run_root, episode_id, grade_root, binding=None, training_seed=None, t0=None, charge='accel',
          phase='exact_grade', path_map=None):
    """Grade ONE plan of one real completed training episode, once (idempotent by receipt).

    Episode id format warm_u<UUU>_<B|R>_<world seed> (accel_runner.wave). Runtime, SUMO checkpoint,
    accel bounds and warm_history come from the run's binding JSON (default <run_root>/WARM_BINDING.json)."""
    import re
    from .accel_storage import read
    from .protocol import sha
    m = re.fullmatch(r'warm_u(\d{3})_([BR])_(\d+)', episode_id)
    if not m:
        raise ValueError('expected a training episode id warm_uUUU_<B|R>_<seed>')
    u, arm = int(m.group(1)), m.group(2)
    run_root = Path(run_root)
    if training_seed is None:
        from .warm_protocol import SEED as training_seed
    b = read(Path(binding) if binding else run_root/'WARM_BINDING.json')
    missing = [k for k in ('runtime', 'checkpoint', 'accel_bounds') if k not in b]
    if missing:
        raise ValueError(f'binding lacks {missing}; pass --binding with the operational runtime binding')
    result_path = run_root/'jobs'/episode_id/'RESULT.json';result = read(result_path)
    ckpt = run_root/f'{arm}_checkpoint_{u-1:03}.pt'
    meta = dict(run_root=str(run_root), result_path=str(result_path), result_sha256=digest(result), update=u, arm=arm,
                checkpoint_path=str(ckpt), checkpoint_sha256=sha(ckpt), checkpoint_update=u-1,
                noise_seed=train_noise_seed(result['world']['seed'], u, training_seed), training_seed=training_seed,
                runtime=b['runtime'], sumo_checkpoint=b['checkpoint'], beta=b.get('beta', 1.),
                warm_history=b.get('warm_history', False), accel_bounds=b['accel_bounds'])
    specs = [x for x in job_specs(result, meta) if t0 is None or x['window']['start_s'] == int(t0)]
    specs = [x for x in specs if not x['overlap_unresolved']] or specs
    if not specs:
        raise ValueError('no executed plan to grade in this episode')
    q = GradeQueue(grade_root, charge=accel_charge(run_root, phase) if charge == 'accel' else None)
    q.enqueue(specs[0])
    return q.run_one(specs[0]['grade_id'], AccelPlantGradeRunner(grade_root, path_map),
                     lambda x: load_original_from_disk(x, path_map))


def summary(r):
    keys = ('grade_id', 'status', 'grade_veh_s', 'J_with_veh_s', 'J_without_veh_s', 'charged_sim_s', 'reason')
    out = {k: r.get(k) for k in keys}
    out.update(window=r.get('window'), prefix_pass=(r.get('prefix') or {}).get('pass_'),
               crn_verified_t0=(r.get('noise') or {}).get('crn_verified_t0'),
               noise_mode_used=(r.get('noise') or {}).get('noise_mode_used'),
               aligned_until_s=(r.get('noise') or {}).get('aligned_until_s'),
               replayed_plans=(r.get('noise') or {}).get('replayed_plans'))
    return out


# ---------------------------------------------------------------- CLI

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    one = sub.add_parser('run-one');one.add_argument('--root', type=Path, required=True)
    one.add_argument('--grade-id', required=True)
    sup = sub.add_parser('supervise');sup.add_argument('--root', type=Path, required=True)
    sup.add_argument('--workers', type=int, default=4);sup.add_argument('--poll', type=float, default=10.)
    sup.add_argument('--once', action='store_true')
    sm = sub.add_parser('smoke', help='grade one plan of one completed episode once')
    sm.add_argument('--run-root', type=Path, required=True);sm.add_argument('--episode-id', required=True)
    sm.add_argument('--grade-root', type=Path, required=True);sm.add_argument('--binding', type=Path)
    sm.add_argument('--t0', type=int);sm.add_argument('--charge', choices=('accel', 'local'), default='accel')
    sm.add_argument('--phase', default='exact_grade');sm.add_argument('--path-map', default='{}')
    a = p.parse_args(argv)
    if a.cmd == 'smoke':
        if os.name == 'posix':
            os.nice(max(0, 19-os.nice(0)))
        torch.set_num_threads(1)
        r = smoke(a.run_root, a.episode_id, a.grade_root, a.binding, t0=a.t0, charge=a.charge, phase=a.phase,
                  path_map=json.loads(a.path_map))
        print(json.dumps(summary(r), sort_keys=True))
        return r
    if a.cmd == 'supervise':
        return supervise(a.root, a.workers, a.poll, once=a.once)
    if os.name == 'posix':
        os.nice(max(0, 19-os.nice(0)))
    torch.set_num_threads(1)
    cfg = _read(a.root/'GRADER_CONFIG.json')     # dict(run_root=..., phase='exact_grade', path_map={})
    q = GradeQueue(a.root, charge=accel_charge(cfg['run_root'], cfg.get('phase', 'exact_grade')))
    pm = cfg.get('path_map')
    r = q.run_one(a.grade_id, AccelPlantGradeRunner(a.root, pm), lambda s: load_original_from_disk(s, pm))
    print(json.dumps(dict(grade_id=a.grade_id, status=None if r is None else r['status'],
                          grade_veh_s=None if r is None else r.get('grade_veh_s'))))


if __name__ == '__main__':
    main()
