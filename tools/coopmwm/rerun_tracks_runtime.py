"""T2C orchestration, shared by qualified CPU integrations and deterministic CPU plants.

Sibling constructors are supplied as adapters; their *consumed* protocol is documented
in RERUN_TRACKS_READY.md. This coordinator owns all arms, commands, charges and worker
permits. Backends may not dispatch outside run_branch/new_episode. No legacy runner
admission or global warm-runner patches are used.
"""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean
import threading

from . import rerun_cpus as rc
from .exact_grade import _seal
from .rerun_tracks_monitor import write_monitor, SAFETY
from .rerun_tracks_bank import training_batch, batch_receipt


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dense_label(label, proposals, cfg):
    """Consumed T2B/T1C boundary: validate provenance, derive eligibility ourselves.

    Calibration is for this target beating the base, including R's random target.
    A missing target is permitted; it masks imitation only, never PPO/group credit.
    """
    if not isinstance(label, dict):
        raise ValueError('REFUSED: dense label needs calibration/verification provenance')
    target = label.get('target')
    if target is not None and target not in proposals:
        raise ValueError('REFUSED: dense label outside common proposal distribution')
    provenance = label.get('provenance', {})
    kind = provenance.get('kind')
    if (kind not in ('planner_calibration', 'sumo_verified', 'no_target') or
            not re.fullmatch('[0-9a-f]{64}', provenance.get('receipt_sha256', ''))):
        raise ValueError('REFUSED: dense label needs auditable provenance receipt')
    p = label.get('calibrated_probability')
    if ((p is not None and (type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1)) or
            (kind == 'planner_calibration' and p is None)):
        raise ValueError('REFUSED: dense label requires calibrated probability')
    if kind == 'no_target' and target is not None:
        raise ValueError('REFUSED: no-target provenance carries a target')
    eligible = target is not None and (kind == 'sumo_verified' or
        (kind == 'planner_calibration' and p >= cfg['dense_teacher']['calibrated_p_min']))
    if label.get('imitation_eligible') is not eligible:
        raise ValueError('REFUSED: teacher imitation eligibility disagrees with registered filter')
    return dict(target=target, calibrated_probability=p, provenance=deepcopy(provenance),
                imitation_eligible=eligible)


class WorkerLimit:
    """One aggregate pool across ALL arms, checks, references and diagnostic branches.

    A coordinator is the sole dispatcher; per-class pools prevent diagnostics taking
    the 24 training slots. Backend subprocess lifetime must fit inside the permit.
    """
    def __init__(self):
        self.total = threading.BoundedSemaphore(rc.CPU['max_sumo_workers'])
        self.pools = {k: threading.BoundedSemaphore(n) for k, n in
                      (('episode', rc.CPU['workers']), ('diagnostic', rc.CPU['diagnostic_workers']))}
        self.lock = threading.Lock()
        self.active = self.peak = 0

    @contextmanager
    def permit(self, kind):
        with self.pools[kind], self.total:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                yield
            finally:
                with self.lock:
                    self.active -= 1


def diagnostic_identity(states):
    """Bind observable TRAIN snapshots, group choices and original FULL windows."""
    from .sumo_verify import snapshot_sha
    descriptions = [dict(state_id=r['verification_id'], snapshot_sha256=snapshot_sha(s),
                         members=r['chosen_plan']['members']) for s, r in states]
    windows = [r['window'] for _, r in states]
    return dict(state_set_sha256=_seal(dict(states=descriptions), 'sha256')['sha256'],
                window_set_sha256=_seal(dict(windows=windows), 'sha256')['sha256'])


def isolated_expansion(job, states, new_weights, common_weights):
    """Reuse existing full-window specs_for, with an ALL-BASE reference.

    Drop the legacy background-new branch; the paired design requires common u0
    background and reference for A/B/R, never arm-specific legacy u3 states.
    """
    from . import policy_synergy as ps
    identity = diagnostic_identity(states)
    if any(job[k] != identity[k] for k in identity):
        raise ValueError('REFUSED: diagnostic states/windows differ from registration')
    out = []
    for snapshot, receipt in states:
        if (snapshot['episode']['world']['seed'] != job['world_seed']
                or snapshot['episode']['role'] != 'train'):
            raise ValueError('REFUSED: diagnostic requires paired registered TRAIN worlds')
        group = receipt['chosen_plan']['members']
        if not 2 <= len(group) <= 4:
            raise ValueError('REFUSED: isolated group must have 2..4 members')
        branches = ps.specs_for(snapshot, receipt, job['update'], job['arm'],
                               new_weights, common_weights, isolated=True)[:-1]
        reference = ps.specs_for(snapshot, receipt, job['update'], job['arm'],
                                common_weights, common_weights, isolated=True)[0]
        spec, snap = reference
        spec = {k: v for k, v in spec.items() if k != 'branch_sha256'}
        spec.update(branch='reference', verification_id=spec['verification_id']+'_common_reference')
        branches.append((_seal(spec, 'branch_sha256'), snap))
        out.append((group, branches))
    if sum(spec['sim_s'] for _, bs in out for spec, _ in bs) > job['sim_s']:
        raise ValueError('REFUSED: expanded isolated windows exceed registered phase allowance')
    return out


class Runtime:
    def __init__(self, cfg, registration, backend, *, smoke=False):
        rc.check_pins(cfg)
        if not cfg.get('track2'):
            raise ValueError('REFUSED: Runtime requires explicit Track 2 opt-in')
        import torch
        torch.set_num_threads(cfg['cpu']['torch_threads'])
        self.cfg, self.registration, self.backend = deepcopy(cfg), deepcopy(registration), backend
        self.root = Path(rc.guard_path(cfg['run_root']))
        rc.guard_path(self.root.resolve())
        rc.own_ledger(self.root)
        self.smoke = smoke
        self.synthetic = backend.cpu_plant is True
        if not self.synthetic:
            # Admission binds preparation to THIS config and registration, rather
            # than accepting a stale preflight boolean from another experiment.
            preflight = rc.preflight_dry(backend.prep_dir, stage_dir=backend.stage_dir)
            approved_cfg = json.loads((Path(backend.prep_dir)/rc.CONFIG_FILE).read_text())
            approved_reg = json.loads((Path(backend.prep_dir)/rc.ISOLATED_REGISTRATION_FILE).read_text())
            if approved_cfg != cfg or approved_reg != registration or not preflight['preflight_pass']:
                raise ValueError('REFUSED: runtime differs from approved preparation')
            if (backend.host != 'cpuS' or backend.release.get('stub', True)
                    or not all(backend.release.get(k) for k in ('cpu_only', 'qualified',
                        'single_writer', 'frozen_base', 'default_off_byte_identical'))):
                raise ValueError('REFUSED: qualified cpuS CPU integration required')
        self.jobs = rc.isolated_jobs(registration)
        self.final_worlds = rc.final_world_registration(registration)
        self.limiter = WorkerLimit()
        self.models, self.components = {}, {}
        self.spend = {p: 0 for p in cfg['budget']['phases']}
        self.reserve_lock = threading.Lock()
        self.seen_jobs = set()
        self.isolated_records = []
        self.common_weights = None
        self.base_sha = {}
        # Immutable claim prevents a second coordinator or duplicate smoke from
        # dispatching into the root. Interrupted claims are investigated, never adopted.
        with (self.root/'T2C_RUNTIME.claim').open('x', encoding='utf-8') as f:
            f.write(json.dumps(dict(synthetic=self.synthetic, smoke=smoke))+'\n')
        for arm in ('A', 'B', 'R'):
            path = self.root/f'{arm}_checkpoint_000.pt'
            if rc.sha(path) != cfg['common_checkpoint_sha256']:
                raise ValueError('REFUSED: common u0 checkpoint differs for '+arm)
            model = backend.load_base(path, arm=arm, cfg=deepcopy(cfg))
            if any(p.device.type != 'cpu' for p in model.parameters()):
                raise ValueError('REFUSED: CPU models only')
            self.models[arm] = model
            self.base_sha[arm] = state_sha(model)
            if arm == 'A':
                self.common_weights = deepcopy(model.state_dict())
            elif self.base_sha[arm] != self.base_sha['A']:
                raise ValueError('REFUSED: common checkpoint loaded into different base states')
            if arm != 'A':
                model.requires_grad_(False)
                model.eval()
            self.components[arm] = rc.track2_components(cfg, arm, adapters=backend.adapters(model))
            if arm != 'A':
                self._check_optimizer(arm)

    def _check_optimizer(self, arm):
        comp = self.components[arm]
        params = {id(p) for group in comp['coop_layer'].optimizer.param_groups for p in group['params']}
        base = {id(p) for p in self.models[arm].parameters()}
        if not params or params & base:
            raise ValueError('REFUSED: cooperation layer needs separate optimizer; frozen base excluded')
        trainable = {id(p) for name in ('coop_layer', 'trigger_head')
                     for p in comp[name].parameters() if p.requires_grad}
        if any(p.device.type != 'cpu' for name in ('coop_layer', 'trigger_head')
               for p in comp[name].parameters()):
            raise ValueError('REFUSED: CPU layer/trigger only')
        if params != trainable:
            raise ValueError('REFUSED: layer optimizer must own exactly the layer and trigger parameters')

    def _write(self, relative, value):
        path = self.root/relative
        rc.guard_path(path)
        rc.guard_path(path.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            f.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')

    def _reserve(self, ident, sim_s, phase):
        self._reserve_batch([dict(id=ident, sim_s=sim_s)], phase)

    def _reserve_batch(self, jobs, phase):
        if self.smoke:
            phase = 'track2_smoke'  # a separately quoted, hard-capped smoke allocation
        with self.reserve_lock:
            ids = [j['id'] for j in jobs]
            if len(set(ids)) != len(ids) or set(ids) & self.seen_jobs:
                raise ValueError('REFUSED: duplicate runtime job')
            if any(type(j['sim_s']) is not int or j['sim_s'] <= 0 for j in jobs):
                raise ValueError('REFUSED: positive whole-window sim-s required')
            sim_s = sum(j['sim_s'] for j in jobs)
            if self.spend[phase]+sim_s > self.cfg['budget']['phases'][phase]:
                raise ValueError('REFUSED: runtime phase ceiling exceeded: '+phase)
            if self.smoke and sum(self.spend.values())+sim_s > self.cfg['smoke']['total_sim_s']:
                raise ValueError('REFUSED: registered smoke budget exceeded')
            if self.synthetic:
                # Virtual work is never presented as real charges or SUMO execution.
                with (self.root/'SYNTHETIC_CHARGES.jsonl').open('a', encoding='utf-8') as f:
                    for job in jobs:
                        f.write(json.dumps(dict(job, phase=phase, synthetic=True))+'\n')
            else:
                rc.charge(self.root, jobs, phase, budget=self.cfg['budget'])
            self.seen_jobs.update(ids)
            self.spend[phase] += sim_s

    def _branch(self, spec, snapshot, phase, ident, *, precharged=False):
        if not precharged:
            self._reserve(ident, spec['sim_s'], phase)
        with self.limiter.permit('diagnostic'):
            result = self.backend.run_branch(spec, snapshot)
        if result.get('status') != 'graded':
            raise ValueError('REFUSED: ungraded diagnostic; never drop/impute/retry')
        if not math.isfinite(result['J_veh_s']):
            raise ValueError('REFUSED: nonfinite diagnostic grade')
        if any(type(result['safety'].get(k)) is not int or result['safety'][k] < 0
               for k in rc.ISOLATED_TEST['safety_counters']):
            raise ValueError('REFUSED: diagnostic safety counters missing/invalid')
        return result

    def _branches(self, pairs, phase, ident):
        jobs = [dict(id=ident+f'_{i}', sim_s=s['sim_s']) for i, (s, _) in enumerate(pairs)]
        self._reserve_batch(jobs, phase)  # WHOLE wave before any branch dispatch
        with ThreadPoolExecutor(max_workers=rc.CPU['diagnostic_workers']) as pool:
            futures = [pool.submit(self._branch, s, snap, phase, j['id'], precharged=True)
                       for (s, snap), j in zip(pairs, jobs)]
            return [f.result() for f in futures]

    def _verify(self, arm, update, kind, rows, fraction):
        count = math.ceil(len(rows)*fraction)
        verified = []
        selected = random.Random(20261104+update).sample(rows, count)
        for i, row in enumerate(selected):
            ident = f't2c_u{update:03}_{arm}_{kind}_{i:05}'
            payload = dict(row, arm=arm, update=update, id=ident)
            pairs = self.backend.expand_verification(kind, payload, cfg=deepcopy(self.cfg))
            if not pairs or sum(s['sim_s'] for s, _ in pairs) > rc.GATE_SIM_S:
                raise ValueError('REFUSED: whole verification exceeds registered allowance')
            for spec, _ in pairs:
                if spec['window'] != row['window']:
                    raise ValueError('REFUSED: verification shortened/changed the full window')
            results = self._branches(pairs, kind+'_verify', ident)
            grade = self.backend.grade_verification(kind, payload, results)
            if grade.get('status') != 'graded':
                raise ValueError('REFUSED: verification must produce graded receipt')
            if kind == 'teacher' and type(grade.get('label_correct')) is not bool:
                raise ValueError('REFUSED: teacher verification needs realised label_correct')
            verified.append(dict(grade, row_id=row.get('row_id', row.get('id')),
                                 target=repr(row.get('target')) if kind == 'teacher' else None,
                                 verification_id=ident))
        self._write(f'components/u{update:03}_{arm}_{kind}_verification.json',
                    dict(synthetic=self.synthetic, selected=count, proposals=len(rows), receipts=verified))
        return verified

    def episode(self, arm, update, index, start):
        from . import moves
        comp, model = self.components[arm], self.models[arm]
        ident = f't2c_u{update:03}_{arm}_episode_{index:03}'
        self._reserve(ident, rc.EPISODE_SIM_S, 'v2v_train')
        # The proposal RNG is arm-independent: same observable state -> SAME slate.
        rng = random.Random(20261104+10000*update+index)
        label_rng = random.Random(20261105+10000*update+index)
        rows, missing = [], []
        with self.limiter.permit('episode'):
            plant = self.backend.new_episode(arm=arm, update=update, index=index, start=start,
                horizon_s=rc.EPISODE_SIM_S, options=rc.episode_options(self.cfg, arm))
            active = None
            for t in range(rc.EPISODE_SIM_S):
                obs = plant.observe()
                if set(obs) != {'time_s', 'own_sensing', 'v2v_reports', 'cav_ids', 'bounds'}:
                    raise ValueError('REFUSED: controller accepts observable-only envelope')
                if obs['time_s'] != t:
                    raise ValueError('REFUSED: episode must retain full per-second horizon')
                obs = deepcopy(obs)
                obs['v2v_reports'] = [r for r in obs['v2v_reports'] if r['time_s'] <= t-1]
                base = self.backend.base_actions(model, obs)
                if set(base) != set(obs['cav_ids']):
                    raise ValueError('REFUSED: base may command only visible CAVs')
                actions = dict(base)
                if comp:
                    features = comp['merge_features'].encode(obs)
                    missing.append(float(features['missing_share']))
                    if t % self.cfg['trigger_head']['decision_interval_s'] == 0:
                        proposals = moves.candidates(obs, rng, self.cfg['dense_teacher']['candidates'])
                        if not proposals or not any(m.family == 'none' for m in proposals):
                            raise ValueError('REFUSED: move slate must include none')
                        move = comp['trigger_head'].choose(obs, features, proposals, rng)
                        if move not in proposals or not set(move.members).issubset(obs['cav_ids']):
                            raise ValueError('REFUSED: trigger selected unobservable/non-CAV members')
                        label = dense_label(comp['dense_teacher'].label(obs, proposals,
                                     kind=rc.teacher_kind(self.cfg, arm), rng=label_rng), proposals, self.cfg)
                        advantage = float(comp['group_credit'].estimate(obs, move, model))
                        if not math.isfinite(advantage):
                            raise ValueError('REFUSED: nonfinite group credit')
                        active = move
                        rows.append(dict(observation=obs, features=features, move=move,
                             **label, row_id=ident+f'_t{t:04}',
                             group_advantage=advantage, family=move.family,
                            window=plant.full_window(), episode=index))
                    offsets = moves.offsets(active, t) if active is not None else {}
                    residual = comp['coop_layer'].residual(obs, features)
                    if (not set(offsets).issubset(base) or not set(residual).issubset(base)):
                        raise ValueError('REFUSED: HDVs/unobserved members never commanded')
                    for vid, value in base.items():
                        lo, hi = obs['bounds'][vid]
                        actions[vid] = min(hi, 1.2, max(lo, value+offsets.get(vid, 0.)+residual.get(vid, 0.)))
                plant.step(actions, speed_mode=27)  # sole command writer
            result = plant.finish()
        if arm != 'A' and state_sha(model) != self.base_sha[arm]:
            raise ValueError('REFUSED: frozen base changed during episode')
        return dict(rows=rows, missing=missing, reward=result['reward'], safety=result['safety'],
                    training=result['training'], safety_stop=result['safety_stop'])

    def update(self, arm, update, *, episodes=None):
        from . import moves
        n = episodes or rc.EPISODES_PER_UPDATE
        comp = self.components[arm]
        starts, augmented = [None]*n, []
        if comp:
            bank = self.backend.bank_examples(arm, update, cfg=deepcopy(self.cfg))
            if not bank or any(e['arm'] != arm for e in bank):
                raise ValueError('REFUSED: B/R must consume their own nonempty banks')
            count = self.cfg['smoke']['augmented_examples_per_arm'] if self.smoke else self.cfg['plan_augment']['examples_per_update']
            augmented = comp['plan_augment'].augment(bank, count=count, arm=arm)
            if len(augmented) != count or any(e['arm'] != arm for e in augmented):
                raise ValueError('REFUSED: augmentation lost own-bank identity or registered count')
            bank_batch = training_batch(bank, augmented, arm=arm, encoder=comp['merge_features'])
            bank_receipt = batch_receipt(bank_batch)
            self._write(f'components/u{update:03}_{arm}_bank_batch.json', bank_receipt)
            self._verify(arm, update, 'augment', augmented, self.cfg['plan_augment']['verify_fraction'])
            count = math.ceil(n*self.cfg['practice_weighting']['fraction'])
            selected = comp['practice_weighting'].select(bank, augmented, count=count,
                                                       repeat_cap=self.cfg['practice_repeat_cap'])
            if len(selected) != count or any(e['arm'] != arm for e in selected):
                raise ValueError('REFUSED: unequal/foreign practice-weighted starts')
            ids = [e['id'] for e in selected]
            if any(ids.count(ident) > self.cfg['practice_repeat_cap'] for ident in ids):
                raise ValueError('REFUSED: weighted start repeat cap exceeded')
            if any(e not in bank+augmented for e in selected):
                raise ValueError('REFUSED: weighted start not from own bank/augmentation')
            starts[-count:] = selected
        with ThreadPoolExecutor(max_workers=rc.CPU['workers']) as pool:
            futures = [pool.submit(self.episode, arm, update, i, start) for i, start in enumerate(starts)]
            episodes_out = [f.result() for f in futures]
        rows = [r for ep in episodes_out for r in ep['rows']]
        safety = {k: sum(ep['safety'][k] for ep in episodes_out) for k in SAFETY}
        reward = fmean(ep['reward'] for ep in episodes_out)
        metrics = dict(reward=reward, safety=safety,
            safety_stop=any(ep['safety_stop'] for ep in episodes_out),
            families=list(moves.FAMILIES), decisions=len(rows),
            trigger_counts={f: sum(r['family'] == f for r in rows) for f in moves.FAMILIES},
             credit_rows=len(rows), credit_checked_rows=0, dense_labels=len(rows),
             dense_eligible_labels=0, dense_checked_labels=0,
            augmented_examples=len(augmented), weighted_starts=sum(s is not None for s in starts))
        if not comp:
            self.backend.ppo_update(self.models[arm], [ep['training'] for ep in episodes_out])
            metrics.update({k: None for k in ('group_credit_wm', 'group_credit_checked',
                'dense_ce', 'dense_accuracy', 'dense_filtered_share', 'dense_checked_accuracy',
                'bank_move_ce', 'layer_kl', 'merge_missing_share')})
        else:
            credit = self._verify(arm, update, 'group_credit', rows, self.cfg['group_credit']['verify_fraction'])
            teacher_checks = self._verify(arm, update, 'teacher', rows, self.cfg['dense_teacher']['verify_fraction'])
            mask = [r['imitation_eligible'] for r in rows]
            self._write(f'components/u{update:03}_{arm}_dense_labels.json', dict(
                arm=arm, update=update, synthetic=self.synthetic,
                rows=[dict({k: r[k] for k in ('row_id', 'calibrated_probability', 'provenance',
                                             'imitation_eligible')}, target=repr(r['target'])) for r in rows],
                checked=teacher_checks))
            # Group advantages enter ONLY the layer/trigger update. Reward and A's
            # PPO stream remain untouched. Base params are frozen and excluded.
            self._check_optimizer(arm)
            losses = comp['coop_layer'].update(rows=rows, bank_batch=bank_batch,
                      bank_options=deepcopy(self.cfg['bank_training']),
                      group_advantages=[r['group_advantage'] for r in rows],
                       dense_targets=[r['target'] for r in rows],
                       dense_mask=mask,
                      trigger=comp['trigger_head'], kl_cap=self.cfg['coop_layer']['kl_cap'])
            if state_sha(self.models[arm]) != self.base_sha[arm]:
                raise ValueError('REFUSED: frozen MAPPO base changed after layer update')
            if losses['layer_kl'] > self.cfg['coop_layer']['kl_cap']:
                raise ValueError('REFUSED: cooperation layer KL budget exceeded')
            consumed = losses.get('bank_receipt', {})
            if (consumed.get('batch_sha256') != bank_receipt['batch_sha256'] or
                    consumed.get('move_rows') != len(bank_batch) or
                    consumed.get('effective_rows') != len(bank_batch) or
                    consumed.get('options') != self.cfg['bank_training'] or
                    type(consumed.get('effective_loss')) not in (int, float) or
                    not math.isfinite(consumed['effective_loss'])):
                raise ValueError('REFUSED: learner did not receipt complete bank/effective supervision')
            self._write(f'components/u{update:03}_{arm}_bank_learning.json', consumed)
            metrics.update({k: losses[k] for k in ('dense_ce', 'dense_accuracy', 'bank_move_ce', 'layer_kl')})
            metrics.update(group_credit_wm=fmean(r['group_advantage'] for r in rows),
                 dense_eligible_labels=sum(mask), dense_filtered_share=1-sum(mask)/len(rows),
                 dense_checked_labels=len(teacher_checks),
                 dense_checked_accuracy=fmean(r['label_correct'] for r in teacher_checks),
                group_credit_checked=fmean(r['advantage'] for r in credit), credit_checked_rows=len(credit),
                merge_missing_share=fmean(v for ep in episodes_out for v in ep['missing']))
            # Each weighted start gets its matched all-base reference, same seed and
            # snapshot. The backend's expansion must retain that full restart window.
            for i, start in enumerate(s for s in starts if s is not None):
                spec, snap = self.backend.practice_reference(start, common_weights=self.common_weights)
                if spec['window'] != start['window']:
                    raise ValueError('REFUSED: practice reference must use matched full window')
                self._branch(spec, snap, 'v2v_train', f't2c_u{update:03}_{arm}_practice_ref_{i}')
        monitor = write_monitor(self.root, arm, update, metrics)
        self.backend.save_checkpoint(self.root/f'{arm}_checkpoint_{update:03}.pt',
                                     self.models[arm], comp)
        self._write(f'updates/u{update:03}_{arm}.json', dict(synthetic=self.synthetic,
            arm=arm, update=update, episodes=n, monitor_sha256=monitor['monitor_sha256'],
            frozen_base_sha256=self.base_sha[arm] if comp else None, metrics=metrics))
        if monitor['action'] == 'PAUSE_SAFETY':
            raise ValueError('REFUSED: safety pause (learning diagnostics never pause)')
        return monitor

    def diagnostics(self, update, *, dry=False, weights_by_arm=None):
        from .policy_synergy import metrics as synergy_metrics
        jobs = [j for j in self.jobs if j['update'] == update]
        if update in rc.EARLY_PROBES['updates']:
            jobs += [dict(self.jobs[0], id=f't2c_probe_u{update:03}_{a}', arm=a, update=update,
                          sim_s=rc.EARLY_PROBES['sim_s_per_arm'], phase='early_probes') for a in ('A', 'B', 'R')]
        records, branches = [], 0
        for job in jobs:
            states = self.backend.resolve_states(job)
            if job['phase'] == 'early_probes':
                if len(states) < 3:
                    raise ValueError('REFUSED: early probe requires three registered states')
                # Validate full registered set before selecting the registered first three.
                if diagnostic_identity(states) != {k: job[k] for k in diagnostic_identity(states)}:
                    raise ValueError('REFUSED: early probe state/window identity mismatch')
                states = states[:3]
                job = dict(job, **diagnostic_identity(states))
            weights = (weights_by_arm[job['arm']] if weights_by_arm is not None else
                       self.backend.diagnostic_weights(self.models[job['arm']], self.components[job['arm']]))
            expanded = isolated_expansion(job, states, weights, self.common_weights)
            branches += sum(len(bs) for _, bs in expanded)
            if dry:
                continue
            gains, counters = [], {k: 0 for k in rc.ISOLATED_TEST['safety_counters']}
            for i, (group, bs) in enumerate(expanded):
                results = self._branches(bs, job['phase'], job['id']+f'_s{i}')
                gain = synergy_metrics(results[-1]['J_veh_s'], results[:-1], group)
                gains.append(gain['policy_cooperation_gain_veh_s'])
                for result in results:
                    for k in counters:
                        counters[k] += result['safety'][k]
            records.append(dict(job, status='graded', metric=fmean(gains), **counters))
        receipt = dict(update=update, dry=dry, synthetic=self.synthetic, cells=len(jobs),
                       branches=branches, records=records, new_sim_s=0 if dry or self.synthetic else None)
        self._write(f'diagnostics/u{update:03}_{"dry" if dry else "graded"}.json', receipt)
        self.isolated_records.extend(r for r in records if r['phase'] == 'isolated_replicated')
        return receipt

    def scheduled_checks(self, update):
        """Consume the existing lesson and fixed-test integrations through the same pool."""
        for arm in ('A', 'B', 'R'):
            kinds = []
            if update in self.cfg['diagnostic_schedule'][arm]['lessons']:
                kinds.append(('lesson', 'v2v_lessons'))
            if update == 8:
                kinds.append(('fixed_test', 'v2v_interim'))
            for kind, phase in kinds:
                pairs = self.backend.expand_scheduled(kind, arm=arm, update=update,
                                                       cfg=deepcopy(self.cfg))
                if not pairs:
                    raise ValueError('REFUSED: scheduled '+kind+' produced no branches')
                if kind == 'fixed_test' and (len(pairs) != rc.TEST_EPISODES or
                        any(s['sim_s'] != rc.EPISODE_SIM_S for s, _ in pairs)):
                    raise ValueError('REFUSED: fixed tests must retain 96 full episodes per arm')
                results = self._branches(pairs, phase, f't2c_u{update:03}_{arm}_{kind}')
                receipt = self.backend.record_scheduled(kind, arm=arm, update=update,
                                                        results=results, cfg=deepcopy(self.cfg))
                if receipt.get('status') != 'graded':
                    raise ValueError('REFUSED: scheduled '+kind+' receipt missing/ungraded')
                self._write(f'diagnostics/u{update:03}_{arm}_{kind}.json', receipt)
        if update == rc.FINAL_EVALUATION['update']:
            return self.final_evaluation()

    def final_evaluation(self):
        """Explicit T1D integration boundary, tested using a test-only sibling module.

        T1D adapter exports prepare_track2 / analyse_track2; it maps its native paired
        harness to this contract. No opaque independent per-arm final evaluator.
        """
        contract = deepcopy(self.cfg['final_evaluation'])
        harness = importlib.import_module(contract['module'])
        checkpoints = {}
        for arm in contract['arms']:
            path = self.root/f'{arm}_checkpoint_{contract["update"]:03}.pt'
            rc.guard_path(path.resolve())
            if self.backend.checkpoint_components(path) != contract['checkpoint_components'][arm]:
                raise ValueError('REFUSED: incomplete final checkpoint components')
            checkpoints[arm] = dict(path=str(path), file_sha256=rc.sha(path),
                components=contract['checkpoint_components'][arm], update=contract['update'])
        worlds = deepcopy(self.final_worlds)
        pairs = harness.prepare_track2(registration=deepcopy(worlds),
            checkpoints=deepcopy(checkpoints), contract=deepcopy(contract))
        expected = {(a, w['world_id']): w for a in contract['arms'] for w in worlds['worlds']}
        seen = set()
        for spec, snapshot in pairs:
            key = (spec.get('arm'), spec.get('world_id'))
            world = expected.get(key)
            if (world is None or key in seen or spec.get('world') != world['world'] or
                    spec.get('checkpoint') != checkpoints.get(key[0]) or
                    spec.get('sim_s') != contract['horizon_s'] or
                    spec.get('window') != dict(start_s=0, end_s=contract['horizon_s']) or
                    spec.get('final_worlds_sha256') != worlds['final_worlds_sha256']):
                raise ValueError('REFUSED: unpaired/incomplete shared final evaluation expansion')
            seen.add(key)
        if seen != set(expected) or sum(s['sim_s'] for s, _ in pairs) != contract['sim_s']:
            raise ValueError('REFUSED: incomplete shared final evaluation grid/budget')
        results = self._branches(pairs, contract['phase'], 't2c_final_planner_eval')
        records = []
        for (spec, _), result in zip(pairs, results):
            if any(result.get(k) != spec[k] for k in
                   ('arm', 'world_id', 'world', 'checkpoint', 'window', 'final_worlds_sha256')):
                raise ValueError('REFUSED: shared final result world/checkpoint/window identity mismatch')
            records.append(result)
        analysis = harness.analyse_track2(registration=deepcopy(worlds),
            checkpoints=deepcopy(checkpoints), records=deepcopy(records), contract=deepcopy(contract))
        if (analysis.get('status') != 'graded' or analysis.get('bootstrap') != contract['bootstrap'] or
                analysis.get('final_worlds_sha256') != worlds['final_worlds_sha256'] or
                analysis.get('checkpoints') != checkpoints or
                set(analysis.get('paired_ci', {})) != set(contract['comparisons'])):
            raise ValueError('REFUSED: incomplete shared paired CI analysis')
        by_key = {(r['arm'], r['world_id']): r for r in records}
        for comparison, ci in analysis['paired_ci'].items():
            a, b = comparison.split('-')
            mean = fmean(by_key[a, w['world_id']]['J_veh_s']-
                         by_key[b, w['world_id']]['J_veh_s'] for w in worlds['worlds'])
            if (ci.get('n') != len(worlds['worlds']) or
                    any(type(ci.get(k)) not in (int, float) or not math.isfinite(ci[k])
                        for k in ('mean', 'low', 'high')) or
                    not math.isclose(ci['mean'], mean, abs_tol=1e-10) or ci['low'] > ci['high']):
                raise ValueError('REFUSED: invalid shared paired CI')
        safety = {a: {k: sum(r['safety'][k] for r in records if r['arm'] == a)
                      for k in contract['safety_counters']} for a in contract['arms']}
        if analysis.get('safety') != safety:
            raise ValueError('REFUSED: incomplete shared final safety analysis')
        receipt = _seal(dict(synthetic=self.synthetic, harness=contract['module'], contract=contract,
            registration=worlds, checkpoints=checkpoints, records=records, analysis=analysis), 'receipt_sha256')
        self._write('T2C_FINAL_PAIRED_EVAL.json', receipt)
        return receipt

    def run(self):
        """One coordinator owns the complete loop; every update has all-arm receipts."""
        monitors = []
        pending = []
        with ThreadPoolExecutor(max_workers=2) as diagnostics_pool:
            for update in range(1, 2 if self.smoke else rc.UPDATES+1):
                for future in pending:
                    if future.done():
                        future.result()  # propagate failures before more training
                for arm in ('A', 'B', 'R'):
                    monitors.append(self.update(arm, update,
                        episodes=self.cfg['smoke']['episodes_per_arm'] if self.smoke else None))
                if not self.smoke:
                    weights = {a: deepcopy(self.backend.diagnostic_weights(self.models[a], self.components[a]))
                               for a in ('A', 'B', 'R')}
                    pending.append(diagnostics_pool.submit(self.diagnostics, update, weights_by_arm=weights))
                    pending.append(diagnostics_pool.submit(self.scheduled_checks, update))
            for future in pending:
                future.result()
        if self.smoke:
            dry = [self.diagnostics(u, dry=True) for u in rc.ISOLATED_TEST['updates']]
            for row in monitors:
                if row['arm'] == 'A':
                    continue
                m = row['metrics']
                if (any(m[k] <= 0 for k in ('decisions', 'credit_rows', 'dense_labels',
                                           'augmented_examples', 'weighted_starts'))
                        or m['merge_missing_share'] >= 1
                        or sum(v for k, v in m['trigger_counts'].items() if k != 'none') == 0):
                    raise ValueError('REFUSED: smoke component produced no usable receipt')
            result = dict(status='CPU_PLANT_SMOKE_PASS' if self.synthetic else 'SMOKE_PASS',
                synthetic=self.synthetic, target_host='cpuS', episodes=6, updates=1,
                monitors=[r['monitor_sha256'] for r in monitors],
                isolated_dry_cells=sum(r['cells'] for r in dry), worker_peak=self.limiter.peak,
                virtual_sim_s=sum(self.spend.values()) if self.synthetic else 0,
                new_sim_s=0 if self.synthetic else sum(self.spend.values()), spend=self.spend)
            self._write('T2C_SMOKE.json', result)
            return result
        result = rc.evaluate_isolated_test(self.isolated_records, registration=self.registration)
        self._write('T2C_ISOLATED_EVAL.json', result)
        return result


def run(cfg, registration, backend, *, smoke=False, legacy_run=None):
    """OFF = direct call to the previous runtime, with no imports/construction/RNG draws."""
    if not cfg.get('track2'):
        if legacy_run is None:
            raise ValueError('legacy runtime callable required')
        return legacy_run(cfg)
    return Runtime(cfg, registration, backend, smoke=smoke).run()
