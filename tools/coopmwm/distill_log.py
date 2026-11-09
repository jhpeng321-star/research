"""DISTILL teacher-data logger (DIST-LOG 2026-11-09): per-step, per-CAV behaviour-cloning data from the LEARNED_TRIGGER
arm (frozen MAPPO u0 + learned trigger classifier + verified-plan executor) on the a28 plant path.

Schema ``distill_teacher/1``: E:/agent-work/distill/SCHEMA.md (one compressed npz shard per episode).

OFF by default: ``nm_eval_runner._policy_run`` builds a logger only when ``spec['distill_log']`` is set; without it
``distill_tick(None)`` installs nothing and the eval path is unchanged (test_distill_log_cpu proves the off path and
that the logger is read-only: the wrapped tick returns the original's value and env.step receives the same args).

What is recorded (all read-only):
* actor input = ``env.state()`` read at the same point the plant's executor reads it (before ``env.step``), all
  present slots, plus the MAPPO-actor memory augmentation (``CommandMemory``: last commanded accel / lane intent of
  each slot, here fed with the TEACHER's applied command and realised lane delta) and ``LaneDwell`` ages, both with
  the plant's slot-reuse ``keep`` rule (``coopfree.backend.tick``);
* labels = the [1,K] acceleration actually passed to ``env.step`` (after U0 executor + plan setpoints + the single
  safety writer), the lane targets passed to ``env.step`` and the realised WM lane delta over the step;
* plan membership from the LEARNED_TRIGGER core (``owners()``, activations, classifier scores).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SCHEMA = 'distill_teacher/1'
#: never collected (eval panels, sealed / forbidden families); learned_trigger.FORBIDDEN_SEED_RANGES + map forbidden
FORBIDDEN_SEED_RANGES = ((989600, 989699), (993000, 993999), (964000, 964199), (962000, 962999), (977000, 977999))
#: the DISTILL collection seed range (registered TRAIN-role range added in the worker process only)
DISTILL_RANGE = (999000, 999999)
SMOKE_SEEDS = (999990, 999991)
DEFAULT_DWELL_S = 3.0          # wms.control.lanes.LaneRules.dwell_s default
BANNED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'driver_params')
CELL = dict(demand=2800, penetration=0.2, cell_key='D2800_p20')


def guard_path(path):
    low = str(path).lower().replace('\\', '/')
    for tok in BANNED:
        if tok in low:
            raise ValueError(f'refusing banned path token {tok!r}: {path}')
    return path


def check_seeds(seeds, lo=DISTILL_RANGE[0], hi=DISTILL_RANGE[1]):
    seeds = sorted({int(s) for s in seeds})
    bad = [s for s in seeds if any(a <= s <= b for a, b in FORBIDDEN_SEED_RANGES)]
    if bad:
        raise ValueError(f'forbidden seeds: {bad[:5]}')
    out = [s for s in seeds if not lo <= s <= hi]
    if out:
        raise ValueError(f'seeds outside the DISTILL range {lo}-{hi}: {out[:5]}')
    return seeds


def install_distill_range(ms=None, lo=DISTILL_RANGE[0], hi=DISTILL_RANGE[1]):
    """Idempotent, this process only: register [lo, hi] as an extra TRAIN range of map_scenarios (scope 'train',
    repeat = seed - lo; labels only, physics-neutral like amendment 23's eval range)."""
    if ms is None:
        from . import map_scenarios as ms
    if any(a <= lo <= b or a <= hi <= b for a, b in FORBIDDEN_SEED_RANGES):
        raise ValueError('DISTILL range overlaps a forbidden range')
    rng = (int(lo), int(hi))
    if rng not in tuple(ms.EXTRA_TRAIN_RANGES):
        ms.EXTRA_TRAIN_RANGES = tuple(ms.EXTRA_TRAIN_RANGES) + (rng,)
    return True


def _np(x, dtype=None):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(dtype) if dtype is not None else x


# ---------------------------------------------------------------- the logger

class DistillLogger:
    """Collects one episode in memory (~a few MB) and writes the shard on ``close``. Read-only on the plant."""

    def __init__(self, out_path, *, meta=None, hook=None, dwell_s=DEFAULT_DWELL_S):
        self.out_path = Path(guard_path(out_path))
        self.meta = dict(meta or {})
        self.hook = hook
        self.dwell_s = float(dwell_s)
        self.families = list(getattr(getattr(hook, 'core', None), 'model', {}).get('families', ()) or ())
        self.meta.update(schema=SCHEMA, dwell_s=self.dwell_s, families=self.families)
        self.K = None
        self.prev_ids = None
        self.mem_accel = self.mem_intent = self.mem_present = None
        self.age = self.last_lane = self.dw_present = None
        self.vid_index = {}
        self.carried = {}               # vid -> last known WM lane (junction keeps it)
        self.frames = []                # per step dict of arrays
        self.cav = []                   # per step dict of CAV label arrays
        self.score_rows = []
        self._score_n = 0
        self._sent = None
        self._pending = None            # (step index, vids, L_t) awaiting the next frame for lane labels
        self.closed = False

    # -- per step ------------------------------------------------------------
    def before(self, env):
        """Pre-step: the exact state the executor reads this second (pure read)."""
        state = env.state()
        now = int(round(float(env.now)))
        ids = _np(env.frame.vid_index).reshape(-1).astype(np.int64)
        slot_vids = list(env.slot_vehicle_ids())
        s = _np(state['s_m'], np.float32).reshape(-1)
        K = s.shape[0]
        if self.K is None:
            self.K = K
            self.meta['K'] = K
        elif K != self.K:
            raise ValueError('slot count changed within an episode')
        present = _np(state['present']).reshape(-1).astype(bool)
        is_cav = _np(state['is_cav']).reshape(-1).astype(bool) & present
        lane = _np(state['lane']).reshape(-1).astype(np.int64)
        keep = np.zeros(K, bool) if self.prev_ids is None else (ids == self.prev_ids) & (ids >= 0)
        self.prev_ids = ids.copy()
        rows = np.flatnonzero(present)
        vids = []
        for i in rows:
            vid = slot_vids[i]
            if vid is None:
                raise ValueError('present slot without a vehicle id')
            vids.append(str(vid))
        vidx = np.array([self.vid_index.setdefault(v, len(self.vid_index)) for v in vids], np.int32)
        # lane labels of the previous step's CAVs (realised delta over that step)
        cur = {}
        for v, ln in zip(vids, lane[rows]):
            if ln >= 0:
                self.carried[v] = int(ln)
            if v in self.carried:
                cur[v] = self.carried[v]
        self._resolve_pending(cur)
        # CommandMemory.augment with the plant keep rule
        if self.mem_accel is None:
            cmd_mem = np.zeros(K, np.float32)
            intent_mem = np.zeros(K, np.float32)
        else:
            ok = present & self.mem_present & keep
            cmd_mem = np.where(ok, self.mem_accel, 0.).astype(np.float32)
            intent_mem = np.where(ok, self.mem_intent, 0.).astype(np.float32)
        # LaneDwell.step with the plant keep rule (dwell.present &= keep before the step)
        if self.age is None:
            age = np.full(K, self.dwell_s, np.float32)
            last = np.where(present, lane, -1)
        else:
            prev = self.dw_present & keep
            age = self.age + 1.0
            known = (self.last_lane >= 0) & (lane >= 0) & present & prev
            age = np.where(known & (lane != self.last_lane), 0., age)
            new = present & ~prev
            age = np.where(new, self.dwell_s, age)
            last = np.where(new, -1, self.last_lane)
            last = np.where(lane >= 0, lane, last)
        age = np.where(present, age, self.dwell_s).astype(np.float32)
        self.age, self.last_lane, self.dw_present = age, last, present
        frame = dict(t=now, slot=rows.astype(np.int16), vid=vidx,
                     s_m=s[rows], v_mps=_np(state['v_mps'], np.float32).reshape(-1)[rows],
                     a_mps2=_np(state['a_mps2'], np.float32).reshape(-1)[rows], lane=lane[rows].astype(np.int8),
                     junction=(_np(state['junction']).reshape(-1).astype(bool)[rows] if 'junction' in state
                               else np.zeros(len(rows), bool)),
                     is_cav=is_cav[rows], length_m=_np(state['length_m'], np.float32).reshape(-1)[rows],
                     cmd_accel_mps2=cmd_mem[rows], lane_intent=intent_mem[rows], dwell_age_s=age[rows])
        if 'entry_queue' in state:
            frame['entry_queue'] = float(_np(state['entry_queue']).reshape(-1)[0])
        self._now, self._present, self._is_cav, self._lane, self._slot_vids = now, present, is_cav, lane, slot_vids
        self._rows, self._vids, self._step = rows, vids, len(self.frames)
        self._cur_carried = cur
        self.frames.append(frame)
        self._sent = None

    def sent(self, accel, lanes):
        """The [1,K] command and lane targets actually passed to env.step this second."""
        self._sent = (_np(accel, np.float64).reshape(-1).copy(),
                      None if lanes is None else _np(lanes).reshape(-1).astype(np.int64).copy())

    def after(self, audit, env=None):
        if self._sent is None:
            raise ValueError('env.step was not observed this tick')
        accel, lanes = self._sent
        step, rows, vids = self._step, self._rows, self._vids
        base = sum(len(f['slot']) for f in self.frames[:-1])
        owners, member = self._plan_members()
        u0 = {r['vid']: float(r['mappo_safe']) for r in (audit or ()) if 'mappo_safe' in r}
        cav_pos = [j for j, i in enumerate(rows) if self._is_cav[i]]
        n = len(cav_pos)
        lab = dict(step=np.full(n, step, np.int32), row=np.empty(n, np.int64), accel=np.empty(n, np.float32),
                   accel_u0=np.empty(n, np.float32), src=np.empty(n, np.int8), lane_cmd=np.zeros(n, np.int8),
                   plan_active=np.zeros(n, bool), plan_family=np.full(n, -1, np.int16),
                   plan_role=np.full(n, -1, np.int8), plan_act=np.full(n, -1, np.int32),
                   plan_onset=np.zeros(n, bool), trigger_prob=np.full(n, np.nan, np.float32))
        pend_vids, pend_lane = [], []
        for k, j in enumerate(cav_pos):
            i, vid = rows[j], vids[j]
            a = float(accel[i]) if i < accel.shape[0] else float('nan')
            lab['row'][k] = base + j
            lab['accel'][k] = a
            lab['accel_u0'][k] = u0.get(vid, a)
            lab['src'][k] = 0 if not math.isfinite(a) else (2 if vid in owners else 1)
            if lanes is not None and lanes[i] >= 0 and self._lane[i] >= 0:
                lab['lane_cmd'][k] = int(np.clip(lanes[i] - self._lane[i], -1, 1))
            if vid in member:
                fam, act_i, role, onset, prob = member[vid]
                lab['plan_active'][k] = True
                lab['plan_family'][k] = self.families.index(fam) if fam in self.families else -1
                lab['plan_role'][k], lab['plan_act'][k], lab['plan_onset'][k] = role, act_i, onset
                lab['trigger_prob'][k] = np.nan if prob is None else prob
            pend_vids.append(vid)
            pend_lane.append(self._cur_carried.get(vid))
        self.cav.append(lab)
        self._pending = (step, pend_vids, pend_lane, accel, rows, self._is_cav)
        self._scores()

    def _resolve_pending(self, cur):
        """Lane labels of the previous step from this frame; then the teacher memory record of that step."""
        if self._pending is None:
            return
        step, vids, lanes_t, accel, rows, is_cav = self._pending
        lab = self.cav[step]
        n = len(vids)
        label = np.zeros(n, np.int8)
        valid = np.zeros(n, bool)
        for k, (v, l0) in enumerate(zip(vids, lanes_t)):
            l1 = cur.get(v)
            if l0 is not None and l1 is not None:
                label[k] = int(np.clip(l1 - l0, -1, 1))
                valid[k] = True
        lab['lane_label'], lab['lane_valid'] = label, valid
        # CommandMemory.record(state, where(cav, applied, 0), where(cav, intent, 0)) of that step
        K = self.K
        present = np.zeros(K, bool)
        present[rows] = True
        acc = np.where(is_cav, np.nan_to_num(accel[:K], nan=0.0), 0.0)
        intent = np.zeros(K)
        slot_of = {self._prev_slot_vids[i]: i for i in rows} if hasattr(self, '_prev_slot_vids') else {}
        for k, v in enumerate(vids):
            i = slot_of.get(v)
            if i is not None:
                intent[i] = label[k]
        self.mem_accel = np.where(present, acc, 0.).astype(np.float32)
        self.mem_intent = np.where(present, intent, 0.).astype(np.float32)
        self.mem_present = present
        self._pending = None

    def _plan_members(self):
        core = getattr(self.hook, 'core', None)
        if core is None:
            return {}, {}
        owners = dict(core.owners())
        index = {id(a): n for n, a in enumerate(core.acts)}
        member = {}
        for c, p in zip(core.cores, core.plans):
            for vid, m in c.active.items():
                act = m['act']
                mem = list(act.get('members') or ())
                member[vid] = (p['family'], index.get(id(act), -1), mem.index(vid) if vid in mem else -1,
                               int(act.get('t', -1)) == self._now, act.get('prob'))
        return owners, member

    def _scores(self):
        core = getattr(self.hook, 'core', None)
        scores = getattr(core, 'scores', None)
        if scores is None:
            return
        for t, fam, p, fire in scores[self._score_n:]:
            self.score_rows.append((int(t), self.families.index(fam) if fam in self.families else -1, float(p),
                                    bool(fire)))
        self._score_n = len(scores)

    def tick_done(self):
        self._prev_slot_vids = self._slot_vids

    # -- close -----------------------------------------------------------------
    def close(self):
        """Finalize (last step: lane labels invalid, a_next NaN) and write the shard atomically. Returns a receipt."""
        if self.closed:
            raise ValueError('logger already closed')
        self.closed = True
        if self._pending is not None:
            step, vids = self._pending[0], self._pending[1]
            self.cav[step]['lane_label'] = np.zeros(len(vids), np.int8)
            self.cav[step]['lane_valid'] = np.zeros(len(vids), bool)
            self._pending = None
        arrays = self.arrays()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_name(self.out_path.name + '.partial.npz')
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, self.out_path)
        h = hashlib.sha256(self.out_path.read_bytes()).hexdigest()
        return dict(file=self.out_path.name, sha256=h, bytes=self.out_path.stat().st_size, steps=len(self.frames),
                    vehicle_rows=int(arrays['off'][-1]), cav_rows=int(arrays['cav_step'].shape[0]),
                    plan_rows=int(arrays['plan_active'].sum()), lane_changes=int(np.abs(arrays['lane_label']).sum()),
                    scores=int(arrays['score_t'].shape[0]))

    def arrays(self):
        F = self.frames
        off = np.zeros(len(F) + 1, np.int64)
        off[1:] = np.cumsum([len(f['slot']) for f in F])
        cat = (lambda k, dt: np.concatenate([f[k] for f in F]).astype(dt) if F else np.zeros(0, dt))  # noqa: E731
        out = dict(schema=np.array(SCHEMA), K=np.array(self.K or 0, np.int64),
                   families=np.array(self.families, dtype=str), t=np.array([f['t'] for f in F], np.int32), off=off,
                   slot=cat('slot', np.int16), vid=cat('vid', np.int32), s_m=cat('s_m', np.float32),
                   v_mps=cat('v_mps', np.float32), a_mps2=cat('a_mps2', np.float32), lane=cat('lane', np.int8),
                   junction=cat('junction', bool), is_cav=cat('is_cav', bool), length_m=cat('length_m', np.float32),
                   cmd_accel_mps2=cat('cmd_accel_mps2', np.float32), lane_intent=cat('lane_intent', np.float32),
                   dwell_age_s=cat('dwell_age_s', np.float32))
        if F and all('entry_queue' in f for f in F):
            out['entry_queue'] = np.array([f['entry_queue'] for f in F], np.float32)
        names = [None] * len(self.vid_index)
        for v, i in self.vid_index.items():
            names[i] = v
        out['vid_names'] = np.array(names, dtype=str)
        C = self.cav
        cc = (lambda k, dt: np.concatenate([c[k] for c in C]).astype(dt) if C else np.zeros(0, dt))  # noqa: E731
        out.update(cav_step=cc('step', np.int32), cav_row=cc('row', np.int64), accel_applied=cc('accel', np.float32),
                   accel_u0=cc('accel_u0', np.float32), ctrl_src=cc('src', np.int8), lane_label=cc('lane_label', np.int8),
                   lane_valid=cc('lane_valid', bool), lane_cmd=cc('lane_cmd', np.int8),
                   plan_active=cc('plan_active', bool), plan_family=cc('plan_family', np.int16),
                   plan_role=cc('plan_role', np.int8), plan_act=cc('plan_act', np.int32),
                   plan_onset=cc('plan_onset', bool), trigger_prob=cc('trigger_prob', np.float32))
        # a_next: measured accel of the same vehicle at the next frame
        a_next = np.full(out['cav_step'].shape[0], np.nan, np.float32)
        for k in range(a_next.shape[0]):
            i = int(out['cav_step'][k])
            if i + 1 < len(F):
                v = out['vid'][out['cav_row'][k]]
                nxt = F[i + 1]
                hit = np.flatnonzero(nxt['vid'] == v)
                if hit.size:
                    a_next[k] = nxt['a_mps2'][hit[0]]
        out['a_next'] = a_next
        S = self.score_rows
        out.update(score_t=np.array([r[0] for r in S], np.int32), score_family=np.array([r[1] for r in S], np.int16),
                   score_prob=np.array([r[2] for r in S], np.float32), score_fire=np.array([r[3] for r in S], bool))
        out['meta_json'] = np.array(json.dumps(self.meta, sort_keys=True, default=str))
        return out


@contextmanager
def distill_tick(logger, cf=None):
    """Wrap ``tools.coopfree.backend.tick`` (the plant's 1 s step) with the read-only logger. ``logger is None``:
    nothing is patched (the eval path stays unchanged)."""
    if logger is None:
        yield None
        return
    if cf is None:
        from tools.coopfree import backend as cf
    original = cf.tick

    def tick(b, controller=None, trace=None):
        env = b.env
        logger.before(env)
        step = env.step

        def spy(accel_by_slot=None, *args, **kwargs):
            logger.sent(accel_by_slot, kwargs.get('lanes'))
            return step(accel_by_slot, *args, **kwargs)
        env.step = spy
        try:
            audit = original(b, controller, trace)
        finally:
            try:
                del env.step                # back to the class method (no instance attribute was there before)
            except AttributeError:
                env.step = step
        logger.after(audit, env)
        logger.tick_done()
        return audit
    cf.tick = tick
    try:
        yield logger
    finally:
        cf.tick = original


def episode_meta(spec, world):
    ck = spec.get('checkpoint') or {}
    return dict(seed=int(world['seed']), world=world, arm=spec.get('arm'), cell_key=world.get('cell_key'),
                warmup_s=300, horizon_s=int(spec.get('sim_s', 1200)), executor_sha256=ck.get('executor_sha256'),
                u0_residual_sha256=ck.get('file_sha256'), learned_trigger_sha256=ck.get('learned_trigger_sha256'),
                model_sha256=(ck.get('learned_trigger_model') or {}).get('file_sha256'),
                bank_sha256=(ck.get('oracle_bank') or {}).get('file_sha256'),
                lane_convention='WM lane 0 = left mainline; label = clip(L(t+1)-L(t),-1,1); '
                                'MAPPO lane choice {0:stay,1:-1,2:+1}', distill_commit=os.environ.get('DISTILL_COMMIT'))


# ---------------------------------------------------------------- reader helpers (DIST-TRAIN)

def load_shard(path):
    with np.load(guard_path(path), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if str(out['schema']) != SCHEMA:
        raise ValueError(f'not a {SCHEMA} shard')
    out['meta'] = json.loads(str(out['meta_json']))
    return out


def actor_states(shard, *, torch_tensors=True):
    """Yield (step index, t, state dict [1,K]) = the MAPPO actor's input (memory-augmented ``ActorStep`` state +
    ``dwell_age_s``); absent slots: present False, zeros, lane -1, length 5."""
    K = int(shard['K'])
    off = shard['off']
    for i in range(len(shard['t'])):
        a, b = int(off[i]), int(off[i+1])
        slot = shard['slot'][a:b].astype(np.int64)
        st = dict(s_m=np.zeros(K, np.float32), v_mps=np.zeros(K, np.float32), a_mps2=np.zeros(K, np.float32),
                  lane=np.full(K, -1, np.int64), present=np.zeros(K, bool), is_cav=np.zeros(K, bool),
                  junction=np.zeros(K, bool), length_m=np.full(K, 5.0, np.float32),
                  cmd_accel_mps2=np.zeros(K, np.float32), lane_intent=np.zeros(K, np.float32),
                  dwell_age_s=np.full(K, float(shard['meta'].get('dwell_s', DEFAULT_DWELL_S)), np.float32))
        st['present'][slot] = True
        for k in ('s_m', 'v_mps', 'a_mps2', 'lane', 'is_cav', 'junction', 'length_m', 'cmd_accel_mps2', 'lane_intent',
                  'dwell_age_s'):
            st[k][slot] = shard[k][a:b]
        if 'entry_queue' in shard:
            st['entry_queue'] = np.array([shard['entry_queue'][i]], np.float32)
        if torch_tensors:
            import torch
            st = {k: torch.as_tensor(v)[None] if v.ndim == 1 and k != 'entry_queue' else torch.as_tensor(v)
                  for k, v in st.items()}
        else:
            st = {k: v[None] if k != 'entry_queue' else v for k, v in st.items()}
        yield i, int(shard['t'][i]), st


# ---------------------------------------------------------------- worker entry (one episode per fresh process)

def checkpoint_from_stage(stage):
    """The a28 LEARNED_TRIGGER checkpoint dict (ops/nm_finaleval.episode_specs, BANK_ARMS branch) from the staged
    copies: BOUND_U0.json (sealed U0 binding), ORACLE_BINDING.json (bank + model + digests), local file paths."""
    u0 = json.loads(Path(stage['bound_u0']).read_text(encoding='utf-8'))
    ob = json.loads(Path(stage['oracle_binding']).read_text(encoding='utf-8'))
    return dict(kind='learned_trigger', comparator_kind=u0['kind'], label=u0['label'], executor_path=u0['path'],
                executor_sha256=u0['sha256'], binding_sha256=u0['binding_sha256'], binding_path=stage['bound_u0'],
                path=stage['residual'], file_sha256=u0['residual_sha256'], update=0, arm='A', cell_key=CELL['cell_key'],
                oracle_bank=dict(path=stage['bank'], file_sha256=ob['bank']['file_sha256']),
                oracle_binding_sha256=ob['receipt_sha256'], learned_trigger_sha256=ob['learned_trigger_sha256'],
                learned_trigger_model=dict(path=stage['model'],
                                           file_sha256=ob['learned_trigger_model']['file_sha256']))


def episode_spec(seed, world, checkpoint):
    return dict(job_id=f'distill_{seed}', arm='LEARNED_TRIGGER', world_id=f'distill_{seed}', world=world,
                checkpoint=checkpoint, sim_s=1200, window=dict(start_s=0, end_s=1200), noise_seed=int(seed),
                deterministic=True, panel_sha256=None, capture=None)


def run_episode(stage_path, seed, work, shard, stop=1200):
    """Fresh interpreter on the collection host: one LEARNED_TRIGGER plant episode (the a28 ``_policy_run``: same
    plant CONFIG, U0 binding, bank, model, code; role 'train', the DISTILL seed range installed in this process) with
    the logger ON. Returns the runner record (+ 'distill' receipt)."""
    from . import map_scenarios as ms
    from . import nm_eval_runner as R
    stage = json.loads(Path(stage_path).read_text(encoding='utf-8'))
    seed, stop = int(seed), int(stop)
    check_seeds([seed])
    if stop != 1200 and not (seed in SMOKE_SEEDS and 300 < stop < 1200):
        raise ValueError('a shortened episode is allowed for the smoke seeds only')
    for k in ('cfg', 'bound_u0', 'oracle_binding', 'residual', 'bank', 'model'):
        guard_path(stage[k])
        if hashlib.sha256(Path(stage[k]).read_bytes()).hexdigest() != stage['sha256'][k]:
            raise ValueError(f'staged {k} differs from its staged sha256')
    cfg = json.loads(Path(stage['cfg']).read_text(encoding='utf-8'))
    install_distill_range(ms)
    world = ms.world(seed, CELL['demand'], CELL['penetration'])
    if world.get('cell_key') != CELL['cell_key']:
        raise ValueError('DISTILL world is not the D2800_p20 cell')
    spec = episode_spec(seed, world, checkpoint_from_stage(stage))
    spec['distill_log'] = dict(path=str(shard), dwell_s=float(stage.get('dwell_s', DEFAULT_DWELL_S)))
    Path(work).mkdir(parents=True, exist_ok=True)
    (Path(work)/'distill_spec.json').write_text(json.dumps(spec, sort_keys=True), encoding='utf-8')
    if stop != 1200:
        spec.update(sim_s=stop, window=dict(start_s=0, end_s=stop))
    return R._policy_run(spec, cfg, work, world=world, stop=stop, role='train')


def main(argv=None):
    p = argparse.ArgumentParser(prog='distill_log')
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('run-episode')
    e.add_argument('--stage', required=True)
    e.add_argument('--seed', required=True, type=int)
    e.add_argument('--stop', type=int, default=1200, help='1200 (collection); 300<stop<1200 for the smoke seeds only')
    e.add_argument('--work', required=True)
    e.add_argument('--shard', required=True)
    e.add_argument('--out', required=True)
    i = sub.add_parser('inspect')
    i.add_argument('shard')
    a = p.parse_args(argv)
    if a.cmd == 'run-episode':
        rec = run_episode(a.stage, a.seed, a.work, a.shard, a.stop)
        Path(a.out).write_text(json.dumps(rec, sort_keys=True, default=str), encoding='utf-8')
        print(json.dumps(dict(status='complete', distill=rec.get('distill'), J_veh_s=rec.get('J_veh_s'),
                              safety=rec.get('safety')), default=str))
    else:
        s = load_shard(a.shard)
        print(json.dumps(dict(meta=s['meta'], steps=int(len(s['t'])), vehicle_rows=int(s['off'][-1]),
                              cav_rows=int(len(s['cav_step'])), plan_rows=int(s['plan_active'].sum()),
                              lane_labels={str(v): int((s['lane_label'][s['lane_valid']] == v).sum()) for v in (-1, 0, 1)},
                              native=int((s['ctrl_src'] == 0).sum()), u0=int((s['ctrl_src'] == 1).sum()),
                              plan=int((s['ctrl_src'] == 2).sum())), default=str, indent=1))


if __name__ == '__main__':
    main()
