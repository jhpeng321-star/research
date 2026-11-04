"""Track 2 T2B: DAgger-style dense WM teacher for the T2A trigger head (branch c4/dense-teacher).

Owner spec (tracks_spec.md, Track 2 T2B, MAIN 2026-11-04): during training episodes, at
decision points, the WM planner (T1B scoring, imported lazily) labels the best move per state
over the T1A move library (K candidate moves + 'none'); a REGISTERED fraction
(``dense_teacher_fraction``) of the labels is SUMO-verified through the EXISTING all8 machinery
(``verify_perseed.confirm_pending_all8`` with verify_mode 'all8_both_seeds', sealed per_seed_v1
examples); the labels are imitation targets (cross-entropy classes) for the T2A discrete head;
the R arm is labelled by a RANDOM teacher drawn from the SAME move distribution (same T1A
generator, same K, own seeded stream, uniform over the same candidate support + 'none'), so a
B-vs-R selection differs only in WM ranking; both use the shared calibrated eligibility gate.

DEFAULT OFF: every knob is a registered ``dense_teacher_*`` config key and ``split_config``
mirrors ``variety.split_config`` (unknown keys refused, everything validated even while off,
``None`` when off so callers stay byte-identical). No SUMO and no torch anywhere in here; the
all8 path is exercised in tests on the deterministic plants of ``test_verify_perseed_cpu``.

T1A (``tools/coopmwm/moves.py``) and T1B (``tools/coopmwm/planner_online.py``) are parallel
tasks: this module codes against their registered APIs through two LAZY seams, both injectable
(test stubs only, never in the module):

  moves_mod.candidates(obs_state, rng, n) -> list[Move]      T1A API (tracks_spec.md); a Move is
  moves_mod.FAMILIES                                         (family, members, onset, hold, depth)
                                                             and FAMILIES is the variety families
                                                             + 'none'
  scorer(obs_state, candidates) -> [{'p_better': float,      T1B scoring seam: the CALIBRATED
                     'gain': float}, ...]                    P(move beats MAPPO-only) and
                                                             expected gain, aligned with the
                                                              candidate list

``make_teacher(..., planner_kwargs={'rollout': cpu_rollout, 'calibration': fitted, ...})``
binds the concrete T1B API through ``planner_scorer``. For T2A imitation, pass the actual
``TriggerBatch.tables[i]`` to ``label_state(..., candidates=...)`` (or DAggerBuffer.add),
then pass ``trigger_targets(labels, batch)`` to ``TriggerPolicy.imitation_loss``. Fresh
candidate draws are supported for standalone labelling but must never index another table.

Labelling rule (mirrors the T1B firing rule): a candidate move is labelled only when its
calibrated P >= ``dense_teacher_p_star`` (owner rule 10-18); maximize expected gain among
eligible moves, then break ties by candidate order; if none clear the gate label 'none'.
That firing fallback is an ABSTENTION for imitation, not a supervised comparator target.
Both CE/export boundaries require calibrated P >= 0.8 (the minimum cannot be lowered).
Only correctly bound, completed all8 verification bypasses that confidence requirement.
DAgger: the labelled states are decision points of the CURRENT policy's OWN training rollouts
(collected by the T2C loop into ``DAggerBuffer``), never expert states.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 'coopmwm/dense-teacher/v1'
NONE_FAMILY = 'none'
OBSERVABLE_ROW_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
PLANNER_KEYS = ('ego_id', 'group', 'rows', 'time_s', 'world_id', 'state_id', 'commands',
                'policy_residuals', 'policy_state', 'history', 'committed_offsets', 'policy_radio')
MIN_IMITATION_P = 0.8

# Registered option config (all of it; unknown dense_teacher_* keys are refused even when the
# option is off, mirroring coop_search.split_config / variety.split_config).
TEACHER_REGISTERED = dict(
    dense_teacher=False,             # master switch (OFF: nothing here ever runs)
    dense_teacher_random=False,      # R arm: random teacher over the SAME move distribution
    dense_teacher_fraction=0.25,     # registered SUMO-verified fraction (all8_both_seeds)
    dense_teacher_candidates=8,      # K candidate moves scored per decision point (+ 'none')
    dense_teacher_p_star=0.8,        # owner rule 10-18 gate, same as the T1B firing rule
    dense_teacher_buckets=4,         # CE head buckets per (onset, hold, depth) dimension
    dense_teacher_cadence_s=60,      # registered decision-interval cadence (T1B cadence)
    dense_teacher_seed=20261104,     # teacher rng stream (candidates + random teacher draws)
    dense_teacher_extra_seeds=0,     # all8 CRN seeds 3..2+n pass-through (DEFAULT 0 = off)
)


def default_windows():
    """Bucket edges for the CE head target space: the REGISTERED unsourced variety ranges
    (the same windows the move library samples from - no new hand-set thresholds)."""
    from .variety import VARIETY_REGISTERED as V
    return dict(onset_s=tuple(V['variety_onset_window_s']),
                hold_s=tuple(V['variety_hold_window_s']),
                depth=tuple(V['variety_depth_window']))


def split_config(cfg):
    """Pop every ``dense_teacher_*`` key from ``cfg`` (in place). Returns the resolved teacher
    config, or None when the option is off (then cfg carries no dense_teacher key:
    byte-identical output). Mirrors ``variety.split_config``."""
    raw = {k: cfg.pop(k) for k in list(cfg)
           if k == 'dense_teacher' or k.startswith('dense_teacher_')}
    unknown = set(raw)-set(TEACHER_REGISTERED)
    if unknown:
        raise ValueError(f'unregistered dense_teacher config keys {sorted(unknown)}')
    v = dict(TEACHER_REGISTERED, **raw)
    for key in ('dense_teacher', 'dense_teacher_random'):
        if not isinstance(v[key], bool):
            raise ValueError(f'{key} must be a bool')
    for key in ('dense_teacher_candidates', 'dense_teacher_buckets', 'dense_teacher_seed',
                'dense_teacher_extra_seeds'):
        x = v[key]
        if isinstance(x, bool) or not math.isfinite(float(x)) or int(x) != float(x):
            raise ValueError(f'{key} must be an integer')
        v[key] = int(x)
    if v['dense_teacher_seed'] < 0:
        raise ValueError('dense_teacher_seed must be >= 0')
    if not 0. <= float(v['dense_teacher_fraction']) <= 1:
        raise ValueError('dense_teacher_fraction must be within [0, 1]')
    if int(v['dense_teacher_candidates']) < 1:
        raise ValueError('dense_teacher_candidates must be >= 1')
    if not MIN_IMITATION_P <= float(v['dense_teacher_p_star']) <= 1:
        raise ValueError('dense_teacher_p_star must be within [0.8, 1] (owner minimum)')
    if int(v['dense_teacher_buckets']) < 2:
        raise ValueError('dense_teacher_buckets must be >= 2')
    if not (math.isfinite(float(v['dense_teacher_cadence_s']))
            and float(v['dense_teacher_cadence_s']) > 0):
        raise ValueError('dense_teacher_cadence_s must be > 0')
    if int(v['dense_teacher_extra_seeds']) < 0:
        raise ValueError('dense_teacher_extra_seeds must be >= 0')
    if not v.get('dense_teacher'):
        return None
    return v


def plan_config(config=None):
    """Resolved teacher config from a merged config dict (a COPY is split; the caller's dict
    keeps its keys). None when the option is off."""
    cfg = dict(config or {})
    return split_config(cfg)


def at_decision_point(t_s, cadence_s):
    """The registered decision-interval cadence (60 s): teacher labels land only on these ticks."""
    t, cadence = float(t_s), float(cadence_s)
    if not (math.isfinite(t) and t >= 0 and math.isfinite(cadence) and cadence > 0):
        raise ValueError('time must be finite/nonnegative and cadence finite/positive')
    return math.isclose(t/cadence, round(t/cadence), rel_tol=0., abs_tol=1e-9)


def observable_rows(obs_state):
    """The teacher's stored per-label observation: ONLY the allowlisted observable row keys
    (V2V/sensing state). No hidden driver type, vType or global-state field ever enters a label."""
    if not isinstance(obs_state, dict) or 'rows' not in obs_state:
        raise TypeError('obs_state must be a dict with observable "rows"')
    return copy.deepcopy([{k: row[k] for k in OBSERVABLE_ROW_KEYS if k in row}
                          for row in obs_state['rows']])


def observable_state(obs_state):
    """Sensing/V2V boundary shared by capture, candidate generation AND WM scoring.

    ``rows`` must already be the acting CAV's locally visible, timestamp-aligned reports,
    not a global scene. T2C owns visibility/t-1 filtering. Only observable fields cross this
    seam; a hidden field on a supplied row is never forwarded to a dependency.
    """
    out = {'rows': observable_rows(obs_state)}
    for key in (*PLANNER_KEYS, 'self', 'cav', 'sensing_range_m', 'v2v_partner_ids'):
        if key in ('rows', 'history'):
            continue
        if key in obs_state:
            out[key] = copy.deepcopy(obs_state[key])
    if 'history' in obs_state:
        out['history'] = [dict(time_s=f['time_s'], rows=observable_rows(f),
                               entrance_counts=copy.deepcopy(f.get('entrance_counts')))
                           for f in obs_state['history']]
    if 'policy_radio' in out:
        _validate_radio(out)
    return out


def _validate_radio(record):
    """Validate the local continuation at capture, before it can enter a stored label.

    Use the production T1B contract, rather than retaining an unvalidated global board.
    A supplied radio payload requires that dependency even on the capture-only path.
    """
    from .planner_online import validate_observation
    validate_observation({k: record[k] for k in PLANNER_KEYS if k in record})


def _move_fields(move):
    """(family, members, onset, hold, depth) from a dict-like or attribute-like Move (the real
    Move dataclass is T1A's; both access paths keep this module land-order independent)."""
    get = (lambda key: move[key]) if isinstance(move, dict) else (lambda key: getattr(move, key))
    family = get('family')
    members = list(get('members'))
    def parameter(name):
        try:
            return get(name)
        except (KeyError, AttributeError):
            return get(name+'_s')
    onset, hold, depth = float(parameter('onset')), float(parameter('hold')), float(get('depth'))
    for name, x in (('onset', onset), ('hold', hold), ('depth', depth)):
        if not math.isfinite(x):
            raise ValueError(f'move {name} is not finite')
    return family, members, onset, hold, depth


# ---------------------------------------------------------------- CE target space

class TargetEncoder:
    """Fixed discrete head target space for the T2A trigger head: class 0 = 'none', then
    family x (onset, hold, depth) parameter-bucket triples over the REGISTERED variety windows.
    ``spec()`` is the head-size contract for T2A; ``cross_entropy`` validates the targets."""

    def __init__(self, families, buckets=4, windows=None):
        families = tuple(families)
        if not families:
            raise ValueError('TargetEncoder needs at least one move family')
        if NONE_FAMILY in families:
            raise ValueError("'none' is class 0 and is never a family class")
        if len(set(families)) != len(families):
            raise ValueError('duplicate families')
        buckets = int(buckets)
        if buckets < 2:
            raise ValueError('buckets must be >= 2')
        w = dict(default_windows() if windows is None else windows)
        for key in ('onset_s', 'hold_s', 'depth'):
            lo, hi = (float(x) for x in w[key])
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
                raise ValueError(f'window {key} must be an ordered finite (lo, hi) with lo >= 0')
            w[key] = (lo, hi)
        self.families = families
        self.buckets = buckets
        self.windows = w

    @property
    def n_classes(self):
        return 1 + len(self.families)*self.buckets**3

    def bucket(self, x, key):
        lo, hi = self.windows[key]
        if hi == lo:
            return 0
        return int(min(self.buckets-1, max(0, math.floor((float(x)-lo)/(hi-lo)*self.buckets))))

    def encode(self, move):
        """Cross-entropy target of one move (or None = 'none'): action_class plus the decoded
        fields (family index, buckets, members) so T2A can also shape auxiliary heads."""
        if move is None:
            return dict(action_class=0, family=None, family_index=-1, buckets=None, members=[])
        family, members, onset, hold, depth = _move_fields(move)
        if family not in self.families:
            raise ValueError(f'unknown move family {family!r}')
        fi = self.families.index(family)
        b = [self.bucket(onset, 'onset_s'), self.bucket(hold, 'hold_s'), self.bucket(depth, 'depth')]
        cls = 1 + fi*self.buckets**3 + (b[0]*self.buckets + b[1])*self.buckets + b[2]
        return dict(action_class=cls, family=family, family_index=fi, buckets=b, members=list(members))

    def spec(self):
        return dict(n_classes=self.n_classes, families=list(self.families), buckets=self.buckets,
                    windows={k: list(v) for k, v in self.windows.items()})


def softmax(logits):
    z = _logits(logits)
    z = z-z.max()
    e = np.exp(z)
    return e/e.sum()


def cross_entropy(logits, class_index):
    """Numerically stable NLL of ``class_index`` under ``logits`` (numpy; the target-side check
    for the T2A head's imitation loss - the head itself owns its torch CE)."""
    z = _logits(logits)
    if isinstance(class_index, bool) or int(class_index) != class_index or not 0 <= class_index < len(z):
        raise ValueError('class_index is outside the categorical head')
    return float(np.log(np.exp(z-z.max()).sum())-(z[int(class_index)]-z.max()))


def _logits(logits):
    z = np.asarray(logits, float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError('logits must be a nonempty finite vector')
    return z


# ---------------------------------------------------------------- labels

def _label_id(episode, step, t_s, cav):
    from .protocol import digest
    return digest(dict(episode=str(episode), step=int(step), t_s=float(t_s), cav=str(cav)))[:24]


def make_label(episode, step, t_s, cav, obs_state, move, source, encoder, scores=None, draw=None):
    """One DAgger label: identity (deterministic digest id), observable obs rows, the labelled
    move (None = 'none'), its CE target, teacher source and score/draw provenance. Labels never
    carry wall-clock time (byte-stable datasets)."""
    if source not in ('wm', 'random'):
        raise ValueError(f'unknown teacher source {source!r}')
    label = dict(schema=SCHEMA, id=_label_id(episode, step, t_s, cav),
                 episode=episode, step=int(step), t_s=float(t_s), cav=str(cav), source=source,
                 obs_rows=observable_rows(obs_state), move=None, scores=None, draw=None,
                 target=encoder.encode(move), verified=False, selected_for_verification=False)
    if move is not None:
        family, members, onset, hold, depth = _move_fields(move)
        label['move'] = dict(family=family, members=members, onset=onset, hold=hold, depth=depth)
    if scores is not None:
        label['scores'] = dict(p_better=float(scores['p_better']), gain=float(scores['gain']))
    if draw is not None:
        label['draw'] = int(draw)
    return label


# ---------------------------------------------------------------- teacher seams

def default_moves():
    """The T1A move library, imported lazily so this module stays land-order independent."""
    try:
        from . import moves
    except ImportError as e:
        raise ImportError('the move library (T1A tools/coopmwm/moves.py) is not on this branch '
                          'yet; inject moves_mod=<stub> (tests only) or land T1A') from e
    return moves


def planner_scorer(*, rollout=None, calibration=None, config=None, calibrate=None, move_api=None):
    """Bind the concrete T1B scoring API to an EXACT recorded candidate table.

    T1B owns paired WM samples, the full 600-second horizon and mandatory calibration.
    This adapter supplies its candidate provider so no second, mismatched table is sampled.
    ``rollout`` should be T1B.WMRollout(cpu_runtime); no WM is built or dispatched here.
    """
    try:
        from . import planner_online as po
    except ImportError as e:
        raise ImportError('the WM planner (T1B tools/coopmwm/planner_online.py) is not on this '
                          'branch yet; inject scorer=<callable> (tests only) or land T1B') from e
    if rollout is None or calibration is None:
        raise ValueError('T1B scoring requires explicit CPU rollout and fitted calibration')

    def score(obs_state, candidates):
        from dataclasses import replace
        from types import SimpleNamespace
        api = default_moves() if move_api is None else move_api
        fixed = SimpleNamespace(FAMILIES=api.FAMILIES, offsets=api.offsets,
                                candidates=lambda record, rng, n: copy.deepcopy(candidates))
        cfg = po.Config() if config is None else config
        cfg = replace(cfg, n_candidates=len(candidates))
        record = {k: copy.deepcopy(obs_state[k]) for k in PLANNER_KEYS if k in obs_state}
        result = po.score_moves(record, None, rollout=rollout, calibration=calibration,
                                config=cfg, calibrate=calibrate, move_api=fixed)
        return [dict(p_better=s['p_calibrated'], gain=s['expected_gain'],
                     confidence_provenance='planner_calibration/T1C')
                for s in result['scores']]
    return score


class _TeacherBase:
    """Shared plumbing: registered config, seeded rng stream, lazy moves module, lazy CE encoder
    built from moves.FAMILIES (minus 'none', which is class 0)."""

    def __init__(self, moves_mod=None, cfg=None, rng=None, encoder=None):
        resolved = dict(TEACHER_REGISTERED, **(cfg or {}))
        # Constructors also validate, including while disabled; no bypass through the factory.
        split_config(dict(resolved))
        self.cfg = resolved
        self._moves_mod = moves_mod
        self._encoder = encoder
        self.rng = np.random.default_rng(int(self.cfg['dense_teacher_seed'])) if rng is None else rng
        # A separate choice stream makes BOTH arms' candidate streams identical for the same
        # seed and observations across arbitrarily many calls, not merely the first call.
        self.choice_rng = np.random.default_rng(np.random.SeedSequence(
            [int(self.cfg['dense_teacher_seed']), 1]))

    @property
    def moves(self):
        if self._moves_mod is None:
            self._moves_mod = default_moves()
        return self._moves_mod

    @property
    def encoder(self):
        if self._encoder is None:
            families = tuple(f for f in self.moves.FAMILIES if f != NONE_FAMILY)
            self._encoder = TargetEncoder(families, buckets=int(self.cfg['dense_teacher_buckets']))
        return self._encoder

    def _candidates(self, obs_state, supplied=None):
        """The shared move distribution: BOTH teachers draw their candidates from the SAME T1A
        generator with the same K and their own seeded streams (R fairness, like
        variety.random_mix_candidates)."""
        budget = int(self.cfg['dense_teacher_candidates'])
        if supplied is None:
            view = dict(obs_state, self=obs_state['cav'])
            if 'group' in view:
                view['rows'] = [r for r in view['rows'] if r['vid'] in view['group']
                                or not r.get('is_cav')]
            cands = list(self.moves.candidates(view, self.rng, budget))
        else:
            cands = list(copy.deepcopy(supplied))
            if not cands or cands[0] is not None:
                raise ValueError('recorded trigger table must start with None')
            cands = cands[1:]
            while cands and cands[-1] is None:
                cands.pop()
            if any(c is None or _move_fields(c)[0] == NONE_FAMILY for c in cands):
                raise ValueError('recorded trigger table has a hole or duplicate none')
        if len(cands) > budget:
            raise ValueError('candidate table exceeded registered budget')
        fams = self.moves.FAMILIES
        for c in cands:
            family = _move_fields(c)[0]
            if family not in fams:
                raise ValueError(f'candidate family {family!r} outside moves.FAMILIES')
            if family != NONE_FAMILY:
                members = _move_fields(c)[1]
                visible = {str(r['vid']) for r in obs_state['rows'] if r.get('is_cav') is True}
                if not 1 <= len(members) <= 4 or len(set(members)) != len(members):
                    raise ValueError('candidate needs 1..4 distinct CAV members')
                if not set(map(str, members)) <= visible or str(obs_state['cav']) not in set(map(str, members)):
                    raise ValueError('candidate members must be visible CAVs including the acting CAV')
        cands = [c for c in cands if _move_fields(c)[0] != NONE_FAMILY]
        return cands

    def _label(self, episode, step, t_s, cav, obs_state, cands, index, source, scores=None):
        from .protocol import digest
        move = None if index == 0 else cands[index-1]
        lb = make_label(episode, step, t_s, cav, obs_state, move, source, self.encoder,
                        scores=scores, draw=index if source == 'random' else None)
        support = []
        for c in cands:
            f, m, o, h, d = _move_fields(c)
            support.append(dict(family=f, members=m, onset=o, hold=h, depth=d))
        lb.update(candidates=support, candidate_index=None if index == 0 else index-1,
                  candidate_class=index, n_candidate_classes=len(cands)+1,
                  support_sha256=digest(support), obs_state=observable_state(obs_state))
        return lb

    def label_state(self, episode, step, t_s, cav, obs_state):
        raise NotImplementedError


class WMDenseTeacher(_TeacherBase):
    """The B-arm teacher: the WM planner labels the best move per state (T1B scoring seam;
    label only when calibrated P >= dense_teacher_p_star and expected gain > 0;
    maximize gain, ties by candidate order,
    otherwise 'none')."""

    def __init__(self, moves_mod=None, scorer=None, cfg=None, rng=None, encoder=None,
                 planner_kwargs=None):
        super().__init__(moves_mod=moves_mod, cfg=cfg, rng=rng, encoder=encoder)
        self._scorer = scorer
        self._planner_kwargs = dict(planner_kwargs or {})

    @property
    def scorer(self):
        if self._scorer is None:
            self._scorer = planner_scorer(**self._planner_kwargs)
        return self._scorer

    def _score(self, obs_state, cands):
        if not cands:
            return []
        scores = self.scorer(obs_state, cands)
        if len(scores) != len(cands):
            raise ValueError('scorer returned a misaligned score list')
        for s in scores:
            for key in ('p_better', 'gain'):
                if not math.isfinite(float(s[key])):
                    raise ValueError(f'scorer {key} is not finite')
            if not 0 <= float(s['p_better']) <= 1:
                raise ValueError('scorer probability is outside [0, 1]')
        return scores

    def label_state(self, episode, step, t_s, cav, obs_state, candidates=None):
        obs_state = dict(observable_state(obs_state), cav=str(cav), ego_id=str(cav), self=str(cav))
        cands = self._candidates(obs_state, candidates)
        scores = self._score(obs_state, cands)
        p_star = float(self.cfg['dense_teacher_p_star'])
        gated = [(i, s) for i, s in enumerate(scores)
                 if float(s['p_better']) >= p_star and float(s['gain']) > 0]
        best = max(gated, key=lambda t: (t[1]['gain'], -t[0])) if gated else None
        lb = self._label(episode, step, t_s, cav, obs_state, cands,
                         0 if best is None else best[0]+1, 'wm',
                         scores=None if best is None else best[1])
        return self._confidence(lb, scores)

    def _confidence(self, lb, scores):
        from .protocol import digest
        lb['candidate_scores'] = copy.deepcopy(scores)
        selected = lb['candidate_index']
        score = None if selected is None else scores[selected]
        lb['confidence'] = None if score is None else dict(
            p_calibrated=float(score['p_better']),
            provenance=score.get('confidence_provenance', 'injected_calibrated_scorer'),
            support_sha256=lb['support_sha256'], candidate_class=lb['candidate_class'],
            observation_sha256=digest(lb['obs_state']))
        lb['imitation_p_star'] = float(self.cfg['dense_teacher_p_star'])
        lb['abstained'] = selected is None  # no calibrated P for the comparator itself
        lb['original_teacher_target'] = copy.deepcopy(lb['target'])
        return lb


class RandomTeacher(WMDenseTeacher):
    """The R-arm teacher: a uniformly random label over the SAME candidate support + 'none'
    (same T1A generator, same K, own seeded stream). Scoring is ONLY for shared imitation
    eligibility; it never changes the random draw or consumes the proposal RNG."""

    def label_state(self, episode, step, t_s, cav, obs_state, candidates=None):
        obs_state = dict(observable_state(obs_state), cav=str(cav), ego_id=str(cav), self=str(cav))
        cands = self._candidates(obs_state, candidates)
        draw = int(self.choice_rng.integers(0, len(cands)+1))
        scores = self._score(obs_state, cands)
        lb = self._label(episode, step, t_s, cav, obs_state, cands, draw, 'random',
                         scores=None if draw == 0 else scores[draw-1])
        return self._confidence(lb, scores)


def make_teacher(cfg, moves_mod=None, scorer=None, rng=None, encoder=None, planner_kwargs=None):
    """Factory used by the T2C loop: ``dense_teacher_random`` -> RandomTeacher (R), else
    WMDenseTeacher (B). Refuses to build anything while the option is off."""
    cfg = dict(TEACHER_REGISTERED, **(cfg or {}))
    split_config(dict(cfg))
    if not cfg.get('dense_teacher'):
        raise ValueError('dense_teacher is OFF: resolve its config (plan_config) first')
    if cfg.get('dense_teacher_random'):
        return RandomTeacher(moves_mod=moves_mod, scorer=scorer, cfg=cfg, rng=rng,
                             encoder=encoder, planner_kwargs=planner_kwargs)
    return WMDenseTeacher(moves_mod=moves_mod, scorer=scorer, cfg=cfg, rng=rng, encoder=encoder,
                          planner_kwargs=planner_kwargs)


# ---------------------------------------------------------------- DAgger buffer

class DAggerBuffer:
    """Decision-point states of the CURRENT policy's OWN training rollouts (the DAgger part:
    the teacher labels states the learner actually visits, never expert states). The T2C loop
    calls ``add`` at every control tick it wants considered; ``label`` runs the teacher."""

    def __init__(self, cadence_s=None):
        self.cadence_s = None if cadence_s is None else float(cadence_s)
        if self.cadence_s is not None:
            at_decision_point(0, self.cadence_s)
        self.records = []
        self._ids = set()
        self._labels = {}

    def add(self, episode, step, t_s, cav, obs_state, candidates=None):
        at = True if self.cadence_s is None else at_decision_point(t_s, self.cadence_s)
        if not at:
            return False
        ident = _label_id(episode, step, t_s, cav)
        if ident in self._ids:
            return False
        self._ids.add(ident)
        self.records.append(dict(episode=episode, step=int(step), t_s=float(t_s), cav=str(cav),
                                 obs_state=observable_state(obs_state), at_decision_point=at,
                                 candidates=copy.deepcopy(candidates)))
        return at

    def decision_points(self):
        return [r for r in self.records if r['at_decision_point']]

    def label(self, teacher, decisions_only=True):
        rows = self.decision_points() if decisions_only else list(self.records)
        for r in rows:
            ident = _label_id(r['episode'], r['step'], r['t_s'], r['cav'])
            if ident not in self._labels:
                self._labels[ident] = teacher.label_state(
                    r['episode'], r['step'], r['t_s'], r['cav'], r['obs_state'],
                    candidates=r['candidates'])
        return [self._labels[_label_id(r['episode'], r['step'], r['t_s'], r['cav'])] for r in rows]


# ---------------------------------------------------------------- SUMO verification (all8)

def select_for_verification(labels, fraction):
    """Deterministically pick the REGISTERED fraction for SUMO verification: a pure function of
    the label id (same labels -> same subset; order-independent), annotated on every label."""
    fraction = float(fraction)
    if not 0. <= fraction <= 1:
        raise ValueError('fraction must be within [0, 1]')
    from .protocol import digest
    cut = round(fraction*1_000_000)
    out = []
    for lb in labels:
        hit = int(digest(dict(id=lb['id'], purpose='dense-teacher-verify'))[:15], 16) % 1_000_000 < cut
        lb['selected_for_verification'] = bool(hit)
        if hit:
            out.append(lb)
    return out


def prepare_verification_plan(label, snapshot, base_profile=None, *, moves_mod=None, plan_id=None,
                              candidate_plans=None, candidate_base_profiles=None):
    """Freeze the execution contract BEFORE dispatch; return an all8 absolute-profile plan.

    ``base_profile`` is the caller's independently recorded base-policy command schedule,
    member x second, from this exact snapshot (never inferred from the submitted plan).
    The existing branch executor consumes absolute profiles: every command here is exactly
    base + moves.offsets, before its ordinary feasibility/ownership clamp. This certifies
    that explicit base schedule, not an unrecorded future feedback-policy trajectory.
    None uses policy-continues, but must freeze its ENTIRE ordered alternative support via
    ``candidate_plans`` and independently supplied ``candidate_base_profiles`` before dispatch.
    One plan per recorded alternative is required, including duplicates (no implicit dedup).
    """
    from . import sumo_verify as sv
    from .protocol import digest
    if (snapshot['episode']['id'] != label['episode'] or snapshot['t0'] != label['t_s']
            or snapshot['episode'].get('role') != 'train'):
        raise ValueError('execution snapshot belongs to another training decision')
    if snapshot.get('truncate_on_accept'):
        raise ValueError('dense verification cannot truncate existing commitments')
    execution = dict(schema='coopmwm/dense-execution/v1', label_id=label['id'],
                     snapshot_sha256=sv.snapshot_sha(snapshot),
                     policy_sha256=digest(snapshot['policy']), start_s=label['t_s'],
                     control_dt_s=1, mode='policy_continues' if label['move'] is None
                     else 'absolute_profile_from_base_plus_offsets')
    if label['move'] is None:
        if base_profile is not None:
            raise ValueError('none verification uses policy-continues, not a base profile')
        support = label['candidates']
        if (not support or candidate_plans is None or candidate_base_profiles is None
                or len(candidate_plans) != len(support)
                or len(candidate_base_profiles) != len(support)):
            raise ValueError('none requires exact candidate support plans and independent base schedules')
        commitments = []
        for i, (move, submitted, base) in enumerate(zip(
                support, candidate_plans, candidate_base_profiles)):
            # Rebuild from the independent base, never infer it from submitted policy_profile.
            probe = dict(id=submitted.get('dense_execution', {}).get('label_id', label['id']),
                         episode=label['episode'], t_s=label['t_s'], move=copy.deepcopy(move))
            expected = prepare_verification_plan(probe, snapshot, base, moves_mod=moves_mod,
                                                 plan_id=submitted.get('id'))
            if digest(submitted) != digest(expected):
                raise ValueError('none candidate support differs from exact base-plus-offset execution')
            commitments.append(dict(candidate_index=i, move=copy.deepcopy(move),
                                    plan_sha256=digest(expected),
                                    execution=copy.deepcopy(probe['verification_execution'])))
        execution.update(support_sha256=digest(support), candidate_support=commitments)
        plan = None
    else:
        api = default_moves() if moves_mod is None else moves_mod
        family, members, onset, hold, depth = _move_fields(label['move'])
        # Concrete T1A offsets requires its actual Move, not the JSON label dict.
        selected = api.Move(family, tuple(members), int(onset), int(hold), depth)
        if onset != int(onset) or hold != int(hold):
            raise ValueError('execution requires integer move seconds')
        base = np.asarray(base_profile, float)
        if (base.ndim != 2 or base.shape[0] != len(members) or not np.isfinite(base).all()
                or base.shape[1] < math.ceil(onset+hold)
                or snapshot['t0']+base.shape[1] > snapshot.get('end_s', sv.EPISODE_END_S)):
            raise ValueError('base profile must cover the full labelled move, members x seconds')
        duration = base.shape[1]
        offsets = []
        for second in range(duration):
            row = api.offsets(selected, second)
            if set(row) != set(members) or not all(math.isfinite(float(v)) for v in row.values()):
                raise ValueError('move offsets must resolve every member and be finite')
            offsets.append([float(row[v]) for v in members])
        offsets = np.asarray(offsets, float).T
        execution.update(move=copy.deepcopy(label['move']), members=members, duration_s=duration,
                         base_profile=base.tolist(), offset_profile=offsets.tolist(),
                         profile=(base+offsets).tolist())
        plan = dict(members=members, duration_s=duration, profile=execution['profile'],
                    policy_profile=execution['base_profile'],
                    perturbation_profile=execution['offset_profile'], family=family,
                    variety_family=family, dense_execution=copy.deepcopy(execution))
        plan['id'] = plan_id or 'dense_'+digest(execution)[:24]
    if label.get('verification_execution_sha256', digest(execution)) != digest(execution):
        raise ValueError('execution contract is already frozen to this base schedule/snapshot')
    label['verification_execution'] = copy.deepcopy(execution)
    label['verification_execution_sha256'] = digest(execution)
    return copy.deepcopy(plan)


def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _receipt_measurements(receipt, candidate_index):
    """A decision can be rejected/accepted while a particular branch has failed."""
    branches = receipt.get('branches') or []
    indices = [0] + ([candidate_index+1] if candidate_index is not None
                     else list(range(1, receipt['k']+1)))
    outcomes = []
    for i in indices:
        branch = branches[i] if i < len(branches) else {}
        measured = branch.get('status') == 'graded' and _finite_number(branch.get('J_veh_s'))
        values = receipt.get('J_candidates_veh_s') or []
        total = receipt.get('J_policy_veh_s') if i == 0 else (
            values[i-1] if i <= len(values) else None)
        measured = measured and _finite_number(total) and total == branch.get('J_veh_s')
        outcomes.append(dict(branch=branch.get('branch', 'policy' if i == 0 else f'cand{i}'),
                             status=branch.get('status', 'missing'), measured=measured,
                             J_veh_s=total, reason=branch.get('reason')))
    return dict(measured=all(b['measured'] for b in outcomes), branches=outcomes)


def bind_receipt(label, receipt, candidate_index, plan=None, moves_mod=None, *,
                 replace_incomplete=False):
    """Bind a label to the completed first-seed verification receipt its move was executed in
    (the T2C branch runner produces it through the existing sv.verify machinery); the 0-based
    candidate index identifies the labelled move inside that decision."""
    from .exact_grade import verify_sealed
    from .protocol import digest
    verify_sealed(receipt, 'receipt_sha256')
    if receipt['status'] not in ('accepted', 'rejected'):
        raise ValueError('verification must be completed')
    if not label.get('selected_for_verification'):
        raise ValueError('only selected labels may be bound')
    if receipt['t0'] != label['t_s'] or receipt['episode']['id'] != label['episode']:
        raise ValueError('receipt belongs to another decision state')
    if receipt['episode'].get('role') != 'train':
        raise ValueError('dense teacher verification requires training episodes')
    execution = label.get('verification_execution')
    if execution is None or digest(execution) != label.get('verification_execution_sha256'):
        raise ValueError('prepare the exact execution contract before dispatch')
    if execution['snapshot_sha256'] != receipt['snapshot_sha256']:
        raise ValueError('receipt differs from the execution snapshot/base policy')
    if not _target_bound(label):
        raise ValueError('candidate support/target differs from the exact labelled move')
    if label['move'] is None:
        if candidate_index is not None or plan is not None:
            raise ValueError('none label binds to policy-continues, not a move')
        commitments = execution.get('candidate_support', [])
        if (execution.get('support_sha256') != digest(label['candidates'])
                or [c['move'] for c in commitments] != label['candidates']
                or receipt['k'] != len(commitments)
                or receipt['candidate_plan_sha256'] != [c['plan_sha256'] for c in commitments]):
            raise ValueError('receipt differs from frozen none candidate support')
    else:
        if (candidate_index is None or isinstance(candidate_index, bool)
                or int(candidate_index) != candidate_index or not 0 <= candidate_index < receipt['k']):
            raise ValueError('candidate index outside receipt')
        candidate_index = int(candidate_index)
        if plan is None or digest(plan) != receipt['candidate_plan_sha256'][candidate_index]:
            raise ValueError('candidate plan differs from sealed receipt')
        wanted = dict(members=execution['members'], duration_s=execution['duration_s'],
                      profile=execution['profile'], policy_profile=execution['base_profile'],
                      perturbation_profile=execution['offset_profile'],
                      family=label['move']['family'], variety_family=label['move']['family'],
                      dense_execution=execution)
        if (execution.get('move') != label['move']
                or any(plan.get(k) != v for k, v in wanted.items())
                or plan.get('overlap_truncated_plan_ids')):
            raise ValueError('candidate execution differs from the exact labelled move')
    vid = receipt['verification_id']
    if label.get('verification_id', vid) != vid:
        if not replace_incomplete or label.get('verified'):
            raise ValueError('label is already bound to another verification')
    elif replace_incomplete:
        raise ValueError('recovery requires a fresh verification id; receipts are immutable')
    label['verification_id'] = vid
    label['receipt_status'] = receipt['status']
    label['verification_candidate_index'] = candidate_index
    label['verification_receipt_sha256'] = receipt['receipt_sha256']
    label['verification_plan_sha256'] = None if plan is None else digest(plan)
    label['verification_k'] = receipt['k']
    label['verification_first_seed'] = _receipt_measurements(receipt, candidate_index)
    label.update(verified=False, verified_replicated=False, verified_cooperative=False,
                 verified_flip=False, verified_admitted_seeds=[], verification_status='incomplete',
                  verification_seed_status={'1': copy.deepcopy(label['verification_first_seed']),
                                           '2': dict(measured=False, status='pending')})
    label.pop('verification_completion_sha256', None)
    return label


def verify_selected(labels, fraction, submit_first_seed, verifier, moves_mod=None, *,
                    recover_incomplete=None):
    """T2C hook: select -> first-seed dispatch -> existing all8 confirmation -> consume.

    ``submit_first_seed(label)`` returns ``(sealed_receipt, index, plan)`` and must persist
    the decision/snapshot through the existing bank runner, with its usual ledger/health
    hooks. For 'none', index/plan are None: all8 still measures the candidate support against
    policy-continues. Dispatch must first call prepare_verification_plan with an independent
    base schedule/snapshot. Bound labels are not automatically redispatched: all8 seals even
    incomplete outcomes. Optional recover_incomplete(label) explicitly submits a fresh,
    budgeted decision (new plan/verification id), returning the same tuple. Its old binding
    and per-seed failures are retained in verification_history. The caller must serialize with its bank
    dispatcher (as for varcollect); this module starts no background jobs.
    """
    selected = select_for_verification(labels, fraction)
    bound = [lb for lb in selected if lb.get('verification_id')]
    if bound:
        verifier.consume(bound)
    for lb in selected:
        if not lb.get('verification_id'):
            receipt, index, plan = submit_first_seed(lb)
            bind_receipt(lb, receipt, index, plan, moves_mod=moves_mod)
        elif not lb.get('verified') and recover_incomplete is not None:
            previous = copy.deepcopy(lb)
            try:
                receipt, index, plan = recover_incomplete(lb)
                if lb.get('verification_execution_sha256') != previous['verification_execution_sha256']:
                    raise ValueError('recovery must retain the same execution/base snapshot')
                bind_receipt(lb, receipt, index, plan, moves_mod=moves_mod, replace_incomplete=True)
            except Exception:
                lb.clear()
                lb.update(previous)
                raise
            lb.setdefault('verification_history', []).append(
                {k: v for k, v in previous.items() if (k.startswith('verification_')
                 and k != 'verification_history') or k.startswith('verified')})
    if any(not lb.get('verified') for lb in selected):
        verifier.flush()
    return verifier.consume(selected) if selected else dict(
        labels=0, unbound=0, missing=0, sealed=0, verified=0, incomplete=0,
        missing_seed_measurements=0, replicated=0, flipped=0)


class All8Verifier:
    """SUMO verification of the selected fraction through the EXISTING all8 machinery:
    ``verify_perseed.confirm_pending_all8`` (verify_mode 'all8_both_seeds', every candidate +
    policy-continues on both CRN seeds, sealed per_seed_v1 records) and consumption of the
    sealed records back onto the labels. No SUMO is run here - the first-seed receipts and
    snapshots come from the live branch runner (T2C wiring), exactly like varcollect.collect."""

    def __init__(self, root, arm, runner_for, run_root, phase, workers=2, extra_seeds=0, health=None):
        self.root, self.arm, self.runner_for = root, arm, runner_for
        self.run_root, self.phase = run_root, phase
        self.workers, self.extra_seeds, self.health = int(workers), int(extra_seeds), health

    def flush(self, limit=None):
        """Confirm every pending bound decision (accepted OR rejected) -> sealed all8 record."""
        from . import verify_perseed as vp
        return vp.confirm_pending_all8(self.root, self.arm, self.runner_for, self.run_root,
                                       self.phase, workers=self.workers, limit=limit,
                                       health=self.health, extra_seeds=self.extra_seeds)

    def consume(self, labels):
        """Read MEASURED paired outcomes, not merely sealed records, onto bound labels.
        Missing/failed seed outcomes remain incomplete and eligible for explicit recovery.
        A measured statistical/safety rejection is completed verification, not admission.
        Also attach per-seed admission
        (``verified_replicated`` = admitted on BOTH seeds 1/2), and ``verified_flip`` when the
        labelled candidate lost to another admitted candidate."""
        from . import verify_perseed as vp
        from .protocol import digest
        recs = vp.records(self.root, self.arm)
        summary = dict(labels=len(labels), unbound=0, missing=0, sealed=0, verified=0,
                       incomplete=0, missing_seed_measurements=0, replicated=0, flipped=0)
        for lb in labels:
            # Recompute on every consume, including resumed labels with stale success flags.
            lb.update(verified=False, verified_replicated=False, verified_cooperative=False,
                      verified_flip=False, verified_admitted_seeds=[], verification_status='incomplete')
            lb.pop('verification_completion_sha256', None)
            lb['label_correct'] = None
            vid = lb.get('verification_id')
            if vid is None:
                summary['unbound'] += 1
                summary['incomplete'] += 1
                continue
            execution = lb.get('verification_execution')
            if (not execution or digest(execution) != lb.get('verification_execution_sha256')
                    or 'verification_first_seed' not in lb):
                raise ValueError('bound label lacks an exact execution/measurement contract')
            if not _target_bound(lb):
                raise ValueError('candidate support/target differs from the exact labelled move')
            rec = recs.get(vid)
            if rec is None:
                summary['missing'] += 1
                summary['incomplete'] += 1
                lb['verification_seed_status'] = {
                    '1': copy.deepcopy(lb['verification_first_seed']),
                    '2': dict(measured=False, status='record_missing')}
                continue
            summary['sealed'] += 1
            if rec['first_seed']['receipt_sha256'] != lb['verification_receipt_sha256']:
                raise ValueError('all8 record differs from bound first-seed receipt')
            cand = lb.get('verification_candidate_index')
            examples = rec['per_seed_v1']['examples']
            mine = [e for e in examples if e['candidate_index'] == cand and e['seed_index'] in (1, 2)]
            if any(digest(e['plan']) != lb['verification_plan_sha256'] for e in mine):
                raise ValueError('all8 candidate differs from bound plan')
            # None needs the complete paired support, not a vacuous empty `mine` list.
            if cand is None:
                support = execution.get('candidate_support', [])
                if (execution.get('support_sha256') != digest(lb['candidates'])
                        or [c['move'] for c in support] != lb['candidates']
                        or len(support) != lb['verification_k']):
                    raise ValueError('none candidate support commitment changed')
                needed = [c['candidate_index'] for c in support]
                if (rec['K'] != len(support)
                        or [(c['index'], c['plan_sha256']) for c in rec['candidates']]
                        != [(c['candidate_index'], c['plan_sha256']) for c in support]):
                    raise ValueError('all8 none candidate differs from frozen support')
                for e in examples:
                    index = e['candidate_index']
                    if (index not in needed
                            or digest(e['plan']) != support[index]['plan_sha256']):
                        raise ValueError('all8 none candidate differs from frozen support')
            else:
                if execution.get('move') != lb['move']:
                    raise ValueError('verified target differs from exact labelled move')
                needed = [cand]
            statuses = {}
            for seed in (1, 2):
                rows = []
                for index in needed:
                    found = [e for e in examples
                             if e['seed_index'] == seed and e['candidate_index'] == index]
                    if len(found) != 1:
                        rows.append(dict(candidate_index=index, measured=False,
                                         status='missing' if not found else 'duplicate'))
                    else:
                        e = found[0]
                        branch = e.get('tape', {}).get('branch', {})
                        measured = (e.get('status') in ('admitted', 'rejected')
                                    and _finite_number(e.get('advantage_veh_s'))
                                    and _finite_number(e.get('lower95_one_sided_veh_s'))
                                    and _finite_number(e.get('n_units')) and e['n_units'] > 0
                                    and branch.get('status') == 'graded'
                                    and _finite_number(branch.get('J_veh_s'))
                                    and not e.get('later_harm_missing')
                                    and (e['lower95_one_sided_veh_s'] <= 0
                                         or e.get('later_harm') in ('pass', 'fail')))
                        rows.append(dict(candidate_index=index, measured=bool(measured),
                                         status=e.get('status', 'missing'), reason=e.get('reason'),
                                         later_harm=e.get('later_harm'), admitted=bool(e.get('admitted')),
                                         plan_sha256=digest(e['plan']),
                                         advantage_veh_s=e.get('advantage_veh_s'),
                                         lower95_one_sided_veh_s=e.get('lower95_one_sided_veh_s')))
                comparator = (lb['verification_first_seed']['measured'] if seed == 1 else
                              rec['seed2'].get('status') in ('accepted', 'rejected')
                              and _finite_number(rec['seed2'].get('J_policy_veh_s')))
                statuses[str(seed)] = dict(measured=bool(comparator and rows
                                                        and all(r['measured'] for r in rows)),
                                           comparator_measured=bool(comparator), examples=rows,
                                           receipt_sha256=(lb['verification_receipt_sha256']
                                               if seed == 1 else rec['seed2']['receipt_sha256']))
            lb['verification_seed_status'] = statuses
            missing_seeds = sum(not s['measured'] for s in statuses.values())
            if missing_seeds:
                summary['incomplete'] += 1
                summary['missing'] += 1
                summary['missing_seed_measurements'] += missing_seeds
                continue
            admitted = sorted(e['seed_index'] for e in mine if e.get('admitted'))
            replicated = {1, 2} <= set(admitted)
            others = any(e.get('admitted') for e in rec['per_seed_v1']['examples']
                         if e['candidate_index'] != cand and e['seed_index'] in (1, 2))
            lb.update(verified=True, verification_status='measured', verified_admitted_seeds=admitted,
                      verified_replicated=replicated,
                      verified_cooperative=bool(any(e.get('cooperative') is True
                                                    for e in mine if e.get('admitted'))),
                      verified_flip=bool((not replicated) and others))
            # Registered realised outcome: move beats base iff all8 admits it on BOTH CRN
            # seeds (paired lower95/safety/later-harm rules). None is correct only when no
            # alternative is admitted on either required seed; its full support was measured.
            lb['label_correct'] = bool(replicated if cand is not None else not others)
            lb['verification_completion_sha256'] = digest(_completion_payload(lb))
            summary['verified'] += 1
            summary['replicated'] += int(replicated)
            summary['flipped'] += int(lb['verified_flip'])
        return summary


def verification_quote(labels, k, duration_s=None, members=4, end_s=1200, extra_seeds=0):
    """Upper-bound all8_both_seeds sim-s quote for the SELECTED labels: every candidate wins
    on every seed (including all solos and extra seeds), charged via
    the real ``verify_perseed.quote_decision`` arithmetic. Pure math - never runs SUMO."""
    from . import verify_perseed as vp
    k = int(k)
    if k < 1:
        raise ValueError('k must be >= 1')
    if duration_s is None:
        windows = default_windows()
        duration_s = math.ceil(windows['onset_s'][1]+windows['hold_s'][1])
    if int(duration_s) < 1 or not 1 <= int(members) <= 4 or int(extra_seeds) < 0:
        raise ValueError('positive duration, 1..4 members and nonnegative extra_seeds required')
    durations = [int(duration_s)]*k
    mem = [['m%d' % i for i in range(int(members))]]*k
    rows = [vp.quote_decision(int(lb['t_s']), durations, mem, {1: list(range(k)), 2: list(range(k))},
                              first_seed_positive=True, end_s=int(end_s), extra_seeds=int(extra_seeds))
            for lb in labels if lb.get('selected_for_verification')]
    return dict(n_selected=len(rows), k=k,
                all8_total_sim_s=sum(r['all8_total_sim_s'] for r in rows),
                all8_total_with_extra_sim_s=sum(r.get('all8_total_with_extra_sim_s',
                                                      r['all8_total_sim_s']) for r in rows),
                decisions=rows)


# ---------------------------------------------------------------- imitation dataset

def _completion_payload(lb):
    """A bare verified=True is never a ground-truth bypass."""
    keys = ('id', 'episode', 't_s', 'cav', 'source', 'obs_state',
            'move', 'target', 'candidates', 'candidate_class',
            'original_teacher_target', 'verification_id', 'verification_execution',
            'verification_execution_sha256', 'verification_receipt_sha256',
            'verification_plan_sha256', 'verification_candidate_index', 'verification_k',
            'verification_first_seed', 'verification_seed_status', 'label_correct')
    return {k: copy.deepcopy(lb.get(k)) for k in keys}


def _target_bound(lb):
    from .protocol import digest
    support, index = lb.get('candidates', []), lb.get('candidate_class', -1)
    return bool(type(index) is int and 0 <= index <= len(support)
                and lb.get('support_sha256') == digest(support)
                and lb.get('candidate_index') == (index-1 if index else None)
                and lb.get('move') == (support[index-1] if index else None)
                and lb.get('target') == lb.get('original_teacher_target'))


def _completed(lb):
    from .protocol import digest
    statuses = lb.get('verification_seed_status', {})
    return bool(_target_bound(lb) and lb.get('verified') and lb.get('verification_status') == 'measured'
                and all(statuses.get(str(s), {}).get('measured') is True for s in (1, 2))
                and lb.get('verification_completion_sha256') == digest(_completion_payload(lb)))


def imitation_eligibility(lb):
    """Shared, fail-closed CE/dataset boundary. Unverified none means abstention, not class 0.

    Scores supplied through the injectable scorer ABI must already be T1C-calibrated; the
    production factory's planner adapter requires fitted calibration and marks provenance.
    Only a bound, measured all8 completion may bypass confidence, for either arm.
    """
    from .protocol import digest
    if _completed(lb):
        return dict(eligible=True, reason='completed_all8', p_calibrated=None,
                    provenance='all8_both_seeds', verified_bypass=True)
    confidence = lb.get('confidence') or {}
    reason = 'missing_calibration'
    p = confidence.get('p_calibrated')
    threshold = max(MIN_IMITATION_P, float(lb.get('imitation_p_star', MIN_IMITATION_P)))
    index = lb.get('candidate_class', 0)
    candidates = lb.get('candidates', [])
    bound = (_target_bound(lb) and 0 < index <= len(candidates)
             and confidence.get('candidate_class') == index
             and confidence.get('support_sha256') == digest(candidates)
             and confidence.get('observation_sha256') == digest(lb.get('obs_state')))
    valid = (_finite_number(p) and 0 <= p <= 1 and bool(confidence.get('provenance')) and bound)
    if lb.get('move') is None:
        reason = 'abstention_without_measured_none'
    elif valid:
        reason = 'calibrated' if p >= threshold else 'below_calibrated_threshold'
    return dict(eligible=bool(valid and p >= threshold and lb.get('move') is not None),
                reason=reason, p_calibrated=p if valid else None,
                provenance=confidence.get('provenance') if valid else None,
                verified_bypass=False)


def update_receipt(labels, update, arm):
    """T2C learning-monitor payload. Bind once to an update and B/R arm, then recompute.

    Accuracy is against the ORIGINAL teacher choice, never completion or flip rate. The
    denominator contains only selected, correctly bound, fully measured paired checks.
    An empty subset has accuracy=None. All counts are label counts (not candidate/seed rows).
    """
    from .protocol import digest
    if type(update) is not int or update < 0 or arm not in ('B', 'R'):
        raise ValueError('receipt requires nonnegative integer update and B/R arm')
    for lb in labels:
        if lb['source'] != {'B': 'wm', 'R': 'random'}[arm]:
            raise ValueError('label source differs from receipt arm')
        if lb.get('update', update) != update or lb.get('arm', arm) != arm:
            raise ValueError('label is already bound to another update/arm')
    for lb in labels:
        lb.update(update=update, arm=arm)
    decisions = [imitation_eligibility(lb) for lb in labels]
    selected = [lb for lb in labels if lb.get('selected_for_verification')]
    measured = [lb for lb in selected if _completed(lb)]
    correct = sum(lb['label_correct'] is True for lb in measured)
    eligible = sum(e['eligible'] for e in decisions)
    counts = dict(attempted=len(labels), selected=len(selected), measured=len(measured),
                  incomplete=len(selected)-len(measured), eligible=eligible,
                  filtered=len(labels)-eligible, accuracy_correct=correct,
                  accuracy_measured=len(measured))
    return dict(schema='coopmwm/dense-update/v1', update=update, arm=arm, counts=counts,
                filtered_denominator=len(labels),
                filtered_share=(len(labels)-eligible)/len(labels) if labels else None,
                accuracy_numerator=correct, accuracy_denominator=len(measured),
                label_accuracy=correct/len(measured) if measured else None,
                accuracy_definition='move: all8 admission on both required seeds; '
                'none: no support alternative admitted on either required seed',
                labels=[dict(id=lb['id'], original_teacher_target=copy.deepcopy(
                    lb.get('original_teacher_target', lb['target'])),
                    candidate_class=lb['candidate_class'], eligibility=e,
                    selected=bool(lb.get('selected_for_verification')),
                    measured=_completed(lb),
                    correct=lb.get('label_correct') if _completed(lb) else None,
                    verification_id=lb.get('verification_id'),
                    receipt_sha256=lb.get('verification_receipt_sha256'),
                    execution_sha256=lb.get('verification_execution_sha256'),
                    paired_evidence=copy.deepcopy(lb.get('verification_seed_status')),
                    support_sha256=digest(lb['candidates'])) for lb, e in zip(labels, decisions)])


def trigger_targets(labels, batch):
    """Indices for T2A ``TriggerPolicy.imitation_loss(batch, targets)``.

    Refuse a different/reordered table: CE must index the learner's recorded choices,
    not a fresh draw. Labels can be passed in any CAV order. No torch dependency.
    """
    by_cav = {lb['cav']: lb for lb in labels}
    if len(by_cav) != len(labels) or set(by_cav) != set(batch.context['members']):
        raise ValueError('one label per recorded trigger CAV is required')
    targets = []
    for i, (cav, table) in enumerate(zip(batch.context['members'], batch.tables)):
        lb = by_cav[cav]
        actual = [_move_fields(c) for c in table[1:] if c is not None]
        if actual != [_move_fields(c) for c in lb['candidates']]:
            raise ValueError('label candidate support differs from recorded trigger table')
        index = int(lb['candidate_class'])
        if index >= len(table) or not bool(batch.mask[i, index]):
            raise ValueError('label selects a padded/masked trigger slot')
        if not imitation_eligibility(lb)['eligible']:
            raise ValueError('filtered dense label cannot enter trigger CE; subset the batch first')
        targets.append(index)
    if len(targets) != len(batch.context['members']):
        raise ValueError('trigger table count mismatch')
    return targets


def build_dataset(labels, encoder, *, update=None, arm=None):
    """Export ONLY eligible imitation targets, retaining ALL original labels for audit/checks.

    Pass update/arm to bind and attach T2C's per-update accuracy/filter receipt. Omission is
    supported for standalone datasets, with explicit null update/arm fields.
    """
    if (update is None) != (arm is None):
        raise ValueError('supply both update and arm for a learning-monitor dataset')
    receipt = None if update is None else update_receipt(labels, update, arm)
    targets, histogram = [], {}
    counts = dict(n=len(labels), moves=0, none=0, wm=0, random=0, verified=0, replicated=0,
                  selected=0, incomplete=0, eligible=0, filtered=0)
    for lb in labels:
        t = lb['target']
        histogram[t['action_class']] = histogram.get(t['action_class'], 0)+1
        counts['moves'] += int(lb['move'] is not None)
        counts['none'] += int(lb['move'] is None)
        counts[lb['source']] += 1
        completed = _completed(lb)
        counts['verified'] += int(completed)
        counts['replicated'] += int(bool(lb.get('verified_replicated')))
        counts['selected'] += int(bool(lb.get('selected_for_verification')))
        counts['incomplete'] += int(bool(lb.get('selected_for_verification'))
                                    and not completed)
        eligibility = imitation_eligibility(lb)
        counts['eligible'] += int(eligibility['eligible'])
        counts['filtered'] += int(not eligibility['eligible'])
        if not eligibility['eligible']:
            continue
        targets.append(dict(id=lb['id'], episode=lb['episode'], step=lb['step'], t_s=lb['t_s'],
                            cav=lb['cav'], source=lb['source'], action_class=t['action_class'],
                            family_index=t['family_index'], buckets=t['buckets'],
                            members=t['members'], obs_state=copy.deepcopy(lb['obs_state']),
                            candidates=copy.deepcopy(lb['candidates']),
                            candidate_class=lb['candidate_class'],
                            n_candidate_classes=lb['n_candidate_classes'],
                            update=lb.get('update'), arm=lb.get('arm'), eligibility=eligibility,
                            weight=1.0, verified=bool(lb.get('verified')),
                            verified_replicated=bool(lb.get('verified_replicated')),
                            selected_for_verification=bool(lb.get('selected_for_verification'))))
    return dict(schema=SCHEMA, spec=dict(encoder.spec(),
                candidate_target='candidate_class: 0=none, i+1=candidates[i]; exact members/params',
                action_target='action_class: family/parameter buckets; members is an auxiliary target'),
                update=update, arm=arm, update_receipt=receipt,
                counts=counts, labels=copy.deepcopy(labels),
                 histogram={str(k): v for k, v in sorted(histogram.items())}, targets=targets)


def save_dataset(dataset, path):
    """Canonical JSON with a content seal (digest of the payload; byte-stable, no timestamps)."""
    from .protocol import digest
    from tools.coopnative.storage import canonical
    payload = {k: v for k, v in dataset.items() if k != 'sha256'}
    Path(path).write_text(canonical(dict(payload, sha256=digest(payload))), encoding='utf-8')
    return str(path)


def load_dataset(path):
    from .protocol import digest
    blob = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = blob.pop('sha256', None)
    if blob.get('schema') != SCHEMA:
        raise ValueError('not a dense-teacher dataset')
    if sha != digest(blob):
        raise ValueError('dataset seal mismatch')
    return blob
