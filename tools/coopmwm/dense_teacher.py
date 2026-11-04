"""Track 2 T2B: DAgger-style dense WM teacher for the T2A trigger head (branch c4/dense-teacher).

Owner spec (tracks_spec.md, Track 2 T2B, MAIN 2026-11-04): during training episodes, at
decision points, the WM planner (T1B scoring, imported lazily) labels the best move per state
over the T1A move library (K candidate moves + 'none'); a REGISTERED fraction
(``dense_teacher_fraction``) of the labels is SUMO-verified through the EXISTING all8 machinery
(``verify_perseed.confirm_pending_all8`` with verify_mode 'all8_both_seeds', sealed per_seed_v1
examples); the labels are imitation targets (cross-entropy classes) for the T2A discrete head;
the R arm is labelled by a RANDOM teacher drawn from the SAME move distribution (same T1A
generator, same K, own seeded stream, uniform over the same candidate support + 'none'), so a
B-vs-R difference comes from WM scoring alone.

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

Labelling rule (mirrors the T1B firing rule): a candidate move is labelled only when its
calibrated P >= ``dense_teacher_p_star`` (owner rule 10-18); maximize expected gain among
eligible moves, then break ties by candidate order; if none clear the gate label 'none'.
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
OBSERVABLE_ROW_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav')

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
    if not 0. <= float(v['dense_teacher_p_star']) <= 1:
        raise ValueError('dense_teacher_p_star must be within [0, 1]')
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
    for key in ('ego_id', 'cav', 'time_s', 'sensing_range_m', 'v2v_partner_ids'):
        if key in obs_state:
            out[key] = copy.deepcopy(obs_state[key])
    return out


def _move_fields(move):
    """(family, members, onset, hold, depth) from a dict-like or attribute-like Move (the real
    Move dataclass is T1A's; both access paths keep this module land-order independent)."""
    get = (lambda key: move[key]) if isinstance(move, dict) else (lambda key: getattr(move, key))
    family = get('family')
    members = list(get('members'))
    onset, hold, depth = float(get('onset')), float(get('hold')), float(get('depth'))
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


def planner_scorer():
    """The T1B WM scoring seam, imported lazily: returns
    ``scorer(obs_state, candidates) -> [{'p_better', 'gain'}, ...]`` (calibrated P(move beats
    MAPPO-only) from T1C + expected gain over the full look-ahead horizon)."""
    try:
        from . import planner_online as po
    except ImportError as e:
        raise ImportError('the WM planner (T1B tools/coopmwm/planner_online.py) is not on this '
                          'branch yet; inject scorer=<callable> (tests only) or land T1B') from e
    fn = getattr(po, 'score_moves', None) or getattr(po, 'score_candidates', None)
    if fn is None:
        raise ValueError('planner_online exposes no score_moves/score_candidates: the seam '
                         'contract is scorer(obs_state, candidates) -> [{"p_better", "gain"}]')
    return fn


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

    def _candidates(self, obs_state):
        """The shared move distribution: BOTH teachers draw their candidates from the SAME T1A
        generator with the same K and their own seeded streams (R fairness, like
        variety.random_mix_candidates)."""
        cands = list(self.moves.candidates(obs_state, self.rng,
                                           int(self.cfg['dense_teacher_candidates'])))
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
    label only when calibrated P >= dense_teacher_p_star, ties by gain then candidate order,
    otherwise 'none')."""

    def __init__(self, moves_mod=None, scorer=None, cfg=None, rng=None, encoder=None):
        super().__init__(moves_mod=moves_mod, cfg=cfg, rng=rng, encoder=encoder)
        self._scorer = scorer

    @property
    def scorer(self):
        if self._scorer is None:
            self._scorer = planner_scorer()
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

    def label_state(self, episode, step, t_s, cav, obs_state):
        obs_state = dict(observable_state(obs_state), cav=str(cav))
        cands = self._candidates(obs_state)
        scores = self._score(obs_state, cands)
        p_star = float(self.cfg['dense_teacher_p_star'])
        gated = [(i, s) for i, s in enumerate(scores) if float(s['p_better']) >= p_star]
        best = max(gated, key=lambda t: (t[1]['gain'], -t[0])) if gated else None
        return self._label(episode, step, t_s, cav, obs_state, cands,
                           0 if best is None else best[0]+1, 'wm',
                           scores=None if best is None else best[1])


class RandomTeacher(_TeacherBase):
    """The R-arm teacher: a uniformly random label over the SAME candidate support + 'none'
    (same T1A generator, same K, own seeded stream). No WM scoring exists here at all - the
    class has no scorer seam, so R's imitation signal is distribution-matched to B's by
    construction and differs only by the (absent) planner."""

    def label_state(self, episode, step, t_s, cav, obs_state):
        obs_state = dict(observable_state(obs_state), cav=str(cav))
        cands = self._candidates(obs_state)
        draw = int(self.choice_rng.integers(0, len(cands)+1))
        return self._label(episode, step, t_s, cav, obs_state, cands, draw, 'random')


def make_teacher(cfg, moves_mod=None, scorer=None, rng=None, encoder=None):
    """Factory used by the T2C loop: ``dense_teacher_random`` -> RandomTeacher (R), else
    WMDenseTeacher (B). Refuses to build anything while the option is off."""
    cfg = dict(TEACHER_REGISTERED, **(cfg or {}))
    split_config(dict(cfg))
    if not cfg.get('dense_teacher'):
        raise ValueError('dense_teacher is OFF: resolve its config (plan_config) first')
    if cfg.get('dense_teacher_random'):
        if scorer is not None:
            raise ValueError('the random teacher takes no WM scorer (R fairness)')
        return RandomTeacher(moves_mod=moves_mod, cfg=cfg, rng=rng, encoder=encoder)
    return WMDenseTeacher(moves_mod=moves_mod, scorer=scorer, cfg=cfg, rng=rng, encoder=encoder)


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

    def add(self, episode, step, t_s, cav, obs_state):
        at = True if self.cadence_s is None else at_decision_point(t_s, self.cadence_s)
        if not at:
            return False
        ident = _label_id(episode, step, t_s, cav)
        if ident in self._ids:
            return False
        self._ids.add(ident)
        self.records.append(dict(episode=episode, step=int(step), t_s=float(t_s), cav=str(cav),
                                 obs_state=observable_state(obs_state), at_decision_point=at))
        return at

    def decision_points(self):
        return [r for r in self.records if r['at_decision_point']]

    def label(self, teacher, decisions_only=True):
        rows = self.decision_points() if decisions_only else list(self.records)
        for r in rows:
            ident = _label_id(r['episode'], r['step'], r['t_s'], r['cav'])
            if ident not in self._labels:
                self._labels[ident] = teacher.label_state(
                    r['episode'], r['step'], r['t_s'], r['cav'], r['obs_state'])
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


def bind_receipt(label, receipt, candidate_index, plan=None, moves_mod=None):
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
    if label['move'] is None:
        if candidate_index is not None or plan is not None:
            raise ValueError('none label binds to policy-continues, not a move')
    else:
        if (candidate_index is None or isinstance(candidate_index, bool)
                or int(candidate_index) != candidate_index or not 0 <= candidate_index < receipt['k']):
            raise ValueError('candidate index outside receipt')
        candidate_index = int(candidate_index)
        if plan is None or digest(plan) != receipt['candidate_plan_sha256'][candidate_index]:
            raise ValueError('candidate plan differs from sealed receipt')
        if list(plan['members']) != label['move']['members']:
            raise ValueError('candidate plan members differ from labelled move')
        moves_mod = default_moves() if moves_mod is None else moves_mod
        if _move_fields(moves_mod.from_plan(plan)) != _move_fields(label['move']):
            raise ValueError('candidate plan does not map to the labelled move')
    vid = receipt['verification_id']
    if label.get('verification_id', vid) != vid:
        raise ValueError('label is already bound to another verification')
    label['verification_id'] = vid
    label['receipt_status'] = receipt['status']
    label['verification_candidate_index'] = candidate_index
    label['verification_receipt_sha256'] = receipt['receipt_sha256']
    label['verification_plan_sha256'] = None if plan is None else digest(plan)
    return label


def verify_selected(labels, fraction, submit_first_seed, verifier, moves_mod=None):
    """T2C hook: select -> first-seed dispatch -> existing all8 confirmation -> consume.

    ``submit_first_seed(label)`` returns ``(sealed_receipt, index, plan)`` and must persist
    the decision/snapshot through the existing bank runner, with its usual ledger/health
    hooks. For 'none', index/plan are None: all8 still measures the candidate support against
    policy-continues. Already bound labels are never dispatched again. Existing all8 pending
    detection handles completed confirmations. The caller must serialize this with its bank
    dispatcher (as for varcollect); this module starts no background jobs.
    """
    selected = select_for_verification(labels, fraction)
    for lb in selected:
        if not lb.get('verification_id'):
            receipt, index, plan = submit_first_seed(lb)
            bind_receipt(lb, receipt, index, plan, moves_mod=moves_mod)
    if any(not lb.get('verified') for lb in selected):
        verifier.flush()
    return verifier.consume(selected) if selected else dict(
        labels=0, unbound=0, missing=0, sealed=0, replicated=0, flipped=0)


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
        """Read the sealed records onto the bound labels: verified flag, per-seed admission
        (``verified_replicated`` = admitted on BOTH seeds 1/2), and ``verified_flip`` when the
        labelled candidate lost to another admitted candidate."""
        from . import verify_perseed as vp
        recs = vp.records(self.root, self.arm)
        summary = dict(labels=len(labels), unbound=0, missing=0, sealed=0, replicated=0, flipped=0)
        for lb in labels:
            vid = lb.get('verification_id')
            if vid is None:
                summary['unbound'] += 1
                continue
            rec = recs.get(vid)
            if rec is None:
                summary['missing'] += 1
                continue
            summary['sealed'] += 1
            if rec['first_seed']['receipt_sha256'] != lb['verification_receipt_sha256']:
                raise ValueError('all8 record differs from bound first-seed receipt')
            cand = lb.get('verification_candidate_index')
            mine = [e for e in rec['per_seed_v1']['examples'] if e['candidate_index'] == cand]
            if cand is not None and ({1, 2} - {e['seed_index'] for e in mine}):
                raise ValueError('all8 record is missing candidate seed examples')
            from .protocol import digest
            if any(digest(e['plan']) != lb['verification_plan_sha256'] for e in mine):
                raise ValueError('all8 candidate differs from bound plan')
            admitted = sorted(e['seed_index'] for e in mine if e.get('admitted'))
            replicated = {1, 2} <= set(admitted)
            others = any(e.get('admitted') for e in rec['per_seed_v1']['examples']
                         if e['candidate_index'] != cand)
            lb.update(verified=True, verified_admitted_seeds=admitted,
                      verified_replicated=replicated,
                      verified_cooperative=bool(any(e.get('cooperative') is True
                                                    for e in mine if e.get('admitted'))),
                      verified_flip=bool((not replicated) and others))
            summary['replicated'] += int(replicated)
            summary['flipped'] += int(lb['verified_flip'])
        return summary


def verification_quote(labels, k, duration_s=120, members=2, end_s=1200, extra_seeds=0):
    """Upper-bound all8_both_seeds sim-s quote for the SELECTED labels: every candidate wins
    on every seed (including all solos and extra seeds), charged via
    the real ``verify_perseed.quote_decision`` arithmetic. Pure math - never runs SUMO."""
    from . import verify_perseed as vp
    k = int(k)
    if k < 1:
        raise ValueError('k must be >= 1')
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

def build_dataset(labels, encoder):
    """Imitation targets for the T2A head: one row per label (id, context, CE class target,
    decoded fields, weight 1.0), the head spec and the label provenance counts."""
    targets, histogram = [], {}
    counts = dict(n=len(labels), moves=0, none=0, wm=0, random=0, verified=0, replicated=0,
                  selected=0)
    for lb in labels:
        t = lb['target']
        histogram[t['action_class']] = histogram.get(t['action_class'], 0)+1
        counts['moves'] += int(lb['move'] is not None)
        counts['none'] += int(lb['move'] is None)
        counts[lb['source']] += 1
        counts['verified'] += int(bool(lb.get('verified')))
        counts['replicated'] += int(bool(lb.get('verified_replicated')))
        counts['selected'] += int(bool(lb.get('selected_for_verification')))
        targets.append(dict(id=lb['id'], episode=lb['episode'], step=lb['step'], t_s=lb['t_s'],
                            cav=lb['cav'], source=lb['source'], action_class=t['action_class'],
                            family_index=t['family_index'], buckets=t['buckets'],
                            members=t['members'], obs_state=copy.deepcopy(lb['obs_state']),
                            candidates=copy.deepcopy(lb['candidates']),
                            candidate_class=lb['candidate_class'],
                            n_candidate_classes=lb['n_candidate_classes'],
                            weight=1.0, verified=bool(lb.get('verified')),
                            verified_replicated=bool(lb.get('verified_replicated')),
                            selected_for_verification=bool(lb.get('selected_for_verification'))))
    return dict(schema=SCHEMA, spec=dict(encoder.spec(),
                candidate_target='candidate_class: 0=none, i+1=candidates[i]; exact members/params',
                action_target='action_class: family/parameter buckets; members is an auxiliary target'),
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
