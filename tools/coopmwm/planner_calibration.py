"""CPU-only WM P(better) calibration from identity-checked dev receipts.

Operational evidence must be embedded as ``plan.wm_probability_evidence`` BEFORE
verification, so the existing plan/spec/receipt hashes bind it to the candidate.
Old saving_vs_policy and p_best fields are NOT P(better). Missing probability or
full-horizon evidence is unavailable, never repaired from a mean gain or a short
window. The producer must attest the MAPPO-only baseline and scoring objective.
Only first/second all-candidate all8 panels enter the primary population; extra
replicas selected on cooperative outcomes are excluded before any linked read.
No physics, tapes, snapshots, truth files, remote access or device work is needed.
"""

from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
import argparse
import json
import math
from pathlib import Path

from .protocol import digest

SCHEMA = 'coopmwm/planner-calibration/v2'
EVIDENCE_SCHEMA = 'coopmwm/wm-p-better/v1'
PRIMARY_POPULATION = 'all-candidates-before-outcome-selection/v1'
SUMO_SCHEMA = 'coopmwm/sumo_verify/v1'
ALL8_SCHEMA = 'coopmwm/verify-perseed/v1'
EVAL_SEED_RANGES = ((962000, 962999), (964000, 964199), (977000, 977999))
BLOCKED_PATH_PARTS = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
                      'hetdata_truth', 'vtype', 'driver', 'vehicle_params')
DEVELOPMENT_ROLES = ('train', 'dev', 'development', 'calibration', 'discovery')


def safe_path(path):
    """Reject forbidden lexical paths AND symlink targets before any content read."""
    p = Path(path)
    for candidate in (str(p), str(p.resolve())):
        if any(word in candidate.lower() for word in BLOCKED_PATH_PARTS):
            raise ValueError(f'forbidden calibration path: {path}')
    return p


def development_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('world seed must be a nonnegative integer')
    if any(lo <= seed <= hi for lo, hi in EVAL_SEED_RANGES):
        raise ValueError(f'evaluation/sealed seed refused: {seed}')
    return seed


def _allowlist(seeds):
    result = {development_seed(s) for s in seeds}
    if not result:
        raise ValueError('an explicit nonempty development seed allowlist is required')
    return result


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    return float(value)


def _verify(record, key):
    if record.get(key) != digest({k: v for k, v in record.items() if k != key}):
        raise ValueError(f'{key} mismatch')


def _read(path):
    return json.loads(safe_path(path).read_text(encoding='utf-8'))


@dataclass(frozen=True)
class GateContract:
    """The consuming planner supplies its unchanged full horizon and baseline."""
    horizon_s: int
    objective: str
    baseline: str
    policy_state_sha256: str

    def __post_init__(self):
        if (isinstance(self.horizon_s, bool) or not isinstance(self.horizon_s, int)
                or self.horizon_s <= 0 or self.objective != 'team_delay_vs'
                or self.baseline != 'MAPPO-only' or not self.policy_state_sha256):
            raise ValueError('invalid full-horizon objective/baseline contract')


@dataclass(frozen=True)
class WMProbability:
    """Typed paired-sample event, never a scalar gain or p_best.

Each sample is (sample_id, MAPPO-only J, candidate J), same belief/noise draw.
Alternatively a producer may record p_better with n_samples and paired=True.
The envelope itself is sealed and then bound into the verified candidate plan.
"""
    candidate_id: str
    snapshot_sha256: str
    start_s: int
    contract: GateContract
    probability: float
    evidence_sha256: str
    n_samples: int
    duration_s: int

    def __post_init__(self):
        from .mppi_search import final_horizon
        if (not isinstance(self.contract, GateContract)
                or not isinstance(self.candidate_id, str) or not self.candidate_id
                or not isinstance(self.snapshot_sha256, str) or not self.snapshot_sha256
                or isinstance(self.start_s, bool) or not isinstance(self.start_s, int)
                or isinstance(self.n_samples, bool) or not isinstance(self.n_samples, int)
                or isinstance(self.duration_s, bool) or not isinstance(self.duration_s, int)
                or self.duration_s <= 0 or self.contract.horizon_s < final_horizon(self.duration_s)
                or not 300 <= self.start_s < self.start_s+self.contract.horizon_s <= 1200
                or not 0 <= _finite(self.probability, 'WM probability') <= 1
                or self.n_samples < 2 or not self.evidence_sha256):
            raise ValueError('invalid/shortened full-horizon WM probability')

    @classmethod
    def from_dict(cls, evidence):
        _verify(evidence, 'evidence_sha256')
        if evidence.get('schema') != EVIDENCE_SCHEMA or evidence.get('paired') is not True:
            raise ValueError('paired WM probability evidence required')
        contract = GateContract(**evidence['contract'])
        n = evidence['n_samples']
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError('multiple paired WM samples required')
        p = _finite(evidence['p_better'], 'WM P(better)')
        if not 0 <= p <= 1:
            raise ValueError('WM probability outside [0, 1]')
        samples = evidence.get('samples')
        if samples is not None:
            if (not isinstance(samples, list) or len(samples) != n
                    or any(not isinstance(s, (list, tuple)) or len(s) != 3
                           or not isinstance(s[0], str) or not s[0] for s in samples)
                    or len({s[0] for s in samples}) != n):
                raise ValueError('unaligned/duplicate paired sample identities')
            wins = sum(_finite(s[1], 'WM baseline J') > _finite(s[2], 'WM candidate J')
                       for s in samples)
            if not math.isclose(p, wins/n, rel_tol=0., abs_tol=1e-12):
                raise ValueError('recorded probability differs from paired sample outcomes')
        if (not evidence.get('candidate_id') or not evidence.get('snapshot_sha256')
                or isinstance(evidence['start_s'], bool) or not isinstance(evidence['start_s'], int)):
            raise ValueError('missing WM prediction identity')
        return cls(evidence['candidate_id'], evidence['snapshot_sha256'], evidence['start_s'],
                   contract, p, evidence['evidence_sha256'], n, evidence['duration_s'])


@dataclass(frozen=True)
class Example:
    world_seed: int
    verification_id: str
    candidate_id: str
    receipt_sha256: str
    snapshot_sha256: str
    plan_sha256: str
    wm_gain_veh_s: float
    realised_gain_veh_s: float
    window_start_s: int
    window_end_s: int
    wm_prediction: dict | None = None
    population: str = 'unregistered-direct'
    replica: str = 'direct'
    minimum_horizon_s: int = 0
    baseline_is_unmodified_policy: bool = False

    @property
    def better(self):
        return int(self.realised_gain_veh_s > 0)  # ties are not wins


def examples_from_receipt(receipt, specs, dev_seeds):
    """Extract all valid WM-scored group candidates; require own-policy score proof.

    specs are ordinary cand<i>.spec.json branch specifications (not result tapes).
    They prove saving_vs_policy is present, avoiding the legacy receipt fallback to
    the different MAPPO comparator. Failed branches/missing scores are not imputed.
    """
    allowed = _allowlist(dev_seeds)
    _verify(receipt, 'receipt_sha256')
    if receipt.get('schema') != SUMO_SCHEMA:
        raise ValueError('unsupported verification schema')
    ep = receipt['episode']
    seed = development_seed(ep['world']['seed'])
    if seed not in allowed or ep.get('role') not in DEVELOPMENT_ROLES:
        raise ValueError('receipt is not an allowlisted development episode')
    if receipt.get('status') not in ('accepted', 'rejected') or receipt.get('proposer') != 'wm':
        return [], {'unusable_decision': 1}
    ids, predictions, gains = (receipt['candidate_ids'], receipt['predicted_saving'],
                               receipt['J_candidates_veh_s'])
    n = len(ids)
    if not n or len(set(ids)) != n or not all(len(v) == n for v in
            (predictions, gains, receipt['candidate_plan_sha256'], specs)):
        raise ValueError('unaligned candidate fields')
    branches = receipt['branches']
    if len(branches) != n+1 or len(receipt['branch_specs_sha256']) != n+1:
        raise ValueError('unaligned branch fields')
    policy = _finite(receipt['J_policy_veh_s'], 'policy J')
    if branches[0]['branch'] != 'policy' or branches[0]['status'] != 'graded' or branches[0]['J_veh_s'] != policy:
        raise ValueError('invalid policy reference')
    w = receipt['window']
    start, end = w['start_s'], w['end_s']
    if w['metric'] != 'team_delay_vs' or not 300 <= start < end <= 1200:
        raise ValueError('invalid verification window')
    examples, skipped = [], {}
    for i, spec in enumerate(specs):
        _verify(spec, 'branch_sha256')
        plan, branch = spec['plan'], branches[i+1]
        if (spec['branch_sha256'] != receipt['branch_specs_sha256'][i+1]
                or branch['branch_sha256'] != spec['branch_sha256']
                or spec['branch'] != f'cand{i+1}' or branch['branch'] != spec['branch']
                or spec['snapshot_sha256'] != receipt['snapshot_sha256']
                or spec['episode'] != ep or spec['window'] != w
                or spec['verification_id'] != receipt['verification_id']
                or plan['id'] != ids[i] or digest(plan) != receipt['candidate_plan_sha256'][i]
                or spec['plan_sha256'] != digest(plan)):
            raise ValueError('candidate specification identity mismatch')
        if branch['status'] != 'graded' or gains[i] is None:
            skipped['ungraded_candidate'] = skipped.get('ungraded_candidate', 0)+1
            continue
        if plan.get('saving_vs_policy') is None:
            skipped['missing_own_policy_prediction'] = skipped.get('missing_own_policy_prediction', 0)+1
            continue
        predicted = _finite(plan['saving_vs_policy'], 'WM own-policy gain')
        if predicted != _finite(predictions[i], 'predicted saving'):
            raise ValueError('prediction differs from verified candidate')
        realised = policy-_finite(gains[i], 'candidate J')
        if branch['J_veh_s'] != gains[i]:
            raise ValueError('candidate outcome differs from branch')
        if plan.get('wm_probability_evidence') is not None:
            # Validate evidence syntax/semantics even when the row will later be
            # unavailable because the realised verification window is too short.
            probability = WMProbability.from_dict(plan['wm_probability_evidence'])
            if probability.duration_s != plan['duration_s']:
                raise ValueError('WM probability candidate duration mismatch')
        from .mppi_search import final_horizon
        examples.append(Example(seed, receipt['verification_id'], ids[i], receipt['receipt_sha256'],
                                 receipt['snapshot_sha256'], digest(plan), predicted, realised, start, end,
                                 plan.get('wm_probability_evidence'), minimum_horizon_s=final_horizon(plan['duration_s']),
                                 baseline_is_unmodified_policy=spec.get('committed_plan_ids') == []))
    return examples, skipped


def load_verified_records(paths, dev_seeds):
    """Explicit receipt files only; all8 links expand to full group receipts.

    No recursive discovery, no opening tapes/snapshots, no admission-based filtering.
    Absolute links must already be local; relative links are relative to the all8
    file's parent. Missing/integrity-invalid files fail closed rather than selecting
    a more favourable subset. Duplicate receipts (direct + all8) count once.
    """
    allowed = _allowlist(dev_seeds)
    paths = [safe_path(p) for p in paths]
    examples, seen, sources, skipped = [], set(), [], {}
    records = [(p, _read(p)) for p in paths]
    excluded, panels = {}, {}
    # Preflight all manifests before direct receipts: extras cannot sneak back in
    # via an explicitly supplied duplicate, regardless of argument order.
    for path, record in records:
        if record.get('schema') != ALL8_SCHEMA:
            continue
        _verify(record, 'confirmation_sha256')
        if record.get('verify_mode') != 'all8_both_seeds':
            raise ValueError('unsupported all8 verification mode')
        for index, link in sorted(record.get('extra', {}).items()):
            excluded[link['receipt_sha256']] = dict(replica=index, selection='outcome-selected-extra',
                                                   receipt_sha256=link['receipt_sha256'],
                                                   candidate_indices=link.get('candidates'))
        for name in ('first_seed', 'seed2'):
            panels[record[name]['receipt_sha256']] = name
    if excluded.keys() & panels.keys():
        raise ValueError('conflicting primary/outcome-selected replica provenance')

    def read_group(path, expected_sha=None):
        path = safe_path(path)
        r = _read(path)
        _verify(r, 'receipt_sha256')
        if expected_sha is not None and r['receipt_sha256'] != expected_sha:
            raise ValueError('all8 linked receipt hash mismatch')
        # Validate development provenance before skipping even failed/random receipts.
        ep = r['episode']
        seed = development_seed(ep['world']['seed'])
        if seed not in allowed or ep.get('role') not in DEVELOPMENT_ROLES:
            raise ValueError('receipt is not an allowlisted development episode')
        if r['receipt_sha256'] in excluded:
            return seed
        if r['receipt_sha256'] in seen:
            return seed
        usable = r.get('status') in ('accepted', 'rejected') and r.get('proposer') == 'wm'
        specs = []
        if usable:
            vid = r['verification_id']
            if not isinstance(vid, str) or '/' in vid or '\\' in vid or vid in ('.', '..'):
                raise ValueError('unsafe verification id')
            specs = [_read(path.parent.parent/'branches'/vid/f'cand{i+1}.spec.json')
                     for i in range(len(r['candidate_ids']))]
        rows, counts = examples_from_receipt(r, specs, allowed)
        # Standalone receipts have no selection proof. Keep their gain diagnostics
        # but do not permit operational fitting without an all8 primary-panel link.
        replica = panels.get(r['receipt_sha256'], 'unregistered-direct')
        examples.extend(replace(ex, replica=replica,
                                population=PRIMARY_POPULATION if replica in ('first_seed', 'seed2')
                                else 'unregistered-direct') for ex in rows)
        seen.add(r['receipt_sha256'])
        sources.append(dict(path=str(path), receipt_sha256=r['receipt_sha256'], world_seed=seed,
                            n_examples=len(rows), replica=replica,
                            selection='all-candidate-panel' if replica != 'unregistered-direct'
                            else 'unregistered-direct'))
        for k, v in counts.items():
            skipped[k] = skipped.get(k, 0)+v
        return seed

    for path, record in records:
        if record.get('schema') == SUMO_SCHEMA:
            read_group(path)
        elif record.get('schema') == ALL8_SCHEMA:
            _verify(record, 'confirmation_sha256')
            if record.get('verify_mode') != 'all8_both_seeds':
                raise ValueError('unsupported all8 verification mode')
            first, second = record['first_seed'], record['seed2']
            links = [(first, Path(first['branches_dir']).parent), (second, Path(second['root']))]
            seeds = set()
            for link, root in links:
                vid = link['verification_id']
                if not isinstance(vid, str) or '/' in vid or '\\' in vid or vid in ('.', '..'):
                    raise ValueError('unsafe linked verification id')
                root = root if root.is_absolute() else path.parent/root
                seeds.add(read_group(root/'verifications'/f'{vid}.json', link['receipt_sha256']))
            if len(seeds) != 1:
                raise ValueError('all8 replicas disagree on original world seed')
        else:
            raise ValueError('only existing sumo_verify / all8 receipt schemas are accepted')
    return examples, dict(sources=sources, skipped=skipped, n_examples=len(examples),
                          excluded_replicas=sorted(excluded.values(), key=lambda x: x['receipt_sha256']))


def split_by_seed(examples, dev_seeds, validation_fraction=.25, split_seed=20261104):
    """Deterministic grouped split, including CRN replicas and both arms/updates."""
    allowed = _allowlist(dev_seeds)
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    for ex in examples:
        if development_seed(ex.world_seed) not in allowed:
            raise ValueError('non-development example')
        _finite(ex.wm_gain_veh_s, 'WM gain')
        _finite(ex.realised_gain_veh_s, 'realised gain')
    seeds = sorted({ex.world_seed for ex in examples}, key=lambda s: (digest([split_seed, s]), s))
    if len(seeds) < 2:
        raise ValueError('at least two development world seeds are required')
    n = min(len(seeds)-1, max(1, math.ceil(len(seeds)*validation_fraction)))
    validation_seeds = set(seeds[:n])
    return ([ex for ex in examples if ex.world_seed not in validation_seeds],
            [ex for ex in examples if ex.world_seed in validation_seeds])


@dataclass(frozen=True)
class IsotonicCurve:
    """Internal numeric curve. Use Calibrator for the operational planner gate."""
    upper_bounds: tuple
    probabilities: tuple
    min_score: float
    max_score: float

    def __post_init__(self):
        if not self.upper_bounds or len(self.upper_bounds) != len(self.probabilities):
            raise ValueError('invalid isotonic model shape')
        for x in self.upper_bounds:
            _finite(x, 'model bound')
        for p in self.probabilities:
            if not 0 <= _finite(p, 'model probability') <= 1:
                raise ValueError('invalid model probability')
        if (any(a >= b for a, b in zip(self.upper_bounds, self.upper_bounds[1:]))
                or any(a > b for a, b in zip(self.probabilities, self.probabilities[1:]))
                or _finite(self.min_score, 'model minimum') > self.upper_bounds[0]
                or _finite(self.max_score, 'model maximum') != self.upper_bounds[-1]):
            raise ValueError('invalid monotonic model')

    def predict_probability(self, score):
        x = _finite(score, 'isotonic score')
        i = min(bisect_left(self.upper_bounds, x), len(self.upper_bounds)-1)
        return self.probabilities[i]

    def to_dict(self):
        return dict(method='isotonic', **asdict(self))

    @classmethod
    def from_dict(cls, value):
        if value.get('method') != 'isotonic':
            raise ValueError('unsupported calibration method')
        return cls(tuple(value['upper_bounds']), tuple(value['probabilities']),
                   value['min_score'], value['max_score'])


def fit_isotonic(scores, labels):
    """Weighted pool-adjacent-violators, aggregating equal scores before pooling."""
    scores, labels = list(scores), list(labels)
    if not scores or len(scores) != len(labels) or any(y not in (0, 1) for y in labels):
        raise ValueError('nonempty aligned scores and binary labels required')
    pairs = sorted((_finite(x, 'WM gain'), int(y)) for x, y in zip(scores, labels))
    groups = []
    for x, y in pairs:
        if groups and groups[-1][0] == x:
            groups[-1][1] += y
            groups[-1][2] += 1
        else:
            groups.append([x, y, 1])
    blocks = []
    for group in groups:
        blocks.append(group[:])
        while len(blocks) > 1 and blocks[-2][1]/blocks[-2][2] > blocks[-1][1]/blocks[-1][2]:
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([b[0], a[1]+b[1], a[2]+b[2]])
    return IsotonicCurve(tuple(b[0] for b in blocks), tuple(b[1]/b[2] for b in blocks),
                      pairs[0][0], pairs[-1][0])


@dataclass(frozen=True)
class Calibrator:
    curve: IsotonicCurve
    contract: GateContract

    def __post_init__(self):
        if not 0 <= self.curve.min_score <= self.curve.max_score <= 1:
            raise ValueError('calibration input must be WM probability')

    def predict_probability(self, prediction, *, planner_contract, candidate_id, snapshot_sha256, start_s):
        """T1B boundary: typed WM P(better) AND explicit unchanged planner contract.

        A mismatch raises ValueError: the caller must leave MAPPO in control.
        No scalar gain/probability overload and no horizon adaptation is provided.
        """
        if not isinstance(prediction, WMProbability):
            raise ValueError('typed WMProbability required; scalar gains/probabilities refused')
        if planner_contract != self.contract or prediction.contract != self.contract:
            raise ValueError('planner prediction full-horizon objective/baseline mismatch')
        if (prediction.candidate_id != candidate_id or prediction.snapshot_sha256 != snapshot_sha256
                or prediction.start_s != start_s):
            raise ValueError('planner prediction candidate/snapshot/start identity mismatch')
        return self.curve.predict_probability(prediction.probability)

    def to_dict(self):
        return dict(score='wm_p_better', contract=asdict(self.contract), curve=self.curve.to_dict())

    @classmethod
    def from_dict(cls, value):
        if value.get('score') != 'wm_p_better':
            raise ValueError('unsupported calibration input; WM probability required')
        return cls(IsotonicCurve.from_dict(value['curve']), GateContract(**value['contract']))


def _prediction(ex, contract):
    if ex.population != PRIMARY_POPULATION:
        raise ValueError('unregistered or outcome-selected calibration population')
    if ex.wm_prediction is None:
        raise ValueError('planner calibration unavailable: missing paired WM probability evidence')
    if ex.baseline_is_unmodified_policy is not True:
        raise ValueError('realised baseline includes committed moves or lacks baseline provenance')
    p = WMProbability.from_dict(ex.wm_prediction)
    if (p.candidate_id != ex.candidate_id or p.snapshot_sha256 != ex.snapshot_sha256
            or p.start_s != ex.window_start_s):
        raise ValueError('WM probability candidate/snapshot identity mismatch')
    if (p.contract != contract or ex.window_end_s-ex.window_start_s != contract.horizon_s
            or ex.minimum_horizon_s <= 0 or contract.horizon_s < ex.minimum_horizon_s):
        raise ValueError('missing/shortened/mismatched full-horizon evidence')
    return p


def reliability(probabilities, labels, n_bins=10):
    """Equal-width reliability bins, sample-weighted ECE, MCE and Brier score."""
    probabilities, labels = list(probabilities), list(labels)
    if (isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1
            or not probabilities or len(probabilities) != len(labels)
            or any(y not in (0, 1) for y in labels)):
        raise ValueError('aligned nonempty probabilities/binary labels and positive bins required')
    buckets = [[] for _ in range(n_bins)]
    for p, y in zip(probabilities, labels):
        if not 0 <= _finite(p, 'probability') <= 1:
            raise ValueError('probability outside [0, 1]')
        buckets[min(int(p*n_bins), n_bins-1)].append((p, y))
    bins, ece, mce = [], 0., 0.
    for i, bucket in enumerate(buckets):
        mean = sum(p for p, _ in bucket)/len(bucket) if bucket else None
        rate = sum(y for _, y in bucket)/len(bucket) if bucket else None
        gap = abs(mean-rate) if bucket else 0.
        ece += len(bucket)/len(labels)*gap
        mce = max(mce, gap)
        bins.append(dict(lower=i/n_bins, upper=(i+1)/n_bins, count=len(bucket),
                         mean_probability=mean, observed_frequency=rate))
    return dict(n=len(labels), bins=bins, ece=ece, mce=mce,
                brier=sum((p-y)**2 for p, y in zip(probabilities, labels))/len(labels))


def fit_calibration(examples, dev_seeds, validation_fraction=.25, split_seed=20261104, n_bins=10,
                    *, planner_contract):
    """Fit only the fit-world partition; report independent dev-world validation."""
    examples = list(examples)
    if not isinstance(planner_contract, GateContract):
        raise ValueError('explicit planner GateContract required')
    for ex in examples:
        _prediction(ex, planner_contract)
    fit, validation = split_by_seed(examples, dev_seeds, validation_fraction, split_seed)
    if len({ex.better for ex in fit}) < 2:
        raise ValueError('fit partition needs both wins and non-wins')
    curve = fit_isotonic([_prediction(ex, planner_contract).probability for ex in fit],
                         [ex.better for ex in fit])
    model = Calibrator(curve, planner_contract)

    def report(rows):
        value = reliability([model.predict_probability(_prediction(ex, planner_contract),
                              planner_contract=planner_contract, candidate_id=ex.candidate_id,
                              snapshot_sha256=ex.snapshot_sha256, start_s=ex.window_start_s) for ex in rows],
                             [ex.better for ex in rows], n_bins)
        value['raw'] = reliability([_prediction(ex, planner_contract).probability for ex in rows],
                                   [ex.better for ex in rows], n_bins)
        value.update(world_seeds=sorted({ex.world_seed for ex in rows}),
                     out_of_domain=sum(not curve.min_score <= _prediction(ex, planner_contract).probability
                                       <= curve.max_score
                                        for ex in rows))
        return value

    return dict(schema=SCHEMA, model=model.to_dict(), target='realised_gain_veh_s > 0 (ties are non-wins)',
                planner_contract=asdict(planner_contract), population=PRIMARY_POPULATION,
                comparator='same-snapshot MAPPO-only policy continues, paired WM samples',
                split=dict(unit='episode.world.seed (all CRN replicas/candidates/arms/updates together)',
                           seed=split_seed, validation_fraction=validation_fraction,
                           fit_seeds=sorted({ex.world_seed for ex in fit}),
                           validation_seeds=sorted({ex.world_seed for ex in validation})),
                fit=report(fit), validation=report(validation),
                 dataset_sha256=digest(sorted((asdict(ex) for ex in examples), key=lambda x: digest(x))),
                 minimum_horizon_s=max(ex.minimum_horizon_s for ex in examples),
                windows=sorted({(ex.window_start_s, ex.window_end_s) for ex in examples}),
                limitations=[
                    'Selected bank candidates are not the distribution of every online move.',
                    'Isotonic probabilities are empirical; boundary predictions clamp, and 0.8 is not a safety guarantee.',
                    'Model fits fit seeds only; validation seeds are never refitted into this artifact.'])


def gain_diagnostic(examples):
    """Legacy mean-gain analysis only; deliberately no operational model artifact."""
    rows = list(examples)
    return dict(diagnostic_only=True, planner_gate_available=False, n=len(rows),
                mean_wm_gain_veh_s=sum(ex.wm_gain_veh_s for ex in rows)/len(rows) if rows else None,
                observed_win_rate=sum(ex.better for ex in rows)/len(rows) if rows else None)


def reliability_svg(metrics):
    """Standalone SVG reliability diagram from validation bins (no plotting deps)."""
    points = ' '.join(f'{60+400*b["mean_probability"]:.2f},{450-400*b["observed_frequency"]:.2f}'
                      for b in metrics['bins'] if b['count'])
    dots = ''.join(f'<circle cx="{60+400*b["mean_probability"]:.2f}" '
                   f'cy="{450-400*b["observed_frequency"]:.2f}" r="5" fill="#1263ad">'
                   f'<title>n={b["count"]}</title></circle>'
                   for b in metrics['bins'] if b['count'])
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="540" height="530" viewBox="0 0 540 530">'
            '<rect width="540" height="530" fill="white"/>'
            '<g font-family="sans-serif" fill="#172033" font-size="15">'
            f'<text x="60" y="25">Dev seed validation: ECE {metrics["ece"]:.4f}, n={metrics["n"]}</text>'
            '<path d="M60 50V450H460" fill="none" stroke="#172033"/>'
            '<path d="M60 450L460 50" stroke="#999" stroke-dasharray="5 5"/>'
            f'<polyline points="{points}" fill="none" stroke="#1263ad" stroke-width="3"/>'
            f'{dots}'
            '<text x="55" y="475">0</text><text x="455" y="475">1</text>'
            '<text x="35" y="455">0</text><text x="35" y="55">1</text>'
            '<text x="155" y="505">Mean calibrated probability</text>'
            '<text transform="translate(20 340) rotate(-90)">Observed win frequency</text></g></svg>')


def load_calibrator(path, *, planner_contract):
    """Load a CLI artifact after verifying its hash, method and seed partition."""
    artifact = _read(path)
    _verify(artifact, 'artifact_sha256')
    if artifact.get('schema') != SCHEMA:
        raise ValueError('unsupported calibration artifact')
    fit = _allowlist(artifact['split']['fit_seeds'])
    validation = _allowlist(artifact['split']['validation_seeds'])
    if fit & validation:
        raise ValueError('calibration fit/validation seed leakage')
    model = Calibrator.from_dict(artifact['model'])
    if (not isinstance(planner_contract, GateContract) or model.contract != planner_contract
            or artifact.get('planner_contract') != asdict(planner_contract)
            or artifact.get('population') != PRIMARY_POPULATION):
        raise ValueError('artifact planner full-horizon objective/baseline/population mismatch')
    minimum = artifact.get('minimum_horizon_s')
    if (isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0
            or planner_contract.horizon_s < minimum or not artifact.get('windows')
            or any(not 300 <= start < end <= 1200 or end-start != planner_contract.horizon_s
                   for start, end in artifact['windows'])):
        raise ValueError('artifact has incompatible realised horizons')
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', nargs='+', required=True, help='explicit verified/all8 JSON files')
    parser.add_argument('--dev-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', required=True, help='new calibration JSON; existing files refused')
    parser.add_argument('--reliability-svg', required=True, help='new validation reliability SVG')
    parser.add_argument('--validation-fraction', type=float, default=.25)
    parser.add_argument('--split-seed', type=int, default=20261104)
    parser.add_argument('--n-bins', type=int, default=10)
    parser.add_argument('--planner-contract', required=True, help='explicit full-horizon GateContract JSON')
    args = parser.parse_args(argv)
    outputs = [safe_path(args.output), safe_path(args.reliability_svg)]
    if outputs[0].resolve() == outputs[1].resolve():
        raise ValueError('output paths must differ')
    for path in outputs:
        if path.exists() or not path.parent.is_dir():
            raise ValueError('output must be a new file in an existing directory')
    rows, provenance = load_verified_records(args.records, args.dev_seeds)
    artifact = fit_calibration(rows, args.dev_seeds, args.validation_fraction, args.split_seed, args.n_bins,
                               planner_contract=GateContract(**_read(args.planner_contract)))
    artifact['provenance'] = provenance
    artifact['artifact_sha256'] = digest(artifact)
    for path, text in zip(outputs, (json.dumps(artifact, indent=2, allow_nan=False)+'\n',
                                     reliability_svg(artifact['validation']))):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    return artifact


if __name__ == '__main__':
    main()
