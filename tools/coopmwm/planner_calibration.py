"""T1C: CPU-only calibration from existing, identity-checked verification receipts.

Code-level source contract (c4/track-base 13f184b5):
* sumo_verify.verify: predicted_saving[i] comes from saving_vs_policy, falling
  back to saving_vs_mappo; realised gain is J_policy_veh_s-J_candidates_veh_s[i].
  Read ALL graded candidates of accepted AND rejected decisions, never just
  chosen_plan/grade_veh_s (which would select positive outcomes).
* mppi_runtime.label_list8_search and mppi_search final scoring: saving_vs_policy
  is WM mean policy cost minus candidate cost. It is NOT a probability; p_best
  is also not P(candidate beats policy). Fit gain -> empirical P(gain_real > 0).
* verify_perseed.confirm_all8: confirmations_all8 carries first_seed and seed2
  receipt identities, optional extra receipt identities, and per_seed_v1.examples.
  Follow group receipt identities, not admitted-only examples or solo/extended
  outcomes. replay_bank.replicate changes post-t0 RNG, not episode.world.seed:
  split all replicas/candidates/updates/arms by their original world seed.

The fitted probability applies to the stored SUMO verification window. Existing
WM and SUMO horizons may differ (see viability_report); report this limitation,
preserve window provenance, and never shorten an online planner's look-ahead.
No SUMO runner, jobs, dispatch, remote access, policy observations or truth data
are needed. Only explicitly supplied files and linked group receipts are read.
"""

from bisect import bisect_left
from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path

from .protocol import digest

SCHEMA = 'coopmwm/planner-calibration/v1'
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
        examples.append(Example(seed, receipt['verification_id'], ids[i], receipt['receipt_sha256'],
                                receipt['snapshot_sha256'], digest(plan), predicted, realised, start, end))
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
        examples.extend(rows)
        seen.add(r['receipt_sha256'])
        sources.append(dict(path=str(path), receipt_sha256=r['receipt_sha256'], world_seed=seed,
                            n_examples=len(rows)))
        for k, v in counts.items():
            skipped[k] = skipped.get(k, 0)+v
        return seed

    for path in paths:
        record = _read(path)
        if record.get('schema') == SUMO_SCHEMA:
            read_group(path)
        elif record.get('schema') == ALL8_SCHEMA:
            _verify(record, 'confirmation_sha256')
            if record.get('verify_mode') != 'all8_both_seeds':
                raise ValueError('unsupported all8 verification mode')
            first, second = record['first_seed'], record['seed2']
            links = [(first, Path(first['branches_dir']).parent), (second, Path(second['root']))]
            links.extend((x, Path(x['root'])) for _, x in sorted(record.get('extra', {}).items()))
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
    return examples, dict(sources=sources, skipped=skipped, n_examples=len(examples))


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
class Calibrator:
    """P(realised gain > 0 | WM mean own-policy gain); input units are veh-s.

    PAVA step function; out-of-domain scores clamp to the nearest fitted block.
    This API does not accept a raw WM probability in place of a gain.
    """
    upper_bounds: tuple
    probabilities: tuple
    min_gain_veh_s: float
    max_gain_veh_s: float

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
                or _finite(self.min_gain_veh_s, 'model minimum') > self.upper_bounds[0]
                or _finite(self.max_gain_veh_s, 'model maximum') != self.upper_bounds[-1]):
            raise ValueError('invalid monotonic model')

    def predict_probability(self, wm_gain_veh_s):
        x = _finite(wm_gain_veh_s, 'WM mean own-policy gain in veh-s')
        i = min(bisect_left(self.upper_bounds, x), len(self.upper_bounds)-1)
        return self.probabilities[i]

    def to_dict(self):
        return dict(method='isotonic', score='wm_gain_veh_s', **asdict(self))

    @classmethod
    def from_dict(cls, value):
        if value.get('method') != 'isotonic' or value.get('score') != 'wm_gain_veh_s':
            raise ValueError('unsupported calibration input/method')
        return cls(tuple(value['upper_bounds']), tuple(value['probabilities']),
                   value['min_gain_veh_s'], value['max_gain_veh_s'])


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
    return Calibrator(tuple(b[0] for b in blocks), tuple(b[1]/b[2] for b in blocks),
                      pairs[0][0], pairs[-1][0])


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


def fit_calibration(examples, dev_seeds, validation_fraction=.25, split_seed=20261104, n_bins=10):
    """Fit only the fit-world partition; report independent dev-world validation."""
    examples = list(examples)
    fit, validation = split_by_seed(examples, dev_seeds, validation_fraction, split_seed)
    if len({ex.better for ex in fit}) < 2:
        raise ValueError('fit partition needs both wins and non-wins')
    model = fit_isotonic([ex.wm_gain_veh_s for ex in fit], [ex.better for ex in fit])

    def report(rows):
        value = reliability([model.predict_probability(ex.wm_gain_veh_s) for ex in rows],
                            [ex.better for ex in rows], n_bins)
        value.update(world_seeds=sorted({ex.world_seed for ex in rows}),
                     out_of_domain=sum(not model.min_gain_veh_s <= ex.wm_gain_veh_s <= model.max_gain_veh_s
                                       for ex in rows))
        return value

    return dict(schema=SCHEMA, model=model.to_dict(), target='realised_gain_veh_s > 0 (ties are non-wins)',
                comparator='same-receipt policy continues; explicit saving_vs_policy required',
                split=dict(unit='episode.world.seed (all CRN replicas/candidates/arms/updates together)',
                           seed=split_seed, validation_fraction=validation_fraction,
                           fit_seeds=sorted({ex.world_seed for ex in fit}),
                           validation_seeds=sorted({ex.world_seed for ex in validation})),
                fit=report(fit), validation=report(validation),
                dataset_sha256=digest(sorted((asdict(ex) for ex in examples), key=lambda x: digest(x))),
                windows=sorted({(ex.window_start_s, ex.window_end_s) for ex in examples}),
                limitations=[
                    'Existing receipts contain mean gains, not raw WM P(better); gain-to-probability calibration.',
                    'Target is the recorded SUMO window; WM and SUMO horizons can differ. No full-horizon guarantee.',
                    'Selected bank candidates are not the distribution of every online move.',
                    'Isotonic probabilities are empirical; boundary predictions clamp, and 0.8 is not a safety guarantee.',
                    'Model fits fit seeds only; validation seeds are never refitted into this artifact.'])


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


def load_calibrator(path):
    """Load a CLI artifact after verifying its hash, method and seed partition."""
    artifact = _read(path)
    _verify(artifact, 'artifact_sha256')
    if artifact.get('schema') != SCHEMA:
        raise ValueError('unsupported calibration artifact')
    fit = _allowlist(artifact['split']['fit_seeds'])
    validation = _allowlist(artifact['split']['validation_seeds'])
    if fit & validation:
        raise ValueError('calibration fit/validation seed leakage')
    return Calibrator.from_dict(artifact['model'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', nargs='+', required=True, help='explicit verified/all8 JSON files')
    parser.add_argument('--dev-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', required=True, help='new calibration JSON; existing files refused')
    parser.add_argument('--reliability-svg', required=True, help='new validation reliability SVG')
    parser.add_argument('--validation-fraction', type=float, default=.25)
    parser.add_argument('--split-seed', type=int, default=20261104)
    parser.add_argument('--n-bins', type=int, default=10)
    args = parser.parse_args(argv)
    outputs = [safe_path(args.output), safe_path(args.reliability_svg)]
    if outputs[0].resolve() == outputs[1].resolve():
        raise ValueError('output paths must differ')
    for path in outputs:
        if path.exists() or not path.parent.is_dir():
            raise ValueError('output must be a new file in an existing directory')
    rows, provenance = load_verified_records(args.records, args.dev_seeds)
    artifact = fit_calibration(rows, args.dev_seeds, args.validation_fraction, args.split_seed, args.n_bins)
    artifact['provenance'] = provenance
    artifact['artifact_sha256'] = digest(artifact)
    for path, text in zip(outputs, (json.dumps(artifact, indent=2, allow_nan=False)+'\n',
                                     reliability_svg(artifact['validation']))):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    return artifact


if __name__ == '__main__':
    main()
