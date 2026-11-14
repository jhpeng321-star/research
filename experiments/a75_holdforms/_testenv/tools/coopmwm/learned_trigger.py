"""LEARNED_TRIGGER arm (TRIGCLF 2026-11-09): the ORACLE_WM executor with the PRIVILEGED situation-bin admission replaced
by a frozen classifier on CAV-observable features - a deployable, WM-guided "when to cooperate" trigger.

Executor: identical to ORACLE_WM (oracle_plan.OracleCore, plan_source 'wm': the bank's WM-source rows, the vendored
S-WMVERIFY StrategyCore hold executor, per-family bank session rule, member exclusivity, priority order, U0 base,
HDVs never commanded). The ONLY change is the admission inside the per-core select gate: where ORACLE computes
onset_bin(holder, all plant rows, 'privileged') and checks it against the plan's admitted bins (unconditional rows:
no check), LEARNED_TRIGGER computes p = classifier(features) and starts the plan iff p >= the frozen threshold. The
gate applies to every classifier family, situational AND unconditional rows (the classifier decides when; a bank WM
row whose family / strategy_key is not in the classifier is refused 'not_in_classifier' / 'classifier_key_mismatch').

Features (CAV-observable only): the policy's training featurizer observation block, tools.coopnative.policy.encode
(the context 'x' and 'anchors' that track2/demo_extract records carry; parity-tested against DEMO.json records),
for the plan's holder (member on SUMO lane 2, oracle_plan.holder_of) and its partner member: own state + the six
nearest vehicles within the CAV's 100 m sensing range, + the members' longitudinal spread + the plan one-hot.
encode's RSU block, the intent-model v2v blocks and every privileged quantity (bins, all-vehicle speeds) are NOT used.

Model: standardized logistic regression; parameters, feature order, threshold and the training data digests are in
a frozen JSON model doc (file sha256 pinned by the runner). Inference is pure Python floats in fixed order: no RNG, no
clock, no torch/sklearn at decision time (encode imports torch only for its module). Reads the bank doc, the model
doc, the plant rows and the U0 commands - never job / B / R outputs.

Training data (``collect`` on a host, ``train`` locally): the WMVERIFY observable tapes behind DEMO wmonly
(PLAN sha 777cab92): candidates = (tape, second, family, members) where the plan's own trigger (select_members over
the CAV rows + CAV-detected rows) is feasible; positives = verified-beneficial onsets (paired diff < 0) in admitted
bins (or of the unconditional row) on the plan's own tape; negatives = harmful onsets (diff > 0) on the incumbent /
u0 tapes at the onset second, and other feasible seconds (background). Seeds: verification ranges only, asserted.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math

from . import oracle_plan as O

SCHEMA = 'coopmwm.learned_trigger.v1'
VERIFY_RANGES = ((989950, 989999), (991000, 991999))
FORBIDDEN_SEED_RANGES = ((989600, 989699), (993000, 993999), (964000, 964199), (962000, 962999), (977000, 977999))
BANNED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1')
ENC_FIELDS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
X_DIM, A_DIM = 29, 2


def guard_path(path):
    low = str(path).lower().replace('\\', '/')
    for tok in BANNED:
        if tok in low:
            raise ValueError(f'refusing banned path token {tok!r}: {path}')
    return path


def assert_seeds(seeds):
    """Every seed in the verification ranges, none in a forbidden range (9896xx / 993xxx / 9640xx-9641xx / 962xxx /
    977xxx)."""
    seeds = sorted({int(s) for s in seeds})
    bad = [s for s in seeds if any(a <= s <= b for a, b in FORBIDDEN_SEED_RANGES)]
    if bad:
        raise ValueError(f'forbidden seeds: {bad}')
    out = [s for s in seeds if not any(a <= s <= b for a, b in VERIFY_RANGES)]
    if out:
        raise ValueError(f'seeds outside the verification ranges {VERIFY_RANGES}: {out}')
    return True


def feature_names(families):
    out = []
    for who in ('holder', 'partner'):
        out += [f'{who}_x{k}' for k in range(X_DIM)] + [f'{who}_v', f'{who}_loggap']
    out.append('spread_100m')
    out += [f'plan_{f}' for f in families]
    return out


def encode_rows(rows):
    """The training featurizer's observation block (coopnative.encode) over the plant rows (encode's field whitelist)."""
    from tools.coopnative.policy import encode
    return encode([{k: r[k] for k in ENC_FIELDS} for r in rows])


def candidate_features(enc, members, family, families):
    """Feature vector of starting ``family`` with SUMO-frame ``members`` (select_members output) from ``enc`` =
    encode_rows(plant rows at that second). Uses only the two members' own-sensing rows (x: own state + nearest
    vehicle per lane/direction within 100 m; anchors: own speed, log time gap to the sensed leader)."""
    hold = O.holder_of(members)
    partner = next((c for c in members if c['id'] != hold['id']), hold)
    out = []
    for c in (hold, partner):
        i = enc['members'].index(c['id'])
        out += [float(v) for v in enc['x'][i]] + [float(v) for v in enc['anchors'][i]]
    xs = [float(c['x']) for c in members]
    out.append((max(xs) - min(xs)) / 100.0)
    out += [1.0 if family == f else 0.0 for f in families]
    return out


def predict(model, feats):
    """Logistic probability, pure Python in fixed feature order."""
    z = float(model['intercept'])
    for f, m, s, w in zip(feats, model['mean'], model['scale'], model['coef']):
        z += float(w) * (float(f) - float(m)) / float(s)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def check_model(model):
    if model.get('schema') != SCHEMA:
        raise ValueError('learned trigger model: wrong schema')
    n = len(feature_names(model['families']))
    if model['feature_names'] != feature_names(model['families']):
        raise ValueError('learned trigger model: feature order differs from this code')
    for k in ('mean', 'scale', 'coef'):
        if len(model[k]) != n:
            raise ValueError(f'learned trigger model: {k} has the wrong length')
    if any(float(s) <= 0 for s in model['scale']):
        raise ValueError('learned trigger model: non-positive scale')
    if not (0.0 < float(model['threshold']) < 1.0):
        raise ValueError('learned trigger model: threshold outside (0, 1)')
    return model


def load_model(path, sha256=None):
    raw = open(guard_path(path), 'rb').read()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('learned trigger model file sha256 differs from its registration')
    return check_model(json.loads(raw))


class LearnedCore(O.OracleCore):
    """ORACLE_WM (plan_source 'wm') with the privileged bin admission replaced by classifier(p) >= threshold."""

    def __init__(self, bank_doc, cell_key, model, warmup_s=300):
        self.model = check_model(model)
        self.threshold = float(model['threshold'])
        self.scores = []
        self._enc, self._t = None, None
        super().__init__(bank_doc, cell_key, warmup_s, plan_source='wm')
        keep, refused = [], {'not_in_classifier': [], 'classifier_key_mismatch': []}
        for core, p in zip(self.cores, self.plans):
            if p['family'] not in model['families']:
                refused['not_in_classifier'].append(p['family'])
            elif model['strategy_keys'][p['family']] != p['key']:
                refused['classifier_key_mismatch'].append(p['family'])
            else:
                keep.append((core, p))
        self.cores, self.plans = [c for c, _ in keep], [p for _, p in keep]
        self.refused.update({k: v for k, v in refused.items() if v})

    def prob(self, family, members):
        if self._enc is None:
            self._enc = encode_rows(self._rows)
        return predict(self.model, candidate_features(self._enc, members, family, self.model['families']))

    def _gate(self, core, plan):
        def select(s, free, observed, phase=0):
            mem = O.select_members(s, free, observed, phase=phase)
            if not mem:
                return None
            if any(c['id'] in self._busy for c in mem):     # a member is in another family's session
                self.offers[(plan['family'], 'member_busy')] += 1
                return None
            p = self.prob(plan['family'], mem)
            fire = p >= self.threshold
            self.offers[(plan['family'], 'fire' if fire else 'hold')] += 1
            self.scores.append([self._t, plan['family'], round(p, 6), bool(fire)])
            if not fire:
                return None
            self._last = dict(family=plan['family'], prob=round(p, 6), holder=O.holder_of(mem)['id'],
                              row_class=plan['row_class'], plan_source=plan['plan_source'])
            return mem
        return select

    def step(self, rows, commands, t):
        self._enc, self._t = None, int(t)
        return super().step(rows, commands, t)

    def receipts(self):
        r = super().receipts()
        r.update(schema=SCHEMA, executor='learned_trigger', threshold=self.threshold,
                 model_families=list(self.model['families']), scores=len(self.scores),
                 fires=sum(1 for s in self.scores if s[3]),
                 acts=[dict(a, prob=x.get('prob')) for a, x in zip(r['acts'], self.acts)])
        return r


def make_hook(bank_doc, cell_key, model, warmup_s=300):
    """controller_hook (RULE-arm protocol), identical to oracle_plan.make_hook except the core (LearnedCore).
    ``job`` and ``prefixes`` are never read."""

    class LearnedTriggerHook:
        def __init__(self):
            self.core = LearnedCore(bank_doc, cell_key, model, warmup_s)
            self.controller = None

        def __call__(self, controller, job, prefixes):
            self.controller = controller
            previous = controller.apply

            def apply(rows, commands, now):
                desired = previous(rows, commands, now)
                accel, lanes = self.core.step(rows, commands, int(now))
                cav = {r['vid'] for r in rows if r.get('is_cav')}
                bad = (set(accel) | set(lanes)) - cav
                if bad:
                    raise RuntimeError(f'LEARNED_TRIGGER: request for non-CAV rows {sorted(bad)} (HDVs never commanded)')
                owner = self.core.owners()
                for vid, a in accel.items():
                    desired[vid] = dict(requested_raw=a, acceleration_residual=a - commands[vid],
                                        oracle_plan=owner.get(vid))
                if lanes:
                    targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
                    targets.update(lanes)
                    controller.g5_lane_targets = targets
                return desired

            controller.apply = apply

        def receipts(self):
            return self.core.receipts()

    return LearnedTriggerHook()


EXECUTED = ('feature_names', 'encode_rows', 'candidate_features', 'predict', 'check_model', 'LearnedCore', 'make_hook')
CONSTANTS = ('SCHEMA', 'ENC_FIELDS', 'X_DIM', 'A_DIM')


def learned_trigger_digest(module=None):
    """sha256 of the LEARNED_TRIGGER code as executed: oracle_digest() (the shared executor) + the source of every
    function/class executed here + coopnative.encode / leader_gap (the featurizer) + the constants. The model doc
    is pinned separately by its file sha256."""
    import sys
    from tools.coopnative import policy as P
    m = module or sys.modules[__name__]
    parts = [O.oracle_digest()] + [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts += [inspect.getsource(P.encode), inspect.getsource(P.leader_gap), json.dumps(sorted(P.FIELDS))]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- data (offline)
def tape_rows(line):
    return [{k: r[k] for k in ENC_FIELDS if k in r} for r in line['vehicles']]


def feasible(rows, plans):
    """[(family, members)] of the plans whose own trigger (select_members, phase 0) is feasible at these rows (every
    CAV considered free; the gates see the CAV rows + CAV-detected rows, as in StrategyCore)."""
    cav = [O.to_sumo_row(r) for r in rows if r.get('is_cav')]
    obs = O.observed_rows(rows)
    out = []
    for p in plans:
        mem = O.select_members(p['spec'], cav, obs, phase=0)
        if mem:
            out.append((p['family'], mem))
    return out


def tape_seed_arm(key):
    """(root, cell, seed, arm) from a tape key <root>__<cell>_s<seed>_<arm>."""
    import re
    root, job = key.split('__', 1)
    m = re.match(r'^(.+?)_s(\d{6})_(.+)$', job)
    if m is None:
        raise ValueError(f'bad tape key {key!r}')
    return root, m.group(1), int(m.group(2)), m.group(3)


def collect_tape(path, key, plans, families, ticks=(), every=5):
    """Candidate rows of one tape: every feasible (second, family) at seconds t % every == 0 or t in ``ticks``."""
    import gzip
    root, cell, seed, arm = tape_seed_arm(key)
    assert_seeds([seed])
    ticks = set(int(t) for t in ticks)
    out = []
    with gzip.open(guard_path(path), 'rt', encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            t = int(round(float(d['time_s'])))
            if t < 300 or not (t % every == 0 or t in ticks):
                continue
            rows = tape_rows(d)
            cands = feasible(rows, plans)
            if not cands:
                continue
            enc = encode_rows(rows)
            for fam, mem in cands:
                out.append(dict(tape=key, root=root, cell=cell, seed=seed, arm=arm, t=t, family=fam,
                                members=[c['id'] for c in mem], holder=O.holder_of(mem)['id'],
                                f=[round(v, 7) for v in candidate_features(enc, mem, fam, families)]))
    return out


def _collect_one(args):
    path, key, plans, families, ticks, every = args
    return key, collect_tape(path, key, plans, families, ticks, every)


def cmd_collect(a):
    """Host step (nice 19, <= 8 workers): candidate rows of every tape of the PLAN work list -> one jsonl.gz."""
    import gzip
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    plan = json.loads(Path(guard_path(a.plan)).read_bytes())
    bank = json.loads(Path(guard_path(a.guidance)).read_bytes())
    plans, _ = O.bank_plans(bank, a.cell, 'wm')
    families = sorted(p['family'] for p in plans)
    if a.families:
        plans = [p for p in plans if p['family'] in a.families]
        families = sorted(a.families)
    jobs = []
    for key, w in sorted(plan['work'].items()):
        path = Path(a.tapes) / f'{key}.jsonl.gz'
        if path.exists():
            jobs.append((str(path), key, plans, families, w['ticks'], a.every))
    assert_seeds([tape_seed_arm(k)[2] for _, k, *_ in jobs])
    n = 0
    with gzip.open(Path(a.out), 'wt', encoding='utf-8') as fh, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for key, rows in pool.map(_collect_one, jobs, chunksize=2):
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + '\n')
            n += len(rows)
            print(json.dumps(dict(tape=key, rows=len(rows))), flush=True)
    meta = dict(schema=SCHEMA + '.collect', plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),
                guidance_sha256=hashlib.sha256(Path(a.guidance).read_bytes()).hexdigest(), cell=a.cell,
                families=families, strategy_keys={p['family']: p['key'] for p in plans}, every=a.every,
                tapes=len(jobs), missing=len(plan['work']) - len(jobs), rows=n,
                digest=learned_trigger_digest())
    Path(str(a.out) + '.meta.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(meta))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('collect')
    p.add_argument('--plan', required=True)
    p.add_argument('--guidance', required=True)
    p.add_argument('--tapes', required=True)
    p.add_argument('--cell', default='D2800_p20')
    p.add_argument('--families', nargs='*', default=[])
    p.add_argument('--every', type=int, default=5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    cmd_collect(a)


if __name__ == '__main__':
    main()
