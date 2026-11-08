"""LEARNED_TRIGGER (TRIGCLF): featurizer parity with the training featurizer records (DEMO.json) and with the
training-data path, the frozen model doc, seed guards, executor identity with ORACLE_WM except the admission,
determinism, and that the gate never reads job / B / R outputs or privileged plant state (CPU only, no SUMO)."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tools.coopmwm import learned_trigger as L
from tools.coopmwm import oracle_plan as O
from tools.coopmwm.test_oracle_plan_cpu import FREE, bank, cond, role

HERE = Path(__file__).parent
FIXTURE = HERE / 'fixtures' / 'learned_trigger_parity.json'
MODEL = HERE.parent.parent / 'ops' / 'learned_trigger' / 'LEARNED_TRIGGER_MODEL.json'
GUIDANCE = Path('E:/agent-work/c4-fullpol/clean5c/priority/GUIDANCE_UNION_R6_UNC_NOF2B_PRIORITY.json')
#: learned_trigger_digest() prefix; any change to an executed function / constant / the featurizer moves it
PINNED = '41fc8d1a2fb0'


def fixture():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def synthetic_model(families, threshold=0.5, coef=None):
    n = len(L.feature_names(families))
    return dict(schema=L.SCHEMA, families=list(families), strategy_keys={}, feature_names=L.feature_names(families),
                mean=[0.0] * n, scale=[1.0] * n, coef=coef or [0.0] * n, intercept=0.0, threshold=threshold)


def wm_rows(speed=25.0):
    rows = [dict(vid=f'c{ln}', s_m=-1000.0 + 10 * i, v_mps=speed, lane=ln, is_cav=True, length_m=5.0)
            for i, ln in enumerate((0, 1, 2))]
    for k, s in enumerate((-950.0, -500.0, -300.0, -100.0, 0.0)):
        for ln in (0, 1):
            rows.append(dict(vid=f'h{ln}_{k}', s_m=s, v_mps=speed, lane=ln, is_cav=False, length_m=5.0))
    return rows


def fam_bank(fams=('mut_a',)):
    doc = bank([(f, 100.0 - i, cond([role(1), role(2)], FREE), None) for i, f in enumerate(fams)])
    return doc


def keyed(model, doc):
    keys = {t['family']: t['template']['source']['strategy_key'] for t in doc['clean5c']['templates']}
    return dict(model, strategy_keys={f: keys[f] for f in model['families']})


# ----------------------------------------------------------------------------------------------- parity
def test_featurizer_parity_with_training_featurizer_records():
    """encode_rows(tape rows at the decision second) == the context x / anchors the training featurizer
    (demo_extract: track2_backend.episode_components over the v2v Controller) recorded in DEMO.json."""
    fx = fixture()
    assert fx['items']
    for it in fx['items']:
        enc = L.encode_rows(it['rows'])
        assert enc['members'] == it['demo_members']
        for vid, ctx in it['demo_context'].items():
            i = enc['members'].index(vid)
            assert enc['x'][i] == ctx['x'], (it['demo_item'], vid)
            assert enc['anchors'][i] == ctx['anchors'], (it['demo_item'], vid)


@pytest.mark.skipif(not GUIDANCE.is_file(), reason='training guidance doc not on this host')
def test_hook_features_equal_training_data_path():
    """The probability the hook's gate computes == predict(model, the training-path features of the same tape
    second), for every family feasible there (collect: tape_rows -> feasible -> candidate_features)."""
    model = L.load_model(MODEL)
    doc = json.loads(GUIDANCE.read_text(encoding='utf-8'))
    plans, _ = O.bank_plans(doc, 'D2800_p20', 'wm')
    for it in fixture()['items']:
        rows = it['rows']
        enc = L.encode_rows(rows)
        train = {f: L.predict(model, L.candidate_features(enc, mem, f, model['families']))
                 for f, mem in L.feasible(rows, plans)}
        core = L.LearnedCore(doc, 'D2800_p20', model)
        core.step(rows, {r['vid']: 0.0 for r in rows if r['is_cav']}, it['t'])
        got = {fam: p for _, fam, p, _ in core.scores}
        assert got, it['demo_item']
        for fam, p in got.items():
            assert p == round(train[fam], 6)
        assert core.scores[0][0] == it['t']
        if it['kind'] == 'act':
            assert it['family'] in train
        mems = dict(L.feasible(rows, plans))
        for fam, p in train.items():          # every feasible family, through the hook's own prob()
            fresh = L.LearnedCore(doc, 'D2800_p20', model)
            fresh._rows = rows
            assert fresh.prob(fam, mems[fam]) == p


def test_collect_rows_use_the_hook_feature_function():
    src = inspect.getsource(L.collect_tape)
    assert 'candidate_features(enc, mem, fam, families)' in src and 'encode_rows(rows)' in src


# ----------------------------------------------------------------------------------------------- model doc
def test_committed_model_doc():
    raw = MODEL.read_bytes()
    pin = (MODEL.parent / 'LEARNED_TRIGGER_MODEL.json.sha256').read_text().strip()
    assert hashlib.sha256(raw).hexdigest() == pin
    m = L.load_model(MODEL, pin)
    assert m['families'] == sorted(m['families']) and len(m['families']) == 5
    assert all(f.startswith('mut_') for f in m['families'])
    assert set(m['strategy_keys']) == set(m['families'])
    assert m['code_digest'] == m['data']['collect_digest'] == L.learned_trigger_digest()
    L.assert_seeds(m['data']['seeds'])
    with pytest.raises(ValueError):
        L.load_model(MODEL, '0' * 64)


def test_seed_guard():
    assert L.assert_seeds([991200, 989950, 989999])
    for bad in (989600, 989650, 993001, 964100, 962500, 977001, 990000, 989949):
        with pytest.raises(ValueError):
            L.assert_seeds([991200, bad])


def test_banned_paths():
    for p in ('x/sealed/a', 'y/HOLDOUT', 'heldout.json', 'cal_split', 'chunk5', 'J4B', 'inc1'):
        with pytest.raises(ValueError):
            L.guard_path(p)


def test_digest_pinned():
    assert L.learned_trigger_digest().startswith(PINNED)


# ----------------------------------------------------------------------------------------------- executor
def test_gate_replaces_only_the_admission():
    doc = fam_bank()
    m = keyed(synthetic_model(['mut_a'], threshold=0.4), doc)          # p = 0.5 everywhere -> fire
    core = L.LearnedCore(doc, 'D2800_p20', m)
    ref = O.OracleCore(doc, 'D2800_p20', plan_source='wm')
    rows = wm_rows()
    cmds = {r['vid']: 0.0 for r in rows if r['is_cav']}
    a1, l1 = core.step(rows, cmds, 400)
    a2, l2 = ref.step(rows, cmds, 400)     # FREE bin admitted for free-flow rows: ORACLE_WM fires too
    assert (a1, l1) == (a2, l2) and a1
    assert core.acts[0]['members'] == ref.acts[0]['members'] and core.acts[0]['prob'] == 0.5
    m2 = keyed(synthetic_model(['mut_a'], threshold=0.6), doc)          # p = 0.5 < 0.6 -> hold
    core2 = L.LearnedCore(doc, 'D2800_p20', m2)
    assert core2.step(rows, cmds, 400) == ({}, {}) and not core2.acts
    assert core2.offers[('mut_a', 'hold')] == 1
    # session + cooldown rule unchanged
    for t in range(401, 520):
        core.step(rows, cmds, t)
    assert len(core.acts) == 1


def test_refusals_not_in_classifier_and_key_mismatch():
    doc = fam_bank(('mut_a', 'mut_b', 'mut_c'))
    m = keyed(synthetic_model(['mut_a', 'mut_b']), doc)
    m['strategy_keys']['mut_b'] = 'deadbeefdeadbeef'
    core = L.LearnedCore(doc, 'D2800_p20', m)
    assert [p['family'] for p in core.plans] == ['mut_a']
    assert core.refused['not_in_classifier'] == ['mut_c']
    assert core.refused['classifier_key_mismatch'] == ['mut_b']


def test_never_reads_job_prefixes_or_privileged_state(monkeypatch):
    class Trap:
        def __getattr__(self, k):
            raise AssertionError(f'read job/prefixes .{k}')

        def __getitem__(self, k):
            raise AssertionError(f'read job/prefixes [{k}]')

        def get(self, *a):
            raise AssertionError('read job/prefixes get')

    def no_bin(*a, **k):
        raise AssertionError('privileged onset_bin called')
    monkeypatch.setattr(O, 'onset_bin', no_bin)
    doc = fam_bank()
    m = keyed(synthetic_model(['mut_a'], threshold=0.4), doc)
    hook = L.make_hook(doc, 'D2800_p20', m)

    class Ctl:
        g5_lane_targets = {}

        def apply(self, rows, commands, now):
            return {}
    ctl = Ctl()
    hook(ctl, Trap(), Trap())
    rows = wm_rows()
    out = ctl.apply(rows, {r['vid']: 0.0 for r in rows if r['is_cav']}, 400)
    assert out and all(v['oracle_plan'] == 'mut_a' for v in out.values())
    assert set(out) <= {r['vid'] for r in rows if r['is_cav']}
    # static: the module never touches job / B / R outputs or privileged bins
    tree = ast.parse(inspect.getsource(L))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert 'onset_bin' not in names and 'bin_admitted' not in names and 'runtime_situation' not in names
    src = inspect.getsource(L.LearnedCore) + inspect.getsource(L.make_hook)
    for tok in ("job[", "job.get", "prefixes[", "B_checkpoint", "R_checkpoint", "decision_log", "'privileged'"):
        assert tok not in src


def test_features_only_from_member_sensing():
    """Vehicles more than 100 m from both members cannot move the probability (no privileged / global state)."""
    import random
    doc = fam_bank()
    rng = random.Random(0)
    fams = ['mut_a']
    coef = [rng.uniform(-0.05, 0.05) for _ in L.feature_names(fams)]
    m = keyed(synthetic_model(fams, threshold=0.01, coef=coef), doc)
    rows = wm_rows()
    far = [dict(vid=f'f{k}', s_m=-600.0 + 40 * k, v_mps=rng.uniform(1, 30), lane=k % 3, is_cav=False, length_m=5.0)
           for k in range(10)]

    def prob(rs):
        core = L.LearnedCore(doc, 'D2800_p20', m)
        core.step(rs, {r['vid']: 0.0 for r in rs if r['is_cav']}, 400)
        return core.scores[0][2]
    base = prob(rows + far)
    for r in far:
        r['v_mps'] = rng.uniform(1, 30)
    assert prob(rows + far) == base
    near = rows + [dict(vid='n0', s_m=-985.0, v_mps=3.0, lane=0, is_cav=False, length_m=5.0)]
    assert prob(near) != prob(rows)


def test_deterministic():
    doc = fam_bank(('mut_a', 'mut_b'))
    import random
    rng = random.Random(1)
    fams = ['mut_a', 'mut_b']
    m = keyed(synthetic_model(fams, threshold=0.5, coef=[rng.uniform(-1, 1) for _ in L.feature_names(fams)]), doc)

    def run():
        core = L.LearnedCore(doc, 'D2800_p20', m)
        out = []
        for t in range(300, 700):
            rows = wm_rows(speed=20.0 + (t % 17) * 0.5)
            out.append(core.step(rows, {r['vid']: 0.0 for r in rows if r['is_cav']}, t))
        return out, json.dumps(core.receipts(), sort_keys=True), list(core.scores)
    assert run() == run()


def test_runner_wiring():
    from tools.coopmwm import nm_eval_runner as R
    assert R.LEARNED_ARMS == ('LEARNED_TRIGGER',)
    assert R.ORACLE_ARMS == ('ORACLE', 'ORACLE_WM')
    src = inspect.getsource(R.learned_trigger_hook)
    for tok in ("learned_trigger_sha256", "learned_trigger_model", "oracle_bank", "file_sha256", "cell_key"):
        assert tok in src
    with pytest.raises(ValueError):
        R.learned_trigger_hook(dict(cell_key='D2800_p20'), dict(cell_key='D3200_p20'))
