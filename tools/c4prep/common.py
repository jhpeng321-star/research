"""Explicit C4 preparation bindings, exclusive receipts and path guards."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

CODE = Path(__file__).resolve().parents[2]
REL = Path('/root/autodl-tmp/c4/restart20261026')
OUT = REL / 'arm/loop/c4prep'
WM = REL / 'arm/loop/wm/c03.a2/wm_s.pt'
WM_SHA = '41e18f09b6060a1f4e4039f02655fcbcd497e32a360cffb71be716fffeaadc84'
WM_CONFIG = CODE / 'configs/c4_wm_loop14_r16_ft.toml'
GATE = REL / 'arm/loop/checks/c03.a2/HORIZON_GATE.json'
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)
ACTORS = {
    0: (REL / 'arm/loop/policy/c00/policy_latest.pt',
        '1b9949cc0eadb32db3f56b08684fa512fc2e70a26a0b7a360edb5599c6226ab5'),
    1: (REL / 'arm/loop/policy/c01.a3/policy_latest.pt',
        '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'),
    2: (REL / 'arm/loop/policy/c02.h60b.r2/policy_latest.pt',
        '44f3eb66aba9a8598a1fd8727074389cb8f797d1e2d2f6cdf223d29fb0e7f128'),
    3: (REL / 'arm/loop/policy/c03.a6/policy_latest.pt',
        '0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e')}


def dataset(cycle):
    """Bind verified corrected exports without overwriting original datasets."""
    corrected = OUT / f'backlog_import_v2/cpuP/pairs_c{cycle:02}.a1.backlog'
    if corrected.is_dir():
        proof = read(OUT / 'backlog_import_v2/IMPORT.json')
        binding = next(x for x in proof['datasets'] if x['cycle'] == cycle)
        assert str(corrected) == binding['root']
        assert sha(corrected / 'C4_EPISODES_MANIFEST.json') == binding['manifest_sha256']
        return corrected
    return REL / f'arm/loop/data/pairs_c{cycle:02}.a1'


def safe(path):
    p = Path(path)
    if BAD.search(str(p)) or BAD.search(str(p.resolve())):
        raise ValueError('prohibited path')
    return p


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_text())


def complete_jsonl(path):
    """Read only durably terminated records while an owned writer is active."""
    p = safe(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines(keepends=True)
            if line.endswith('\n')]


def completed_updates(path):
    """CLI emits a timestamped completion after the trainer's raw metric write.

    Count unique completed optimizer updates, never those two writes as two
    updates, and retain timestamps for phase rate/ETA.
    """
    rows = {}
    for row in complete_jsonl(path):
        if 't' in row and 'update' in row:
            rows[row['update']] = row
    return list(rows.values())


def write_new(path, doc):
    p = safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def accepted():
    doc = read(REL / 'arm/loop/checks/c03.a2/ACCEPT.json')
    assert doc['decision'] == 'ACCEPT'
    assert doc['inputs']['candidate_sha256'] == WM_SHA == sha(WM)
    assert doc['inputs']['config_sha256'] == sha(WM_CONFIG)
    return doc


def qualified_gate(entrant):
    accepted()
    doc = read(GATE)
    assert doc['wm_sha256'] == WM_SHA
    assert doc['wm_config_sha256'] == sha(WM_CONFIG)
    assert bool(doc.get('entrant_control', False)) == entrant, 'gate entrant signature mismatch'
    threshold = REL / 'runtime_c1/configs/coopw_horizon_gate_13E_b2.json'
    assert sha(threshold).startswith('c479b2de')
    assert doc['threshold_source']['sha256'] == sha(threshold)
    assert doc['threshold'] == read(threshold)['threshold']
    assert all(doc['horizons'][str(h)]['sign_agreement'] >= doc['threshold'] for h in (10, 30, 60))
    passed = [int(h) for h, r in doc['horizons'].items() if r['verdict'] == 'PASS']
    return max([10, *passed]), doc
