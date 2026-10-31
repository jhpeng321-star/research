"""Verify immutable verdict/deliverables without repeating inference or statistics."""
import argparse
import json
from pathlib import Path

from .audit import ARMS, digest, safe
from .core import TIMES, finite
from .evaluate_frozen import FROZEN_SHA
from .finalize_short import OLD_WORKER, markdown
from .orchestrate import save_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = root/'tools/wmvalue/evidence'
    result_path = evidence/'short_test.json'
    result = json.loads(result_path.read_text())
    assert result['verdict'] == 'FAIL' and result['survivors'] == [] and not result['A_evaluated']
    assert result['n_worlds'] == 8 and result['n_pairs'] == 400
    assert result['bootstrap'] == dict(unit='whole world', draws=10000, seed=20261031)
    assert not result['policy_training_authorized']
    for m in ('B', 'C'):
        assert not all(result['contrasts'][m+'-'+ref][metric][0] > 0
                       for ref in ('D', 'E') for metric in ('sign_ci95', 'mae_ci95'))
    model = root/'_out/wmvalue_frozen_value6f334469.joblib'
    real = root/'_out/wmvalue_reserved_real_v1.npz'
    assert digest(model) == result['model_sha256'] == FROZEN_SHA
    assert digest(real) == result['real_arrays_sha256']
    for name, sha in result['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == sha, 'frozen evaluator source changed'
    ack = json.loads((root/'_out/wmvalue_short_v1/ACK.json').read_text())
    seen, source_ref, long_count = set(), None, 0
    assert len(result['task_sha256']) == 320
    for name, sha in result['task_sha256'].items():
        path = safe(root/name)
        assert digest(path) == sha
        d = json.loads(path.read_text())
        key = d['seed'], d['t0'], d['future']
        assert key not in seen
        seen.add(key)
        assert d['arms'] == list(ARMS) and d['K'] == 8 and d['mix']
        assert not d['real_outcomes_opened']
        assert d['end_time'] in (d['t0']+90, 900)
        assert d['results']['steps'] == 2*(d['end_time']-d['t0'])
        long_count += d['end_time'] == 900
        sources = dict(d['source_sha256'])
        assert sources.pop('worker') in {OLD_WORKER, ack['worker_sha256']}
        if source_ref is None:
            source_ref = sources
        assert sources == source_ref
    assert seen == {(s, t, f) for s in range(969432, 969440) for t in TIMES for f in range(8)}
    rows = json.loads((evidence/'short_arm_predictions.json').read_text())
    cells = set()
    for row in rows:
        key = row['seed'], row['t0'], row['arm']
        assert key not in cells and 'A' not in row
        cells.add(key)
        finite([row[m] for m in ('real', 'B', 'C', 'D', 'E')])
    assert len(rows) == 200
    assert cells == {(s, t, arm) for s in range(969432, 969440) for t in TIMES for arm in ARMS}
    report = root/'docs/paper_results/WMVALUE.md'
    text = report.read_text(encoding='utf-8')
    assert text == markdown(result) and len(text.splitlines()) <= 60
    line = (evidence/'final_verdict_registration.md').read_text(encoding='utf-8')
    assert 'resultSHA'+digest(result_path) in line
    registration = safe('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    with registration.open(encoding='utf-8') as stream:
        assert sum(s == line for s in stream) == 1
    receipt = dict(schema='wmvalue_final_verification/1', verification='PASS',
        scientific_verdict='FAIL', statistics_repeated=False, inference_repeated=False,
        all_task_hashes=320, all_arm_cells=200, world_pairs=400,
        reused_long_tasks=long_count, new_short_tasks=320-long_count,
        identical_runtime_sources=len(source_ref), model_sha256=FROZEN_SHA,
        result_sha256=digest(result_path), report_sha256=digest(report),
        report_lines=len(text.splitlines()), exact_verdict_registration_occurrences=1)
    save_new(safe(a.out), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
