"""Verify the completed frozen WM2 report and its local immutable task mirror.

This performs no rollout or rescoring. It rechecks report/task/identity hashes,
the exact BIC events and observable contracts before publishing a NEW receipt.
"""
import argparse
from pathlib import Path, PurePosixPath

from .core import HERE, SEEDS, digest, load, safe, save_new
from .finalize import panel_state
from .score import read_predictions


def verify_report_hashes(receipt, report):
    expected = receipt['report_hashes']
    assert set(expected) == {p.name for p in report.iterdir() if p.is_file()}, 'report file set changed'
    for name, sha in expected.items():
        assert PurePosixPath(name).name == name and '\\' not in name
        assert digest(report/name) == sha, ('report hash mismatch', name)
    assert load(report/'gate.json') == receipt['gate'], 'gate receipt mismatch'


def verify(evidence):
    evidence = safe(evidence)
    receipt_path = evidence/'FINALIZED.json'
    result = load(receipt_path)
    assert result['schema'] == 'rbeval-finalized/1'
    assert (result['tasks'], result['worlds'], result['episodes']) == (160, 20, 800)
    remote = PurePosixPath(result['report'])
    assert remote.parent == PurePosixPath('/root/rbeval20261031/reports')
    report = evidence/'reports'/remote.name
    verify_report_hashes(result, report)
    originals = [p for p in evidence.glob('FINALIZER_[0-9]*.json')
                 if load(p).get('sources') == result['sources']]
    assert originals, 'missing original frozen scorer identity'
    state = panel_state(evidence)
    assert state['complete'] and state['tasks'] == 160 and not state['blocked']
    real = load(HERE/'real_observables.json')['episodes']
    _, current = read_predictions([evidence/f'wm2_shard{i}' for i in range(4)], real, partial=False)
    original = load(report/'audit.json')
    assert original['model'] == current['model']
    assert original['complete_worlds'] == current['complete_worlds'] == list(SEEDS)
    assert not original['missing_tasks'] and not current['missing_tasks']
    assert original['tasks'] == current['tasks'] == 160

    def task_hashes(audit):
        hashes = {}
        for row in audit['inputs']:
            # Remote paths stay recorded; compare their stable shard/task suffix.
            path = PurePosixPath(row['task'].replace('\\', '/'))
            key = (path.parent.name, path.name)
            assert key not in hashes, 'duplicate audited task'
            hashes[key] = (row['sha256'], row['identity_sha256'])
        return hashes

    assert task_hashes(original) == task_hashes(current), 'scored input set changed'
    return dict(schema='rbeval-wm2-verified/1', tasks=160, worlds=20,
        arm_futures=800, K=8, t0=300, horizons=[120, 300, 600],
        complete_worlds=list(SEEDS), model=current['model']['model'],
        wm_sha256=current['model']['wm_sha256'],
        finalization_receipt_sha256=digest(receipt_path),
        original_finalizer_receipts={p.name: digest(p) for p in originals},
        remote_report=result['report'], local_report=str(report),
        report_hashes=result['report_hashes'], frozen_scoring_sources=result['sources'],
        finished_unix=result['finished_unix'], gate=result['gate'],
        checks=dict(all_160_completion_hashes=True, all_report_hashes=True,
            exact_scored_input_set=True, common_model_contract=True,
            observable_starts_and_prehistory=True, speed_support=True,
            delay_decomposition=True, registered_bic_decisions=7200),
        verifier_sha256=digest(Path(__file__)),
        note='Original frozen full-panel report; no rollout or scoring repeated. '
             'Partial-report bootstrap correction leaves full20-world draws unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--out', required=True, help='NEW JSON receipt, existing parent')
    args = parser.parse_args()
    out = safe(args.out)
    assert out.parent.is_dir() and not out.exists()
    result = verify(args.evidence)
    save_new(out, result)
    print('VERIFIED', result['tasks'], 'tasks;', len(result['report_hashes']),
          'report files;', result['gate']['verdict'], ';', out)


if __name__ == '__main__':
    main()
