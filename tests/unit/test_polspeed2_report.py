import json

from tools.polspeed2.benchmark import compare, update_row, memory_failure


def test_parity_selects_cli_rescore_row(tmp_path):
    path = tmp_path/'updates.jsonl'
    rows = [{'update': 181, 'actor_loss': 1.},
            {'update': 181, 'actor_loss': 1., 'independent_J': {'J': 2.}},
            {'update': 182, 'actor_loss': 3.}]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert update_row(path) == rows[1]


def test_parity_checks_missing_none_and_signed_zero():
    assert not compare({'optional': None}, {})['bitwise_identical']
    assert not compare({'x': 0.}, {'x': -0.})['bitwise_identical']
    result = compare({'x': 1., 'wall_time_s': 2.}, {'x': 1., 'wall_time_s': 3.})
    assert result['bitwise_identical'] and result['compared_leaves'] == 1


def test_worker_oom_diagnostic_survives_a_peer_exit(tmp_path):
    (tmp_path/'console.log').write_text('future worker exited (-9)')
    assert not memory_failure(tmp_path)
    worker = tmp_path/'future_workers/pool/2'
    worker.mkdir(parents=True)
    (worker/'2.error').write_text('torch.OutOfMemoryError: CUDA out of memory')
    assert memory_failure(tmp_path)
