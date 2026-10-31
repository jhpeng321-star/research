import numpy as np
import pytest

from tools.rbeval.validate_merge import align_identity, load_episode, score_trajectories


def test_slot_recycling_and_reordering_do_not_create_identity_merge_or_birth():
    data = dict(s=np.array([[70., 30.], [35., 74.], [60., 80.]]),
                v=np.zeros((3, 2)), lane=np.array([[2, 1], [1, 1], [2, 1]]),
                mask=np.ones((3, 2), dtype=bool))
    vid = np.array([[10, 20], [20, 10], [30, 10]])
    aligned = align_identity(data, vid, ['10', '20', '30'])
    score = score_trajectories(aligned, aligned, np.arange(3) * .5, .5, 'test')
    assert score['real_merges'] == 1
    assert score['matched'] == 1
    assert score['insertion_counts']['real'] == 1
    assert score['time_error_s']['mean'] == 0
    assert score['position_error_m']['mean'] == 0


def test_seed_allowlist_rejects_before_any_file_read(tmp_path):
    with pytest.raises(ValueError, match='FIT allowlist'):
        load_episode(tmp_path, 969432, 'native')
