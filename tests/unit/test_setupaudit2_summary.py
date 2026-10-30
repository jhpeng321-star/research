"""Cluster, pairing and bootstrap reproducibility (no checkpoints or GPU)."""
import numpy as np

from tools.setupaudit.summarize2 import cluster_plan, estimate


def test_world_clustering_is_not_replaced_by_start_or_future_pseudoreplication():
    worlds = ['a', 'a', 'a', 'b']
    values = np.array([0., 0., 0., 10.])
    unique, plan = cluster_plan(worlds, draws=5000)
    result = estimate(values, worlds, unique, plan)
    assert result == {'mean': 5., 'lo': 0., 'hi': 10.}
    unique2, plan2 = cluster_plan(worlds*8, draws=5000)
    assert estimate(np.tile(values, 8), worlds*8, unique2, plan2) == result


def test_shared_cluster_plan_preserves_zero_paired_contrast_and_seed():
    worlds = ['a', 'b', 'c']
    unique, plan = cluster_plan(worlds)
    _, repeat = cluster_plan(worlds)
    np.testing.assert_array_equal(plan, repeat)
    treatment = np.array([[100., 100.], [300., 300.], [900., 900.]])
    result = estimate(treatment-treatment[:, :1], worlds, unique, plan)
    for key in ('mean', 'lo', 'hi'):
        assert result[key] == [0., 0.]
