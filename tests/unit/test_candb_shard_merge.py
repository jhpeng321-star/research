"""A resumed prefix must not inflate evidence or hide a conflicting repeat."""
import copy

import pytest

from tools.candB.summarize_shards import METRICS, merge_completed


def test_repeated_completed_start_counted_once_and_conflict_refused():
    row = dict(index=4, metadata=[dict(world='w963110m2400p20t300')],
               value={metric: [1., 2., 3., 4.] if metric.startswith('J') else 2.
                      for metric in METRICS},
               reference={metric: [2., 3., 4., 5.] if metric.startswith('J') else 3.
                          for metric in METRICS},
               native=dict(J90=[3., 4., 5., 6.], J300=[4., 5., 6., 7.]))
    previous, duplicates = [row], []
    repeat = copy.deepcopy(row)
    repeat['value']['elapsed_s'] = 999.  # Scheduling does not change scientific identity.
    merge_completed(previous, [repeat], 'MAPPO_native_lanes', 'tail', duplicates)
    assert len(previous) == 1 and len(duplicates) == 1
    assert duplicates[0]['scientific_arrays_equal']
    repeat['reference']['J300'][0] += .01
    with pytest.raises(AssertionError, match='conflicting repeated'):
        merge_completed(previous, [repeat], 'MAPPO_native_lanes', 'tail', duplicates)
