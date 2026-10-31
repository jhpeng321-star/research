"""Operational gates for unattended publication, independent of SUMO/PyTorch."""
from copy import deepcopy
from datetime import datetime
import unittest
from unittest.mock import patch

from . import supervise


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        _, self.record = supervise.latest()
        self.now = datetime.fromisoformat(self.record['timestamp_utc']).timestamp()

    def test_real_existing_receipt_is_admitted(self):
        supervise.validate(self.record, now=self.now)

    def test_stale_or_future_observer_receipt_refused(self):
        for age in (661, -61):
            with self.assertRaises(AssertionError):
                supervise.validate(self.record, now=self.now+age)

    def test_wrong_build_and_budget_and_replays_refused(self):
        for key, value in (
            ('source_binding', 'wrong-build'), ('own_pid', 1),
            ('historical_charged_sim_s', 0), ('inherited_unresolved_reservations', 0),
            ('training_budget_upper_sim_s', 7012801), ('charged_sim_s', 6999300),
            ('production_reserved_episodes', 7729), ('completed_updates', 170),
            ('failure_count', 1), ('local_evaluation_started', True), ('mechanics_replayed', True),
        ):
            record = deepcopy(self.record)
            record[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                supervise.validate(record, now=self.now)

    def test_counters_cannot_regress(self):
        previous = deepcopy(self.record)
        previous['production_completed_episodes'] += 1
        with self.assertRaises(AssertionError):
            supervise.validate(self.record, previous, now=self.now)

    def test_safety_stop_cannot_be_published_as_healthy(self):
        record = deepcopy(self.record)
        record['completed_result_totals']['collisions'] = 1
        with self.assertRaises(AssertionError):
            supervise.validate(record, now=self.now)

    def test_unrelated_staged_work_blocks_git_add_and_commit(self):
        def git(*args):
            return 'user-work.py\n' if args == ('diff', '--cached', '--name-only') else ''
        with patch.object(supervise, 'git', side_effect=git) as call:
            with self.assertRaises(AssertionError):
                supervise.commit([supervise.REPORT], 'must not commit')
        self.assertFalse(any(c.args[0] in ('add', 'commit') for c in call.call_args_list))

    def test_peer_file_is_never_staged(self):
        with patch.object(supervise, 'git', return_value='') as call:
            with self.assertRaises(AssertionError):
                supervise.commit(['SELMAPPO2_CONTINUATION.md'], 'must not commit')
        self.assertFalse(any(c.args[0] in ('add', 'commit') for c in call.call_args_list))


if __name__ == '__main__':
    unittest.main()
