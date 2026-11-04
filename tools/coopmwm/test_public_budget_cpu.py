"""Accounting-report regression fixtures; no physics or external files."""
import copy
import unittest

from .public_budget import reconcile


class PublicBudgetTest(unittest.TestCase):
    def setUp(self):
        self.digest = 'a' * 64
        self.shared = dict(total=3000000, phases=dict(
            paired_branch=1000000, v2v_train=1000000, bank_verification_ext=900000))
        self.runtime = copy.deepcopy(self.shared)
        self.runtime['phases'].update(paired_branch=654400, v2v_train=1345600)
        self.runtime['transfers'] = [dict(source='paired_branch', target='v2v_train',
            amount=345600, amendment_sha256=self.digest)]

    def test_runtime_transfer_preserves_newer_shared_bank_funding(self):
        original = copy.deepcopy(self.shared)
        self.runtime['phases']['bank_verification_ext'] = 800000
        report = reconcile(self.shared, self.runtime, self.digest)
        self.assertEqual(report['phases']['bank_verification_ext'], 900000)
        self.assertEqual(report['phases']['paired_branch'], 654400)
        self.assertEqual(report['phases']['v2v_train'], 1345600)
        self.assertEqual(sum(report['phases'].values()), 2900000)
        self.assertEqual(self.shared, original)

    def test_already_shared_transfer_is_not_applied_twice(self):
        self.shared['phases'].update(self.runtime['phases'])
        report = reconcile(self.shared, self.runtime, self.digest)
        self.assertEqual(report['phases'], self.shared['phases'])
        self.assertEqual(report['transfers'][0]['state'], 'already_in_shared_binding')

    def test_unregistered_and_duplicate_transfers_refuse(self):
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            reconcile(self.shared, self.runtime, '')
        self.runtime['transfers'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reconcile(self.shared, self.runtime, self.digest)

    def test_conflicting_endpoints_and_scope_refuse(self):
        self.runtime['phases']['v2v_train'] += 1
        with self.assertRaisesRegex(ValueError, 'endpoints conflict'):
            reconcile(self.shared, self.runtime, self.digest)
        self.runtime['transfers'][0]['amount'] = 345601
        with self.assertRaisesRegex(ValueError, 'unrecognised'):
            reconcile(self.shared, self.runtime, self.digest)

    def test_no_transfer_keeps_shared_authority_and_total_mismatch_refuses(self):
        self.runtime.pop('transfers')
        self.assertEqual(reconcile(self.shared, self.runtime, '')['phases'],
                         self.shared['phases'])
        self.runtime['total'] += 1
        with self.assertRaisesRegex(ValueError, 'total caps'):
            reconcile(self.shared, self.runtime, '')


if __name__ == '__main__':
    unittest.main()
