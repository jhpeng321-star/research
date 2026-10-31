"""Bounded handover policy: keep existing work and fail closed on real failures."""
import unittest
from .support import decision


class SupportPolicy(unittest.TestCase):
    def decide(self, **overrides):
        state = dict(finalized=False, live_finalizer=False, blocked_workers=False,
            deadline_expired=True, handover_used=False, before_deadline=True, sources_match=True)
        state.update(overrides)
        return decision(**state)

    def test_live_and_completed_work_never_relaunched(self):
        self.assertEqual(self.decide(live_finalizer=True), 'PRESERVE_LIVE_FINALIZER')
        self.assertEqual(self.decide(finalized=True), 'REPORT_READY')

    def test_only_deadline_allows_one_cpu_extension(self):
        self.assertEqual(self.decide(), 'EXTEND_CPU_FINALIZER_ONCE')
        self.assertEqual(self.decide(handover_used=True), 'ALERT_FINALIZER_EXITED_NO_RETRY')
        self.assertEqual(self.decide(deadline_expired=False), 'ALERT_FINALIZER_EXITED_NO_RETRY')
        self.assertEqual(self.decide(before_deadline=False), 'SUPPORT_DEADLINE')

    def test_worker_loss_or_source_change_never_retried(self):
        self.assertEqual(self.decide(blocked_workers=True), 'ALERT_INFERENCE_BLOCKED')
        self.assertEqual(self.decide(sources_match=False), 'ALERT_SCORING_SOURCE_CHANGED')


if __name__ == '__main__':
    unittest.main()
