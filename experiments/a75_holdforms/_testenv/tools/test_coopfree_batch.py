"""Operational partition/identity gates; no simulation or real process signals."""
import signal
import unittest
from unittest.mock import Mock
from tools.coopfree_batch import identity,tasks,resume


class BatchTests(unittest.TestCase):
    def test_process_identity_with_spaces_and_parentheses(self):
        fields=['S']+['0']*18+['47835664']+['0']*8
        stat='339173 (python (owned worker)) '+' '.join(fields)
        self.assertEqual(identity(stat),'S')
        with self.assertRaises(ValueError):
            identity(stat.replace('47835664','999'))
        with self.assertRaises(ValueError):
            identity(stat.replace('339173','123',1))

    def test_all_member_solos_joint_and_no_existing_duplicates(self):
        plan=dict(windows=[dict(t0=300,sample=dict(context=dict(members=['a','b','c'])))])
        charges=[dict(id='u001_w000_'+a,sim_s=900,phase='train')
                 for a in ('joint','solo0','solo1','solo2')]
        selected=tasks(plan,charges,{'u001_w000_joint','u001_w000_solo0'})
        self.assertEqual([t['id'] for t in selected],['u001_w000_solo1','u001_w000_solo2'])
        self.assertEqual([t['members'] for t in selected],[['b'],['c']])
        with self.assertRaisesRegex(ValueError,'precharged'):
            tasks(plan,charges[:-1],set())

    def test_resumes_original_supervisor_after_failure(self):
        action=Mock()
        with self.assertRaises(RuntimeError):
            try:
                raise RuntimeError('worker failure')
            finally:
                resume(action)
        action.assert_called_once_with(getattr(signal,'SIGCONT',18))


if __name__=='__main__':
    unittest.main()
