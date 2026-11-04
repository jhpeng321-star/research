import copy
import unittest
from .parked_retirement import PARENTS, SPAWN, TRACKER, targets, unchanged


def fixture():
    rows = [dict(pid=p, ppid=1, start_ticks=s, cmd=c, state='T',
                 uid='1000\t1000\t1000\t1000', cwd='/home/ubuntu', cpu_ticks=1)
            for p, (s, c) in PARENTS.items()]
    rows[0]['cwd']='/home/ubuntu/coopmwm20261101_v2 (deleted)'
    rows[1]['cwd']='/home/ubuntu/coopmwm20261101_accel1'
    rows += [dict(pid=2477311, ppid=2477255, start_ticks='55545772',
                  cmd=SPAWN+'spawn_main(tracker_fd=9, pipe_handle=13) --multiprocessing-fork ',
                  state='T', uid=rows[0]['uid'], cwd='/home/ubuntu/coopmwm20261101_accel1', cpu_ticks=2),
             dict(pid=2477310, ppid=2477255, start_ticks='55545771',
                  cmd=TRACKER+'main(8) ', state='S', uid=rows[0]['uid'],
                  cwd='/home/ubuntu/coopmwm20261101_accel1', cpu_ticks=2),
             dict(pid=123, ppid=1, start_ticks='7', cmd=SPAWN+'orphan', state='S',
                  uid=rows[0]['uid'], cwd='/home/ubuntu', cpu_ticks=2)]
    return rows


class ParkedRetirementTests(unittest.TestCase):
    def choose(self, rows):
        return targets(rows, expected_boot='boot', actual_boot='boot')

    def test_only_pinned_parents_and_parked_children_tracker_and_orphans_excluded(self):
        rows=fixture()
        rows.append(dict(rows[3],pid=2146733,ppid=2146725,cwd=rows[0]['cwd']))
        self.assertEqual([r['pid'] for r in self.choose(rows)], [2477311, 2146725, 2477255])

    def test_boot_and_parent_reuse_or_resume_refused(self):
        with self.assertRaises(ValueError):
            targets(fixture(), expected_boot='old', actual_boot='new')
        for key, value in [('start_ticks', 'new'), ('state', 'S'), ('cmd', 'other'),
                           ('uid', '0\t0\t0\t0'), ('ppid', 9)]:
            rows=fixture();rows[0][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):self.choose(rows)

    def test_running_or_foreign_child_and_grandchild_refused(self):
        for key,value in [('state','S'),('cmd','sumo'),('uid','0\t0\t0\t0'),('cwd','/other')]:
            rows=fixture();rows[2][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.choose(rows)
        rows=fixture();rows.append(dict(rows[2],pid=999,ppid=2477311))
        with self.assertRaises(ValueError):self.choose(rows)

    def test_each_pre_signal_identity_field_is_pinned(self):
        r=fixture()[2];unchanged(r,copy.deepcopy(r))
        for key,value in [('pid',999),('ppid',1),('start_ticks','new'),('state','S'),
                          ('cpu_ticks',3),('uid','0'),('cmd','other'),('cwd','/other')]:
            changed=dict(r,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):unchanged(r,changed)

    def test_duplicate_pid_refused(self):
        rows=fixture();rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):self.choose(rows)


if __name__ == '__main__':unittest.main()
