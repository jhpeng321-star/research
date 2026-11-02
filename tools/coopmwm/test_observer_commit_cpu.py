"""Windows-length publication regression tests; no physics or remote processes."""
import subprocess
import unittest
from unittest.mock import patch

from . import warm_v3_watch as watch


class CommitTests(unittest.TestCase):
    def exercise(self,paths,pending,staged=False):
        checks=[];runs=[];stage_reads=iter(['other.txt\n' if staged else '', 'staged\n'])
        def output(argv,**kwargs):
            checks.append(argv)
            if argv[1:]==['diff','--cached','--name-only']:return next(stage_reads)
            if argv[1]=='ls-files':return '\0'.join(pending)+'\0'
            return ''
        def run(argv,**kwargs):runs.append((argv,kwargs))
        with patch.object(watch.subprocess,'check_output',side_effect=output),patch.object(
                watch.subprocess,'run',side_effect=run),patch('builtins.print'):
            if staged:
                with self.assertRaisesRegex(ValueError,'other staged work; preserve it'):watch.commit(paths)
            else:watch.commit(paths)
        return checks,runs

    def test_large_receipt_backlog_uses_bounded_argv_and_exact_nul_stdin(self):
        paths=['docs/paper_results/coopmwm/EXACT_STATUS_'+str(i)+'_'+'a'*100+'.json' for i in range(1500)]
        checks,runs=self.exercise(paths,paths+['unrelated.txt'])
        self.assertTrue(all(len(subprocess.list2cmdline(argv))<=16000 for argv in checks))
        self.assertGreater(len([a for a in checks if a[1:4]==['diff','--stat','--']]),1)
        self.assertEqual(runs[0][0],['git','add','--pathspec-from-file=-','--pathspec-file-nul'])
        self.assertEqual(runs[0][1]['input'],(''.join(':(literal)'+p+'\0' for p in sorted(paths))).encode())
        self.assertEqual(runs[1][0][1],'commit')

    def test_unchanged_receipts_and_unrelated_files_are_not_staged(self):
        checks,runs=self.exercise(['old.json','new space[1].json'],['new space[1].json','unrelated.txt'])
        self.assertEqual(runs[0][1]['input'],b':(literal)new space[1].json\0')
        self.assertEqual([a for a in checks if a[1:4]==['diff','--stat','--']],
            [['git','diff','--stat','--',':(literal)new space[1].json']])

    def test_operator_staged_work_remains_untouched(self):
        checks,runs=self.exercise(['new.json'],['new.json'],staged=True)
        self.assertEqual(len(checks),1);self.assertEqual(runs,[])

    def test_windows_relative_paths_match_git_slash_paths(self):
        _,runs=self.exercise(['docs\\paper_results\\new.json'],['docs/paper_results/new.json'])
        self.assertEqual(runs[0][1]['input'],b':(literal)docs/paper_results/new.json\0')

    def test_no_changed_intended_receipts_does_not_commit(self):
        checks,runs=self.exercise(['old.json'],['unrelated.txt'])
        self.assertEqual(runs,[])
        self.assertFalse(any(a[1:3]==['log','--oneline'] for a in checks))


if __name__=='__main__':unittest.main()
