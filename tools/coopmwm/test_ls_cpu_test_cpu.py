"""The release harness admits metadata fixtures without allowing physics."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
from .ls_cpu_test import metadata_command,guarded_popen


class MetadataBoundary(unittest.TestCase):
    def test_exact_preflight_requires_a_temporary_bank_and_no_extra_arguments(self):
        cmd=[sys.executable,'-m','tools.coopmwm.coop_search','preflight','--sha','a'*64]
        with tempfile.TemporaryDirectory() as d:
            kwargs=dict(env={'COOPMWM_COOP_BANK_DIR':d})
            self.assertTrue(metadata_command(cmd,kwargs))
            for changed in (cmd+['--run'],cmd[:-1]+['bad'],cmd[:3]+['search']+cmd[4:]):
                self.assertFalse(metadata_command(changed,kwargs))
        self.assertFalse(metadata_command(cmd,dict(env={'COOPMWM_COOP_BANK_DIR':'/home/ubuntu/live'})))

    def test_git_is_confined_to_the_exact_temporary_fixture_and_verbs(self):
        with tempfile.TemporaryDirectory() as d:
            prefix=['git','-C',str(Path(d)/'gitsrc')]
            for suffix in (['init','-q'],['add','-A'],['commit','-q','-m','x'],['rev-parse','HEAD'],['diff','--quiet','HEAD','--','*.py']):
                self.assertTrue(metadata_command(prefix+suffix,{}))
            for suffix in (['push'],['clean','-fd'],['config','x','y'],['commit','--amend']):
                self.assertFalse(metadata_command(prefix+suffix,{}))
        self.assertFalse(metadata_command(['git','-C','/home/ubuntu/gitsrc','init','-q'],{}))

    def test_physics_inference_shells_and_executable_overrides_never_dispatch(self):
        original=Mock()
        for args,kwargs in ((['sumo','-c','scenario'],{}),
            ([sys.executable,'-m','tools.coopmwm.warm_runner'],{}),
            ([sys.executable,'-c','do_anything()'],{}),('git status',{'shell':True}),
            (['git','-C','/tmp/gitsrc','init','-q'],{'executable':'sumo'})):
            with self.subTest(args=args),self.assertRaises(AssertionError):guarded_popen(original,args,**kwargs)
        original.assert_not_called()


if __name__=='__main__':unittest.main()
