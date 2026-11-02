"""MAIN23:32: explicit SUMO environment, append-only retry, real branch smoke."""
import os
from pathlib import Path

from .sumo_verify import SubprocessBranchRunner, check_prefix, POLICY
from .warm_spool import immutable


def sumo_environment(home, environ=None):
    home = Path(home)
    binary = home/'bin'/'sumo'
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('registered SUMO binary missing or not executable: '+str(binary))
    env = dict(os.environ if environ is None else environ)
    env.update(SUMO_HOME=str(home), PATH=str(home/'bin')+os.pathsep+env.get('PATH', ''))
    return env


def void_environment_failure(label):
    v = label['verification']
    branches = v.get('branches') or []
    return (v['status'] == 'failed' and v.get('actual_sim_s') == 0 and len(branches) == 9
            and all(b['status'] == 'failed' and 'SUMO binary not found on PATH' in b.get('reason', '')
                    for b in branches))


def label_path(root, key, generation=None):
    return Path(root)/'labels'/(key+('_'+generation if generation else '')+'.json')


class SmokeBranchRunner(SubprocessBranchRunner):
    """Run policy first; reuse it in the nine-branch list (zero extra physics).

    A startup smoke must pass the real executor AND recorded prefix before any
    candidate branch is launched. No synthetic test or --version substitutes.
    """
    def __init__(self, *args, smoke_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoke_path = Path(smoke_path)
        self.policy_result = None
        self.prepare_actual_sim_s = 0

    def prepare(self, specs, snapshot):
        super().prepare(specs, snapshot)
        assert specs[0]['branch'] == POLICY
        result = super().run(specs[0], snapshot)
        self.prepare_actual_sim_s = result.get('actual_sim_s', 0)
        check_prefix(result, snapshot)
        self.policy_result = result
        if not self.smoke_path.exists():
            immutable(self.smoke_path, dict(pass_=True, verification_id=specs[0]['verification_id'],
                branch_sha256=specs[0]['branch_sha256'], actual_sim_s=self.prepare_actual_sim_s,
                prefix_checked_through_s=snapshot['t0'], extra_sim_s=0,
                source='real policy branch reused in this k8 verification'))

    def run(self, spec, snapshot):
        if self.policy_result is None:
            raise ValueError('policy branch smoke has not passed; candidate launch forbidden')
        if spec['branch'] == POLICY:
            return self.policy_result
        return super().run(spec, snapshot)
