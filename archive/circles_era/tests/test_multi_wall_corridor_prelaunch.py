"""Exercise the real production preparation path, stopping before allocation."""
import pytest
from output.multi_wall_probe_v1 import study
from traffic_shepherd import paired_experiments as pe
from traffic_shepherd.multi_wall_probe import MultiWallProbePolicy


def test_actual_corridor_registration_reaches_launch_boundary_without_sumo(tmp_path,monkeypatch):
    reg = study.load_registration()
    reg['assignment'] = 'Registered diagnostic extra regulation or relay'
    job = reg['jobs'][0]
    reached = []
    class PreparedWithoutLaunch(Exception):
        pass
    def no_launch(budget,**kwargs):
        reached.append(str(kwargs['target_dir']))
        raise PreparedWithoutLaunch()
    monkeypatch.setattr(pe.budget_guard,'reserve_and_launch',no_launch)
    monkeypatch.setattr(pe,'start_sumo',lambda *a,**k:pytest.fail('SUMO is forbidden in unit test'))
    output = tmp_path/'prepared'
    with pytest.raises(PreparedWithoutLaunch):
        pe.evaluate_arm(job=job,registration=reg,output=output,
            policy=MultiWallProbePolicy(job,reg,mode=job['policy_spec']['multi_wall_mode']),
            budget=dict(target_dir=str(output)))
    assert reached == [str(output)]
    assert study.read(output/'manifest.json')['status']=='failed'
