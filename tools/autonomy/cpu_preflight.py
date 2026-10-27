"""Zero simulation: imports, exact policy load, native assets, ONNX and SUMO binary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    runtime, policy, expected, cases = sys.argv[1:]
    root = Path(runtime)
    os.chdir(root)
    sys.path[:0] = [str(root / 'src'), str(root)]
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import traffic_shepherd.c4.collect  # noqa: F401
    from traffic_shepherd.config import load_config
    from traffic_shepherd.damping import CirclesController
    from traffic_shepherd.c4.seeds import check_case
    from wms.control.policy import load_policy_fn
    cfg = load_config()
    assert hashlib.sha256(Path(policy).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest() == cfg.model_sha256
    assert Path(cfg.network_path).is_file()
    # Metadata only; never open the route/assignment file.
    assert Path(cfg.scenario_template_path).is_file()
    controller = CirclesController(cfg)
    exe = load_policy_fn(policy, device='cpu')
    assert exe.policy.residual_on_native
    assert shutil.which('sumo')
    for case in json.loads(cases):
        check_case(case['seed'], int(case['demand']))
        assert not 964000 <= case['seed'] <= 964199
    print(json.dumps({'status': 'PASS', 'charged_sim_s': 0, 'sumo_started': False,
                      'policy_sha256': expected, 'onnx_providers': controller._session.get_providers()}))


if __name__ == '__main__':
    main()
