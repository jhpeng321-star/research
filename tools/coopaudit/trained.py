"""Hash-bound original S-SUMOCF deployment adapter, without exploration offsets."""
import importlib.util
from pathlib import Path
import torch
from tools.coopaudit.design import sha
from tools.coopaudit.controllers import BoundController


def coordination(expected):
    source = Path(__file__).with_name('upstream_coordination.py')
    if not source.exists():
        source = Path('E:/agent-work/c4-sumocf/tools/sumocf/coordination.py')
    if sha(source) != expected:
        raise ValueError('trained RSU adapter source changed')
    spec = importlib.util.spec_from_file_location('coopaudit_bound_coordination', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trained_bound(selection):
    from wms.control.policy import load_policy_fn
    path = selection['checkpoint']
    if sha(path) != selection['checkpoint_sha256']:
        raise ValueError('fixed final trained checkpoint changed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'sumocf_rsu' not in payload:
        raise ValueError('trained checkpoint missing RSU weights')
    executor = load_policy_fn(path,device='cpu')
    if not executor.native_lanes or executor.residual_on_native:
        raise ValueError('wrong trained executor native-lane/direct-accel semantics')
    executor.reset()
    adapter = coordination(selection['coordination_source_sha256']).Coordinated(executor,payload)
    if adapter.intervention_offsets is not None:
        raise ValueError('audit must use deployed mean without exploration')
    return BoundController(adapter,'sumocf')
