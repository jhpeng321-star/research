"""Hash-bound adapters to the selector owners' checkpoint/observation interfaces.

PENDING specs are intentional stubs: they fail before importing or loading a
checkpoint, never return a placeholder controller and never start training.
"""
import importlib
import importlib.util
import json
import sys

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical

from .design import CONTROLLERS, PROTOCOL, TIMES
from .observable import anonymous_payload
from .wall import Wall


class DependencyPending(RuntimeError):
    pass


def package(spec, required):
    if spec.get('status') != 'FROZEN_FINAL':
        raise DependencyPending('final169 selector checkpoint/source receipt not delivered')
    root = checked(spec['root']).resolve()
    sources = spec['sources']
    if not {'__init__.py', *required} <= sources.keys():
        raise ValueError('selector peer sources incomplete')
    for name, expected in sources.items():
        path = checked(root/name).resolve()
        if not path.is_relative_to(root) or path.suffix != '.py' or sha(path) != expected:
            raise ValueError('selector peer source changed')
    name = '_c1switch2_peer_'+canonical(dict(root=str(root), sources=sources))[:16]
    if name not in sys.modules:
        definition = importlib.util.spec_from_file_location(name, root/'__init__.py',
                                                           submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(definition)
        sys.modules[name] = module
        definition.loader.exec_module(module)
    return name


def frozen_spec(spec):
    # PENDING has no paths to open, including before final checkpoints exist.
    if spec.get('status') != 'FROZEN_FINAL':
        raise DependencyPending('final169 selector checkpoint/source receipt not delivered')
    path = checked(spec['checkpoint'])
    if sha(path) != spec['checkpoint_sha256']:
        raise ValueError('selector checkpoint hash mismatch')
    frozen_path = checked(spec['frozen_receipt'])
    if sha(frozen_path) != spec['frozen_receipt_sha256']:
        raise ValueError('selector final receipt hash mismatch')
    frozen = json.loads(frozen_path.read_text())
    if frozen['iteration'] != 169 or frozen['checkpoint_sha256'] != spec['checkpoint_sha256']:
        raise ValueError('only final169 checkpoint eligible')
    return path, frozen


class Fleet:
    def __init__(self, spec):
        path, _ = frozen_spec(spec)
        name = package(spec, {'selector.py', 'observations.py', 'learning.py', 'protocol.py'})
        peer = importlib.import_module(name+'.selector')
        self.selector = peer.FrozenFleetSelector(path, spec['checkpoint_sha256'])

    def select(self, payload):
        result = self.selector.select(anonymous_payload(payload, Wall.validate_memory))
        if result not in CONTROLLERS:
            raise ValueError('fleet selector returned unregistered controller')
        return result


class PerCav:
    def __init__(self, spec):
        path, frozen = frozen_spec(spec)
        name = package(spec, {'observations.py', 'learning.py', 'protocol.py'})
        protocol = importlib.import_module(name+'.protocol')
        if protocol.PROTOCOL != PROTOCOL or tuple(protocol.LIBRARY) != CONTROLLERS:
            raise ValueError('per-CAV shared protocol mismatch')
        import torch
        # The owner saves Python/NumPy RNG and optimizer state, not weights alone.
        # Only the caller-pinned final receipt/checkpoint is loaded, once on CPU.
        record = torch.load(path, map_location='cpu', weights_only=False)
        if (record['iteration'] != 169 or record['variant'] != 'per_cav' or
                record['optimization_seed'] != 1 or record['sources'] != frozen['sources'] or
                record['bindings'] != frozen['bindings']):
            raise ValueError('per-CAV final169 source/binding contract mismatch')
        expected = spec['checkpoint_sources']
        if record['sources'] != expected or record['bindings'] != spec['checkpoint_bindings']:
            raise ValueError('per-CAV checkpoint must bind its training sources and integration')
        for stem in ('observations', 'learning', 'protocol'):
            if expected[stem] != spec['sources'][stem+'.py']:
                raise ValueError('evaluation feature/model sources differ from training')
        self.build = importlib.import_module(name+'.observations').build
        self.model = importlib.import_module(name+'.learning').Selector().cpu()
        self.model.load_state_dict(record['model'])
        self.model.eval()
        self.model.requires_grad_(False)
        self.index, self.incumbents = 0, {}
        self.synchronized = False

    def sync_incumbents(self, names):
        if not isinstance(names, dict) or any(type(key) is not str or value not in CONTROLLERS
                                             for key, value in names.items()):
            raise ValueError('invalid observable live per-CAV incumbents')
        self.incumbents = dict(names)
        self.synchronized = True

    def select(self, raw):
        payload = anonymous_payload(raw, Wall.validate_memory)
        if self.index >= len(TIMES) or payload['history'][-1]['now_s'] != TIMES[self.index]:
            raise ValueError('per-CAV picker off-grid or reused without episode reset')
        own = {r['vehicle_id'] for r in payload['history'][-1]['self_reports']}
        if not self.synchronized or set(self.incumbents) != own:
            raise ValueError('synchronize incumbents from current live backend contexts before every pick')
        observation = self.build(payload, variant='per_cav', incumbents=self.incumbents)
        import torch
        with torch.inference_mode():
            actions, _ = self.model.act(observation, deterministic=True)
        if set(actions) != own:
            raise ValueError('per-CAV picker must cover exactly current self IDs')
        result = {key: CONTROLLERS[actions[key]] for key in actions}
        self.incumbents, self.index = result, self.index+1
        self.synchronized = False
        return result


def fleet(binding):
    return Fleet(binding['selector_specs']['fleet_ppo'])


def per_cav(binding):
    return PerCav(binding['selector_specs']['cav_mappo'])
