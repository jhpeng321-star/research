"""Hash-bound atomic file inference handshake with explicit objective contract."""
import argparse
import importlib
import json
from pathlib import Path
import signal
import time

from .core import medians
from .design import checked, sha
from .storage import canonical, exclusive, write


def validate_objective(binding):
    # Legacy team-support bindings remain explicit; MAIN's fleet-only amendment
    # must declare the evaluation mismatch rather than falsely attest team support.
    if binding.get('team_source_entries') is True:
        return
    if not (binding.get('prediction_objective') == 'on_road_fleet_delay' and
            binding.get('evaluation_objective') == 'full_team_delay_fleet_plus_source' and
            binding.get('history_s') == 60 and binding.get('horizon_s') == 60 and
            binding.get('K') == 8 and binding.get('rule') == 'MAIN18:33 fixed before switching outcomes'):
        raise ValueError('unverified prediction/evaluation objective contract')


class FilePredictor:
    def __init__(self, root, binding, timeout_s=900):
        self.root, self.binding, self.timeout_s = checked(root), binding, timeout_s
        validate_objective(binding)

    def predict(self, payload, ident):
        # Payload must already have passed the Belief allowlist boundary.
        if payload.get('schema') != 'c1switch_observable/1':
            raise ValueError('observable belief required')
        request = dict(id=ident, belief=payload, binding=self.binding, K=8, horizon_s=60)
        key = canonical(request)
        request['request_sha256'] = key
        path = self.root / (key+'.request.json')
        if not path.exists():
            write(path, request)
        finish = self.root / (key+'.result.json')
        start = time.monotonic()
        while not finish.exists():
            if (self.root/'STOP').exists() or time.monotonic()-start > self.timeout_s:
                raise TimeoutError('inference stopped/timed out; SUMO stays paused')
            time.sleep(.25)
        result = json.loads(finish.read_text())
        if result['request_sha256'] != key or result['binding'] != self.binding:
            raise ValueError('inference identity mismatch')
        return medians(result['samples'])


def serve(root, adapter_name, binding_path, device):
    """Adapter receives only request belief+binding; support must be verified beforehand.

    No speculative WM adapter is substituted when diagnosis/source contract is missing.
    GPU process owns one model; wall-clock incl loading is charged against per-model cap.
    """
    root = checked(root)
    binding = json.loads(checked(binding_path).read_text())
    validate_objective(binding)
    if not binding.get('diagnosis_sha256'):
        raise ValueError('missing model-selection/source receipt')
    module, func = adapter_name.split(':')
    mod = importlib.import_module(module)
    if sha(mod.__file__) != binding['adapter_sha256']:
        raise ValueError('adapter changed')
    began = time.monotonic()
    with exclusive(root):
        if (root/'GPU_STARTED.json').exists():
            raise ValueError('previous GPU allocation; reconcile before restart')
        cap = binding['gpu_cap_s']
        if cap not in (10800, 3600):
            raise ValueError('unregistered per-model GPU cap')
        write(root/'GPU_STARTED.json', dict(cap_s=cap, unix=time.time()))
        def expired(signum, frame):
            write(root/'STOP', dict(reason='GPU hard wall-time cap', cap_s=cap))
            raise TimeoutError('GPU allocation exhausted')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, cap)
        model = getattr(mod, func)(binding, device=device)
        while not (root/'STOP').exists():
            if time.monotonic()-began >= binding['gpu_cap_s']:
                write(root/'STOP', dict(reason='GPU wall-time cap'))
                return
            for request_path in sorted(root.glob('*.request.json')):
                result_path = request_path.with_name(request_path.name.replace('.request.', '.result.'))
                if result_path.exists():
                    continue
                request = json.loads(request_path.read_text())
                key = request.pop('request_sha256')
                if canonical(request) != key or request['binding'] != binding:
                    raise ValueError('request binding mismatch')
                samples = model.predict(request['belief'], K=8, horizon_s=60)
                medians(samples)
                write(result_path, dict(request_sha256=key, binding=binding, samples=samples,
                                        elapsed_gpu_s=time.monotonic()-began))
            time.sleep(.25)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    serve(a.root, a.adapter, a.binding, a.device)
