"""v7 hash-bound file/RPC protocol, anonymous v2 payloads and mean-of-K scoring."""
import json
import time

from tools.c1switch.design import checked
from tools.c1switch.storage import canonical, write

from .core import means
from .design import HORIZON, K, PROTOCOL
from .inference import validate_engine
from .observable import anonymous_payload


class FilePredictor:
    def __init__(self, roots, binding, wall, timeout_s=1200):
        if len(roots) != 2:
            raise ValueError('both GPU inboxes required')
        validate_engine(binding)
        self.roots = tuple(checked(root) for root in roots)
        self.binding, self.wall, self.timeout_s = binding, wall, timeout_s

    def predict(self, payload, ident):
        request = dict(protocol=PROTOCOL, id=ident,
                       belief=anonymous_payload(payload, self.wall.validate_memory),
                       binding=self.binding, K=K, horizon_s=HORIZON)
        key = canonical(request)
        # Deterministic stateless routing across BOTH GPUs; transport identity
        # chooses a service only, never the scientific stochastic key.
        root = self.roots[int(key[:8], 16) % 2]
        request['request_sha256'] = key
        path, finish = root/(key+'.request.json'), root/(key+'.result.json')
        if not path.exists():
            write(path, request)
        began = time.monotonic()
        while not finish.exists():
            if (root/'STOP').exists() or time.monotonic()-began > self.timeout_s:
                raise TimeoutError('inference unavailable; SUMO remains paused')
            time.sleep(.25)
        result = json.loads(finish.read_text())
        if result['request_sha256'] != key or result['binding'] != self.binding:
            raise ValueError('inference identity mismatch')
        return means(result['samples'])


def serve(root, binding, factory):
    """One GPU service allocation: exactly half its model arm's aggregate cap.

    Each remote service records elapsed wall time INCLUDING startup/idle/validation;
    two simultaneous services accrue two GPU-seconds per elapsed second.
    """
    import os
    import signal
    from tools.c1switch.storage import exclusive
    from .design import GPU_SECONDS
    root = checked(root)
    validate_engine(binding)
    arm = binding['arm']
    cap = GPU_SECONDS[arm]/2
    if binding['service_gpu_cap_s'] != cap or binding.get('prior_gpu_s', 0) != 0:
        raise ValueError('unregistered GPU service allocation/restart')
    with exclusive(root):
        if (root/'GPU_STARTED.json').exists():
            raise ValueError('previous allocation; no implicit GPU retry')
        began = time.monotonic()
        write(root/'GPU_STARTED.json', dict(pid=os.getpid(), cap_s=cap, unix=time.time()))
        def expired(signum, frame):
            raise TimeoutError('GPU service hard allocation exhausted')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, cap)
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            if free < 7*2**30:
                raise ValueError('insufficient free GPU memory for frozen service admission')
            torch.cuda.set_per_process_memory_fraction(2.5*2**30/total)
            predictor = factory(binding)
            write(root/'READY.json', dict(pid=os.getpid(), binding=binding,
                  attestation=predictor.engine.attestation))
            while not (root/'STOP').exists() and not (root/'COMPLETE').exists():
                for path in sorted(root.glob('*.request.json')):
                    finish = path.with_name(path.name.replace('.request.', '.result.'))
                    if finish.exists():
                        continue
                    request = json.loads(path.read_text())
                    key = request.pop('request_sha256')
                    if canonical(request) != key or request['binding'] != binding:
                        raise ValueError('inference request identity mismatch')
                    if request['K'] != K or request['horizon_s'] != HORIZON or request['protocol'] != PROTOCOL:
                        raise ValueError('inference protocol mismatch')
                    samples = predictor.samples(request['belief'])
                    means(samples)
                    write(finish, dict(request_sha256=key, binding=binding, samples=samples))
                time.sleep(.25)
        except BaseException as exc:
            write(root/'STOP', dict(reason=str(exc), pid=os.getpid()))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            write(root/'GPU_FINISHED.json', dict(pid=os.getpid(), elapsed_s=time.monotonic()-began))


def client(binding, root, arm):
    """Runner plugin ABI `(task_binding, task_root)` to the two GPU relay inboxes."""
    from .selectors import DependencyPending
    from .wall import Wall
    spec = binding['inference_specs'][arm]
    if spec.get('status') != 'FROZEN_READY':
        raise DependencyPending('hash-bound two-GPU inference/parity routes not ready')
    engine = spec['engine']
    validate_engine(engine)
    if (engine['arm'] != arm or engine['objective'] != binding['prediction_objective'] or
            engine['objective_design_sha256'] != binding['design_sha256']):
        raise ValueError('inference objective/design differs from registered task binding')
    if set(spec['routes']) != {'gpuA', 'gpuR'}:
        raise ValueError('both registered GPU routes required')
    root = checked(root).resolve()
    roots = []
    for host in ('gpuA', 'gpuR'):
        path = checked(root/spec['routes'][host]).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError('inference relay inbox must exist inside task root')
        roots.append(path)
    return FilePredictor(roots, engine, Wall())


def wm(binding, root):
    return client(binding, root, 'wm')


def bb(binding, root):
    return client(binding, root, 'bb')


def wm_phys(binding, root):
    return client(binding, root, 'wm_phys')
