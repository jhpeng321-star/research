"""One immutable world/time/future task, five closed-loop arms; no real labels."""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

from .audit import ARMS, allowed_seed, digest, safe
from .audit_runtime import read_prefixes
from .core import TIMES
from .restart import current_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefixes', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--t0', type=int, choices=TIMES, required=True)
    p.add_argument('--future', type=int, choices=range(8), required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=int)  # FIT-only canary, never a reserved partial A
    p.add_argument('--short', action='store_true')  # MAIN21:12; B/C only, never A
    a = p.parse_args()
    if a.short and a.seconds is not None:
        raise ValueError('short reserved execution and FIT canary are distinct')
    if a.seconds is not None:
        allowed_seed(a.seed, 'fit')
    else:
        allowed_seed(a.seed, 'test')
    out = safe(a.out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('new immutable task output and existing parent required')
    gate = json.loads(safe(a.gate).read_text())
    assert gate['verdict'] == 'PASS' and gate['device'] == 'cpu'
    for name in ('runtime.py', 'restart.py', 'reconstruct.py'):
        assert digest(Path(__file__).parent/name) == gate['code_sha256'][name]
    print(json.dumps(dict(status='START', pid=os.getpid(), seed=a.seed, t0=a.t0,
                          future=a.future, device=a.device)), flush=True)
    from . import runtime as r
    if a.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        r.torch.cuda.set_per_process_memory_fraction(5*2**30/r.torch.cuda.get_device_properties(0).total_memory)
        r.torch.backends.cuda.matmul.allow_tf32 = False
        r.torch.backends.cudnn.allow_tf32 = False
    r.torch.set_num_threads(1)
    begun = time.monotonic()
    rt = r.load(a.device)
    prefixes, metadata = read_prefixes(a.prefixes, a.seed, a.t0)
    columns = sorted(set().union(*(current_names(prefixes[arm]) for arm in ARMS)))
    batches, events = [], []
    for arm in ARMS:
        b, ev = r.make_batch(prefixes[arm], columns, rt['model'], metadata[arm]['queue300'],
                             metadata[arm]['previous_commands'][str(a.t0)])
        batches.append(b)
        events.append(ev)
    rollout_start = time.monotonic()
    seconds = 90 if a.short else a.seconds
    result = r.rollout(rt, batches, events, a.t0, a.future, a.device, prefixes, columns, seconds=seconds)
    # Hash every imported model/adapter/controller source, never training data.
    sources = {}
    for name, module in sorted(sys.modules.copy().items()):
        if name.startswith(('wms.', 'tools.errsrc.', 'tools.wmvalue.')) or name in ('wm_worker', 'measure', 'panel2'):
            path = getattr(module, '__file__', None)
            if path and path.endswith('.py'):
                sources[name] = digest(safe(path))
    for i, obj in enumerate(rt['dependencies'][1:]):
        if obj is not None:
            try:
                sources['controller_dependency_'+str(i)] = digest(safe(inspect.getfile(obj)))
            except TypeError:
                pass
    from wms.control.native_cav import HEAD_PATH
    sources['native_head_json'] = digest(safe(HEAD_PATH))
    sources['execution_driver_json'] = digest(safe(os.environ['C4_EXECUTION_DRIVER']))
    sources['worker'] = digest(safe(__file__))
    sources['circles_model'] = digest(safe(rt['dependencies'][0].model_path))
    result = dict(schema='wmvalue_rollout/1', seed=a.seed, t0=a.t0, future=a.future,
        start_time=a.t0, end_time=a.t0+(seconds if seconds is not None else 900-a.t0),
        arms=list(ARMS), K=8, mix=True, device=a.device, pid=os.getpid(),
        total_seconds=time.monotonic()-begun, rollout_seconds=time.monotonic()-rollout_start,
        prefixes_sha256=digest(a.prefixes), gate_sha256=digest(a.gate),
        initial_identity_columns=columns, wm=rt['receipt'], source_sha256=sources,
        candidate_sha256=r.frozen.digest(r.frozen.CANDIDATE), export_sha256=rt['binding']['sha256'],
        gpu_peak_reserved_mib=r.torch.cuda.max_memory_reserved()/2**20 if a.device == 'cuda' else 0,
        real_outcomes_opened=False, results=result)
    with out.open('x') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status='DONE', path=str(out), sha256=digest(out),
                          seconds=result['total_seconds'], device=a.device,
                          seed=a.seed, t0=a.t0, future=a.future)), flush=True)


if __name__ == '__main__':
    main()
