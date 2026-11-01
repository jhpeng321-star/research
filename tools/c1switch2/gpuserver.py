"""One registered hard-capped inference allocation on one owned GPU service."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(root):
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0, '/root/autodl-tmp/c4/speedhead20261030/pythondeps')
    from tools.c1switch.design import sha
    from .engine import create
    from .rpc import serve
    binding = json.loads((root/'ENGINE.json').read_text())
    sources = json.loads((root/'STAGED.json').read_text())
    if not all(sha(root/'pytools'/name) == digest for name, digest in sources.items()):
        raise ValueError('immutable GPU source changed')
    allocation_path = root/'GPU_ALLOCATION.json'
    allocation = json.loads(allocation_path.read_text()) if allocation_path.exists() else None
    if allocation is not None:
        from .gpuallocation import allocation_cap
        allocation_cap(binding, allocation)
        old = Path(allocation['predecessor_root'])
        if (sha(old/'rpc/GPU_FINISHED.json') != allocation['finished_sha256'] or
                json.loads((old/'rpc/GPU_FINISHED.json').read_text()) != allocation['finished'] or
                Path('/proc', str(allocation['finished']['pid'])).exists()):
            raise ValueError('continuation requires exact exited predecessor and final debit')
        registration = (root/'CONTINUATION_REGISTRATION.md').read_text()
        if allocation['authority'] not in registration or allocation['plan_sha256'] not in registration:
            raise ValueError('continuation not prospectively registered')
    serve(root/'rpc', binding, create, allocation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(Path(parser.parse_args().root))
