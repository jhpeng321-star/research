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
    serve(root/'rpc', binding, create)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(Path(parser.parse_args().root))
