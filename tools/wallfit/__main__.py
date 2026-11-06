"""python -m tools.wallfit {build|latent|run|smoke} ...

build   (CPU; cpuV and/or gpuA, where the ledgers live)  -> LABELS.jsonl, QUERIES.jsonl.gz, BUILD_RECEIPT.json
latent  (gpuA GPU, frozen WM, zero SUMO)                  -> LATENT.pt (+ .receipt.json)
run     (gpuA; CPU also works)                            -> RESULT.json, PREDICTIONS.jsonl, HEAD_SPLIT.pt, HEAD.pt
smoke   (synthetic fakes, any host)                       -> the same run outputs on synthetic data
"""
import json
from pathlib import Path
import sys


def smoke(argv):
    import argparse
    from . import evaluate, synthetic
    ap = argparse.ArgumentParser(prog='python -m tools.wallfit smoke')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args(argv)
    out = Path(a.out)
    pairs, latents = synthetic.synthetic_pairs()
    labels = synthetic.write_labels(pairs, out / 'SYN_LABELS.jsonl')
    lat = synthetic.write_latents(latents, out / 'SYN_LATENT.pt')
    r = evaluate.run([labels], out / 'run', latent=[lat], config=dict(device=a.device, epochs=200, ensemble=2))
    print(json.dumps(evaluate.summary(r), indent=1, sort_keys=True))
    return r


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'build':
        from .dataset import main as m
    elif cmd == 'latent':
        from .latent import main as m
    elif cmd == 'run':
        from .evaluate import main as m
    elif cmd == 'smoke':
        m = smoke
    else:
        raise SystemExit(f'unknown command {cmd!r}\n' + __doc__)
    m(rest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
