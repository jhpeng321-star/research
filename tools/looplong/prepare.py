"""Write an explicit default-OFF enable config; never mutate state or run a phase."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/c4'))
import loop14  # noqa: E402
import loop14_long  # noqa: E402


def prepare(base, output, pairs300, threshold, *, enable=False, coop_credit=True,
            imagination_horizon_s=None):
    cfg = loop14.load_config(base)
    threshold = Path(loop14.guard(threshold))
    registered = loop14.read_json(threshold)
    if registered.get('schema') != 'coopw_horizon_gate/1' or registered.get('threshold') is None:
        raise ValueError('use the existing registered threshold file')
    cfg['looplong'] = {'enabled': enable, **{flag: True for flag in loop14_long.FLAGS},
        'enable_cycle': 3, 'pairs300': loop14.guard(pairs300), 'threshold': str(threshold),
        'threshold_sha256': hashlib.sha256(threshold.read_bytes()).hexdigest(),
        'groups': registered.get('groups', 96), 'panel_seed': registered.get('panel_seed', 20261026),
        'coop_credit': coop_credit}
    if imagination_horizon_s is not None:
        cfg['looplong']['imagination_horizon_s'] = imagination_horizon_s
    if enable:
        cfg.setdefault('fastloop', {})['overlap_policy'] = True
        cfg.setdefault('resumeprep', {})['enabled'] = True
    loop14_long.validate(cfg, loop14.Refused)
    output = Path(loop14.guard(output))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(cfg, stream, indent=1)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'out', 'pairs300', 'threshold'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--enable', action='store_true')
    ap.add_argument('--coop-credit', choices=('on', 'off'), default='on')
    ap.add_argument('--imagination-horizon-s', type=int, choices=loop14_long.HORIZONS)
    args = ap.parse_args()
    prepare(args.base, args.out, args.pairs300, args.threshold,
            enable=args.enable, coop_credit=args.coop_credit == 'on',
            imagination_horizon_s=args.imagination_horizon_s)
    print(json.dumps({'prepared': args.out, 'enabled': args.enable, 'no_phase_launched': True}))


if __name__ == '__main__':
    main()
