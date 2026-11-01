"""Verify and package the owner-requested frozen LC fit without refitting."""
import argparse
import hashlib
import json
from pathlib import Path

from wms.physics.lane_continuation import LaneContinuation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    fit = LaneContinuation.from_directory(args.source)
    verified = json.loads((args.source/'VERIFIED.json').read_text())
    if verified['status'] != 'PASS' or args.out.exists() or not args.out.parent.is_dir():
        raise ValueError('verified source and new destination required')
    names = ('lc2013_population.json', 'lc2013_shrinkage.json', 'provenance.json', 'VERIFIED.json')
    raw = {name: (args.source/name).read_bytes() for name in names}
    for name, data in raw.items():
        if name in verified['artifact_sha256'] and hashlib.sha256(data).hexdigest() != verified['artifact_sha256'][name]:
            raise ValueError('verified artifact differs: '+name)
    args.out.mkdir()
    for name, data in raw.items():
        (args.out/name).write_bytes(data)
    receipt = dict(source=str(args.source), artifact_sha256={name:hashlib.sha256(data).hexdigest()
        for name, data in raw.items()}, active_profiles=len(fit.active),
        profiles=len(fit.support), fit_worlds=fit.population['fit_worlds'],
        saved_driver_id_lookup=False, population_refit=False,
        note='Per-driver weights reconditioned from own causal motion; original native-ID posterior intentionally not transferred')
    (args.out/'PACKAGE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
