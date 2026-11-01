"""Compare exported causal bins with recorded pre-t0 bins; never score outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from tools.rb2int.audit import ROOT, WORK
from tools.rb2int.prepare_inputs import safe, sha, read, save

ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')
REGIONS = ('upstream', 'merge', 'ramp')


def compare(exported, recorded, worlds, *, tolerance=1e-8):
    expected = {(s, a) for s in worlds for a in ARMS}
    selected = [r for r in recorded if (r['seed'], r['arm']) in expected]
    targets = {(r['seed'], r['arm']): r for r in selected}
    given = {(r['seed'], r['arm']): r for r in exported['episodes']}
    if (set(given) != expected or set(targets) != expected or len(given) != len(exported['episodes'])
            or len(targets) != len(selected)):
        raise ValueError('complete registered causal prefix roster required')
    errors, canonical = [], []
    for key in sorted(expected):
        a, b = given[key], targets[key]
        if a['initial_sha256'] != b['initial_sha256']:
            raise ValueError('observable warmup identity differs')
        bins = {}
        for region in REGIONS:
            x, y = a['bins'][region]['speed'], b['bins'][region]['speed'][:6]
            if len(x) != 6 or len(y) != 6:
                raise ValueError('prefix must have exactly six pre-t0 bins')
            for av, bv in zip(x, y):
                if (av is None) != (bv is None):
                    raise ValueError('causal speed support differs')
                if av is not None:
                    delta = abs(av-bv)
                    if not delta <= tolerance:
                        raise ValueError('causal speed bin differs beyond registered numeric tolerance')
                    errors.append(delta)
            bins[region] = dict(speed=y)
        canonical.append(dict(seed=key[0], arm=key[1], demand=b['demand'],
            penetration=b['penetration'], initial_sha256=b['initial_sha256'], bins=bins))
    return dict(schema='rb2int-causal-prefix/1', interval_s=[240, 300],
        future_outcome_fields=False, episodes=canonical), dict(passed=True,
        worlds=len(worlds), arms=5, regions=3, bins=6,
        max_absolute_prefix_error=max(errors, default=0.), absolute_tolerance=tolerance,
        final_outcome_values_used=False, model_fit_or_selection=False,
        source_fields_used=['seed', 'arm', 'demand', 'penetration', 'initial_sha256', 'bins.*.speed[:6]'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--canonical-out', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    archive, canonical_out, out = map(safe, (args.archive, args.canonical_out, args.out))
    if any(not p.parent.is_dir() or p.exists() for p in (canonical_out, out)):
        raise ValueError('exclusive output files in existing parents required')
    if sha(archive) != args.archive_sha256:
        raise ValueError('causal source archive hash differs')
    with tarfile.open(archive) as stream:
        raw = stream.extractfile('prefix_observables.json').read()
    exported = json.loads(raw)
    reg_path = ROOT/'docs/rb2int/EVALUATION_REGISTRATION.json'
    reg = read(reg_path)
    paths = [ROOT/'tools/rbeval/real_observables.json',
             WORK/'c4-wmbreakdown/docs/paper_results/wmbreakdown/wmdata2_observables.json']
    # Ancillary reduced receipts contain future fields, but this projection uses
    # ONLY the whitelisted initial identity and pre-t0 speed bins. Model/parameter
    # freezes already exist; no future value is inspected, emitted or selected.
    recorded = [row for path in paths for row in read(path)['episodes']]
    canonical, proof = compare(exported, recorded, reg['final_comparison_worlds'])
    proof.update(source_sha256=sha(__file__), registration_sha256=sha(reg_path),
        exported_prefix_sha256=hashlib.sha256(raw).hexdigest(),
        archive_sha256=args.archive_sha256, recorded_receipts={str(p): sha(p) for p in paths})
    save(canonical_out, canonical)
    proof['canonical_prefix_sha256'] = sha(canonical_out)
    save(out, proof)
    print(json.dumps(dict(proof=str(out), passed=True, worlds=proof['worlds'],
        max_absolute_prefix_error=proof['max_absolute_prefix_error'], canonical_sha256=proof['canonical_prefix_sha256']), indent=1))


if __name__ == '__main__':
    main()
