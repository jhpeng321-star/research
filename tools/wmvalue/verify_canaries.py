"""Require equal frozen runtime sources and exact CPU canaries across hosts."""
import argparse
import json

from .audit import digest, safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('first')
    p.add_argument('second')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    one, two = (json.loads(safe(p).read_text()) for p in (a.first, a.second))
    x, y = one['source_sha256'], two['source_sha256']
    differences = [k for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
    assert not differences, 'different runtime source modules: '+repr(differences)
    assert one['results'] == two['results'], 'CPU cross-host canary costs/births differ'
    assert one['initial_identity_columns'] == two['initial_identity_columns']
    result = dict(schema='wmvalue_cross_host/1', verdict='PASS', actual_fit_seed=969400,
        t0=360, seconds=2, exact_costs_and_births=True, identical_sources=len(x),
        source_sha256=x, first_sha256=digest(a.first), second_sha256=digest(a.second),
        reserved_outcomes_opened=False)
    with safe(a.out).open('x') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
