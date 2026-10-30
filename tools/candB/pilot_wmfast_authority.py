"""Regenerated exploratory lean authority for memory-bounded S2 continuation."""
import argparse
import json

from tools.candB import pilot_authority as base, pilot_lean_authority as lean

LABEL = lean.LABEL
ENV = dict(lean.ENV, WMS_SPEED_FORWARD_WORKERS='1',
    WMS_SPEED_FORWARD_MEMORY_FRACTION='.50', WMS_SPEED_ES_STACK_SIDES='2')


def recipe(doc):
    result = lean.recipe(doc)
    result.update(environment=ENV, continuation_only=True,
        owner_amendment='2026-10-31 00:52 MAIN',
        initialization='ordinary full resume; original fresh-zero phase retained',
        wmfast_stack_sides=2,
        memory_selection='S4 production fit unmeasured; S2 one .50 worker retains <=45GiB guard')
    return result


def verify(path, *, exploratory=False):
    doc = base.verify(path, exploratory=exploratory)
    if doc.get('lean_recipe') != recipe(doc):
        raise ValueError('WMFAST continuation recipe identity mismatch')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(base.FLAG, action='store_true', required=True)
    parser.add_argument('command', choices=('build', 'verify'))
    parser.add_argument('authority')
    parser.add_argument('--parent-authority')
    args = parser.parse_args()
    if args.command == 'build':
        parent = json.loads(base.safe_path(args.parent_authority).read_text())
        doc = base.build(parent['inputs'], args.authority, exploratory=True)
        doc['lean_recipe'] = recipe(doc)
        base.safe_path(args.authority).write_text(json.dumps(doc, indent=1)+'\n')
    doc = verify(args.authority, exploratory=True)
    print(json.dumps(dict(label=LABEL, source_commit=doc['source_commit'],
        authority_sha256=base.sha256(args.authority), recipe=doc['lean_recipe']), indent=1))


if __name__ == '__main__':
    main()
