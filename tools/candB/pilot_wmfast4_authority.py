"""Explicit exploratory S4 authority for the now peer-free gpuA boundary."""
import argparse
import json

from tools.candB import pilot_authority as base, pilot_lean_authority as lean

LABEL = lean.LABEL
ENV = dict(lean.ENV, WMS_SPEED_FORWARD_WORKERS='1',
    WMS_SPEED_FORWARD_MEMORY_FRACTION='.75', WMS_SPEED_ES_STACK_SIDES='4',
    WMS_SPEED_FUTURE_WORKERS='2', WMS_SPEED_FUTURE_VJP_SLOTS='2',
    WMS_SPEED_FUTURE_MEMORY_FRACTION='.24')


def recipe(doc):
    result = lean.recipe(doc)
    result.update(environment=ENV, continuation_only=True,
        owner_amendment='2026-10-31 00:52 MAIN; peer-free gpuA S4',
        initialization='ordinary full resume; original fresh-zero phase retained',
        wmfast_stack_sides=4,
        memory_selection='peer-free admission >=46000MiB; one .75 S4 forward worker; '
            'two .24 dense VJP workers; measured coordinator+contexts+peers+2GiB '
            '<=min(90% device,45GiB) checked before EVERY pool')
    return result


def verify(path, *, exploratory=False):
    doc = base.verify(path, exploratory=exploratory)
    if doc.get('lean_recipe') != recipe(doc):
        raise ValueError('WMFAST S4 continuation recipe identity mismatch')
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
        with base.safe_path(args.authority).open('w') as stream:
            json.dump(doc, stream, indent=1)
            stream.write('\n')
    doc = verify(args.authority, exploratory=True)
    print(json.dumps(dict(label=LABEL, source_commit=doc['source_commit'],
        authority_sha256=base.sha256(args.authority), recipe=doc['lean_recipe']), indent=1))


if __name__ == '__main__':
    main()
