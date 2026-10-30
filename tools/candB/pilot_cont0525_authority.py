"""Owner05:25 peer-present S2 continuation authority (exploratory, label unchanged)."""
import argparse
import json

from tools.candB import pilot_authority as base, pilot_lean_authority as lean

LABEL = lean.LABEL
ENV = dict(lean.ENV, WMS_SPEED_FORWARD_WORKERS='1',
    WMS_SPEED_FORWARD_MEMORY_FRACTION='.28', WMS_SPEED_ES_STACK_SIDES='2',
    WMS_SPEED_FUTURE_WORKERS='2', WMS_SPEED_FUTURE_VJP_SLOTS='2',
    WMS_SPEED_FUTURE_MEMORY_FRACTION='.16')


def recipe(doc):
    result = lean.recipe(doc)
    result.update(environment=ENV, continuation_only=True,
        owner_amendment='2026-10-31 05:25 MAIN; peer-present gpuA S2 continuation',
        initialization='ordinary full resume; original fresh-zero phase retained',
        wmfast_stack_sides=2,
        memory_selection='peer-aware admission: sampled peers<=23000MiB and free>=21000MiB; '
            'one .28 S2 forward worker; two .16 dense VJP workers; measured '
            'coordinator+worker caps+contexts+sampled peers+2GiB <=min(90% device,45GiB) '
            'checked before EVERY pool; total gpuA kept <=~44GB next to the verdict peer')
    return result


def verify(path, *, exploratory=False):
    doc = base.verify(path, exploratory=exploratory)
    if doc.get('lean_recipe') != recipe(doc):
        raise ValueError('CONT0525 S2 peer-present continuation recipe identity mismatch')
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
