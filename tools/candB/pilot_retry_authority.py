"""Owner03:00: source-bound zero-progress recovery with WMFAST-B already enabled."""
import argparse
import json

from tools.candB import pilot_authority as base, pilot_wmfast4_authority as s4

LABEL = s4.LABEL
ENV = s4.ENV


def recipe(doc):
    return dict(s4.recipe(doc), continuation_only=False,
        owner_amendment='2026-10-31 03:00 MAIN; repaired zero-progress retry; WMFAST ON',
        initialization='original zero191 full checkpoint; no reinitialization or baseline forwards')


def verify(path, *, exploratory=False):
    doc = base.verify(path, exploratory=exploratory)
    if doc.get('lean_recipe') != recipe(doc):
        raise ValueError('retry execution/authority identity mismatch')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(base.FLAG, action='store_true', required=True)
    parser.add_argument('action', choices=('build','verify'))
    parser.add_argument('authority')
    parser.add_argument('--parent-authority')
    args = parser.parse_args()
    if args.action == 'build':
        parent = json.loads(base.safe_path(args.parent_authority).read_text())
        doc = base.build(parent['inputs'], args.authority, exploratory=True)
        doc['lean_recipe'] = recipe(doc)
        with base.safe_path(args.authority).open('w') as stream:
            json.dump(doc,stream,indent=1)
            stream.write('\n')
    doc = verify(args.authority,exploratory=True)
    print(json.dumps(dict(source=doc['source_commit'],authority_sha256=base.sha256(args.authority),
        label=LABEL,recipe=doc['lean_recipe'])))


if __name__ == '__main__':
    main()
