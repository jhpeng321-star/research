"""Verify local shared inputs or prepare a relocated SUMO config; never launch SUMO."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_shepherd.research_assets import (  # noqa: E402
    INPUT_OPTIONS, legacy_relative, resolve_research_path,
)


def verify():
    manifest_path = ROOT / 'assets/research/migration_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        try:
            local = resolve_research_path(row['source'])
            if local != (ROOT / row['path']).resolve():
                errors.append(f"Mapping disagrees: {row['source']}")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    return dict(task_id='MIG-01', checked=len(manifest['files']), errors=errors,
                total_bytes=manifest['total_bytes'], source_access_required=False,
                preexisting_missing_inputs=manifest['original_input_gaps'])


def prepare(source, output):
    """Create a derived config with resolved inputs, leaving source bytes intact.

    The output config is a relocation artifact, not a new run or registration.
    It does not bypass the project's simulation-budget/identity checks.
    """
    source = resolve_research_path(source).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.is_relative_to(ROOT / 'archive') \
            or output.is_relative_to(ROOT / 'assets'):
        raise ValueError('Put derived replay configurations in a new workspace output directory')
    before = source.read_bytes()
    tree = ET.fromstring(before)
    mappings = []
    for node in tree.iter():
        if node.tag not in INPUT_OPTIONS:
            continue
        items = [s.strip() for s in node.get('value', '').split(',') if s.strip()]
        resolved = []
        for item in items:
            path = resolve_research_path(item) if legacy_relative(item) is not None \
                else (source.parent / item).resolve()
            if not path.is_file():
                raise FileNotFoundError(f'{node.tag}: {path}')
            resolved.append(str(path))
            mappings.append(dict(option=node.tag, original=item, resolved=str(path),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        node.set('value', ','.join(resolved))
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'scenario.sumocfg'
    ET.indent(tree)
    ET.ElementTree(tree).write(target, encoding='utf-8', xml_declaration=True)
    receipt = dict(source=str(source), source_sha256=hashlib.sha256(before).hexdigest(),
                   prepared=str(target), prepared_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   input_mappings=mappings, simulations_started=0,
                   source_unchanged=(source.read_bytes() == before))
    (output / 'relocation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    p = sub.add_parser('resolve')
    p.add_argument('path')
    p = sub.add_parser('prepare')
    p.add_argument('source')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'verify':
        result = verify()
    elif args.command == 'resolve':
        result = dict(path=str(resolve_research_path(args.path)))
    else:
        result = prepare(args.source, args.output)
    print(json.dumps(result, indent=2))
    return 1 if result.get('errors') else 0


if __name__ == '__main__':
    raise SystemExit(main())
