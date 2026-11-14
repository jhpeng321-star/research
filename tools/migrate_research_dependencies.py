"""Preserve required research inputs locally; never delete source data or run SUMO."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('D:/research')
DEST = ROOT / 'assets/research'
RECORD = ROOT / 'output/research_dependency_migration_v1'
MAPS = ('highway_merge_shepherd', 'highway_merge_shepherd_long',
        'highway_merge_shepherd_long2', 'highway_merge_shepherd_long3',
        'highway_merge_shepherd_mid')
INPUT_SUFFIXES = ('.sumocfg', '.net.xml', '.net.xml.gz', '.rou.xml', '.rou.xml.gz',
                  '.add.xml', '.add.xml.gz', '.view.xml', '.nod.xml', '.edg.xml',
                  '.con.xml', '.typ.xml', '.tll.xml', '.poly.xml', '.netccfg')
INPUT_OPTIONS = ('net-file', 'route-files', 'additional-files', 'gui-settings-file')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files_under(root):
    for here, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [n for n in dirs if n not in ('.git', '__pycache__', '.pytest_cache')
                   and not (Path(here, n).lstat().st_file_attributes & 0x400)]
        for name in names:
            path = Path(here, name)
            if not (path.lstat().st_file_attributes & 0x400):
                yield path


def inventory():
    selected = set()
    for name in MAPS:
        for p in files_under(SOURCE / name):
            if p.name.lower().endswith(INPUT_SUFFIXES) or p.suffix.lower() in (
                    '.bat', '.cmd', '.ps1', '.py', '.md', '.txt'):
                selected.add(p.resolve())
    # These are explicit inputs of the archived checkpoint-comparison tool.
    # Preserve them as opaque bytes: do not parse historical outcomes.
    selected.update(p.resolve() for p in
                    (SOURCE / 'traffic_shepherd_rl/runs/eval').glob('all_hdv_*.json'))
    missing = []
    for p in sorted(selected):
        if p.suffix != '.sumocfg':
            continue
        tree = ET.parse(p)
        for node in tree.iter():
            if node.tag not in INPUT_OPTIONS:
                continue
            for item in node.get('value', '').split(','):
                if not item.strip():
                    continue
                dependency = (p.parent / item.strip()).resolve()
                if not dependency.is_relative_to(SOURCE.resolve()):
                    raise ValueError(f'Unexpected external input: {p}: {dependency}')
                if dependency.is_file():
                    selected.add(dependency)
                else:
                    missing.append(dict(config=str(p), option=node.tag,
                                        missing_source=str(dependency)))
    rows = []
    for p in sorted(selected):
        relative = p.relative_to(SOURCE.resolve())
        target = (DEST / relative).resolve()
        if not target.is_relative_to(DEST.resolve()):
            raise ValueError(f'Unsafe migration target: {target}')
        rows.append(dict(source=str(p), path=target.relative_to(ROOT).as_posix(),
                         bytes=p.stat().st_size, sha256=sha(p),
                         role='opaque_historical_reference' if relative.parts[0]
                         == 'traffic_shepherd_rl' else 'scenario_input_or_build_support'))
    return rows, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args(argv)
    RECORD.mkdir(parents=True, exist_ok=True)
    rows, missing = inventory()
    plan = dict(schema='research_dependency_migration/1', task_id='MIG-01',
                source_root=str(SOURCE), destination_root=DEST.relative_to(ROOT).as_posix(),
                source_deleted=False, simulations_started=0,
                selection='Five referenced highway input families and their build support; '
                          'explicit all_hdv_*.json legacy baseline inputs (opaque copy/hash only). '
                          'Unreferenced lane-drop project, old RL source/checkpoints, and generated '
                          'simulation outputs are not migration dependencies.',
                original_input_gaps=missing, files=rows,
                file_count=len(rows), total_bytes=sum(r['bytes'] for r in rows))
    plan_path = RECORD / 'PLAN.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Source inventory changed since migration was planned')
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    if args.copy:
        for row in rows:
            source, target = Path(row['source']), ROOT / row['path']
            if sha(source) != row['sha256']:
                raise ValueError(f'Source changed: {source}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != row['sha256']:
                    raise ValueError(f'Refusing to overwrite different target: {target}')
            else:
                shutil.copy2(source, target)
            if sha(target) != row['sha256']:
                raise ValueError(f'Copy verification failed: {target}')
        manifest = dict(plan, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        status='copied_and_sha256_verified_source_retained')
        manifest_path = DEST / 'migration_manifest.json'
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous['files'] != rows:
                raise ValueError('Existing migration manifest differs')
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(copied=args.copy, files=len(rows), bytes=plan['total_bytes'],
                          original_input_gaps=missing,
                          families=dict(Counter(Path(r['path']).parts[2] for r in rows))),
                     indent=2))


if __name__ == '__main__':
    main()
