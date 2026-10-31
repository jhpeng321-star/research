"""Resolve migrated shared inputs without modifying frozen historical evidence."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PureWindowsPath('D:/research')
INPUT_OPTIONS = frozenset(('net-file', 'route-files', 'additional-files', 'gui-settings-file'))


def legacy_relative(path):
    """Match the old directory exactly, including Windows JSON/backslash spellings."""
    try:
        relative = PureWindowsPath(str(path)).relative_to(LEGACY_ROOT)
    except ValueError:
        return None
    if '..' in relative.parts:
        raise ValueError('Parent traversal in a migrated research path')
    return relative


def resolve_research_path(path, *, root=ROOT):
    """Prefer only a verified local copy; never fall back to the retired source.

    Ordinary local/temporary paths are unchanged. The source pathname remains
    a provenance identifier, not a dependency on the existence of D:/research.
    """
    relative = legacy_relative(path)
    if relative is None:
        return Path(path)
    root = Path(root).resolve()
    base = root / 'assets/research'
    target = (base / Path(*relative.parts)).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError('Migrated path escapes the local asset directory')
    manifest = json.loads((base / 'migration_manifest.json').read_text(encoding='utf-8'))
    key = target.relative_to(root).as_posix().casefold()
    row = next((r for r in manifest['files'] if r['path'].casefold() == key), None)
    if row is None:
        if target.is_dir() and any(r['path'].casefold().startswith(key + '/')
                                   for r in manifest['files']):
            return target
        raise FileNotFoundError(f'No preserved dependency for {path}')
    if (not target.is_file() or target.stat().st_size != row['bytes']
            or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']):
        raise ValueError(f'Migrated input is missing or differs from its source: {target}')
    return target


def sumo_input_overrides(scene, *, root=ROOT):
    """CLI-only overrides keep a saved .sumocfg and its historical hashes intact."""
    scene = resolve_research_path(scene, root=root).resolve()
    if not scene.is_file():
        return []  # SUMO retains responsibility for reporting an absent config.
    overrides = []
    for node in ET.parse(scene).iter():
        if node.tag not in INPUT_OPTIONS:
            continue
        values = [s.strip() for s in node.get('value', '').split(',') if s.strip()]
        if not any(legacy_relative(value) is not None for value in values):
            continue
        paths = []
        for value in values:
            if legacy_relative(value) is not None:
                paths.append(str(resolve_research_path(value, root=root)))
            else:
                paths.append(str((scene.parent / value).resolve()))
        overrides.extend(['--' + node.tag, ','.join(paths)])
    return overrides
