"""Read-only transitive closure of the production bank readers' staging inputs.

``files, manifest = closure(paths)`` takes an iterable of inventory FILE paths.
It returns sorted absolute Paths (including lexical aliases AND physical link
targets), and a JSON-serializable SHA manifest. No torch/SUMO imports, snapshot
deserialization, writes, or broad directory scans occur. Pass the returned files
to the staging copier; include manifest['allowed_roots'] in its root admission.
Both aliases and targets may be materialized as regular files when copying.

Rules mirror replay_bank._receipts/_decision_snapshot/_ls_snapshot,
practice_starts.load_snapshot, accel_storage.read, and solo_seed2_links.wire.
An LS /home/ubuntu reference in a filesystem-root mirror is mapped into that
same mirror (inferred from input paths, or supplied via ``mirror=``).
"""
from collections import deque
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


SCHEMA = 'coopmwm.staging_closure.v1'
READERS = ['replay_bank._receipts', 'replay_bank._decision_snapshot',
           'replay_bank._ls_snapshot', 'practice_starts.load_snapshot',
           'accel_storage.read', 'solo_seed2_links.wire']
_FORBIDDEN = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
              'hetdata_truth', 'vtype', 'driver', 'vehicle_params')


def _guard(path):
    text = str(path).replace('\\', '/').lower()
    if 'sealed' in text.split('/') or any(token in text for token in _FORBIDDEN):
        raise ValueError('REFUSED: prohibited staging path: '+str(path))


def _absolute(path):
    _guard(path)
    # Do not normalize away a prohibited component before checking it.
    path = Path(os.path.abspath(path))
    _guard(path.resolve())
    return path


def _identifier(value, field):
    if not isinstance(value, str) or not value or value in ('.', '..') or any(
            ch in value for ch in '/\\:\x00'):
        raise ValueError('REFUSED: invalid '+field)
    return value


def _json(path):
    if path.name.endswith('.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _hash(path):
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def closure(paths, *, mirror=None):
    """Return ``(list[Path], manifest)``; fail on any missing required dependency.

    ``manifest['entries']`` maps each returned absolute POSIX path to sha256,
    bytes, resolved_path and reader provenance. ``aliases`` maps lexical paths
    to physical files; ``allowed_roots`` contains only actual file-parent roots,
    derived from this closure, never hand-maintained run names. The manifest's
    own SHA is canonical JSON (sort_keys, compact separators, ASCII, no NaN),
    excluding ``manifest_sha256``. All paths are guarded before opening.

    Inputs are the inventory files already selected by the real bank readers,
    not a directory to recursively inventory. Plain/compressed RESULT markers
    and their payload bytes are both included and payload digests validated.
    Snapshots are hashed as opaque bytes, never unpickled.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    initial = [_absolute(p) for p in paths]
    if not initial:
        raise ValueError('REFUSED: staging inventory is empty')
    mirrors = set()
    for p in initial:
        for i in range(len(p.parts)-1):
            if p.parts[i:i+2] == ('home', 'ubuntu'):
                mirrors.add(Path(*p.parts[:i]))
                break
    if mirror is not None:
        mirror = _absolute(mirror)
    elif len(mirrors) == 1:
        mirror = next(iter(mirrors))
    elif len(mirrors) > 1:
        raise ValueError('REFUSED: ambiguous filesystem-root mirrors; pass mirror=')

    def episode_path(value):
        if not isinstance(value, str) or not value:
            raise ValueError('REFUSED: invalid episode_path')
        _guard(value)
        posix = PurePosixPath(value.replace('\\', '/'))
        if mirror is not None and posix.parts[:3] == ('/', 'home', 'ubuntu'):
            return mirror.joinpath(*posix.parts[1:])
        path = Path(value)
        if not path.is_absolute():
            raise ValueError('REFUSED: episode_path must be absolute: '+value)
        return path

    queue = deque((p, 'inventory') for p in initial)
    entries, aliases, hashes, expanded = {}, {}, {}, set()

    def add(path, reader):
        queue.append((_absolute(path), reader))

    def branches(base, receipt):
        if receipt.get('status') not in ('accepted', 'rejected'):
            return
        vid = _identifier(receipt['verification_id'], 'verification_id')
        for row in receipt.get('branches') or ():
            branch = _identifier(row['branch'], 'branch')
            directory = base/vid/branch
            add(directory/'RESULT.json', 'replay_bank._receipts')
            trace = _absolute(directory/'observable.jsonl.gz')
            if trace.is_file():
                add(trace, 'replay_bank.bank_pairs')

    while queue:
        path, reader = queue.popleft()
        key = path.as_posix()
        if key in entries:
            entries[key]['readers'] = sorted(set(entries[key]['readers']) | {reader})
        else:
            resolved = path.resolve(strict=True)
            _guard(resolved)
            if not resolved.is_file():
                raise ValueError('REFUSED: staging input is not a file: '+str(path))
            if resolved not in hashes:
                hashes[resolved] = _hash(resolved)
            sha, size = hashes[resolved]
            entries[key] = dict(sha256=sha, bytes=size, resolved_path=resolved.as_posix(),
                                readers=[reader])
            if resolved != path:
                aliases[key] = resolved.as_posix()
                add(resolved, 'solo_seed2_links.wire:resolved_target')
        # Reader-relative dependencies belong to the lexical path used to open
        # a file, not to its link target's containing directory.
        if reader == 'solo_seed2_links.wire:resolved_target' or key in expanded:
            continue
        expanded.add(key)

        # Match the EXACT generation layout in replay_bank._receipts: one
        # generation directory, not */*/branches. No snapshot runner copies.
        if path.parent.name == 'labels' and path.suffix == '.json':
            label = _json(path)
            search = _identifier(label['search_id'], 'search_id')
            if not path.stem.startswith(search):
                raise ValueError('REFUSED: LS label filename differs from search identity')
            suffix = path.stem[len(search):]
            if suffix and not re.fullmatch(r'_[A-Z0-9]+', suffix):
                raise ValueError('REFUSED: invalid LS label generation')
            src = path.parent.parent
            receipt = label['verification']
            base = src/'label_attempts'/suffix[1:] if suffix else src
            branches(base/'branches', receipt)
            if receipt.get('status') in ('accepted', 'rejected'):
                add(src/'pending'/(search+'.json.gz'), 'replay_bank._ls_snapshot')
        elif path.parent.name == 'pending' and path.name.endswith('.json.gz'):
            envelope = _json(path)
            snapshot = _identifier(envelope['snapshot'], 'snapshot')
            add(path.parent.parent/'snapshots'/(snapshot+'.pt.gz'), 'replay_bank._ls_snapshot')
            add(episode_path(envelope['episode_path'])/'RESULT.json', 'replay_bank._ls_snapshot')
        elif path.parent.name == 'decisions' and path.suffix == '.json':
            link = _json(path)
            state = _identifier(link['state_id'], 'state_id')
            add(path.parent.parent/'snapshots'/(state+'.pt'), 'replay_bank._decision_snapshot')
        elif path.parent.name == 'verifications' and path.suffix == '.json':
            branches(path.parent.parent/'branches', _json(path))
        elif path.name == 'RESULT.json':
            value = _json(path)
            if isinstance(value, dict) and value.get('encoding') == 'coopmwm-lossless-gzip-v1':
                if value.get('file') != 'RESULT.json.gz':
                    raise ValueError('REFUSED: invalid result payload path')
                payload = _absolute(path.parent/value['file'])
                h, length = hashlib.sha256(), 0
                with gzip.open(payload, 'rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        h.update(block)
                        length += len(block)
                if h.hexdigest() != value['sha256']:
                    raise ValueError('REFUSED: compressed result digest mismatch: '+str(path))
                if 'uncompressed_bytes' in value and length != value['uncompressed_bytes']:
                    raise ValueError('REFUSED: compressed result size mismatch: '+str(path))
                add(payload, 'accel_storage.read')

    files = [Path(key) for key in sorted(entries)]
    manifest = dict(schema=SCHEMA, entries=dict(sorted(entries.items())),
                    aliases=dict(sorted(aliases.items())), readers=READERS,
                    allowed_roots=sorted({p.parent.as_posix() for p in files}),
                    file_count=len(files), physical_file_count=len(hashes),
                    physical_bytes=sum(size for _, size in hashes.values()))
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    manifest['manifest_sha256'] = hashlib.sha256(encoded).hexdigest()
    return files, manifest
