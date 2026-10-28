"""CPU-only lossless storage dedup, scoped to immutable owned collector copies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

WORK = Path('/home/ubuntu/c4/main14/restart20261026/work')
RECEIPT = WORK.parent / 'autonomy/C3_DISK_DEDUP.json'
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dedup(apply=False):
    assert WORK.is_dir() and RECEIPT.parent.is_dir()
    if apply:
        assert not RECEIPT.exists()
    files = []
    for world in sorted(WORK.glob('w968*')):
        if not world.name[1:].isdigit() or not 968100 <= int(world.name[1:]) <= 968227:
            continue
        for run in sorted((world / 'C4-LOOP14-COOPON-20261026').glob('c0*')):
            files.extend(p for p in (run / 'inputs/policy.pt', run / 'inputs/wm/wm_s.pt') if p.is_file())
            files.extend((run / 'artifact/pairs/episodes').glob('**/*.npy'))
    before = shutil.disk_usage(WORK).free
    groups, changes, potential = {}, [], 0
    for path in files:
        assert not any(t in str(path).lower() for t in BAD)
        assert not path.is_symlink() and path.resolve().is_relative_to(WORK.resolve())
        stat = path.stat()
        checksum = digest(path)
        key = stat.st_size, checksum
        if key not in groups:
            groups[key] = path
            continue
        anchor = groups[key]
        if (stat.st_dev, stat.st_ino) == (anchor.stat().st_dev, anchor.stat().st_ino):
            continue
        row = {'path': str(path), 'anchor': str(anchor), 'bytes': stat.st_size,
               'sha256': checksum, 'old_inode': stat.st_ino, 'old_mtime_ns': stat.st_mtime_ns}
        potential += stat.st_size
        if apply:
            assert digest(anchor) == checksum
            temp = path.with_name(path.name + '.c3_hardlink_tmp')
            assert not temp.exists() and anchor.stat().st_dev == stat.st_dev
            os.link(anchor, temp)
            assert digest(temp) == checksum
            os.replace(temp, path)
            assert digest(path) == checksum and path.stat().st_ino == anchor.stat().st_ino
            row['new_inode'] = path.stat().st_ino
        changes.append(row)
    result = {'apply': apply, 'charged_sim_s': 0, 'files_examined': len(files),
              'duplicates': len(changes), 'deduplicated_logical_bytes': potential,
              'free_before_bytes': before, 'free_after_bytes': shutil.disk_usage(WORK).free,
              'preserved_content_and_paths': True, 'changes': changes}
    if apply:
        RECEIPT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'changes'}))


if __name__ == '__main__':
    dedup(len(sys.argv) > 1 and sys.argv[1] == 'apply')
