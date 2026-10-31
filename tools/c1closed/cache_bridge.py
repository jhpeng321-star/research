"""Reuse checksum-verified local publication copies in the frozen sync cache."""
from pathlib import Path
import hashlib
import json
import time

from operate import HERE, REMOTE
from probe import ssh
from watch import creation


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    # No competing bulk cache writer; the original collector is still gated on
    # remote COMPLETE and cannot sync yet. This is transport only, never scoring.
    for receipt in (HERE/'evidence').glob('PREFETCH_*_LAUNCH.json'):
        r = json.loads(receipt.read_text())
        assert creation(r['pid']) != r['creation_time'], 'wait for existing cache copy'
    manifest = json.loads(ssh('gpuA', f'''from pathlib import Path
import json,hashlib
p=Path({REMOTE!r});assert not (p/'COMPLETE.json').exists()
print(json.dumps({{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in (p/'panel').glob('*_h300.json')}}))
'''))
    destination = HERE/'out/panel'
    assert destination.is_dir()
    reused, existing, unavailable = [], [], []
    for name, expected in manifest.items():
        target = destination/name
        if target.exists():
            assert digest(target.read_bytes()) == expected
            existing.append(name)
            continue
        source = HERE/'out'/('expand_'+name)
        if not source.exists():
            unavailable.append(name)
            continue
        data = source.read_bytes()
        assert digest(data) == expected, 'preserve and inspect changed transport file'
        with target.open('xb') as f:
            f.write(data)
        reused.append(name)
    receipt = dict(at=time.time(),remote_files=len(manifest),existing=len(existing),
                   reused_local_transport=reused,still_uncached=unavailable,
                   hashes={n:manifest[n] for n in reused},
                   source_sha256=digest(Path(__file__).read_bytes()),
                   role='immutable own publication-copy reuse only; no inference or scoring')
    path = HERE/'evidence'/('BRIDGE_CACHE_'+str(time.time_ns())+'.json')
    with path.open('x') as f:
        json.dump(receipt,f,indent=1)
    print(json.dumps(dict(existing=len(existing),reused=len(reused),still_uncached=len(unavailable),receipt=str(path))))


if __name__ == '__main__':
    main()
