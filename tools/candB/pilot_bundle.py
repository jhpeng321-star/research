"""Committed-blob staging while S-RANKCHECK has explicitly excluded working edits."""
import json
from unittest.mock import patch

from tools.candB import bundle


def check_shared_status(status):
    paths = [line[3:] for line in status.decode().splitlines()]
    allowed = ('tools/candB/evidence/rankcheck_',)
    names = {'tools/candB/rankcheck.py', 'tests/unit/test_candb_rankcheck.py',
             'docs/paper_results/RANKCHECK.md'}
    if any(path not in names and not path.startswith(allowed) for path in paths):
        raise ValueError('commit own pilot files before staging; unexpected dirty paths: '+str(paths))
    return paths


def main():
    original = bundle.subprocess.check_output
    def committed_only(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if command == ['git', 'status', '--porcelain']:
            excluded = check_shared_status(result)
            print(json.dumps(dict(label='EXPLORATORY', committed_blobs_only=True,
                                  excluded_shared_working_edits=excluded)))
            return b''
        return result
    # Test fixture only: the existing allowlist omitted its CPU dependency.
    bundle.ROOTS.append('tools/gated300/cct1.py')
    with patch.object(bundle.subprocess, 'check_output', committed_only):
        bundle.main()


if __name__ == '__main__':
    main()
