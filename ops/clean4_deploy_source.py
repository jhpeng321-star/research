"""Assemble an immutable clean4 source tree on cpuV from a git commit (zero physics).

Usage (laptop): python ops/clean4_deploy_source.py <worktree> <commit> [--name NAME]

Layout = the qualified clean3 tree (/home/ubuntu/coopmwm20261106_clean3_src_3e5bc9f9: runtime_src,
WM_PACKAGE, qualification_baselines, acceptance, ops) with pytools/tools + pytools/tests + pytools/assets
replaced by the commit's files; packages the clean3 tree took from launchcoop_src7 (coopbehav, coopplan,
leverprobe, wmteach, tools/__init__.py) stay the same symlinks. Writes SOURCE_MANIFEST.json
(commit + sha256 of every regular file). Refuses an existing target.
"""
import argparse
import gzip
import hashlib
import io
import subprocess
import sys
from pathlib import Path

BASE = '/home/ubuntu/coopmwm20261106_clean3_src_3e5bc9f9'
LINKED = ('coopbehav', 'coopplan', 'leverprobe', 'wmteach', '__init__.py')
REMOTE = r'''
set -euo pipefail
S={target}; B={base}; L=/home/ubuntu/coopmwm20261104_launchcoop_src7/pytools/tools
test ! -e "$S"
cp -a "$B" "$S"
rm -rf "$S/pytools/tools" "$S/pytools/tests" "$S/pytools/assets" "$S/ops"
tar -xzf {tarball} -C "$S/pytools"
mv "$S/pytools/ops" "$S/ops"
ln -s ../ops "$S/pytools/ops"   # tests import ops/ modules relative to the repo root (= pytools)
for p in {linked}; do test -e "$S/pytools/tools/$p" || ln -s "$L/$p" "$S/pytools/tools/$p"; done
/home/ubuntu/miniconda3/bin/python - "$S" {commit} <<'EOF'
import hashlib, json, sys
from pathlib import Path
src, commit = Path(sys.argv[1]), sys.argv[2]
files = {{}}
for p in sorted(src.rglob('*')):
    rel = p.relative_to(src)
    if any((src/q).is_symlink() for q in list(rel.parents)[:-1]) or p.is_symlink() or not p.is_file():
        continue
    if rel.parts[0] == 'WM_PACKAGE' or '__pycache__' in rel.parts or rel.name == 'SOURCE_MANIFEST.json':
        continue
    files[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
(src/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(commit=commit, experiment='clean4', files=files), indent=1, sort_keys=True))
print(len(files), hashlib.sha256((src/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
EOF
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('worktree')
    ap.add_argument('commit')
    ap.add_argument('--name')
    ap.add_argument('--skip-upload', action='store_true')
    ap.add_argument('--host', default='cpuV')
    a = ap.parse_args()
    commit = subprocess.check_output(['git', '-C', a.worktree, 'rev-parse', a.commit], text=True).strip()
    name = a.name or f'coopmwm20261106_clean4_src_{commit[:8]}'
    target = f'/home/ubuntu/{name}'
    tar = subprocess.check_output(['git', '-C', a.worktree, '-c', 'core.autocrlf=false', 'archive', '--format=tar',
                                   commit, 'tools', 'tests', 'ops', 'assets/research/newmap_candidates'])
    blob = gzip.compress(tar, mtime=0)
    local = Path(a.worktree)/'_out'/f'{name}.tar.gz'
    local.parent.mkdir(exist_ok=True)
    local.write_bytes(blob)
    remote_tar = f'/home/ubuntu/{name}.tar.gz'
    if not a.skip_upload:
        subprocess.check_call(['scp', '-q', '-C', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', str(local),
                               f'{a.host}:{remote_tar}'])
    script = REMOTE.format(target=target, base=BASE, tarball=remote_tar, linked=' '.join(LINKED), commit=commit)
    # bytes stdin: Windows text mode would turn LF into CRLF and break the remote bash script
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', a.host, 'bash -s'],
                         input=script.replace(chr(13), '').encode(), capture_output=True)
    sys.stdout.write(out.stdout.decode(errors='replace')); sys.stderr.write(out.stderr.decode(errors='replace'))
    print(dict(host=a.host, target=target, commit=commit, tar_sha256=hashlib.sha256(blob).hexdigest()))
    raise SystemExit(out.returncode)


if __name__ == '__main__':
    main()
