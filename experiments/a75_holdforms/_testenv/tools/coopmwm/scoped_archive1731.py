"""Opaque-file SHA gated MOVE under MAIN17:31; e3final is live and excluded."""
import json
import os
from pathlib import Path
import time
import archive_move as base
import rolling_archive as rolling

CONTROL=Path('/home/ubuntu/coopmwm20261103_archive1731')
NAMES=('coopfree20261101_owner1511.tar.gz','c1switch20261031.tar.gz')


def safe(p):
    p=Path(p)
    return (p.parent==base.HOME and p.name in NAMES and not p.is_symlink()
        and p.resolve()==p and p.is_file())


def referenced(p,text):
    return str(p) in text


def candidates():
    # Root-only scan: permission failures are fatal, never treated as no reference.
    paths=[base.HOME/n for n in NAMES if safe(base.HOME/n)]
    reasons={str(p):[] for p in paths}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            values=[]
            for field in ('cwd','exe'):
                try:values.append(os.readlink(proc/field))
                except FileNotFoundError:pass
            for field in ('cmdline','environ'):values.append((proc/field).read_bytes().decode(errors='replace'))
            for fd in (proc/'fd').iterdir():
                try:values.append(os.readlink(fd))
                except FileNotFoundError:pass
            for p in paths:
                if any(referenced(p,v) for v in values):reasons[str(p)].append('process:'+proc.name)
        except (FileNotFoundError,ProcessLookupError):pass
    for root in base.HOME.glob('coopmwm*'):
        if not root.is_dir() or root.is_symlink() or base.BAD.search(str(root)):continue
        if 'archive' in root.name or 'rolling' in root.name:continue
        for p in root.glob('*.json'):
            if base.BAD.search(str(p)) or p.is_symlink():continue
            if not any(k in p.name.upper() for k in ('BINDING','CONFIG','MANIFEST','RUNTIME')):continue
            text=p.read_text()
            for q in paths:
                if referenced(q,text):reasons[str(q)].append('metadata:'+str(p))
    return {p:'MAIN1731_opaque_archive' for p,why in reasons.items() if not why},reasons


def run():
    import fcntl,signal,shutil
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    authority=json.loads((CONTROL/'AUTHORITY.json').read_text())
    for name,h in authority['source_hashes'].items():
        assert base.digest_file(CONTROL/name)==h,'source identity changed'
    assert json.loads((CONTROL/'CPU_RESULT.json').read_text())['pass_']
    lock=(CONTROL/'RUN.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (CONTROL/'OWNER.json').exists():raise ValueError('adopt existing scoped archive owner')
    base.write(CONTROL/'OWNER.json',base.owner())
    rolling.CONTROL=CONTROL;rolling.DEST='/root/cpuq_archive_20261103/scoped1731'
    rolling.safe=safe;rolling.candidates=candidates
    try:
        found,reasons=candidates()
        base.write(CONTROL/'AUDIT.json',dict(unix=time.time(),references=reasons,
            e3final='SKIP: prior privileged audit proves live runtime/binding/process references'))
        moved=rolling.batch(sorted(found.items()),0) if found else 0
        base.write(CONTROL/'COMPLETE.json',dict(unix=time.time(),moved_bytes=moved,
            cpu_free=shutil.disk_usage(base.HOME).free,new_sim_s=0))
    except BaseException as e:
        base.write(CONTROL/'FAILED.json',dict(unix=time.time(),error=repr(e),sources_retained_unless_certified=True))
        raise


if __name__=='__main__':run()
