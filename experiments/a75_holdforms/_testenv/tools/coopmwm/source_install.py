"""Atomic owned-path source installation, preserving shared targets and old bytes."""


def install_source(target, backup, payload, expected_prior=None):
    import hashlib
    import os
    import time
    from pathlib import Path
    target, backup = Path(target), Path(backup)
    assert target.parent.is_dir()
    incoming = hashlib.sha256(payload).hexdigest()
    record = dict(path=str(target), installed_sha256=incoming)
    if target.exists() or target.is_symlink():
        original = target.read_bytes()
        observed = hashlib.sha256(original).hexdigest()
        if expected_prior is not None:
            assert observed in (expected_prior, incoming), str(target)
        record.update(original_sha256=observed,
            original_link=os.readlink(target) if target.is_symlink() else None)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            with backup.open('xb') as f:
                f.write(original);f.flush();os.fsync(f.fileno())
            assert hashlib.sha256(backup.read_bytes()).hexdigest() == observed
    temporary = target.with_name('.'+target.name+'.niche.'+str(os.getpid())+'.'+str(time.time_ns()))
    with temporary.open('xb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    # Replacing the directory entry detaches symlinks/hardlinks. Never write
    # through an alias into another slot's already-qualified serving package.
    os.replace(temporary, target)
    assert not target.is_symlink() and hashlib.sha256(target.read_bytes()).hexdigest() == incoming
    return record
