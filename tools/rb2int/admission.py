"""Read-only Linux memory headroom, counting only reclaimable inactive file cache."""
from pathlib import Path


def available_memory(*, limit, current, inactive_file, dirty=0, writeback=0, host_available):
    if limit is None:
        return max(0, int(host_available))
    reclaimable = min(max(0, int(current)), max(0, int(inactive_file)-int(dirty)-int(writeback)))
    return max(0, min(int(limit), int(host_available), int(limit)-int(current)+reclaimable))


def memory_snapshot(cgroup='/sys/fs/cgroup', meminfo='/proc/meminfo'):
    cg = Path(cgroup)
    host = next(int(line.split()[1])*1024 for line in Path(meminfo).read_text().splitlines()
                if line.startswith('MemAvailable:'))
    maximum, current = cg/'memory.max', cg/'memory.current'
    limit = None if not maximum.is_file() or maximum.read_text().strip() == 'max' else int(maximum.read_text())
    used = int(current.read_text()) if current.is_file() else 0
    stats = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines()) if (cg/'memory.stat').is_file() else {}
    inactive = int(stats.get('inactive_file', 0))
    dirty, writeback = int(stats.get('file_dirty', 0)), int(stats.get('file_writeback', 0))
    return dict(limit_bytes=limit, current_bytes=used, inactive_file_bytes=inactive,
        dirty_file_bytes=dirty, writeback_bytes=writeback, host_available_bytes=host,
        effective_available_bytes=available_memory(limit=limit, current=used,
            inactive_file=inactive, dirty=dirty, writeback=writeback, host_available=host),
        method='min(host MemAvailable, cgroup limit-current+clean inactive_file); no active/anonymous cache allowance',
        process_or_cache_mutation=False)
