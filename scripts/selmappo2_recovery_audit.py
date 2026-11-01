"""One-shot, read-only audit of the existing owned trainer and output sizes."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(REPO))
from tools.selmappo2 import ops  # noqa: E402


REMOTE = r'''
import json, os, stat, time
from pathlib import Path
root = Path('/home/ubuntu/selmappo2_20261101/build_c141d5715f26af4e9b50d0866cdf971ef92d55dd1223c4312c4f25bdec5a8b29')
def checked(path):
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('symlink refused')
    return path
dispatch = json.loads(checked(root / 'DISPATCH.json').read_bytes())
assert dispatch['pid'] == 606968 and dispatch['uid'] == os.getuid() == 1000
assert dispatch['root'] == str(root)
proc = Path('/proc/606968')
assert proc.stat().st_uid == dispatch['uid']
fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
assert int(fields[19]) == dispatch['proc_start_ticks'] == 48374316
command = [arg.decode() for arg in (proc / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
assert command == dispatch['command']
assert int(fields[16]) == 19
blocked = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
           'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
groups, skipped = {}, 0
stack = [root]
while stack:
    directory = stack.pop()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_symlink() or any(word in entry.name.lower() for word in blocked):
                skipped += 1
                continue
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                stack.append(Path(entry.path))
            elif stat.S_ISREG(info.st_mode):
                group = Path(entry.path).relative_to(root).parts[0]
                row = groups.setdefault(group, dict(files=0, logical_bytes=0, allocated_bytes=0))
                row['files'] += 1
                row['logical_bytes'] += info.st_size
                row['allocated_bytes'] += info.st_blocks * 512
disk = os.statvfs(root)
print(json.dumps(dict(schema='selmappo2.recovery_audit/1', timestamp_ns=time.time_ns(),
    root=str(root), pid=606968, uid=os.getuid(), proc_start_ticks=int(fields[19]),
    command=command, process_state=fields[0], nice=int(fields[16]),
    process_elapsed_s=time.clock_gettime(time.CLOCK_BOOTTIME)-int(fields[19])/os.sysconf('SC_CLK_TCK'),
    host_uptime_s=time.clock_gettime(time.CLOCK_BOOTTIME),
    disk_available_bytes=disk.f_bavail*disk.f_frsize, disk_total_bytes=disk.f_blocks*disk.f_frsize,
    output_groups=groups, skipped_entries=skipped, file_contents_read=['DISPATCH.json'],
    metadata_only_output_inventory=True, new_sumo_seconds=0)))
'''


def main():
    ops.checked(ops.EVIDENCE, directory=True)
    packed = base64.b64encode(REMOTE.encode()).decode()
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'cpuQ',
               '/home/ubuntu/miniconda3/bin/python -B -c '
               f"'import base64;exec(base64.b64decode(\"{packed}\"))'"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        path = ops.EVIDENCE / f'RECOVERY_AUDIT_ERROR_{time.time_ns()}.json'
        ops.exclusive_json(path, dict(returncode=result.returncode, stdout=result.stdout,
                                     stderr=result.stderr, timestamp_ns=time.time_ns(),
                                     new_sumo_seconds=0))
        raise RuntimeError(f'{path.name}: {result.stderr}')
    data = json.loads(result.stdout)
    data['local_timestamp_ns'] = time.time_ns()
    data['audit_source_sha256'] = ops.sha(Path(__file__).read_bytes())
    data['local_monitors'] = json.loads(subprocess.run([
        'powershell', '-NoProfile', '-Command',
        'Get-CimInstance Win32_Process -Filter "ProcessId=12432 OR ProcessId=46520" | '
        'Select-Object ProcessId,ExecutablePath,CommandLine,CreationDate | ConvertTo-Json -Compress'
    ], check=True, capture_output=True, text=True, timeout=30).stdout)
    path = ops.EVIDENCE / f'RECOVERY_AUDIT_{time.time_ns()}.json'
    ops.exclusive_json(path, data)
    print(json.dumps(dict(receipt=str(path), **data)))


if __name__ == '__main__':
    main()
