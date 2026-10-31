"""Standard-library-only local resource/owned-child incarnation receipts."""
import ctypes
import os
from pathlib import Path
import time


class FileTime(ctypes.Structure):
    _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_uint32)]

    def ticks(self):
        return (self.high << 32) | self.low


def child_incarnation(child):
    if os.name == 'nt':
        created, exited, kernel, user = (FileTime() for _ in range(4))
        ok = ctypes.windll.kernel32.GetProcessTimes(ctypes.c_void_p(int(child._handle)),
            ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            raise ctypes.WinError()
        return dict(create_filetime=created.ticks())
    return dict(start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def local_resources(interval=.25):
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                        *[(n, ctypes.c_uint64) for n in ('total', 'available', 'page_total',
                            'page_available', 'virtual_total', 'virtual_available', 'extended')]]
        def usage():
            idle, kernel, user = FileTime(), FileTime(), FileTime()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                raise ctypes.WinError()
            return idle.ticks(), kernel.ticks()+user.ticks()
        before = usage()
        time.sleep(interval)
        after = usage()
        total, idle = after[1]-before[1], after[0]-before[0]
        memory = Memory()
        memory.length = ctypes.sizeof(Memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
        return dict(cpu_percent=100*(1-idle/total) if total else 0.,
                    available_bytes=memory.available, logical_cpus=os.cpu_count(), gpu_used=False)
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return dict(cpu_percent=100*os.getloadavg()[0]/os.cpu_count(), available_bytes=available,
                logical_cpus=os.cpu_count(), gpu_used=False)
