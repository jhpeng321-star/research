p = 'E:/rd-wt/ws-r3-evalstates3/output/wd01/wm_n1/lpwo/real_t0.py'
s = open(p).read()
a = '''    root = checked_path(str(raw).replace("\\\\", "/"), guard)
'''
assert a in s
s = s.replace(a, a + '''    if not (root / "slot_index.npy").is_file():
        raise MaterializeError("no per-tick identity shard (slot_index.npy absent)")
''')
open(p, 'w').write(s)
