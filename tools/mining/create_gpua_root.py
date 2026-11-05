import json
import os
from pathlib import Path
r=Path('/root/mining20261106')
assert r.parent.is_dir() and not r.exists()
assert not Path('/home/ubuntu').exists() and not Path('/home/ubuntu').is_symlink()
r.mkdir()
(r/'EXPERIMENT_ID').write_text('S-MINING-20261106\n')
(r/'CHARGES.jsonl').touch(exist_ok=False)
(r/'cpuv_home').mkdir()
Path('/home/ubuntu').symlink_to(r/'cpuv_home',target_is_directory=True)
(r/'AUTHORITY.md').write_text('Owner REGISTRATION.md 2026-11-06 00:27 and MAIN01:09 resume.\n'
 'Own mining phase<=500000 inside total6500000; protect existing allocations.\n'
 'Environment/source/input staging only; no simulation until qualified/precharged.\n'
 'cpuV source paths preserved under private empty gpuA /home/ubuntu alias.\n')
print(json.dumps(dict(root=str(r),home_alias=str(Path('/home/ubuntu').resolve()),sim_s=0)))
