import hashlib,json
from pathlib import Path
for name in ('/root/planfix_f631_gpu_src_361a974fd6b9274a.z','/root/planfix_f631_gpu_src_361a974fd6b9274a.z.partial'):
 p=Path(name)
 print('PACKAGE',json.dumps(dict(path=name,exists=p.exists(),bytes=p.stat().st_size if p.exists() else None,
   sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)),flush=True)
