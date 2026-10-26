"""Read-only copy of the pinned collected-policy tensor state for offline selection."""
import json

from remote import ROOT, REMOTE, ssh

source = json.loads((ROOT / 'tools/longcritic/CURRENT_REAL_SOURCE.json').read_text(encoding='utf-8'))
policy = source['policy']
r = ssh('gpuR', "import os\nos.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')\n"
        "import pathlib,hashlib,torch,io,base64,json\ntorch.set_num_threads(4)\n"
        f"p=pathlib.Path({policy['checkpoint']!r}); assert p.is_relative_to(pathlib.Path('/root/main14_loop/policy'))\n"
        f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={policy['sha256']!r}\n"
        "d=torch.load(p,map_location='cpu',weights_only=True); out={k:d[k] for k in ['critic','critic_optimizer','policy','group_encoder']}\n"
        "buf=io.BytesIO(); torch.save(out,buf); print(base64.b64encode(buf.getvalue()).decode())\n")
if r.returncode:
    raise RuntimeError(r.stderr)
r = ssh('gpuA', "import pathlib,base64\n"
        f"p=pathlib.Path('{REMOTE}/COLLECTED_POLICY_STATE.pt'); raw=base64.b64decode({r.stdout.strip()!r})\n"
        "assert not p.exists() or p.read_bytes()==raw\np.write_bytes(raw)\nprint('collected policy tensor snapshot staged')\n")
print(r.stdout, r.stderr)
if r.returncode:
    raise RuntimeError('policy snapshot failed')
