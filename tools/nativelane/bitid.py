"""Compare both flag-OFF imagination paths against the requested base by tensor bytes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

base, changed, output = map(Path, sys.argv[1:4])
assert output.parent.is_dir()
output.mkdir(exist_ok=True)
receipt = {}
for label, flags in (("legacy", []), ("loop14c", ["0,wm_law"])):
    values = []
    for side, repo in (("base", base), ("changed", changed)):
        target = output / f"{label}_{side}.pt"
        subprocess.run([sys.executable, str(changed / "tools/loop14_checks/loop14c_bitid.py"),
                        str(repo), str(target), *flags], env={**os.environ, "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, check=True)
        values.append(torch.load(target, map_location="cpu", weights_only=True))
    a, b = values
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        assert torch.equal(a[key].reshape(-1).view(torch.uint8), b[key].reshape(-1).view(torch.uint8)), key
    receipt[label] = {"tensors": len(a), "bit_identical": True}
(output / "BIT_IDENTITY.json").write_text(json.dumps(receipt, indent=1))
print(json.dumps(receipt))
