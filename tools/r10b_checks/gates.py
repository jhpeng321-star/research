"""gate_g_by_k of each arm (saturating gate), read from the checkpoint tensors.  Usage: gates.py REPO OUT.json arm=ckpt ..."""
import json
import sys

repo, out = sys.argv[1:3]
sys.path[:0] = [repo, repo + "/src"]
import torch  # noqa: E402
from tools.r10_attribution import load_model  # noqa: E402
from wms.train.unroll import GATE_LOG_K  # noqa: E402

res = {}
for spec in sys.argv[3:]:
    arm, ck = spec.split("=", 1)
    _, _, model, _ = load_model("c4_wm_v10round10b.toml", ck)
    g = model.gate(torch.tensor(GATE_LOG_K)).detach()
    res[arm] = {str(k): round(float(v), 5) for k, v in zip(GATE_LOG_K, g)}
json.dump(res, open(out, "w"))
print(res)
