"""Build the r10b_headzero checkpoint: r10b with follower_courtesy.net.4.weight/bias zeroed (exact ablation).
Writes a receipt json (every other tensor must be bit-equal).  Usage: make_headzero.py IN.pt OUT.pt RECEIPT.json"""
import json
import os
import shutil
import sys

import torch

src, dst, receipt = sys.argv[1:4]
payload = torch.load(src, map_location="cpu", weights_only=False)
model = payload["model"]
keys = [k for k in model if k.startswith("follower_courtesy.net.4.")]
assert sorted(keys) == ["follower_courtesy.net.4.bias", "follower_courtesy.net.4.weight"], keys
before = {k: float(model[k].abs().sum()) for k in keys}
out = dict(payload)
out["model"] = {k: (torch.zeros_like(v) if k in keys else v.clone()) for k, v in model.items()}
torch.save(out, dst)
for extra in ("physics.json", "TRAIN_RECEIPT.json"):   # the loader needs the physics next to the checkpoint
    if os.path.exists(os.path.join(os.path.dirname(src), extra)):
        shutil.copy(os.path.join(os.path.dirname(src), extra), os.path.join(os.path.dirname(dst), extra))
chk = torch.load(dst, map_location="cpu", weights_only=False)["model"]
same = all(torch.equal(chk[k], model[k]) for k in model if k not in keys)
zero = all(float(chk[k].abs().sum()) == 0.0 for k in keys)
json.dump({"src": src, "dst": dst, "zeroed": keys, "abs_sum_before": before, "all_other_tensors_equal": same,
           "zeroed_are_zero": zero, "n_tensors": len(model)}, open(receipt, "w"), indent=1)
assert same and zero
print("headzero ok", before)
