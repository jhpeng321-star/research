"""Read only named checkpoints/configuration; optionally export permitted TRAIN starts."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import torch


def allowed(meta):
    text = str(meta)
    if re.search(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth", text, re.I):
        return False
    seeds = [int(x) for x in re.findall(r"\d{6,}", text)]
    return not any(964000 <= s <= 964199 or 977000 <= s <= 977999 for s in seeds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--config", default="configs/c4_policy14_candA_longgrad_skip.toml")
    p.add_argument("--export")
    a = p.parse_args()
    assert allowed(a.candidate) and allowed(a.config)
    payload = torch.load(a.candidate, map_location="cpu", weights_only=False)
    print("CHECKPOINT", hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest())
    print("KEYS", list(payload))
    print("CONFIG", json.dumps(payload.get("config", {}), default=str))
    print("UPDATE", payload.get("update_count"))
    print("GATE", json.dumps(payload.get("gated300", {}), default=str))
    print("COOPW", json.dumps(payload.get("coopw", {}), default=str))
    from wms.control.train_cli import _read_config
    raw = _read_config(a.config)
    print("RESOLVED", json.dumps(raw, default=str))
    if not a.export:
        return
    from wms.control.policy14_cli import load_members
    from wms.control.train_cli import _load_starts
    models, _ = load_members(
        ["/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt"],
        ["/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml"], "cpu")
    train, _ = _load_starts(raw["data"], models, 965140, entries=True)
    assert all(allowed(m) for m in train.metadata.values()), "unsafe TRAIN metadata"
    train.configure_focus_cells(raw["data"]["focus_cell_fraction"])
    # Reproduce the next training draw from the candidate's saved sampler RNG.
    saved = payload.get("start_sampler", payload.get("sampler"))
    print("SAMPLER_SAVED", saved is not None)
    if saved is not None:
        train.load_state_dict(saved)
    batches = []
    for _ in range(2):
        b = train.sample(1, "cpu")
        assert all(allowed(m) for m in b.metadata)
        print("START", b.metadata, "CAV", int(b.state["is_cav"].sum()))
        batches.append(b)
    out = Path(a.export)
    assert not out.exists() and out.parent.is_dir()
    torch.save(batches, out)


if __name__ == "__main__":
    os.environ.setdefault("C4_ALLOWED_ROOTS", "/root/autodl-tmp/c4")
    torch.set_num_threads(2)
    main()
