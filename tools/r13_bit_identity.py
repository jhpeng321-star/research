"""R13F/R13T cross-version bit-identity probe (CPU, synthetic states, no data, no SUMO).

Run once per source tree (old = 079e39ff, new = this branch) in SEPARATE processes, then compare:

    python tools/r13_bit_identity.py dump --src <old>/src --config configs/c4_wm_v10round12.toml --out old.pt
    python tools/r13_bit_identity.py dump --src src --config configs/c4_wm_v10round12.toml --out new.pt
    python tools/r13_bit_identity.py compare old.pt new.pt

``dump`` builds the model of ``--config`` (+ ``--set`` overrides) with ``build_model``, fixed seeds, a
non-zero courtesy head and a non-zero temporal read-out (so both paths are exercised), then records
(a) every output tensor of ``model(state)`` at 3 steps of a synthetic crowded merge sequence and (b) a
6-step ``UnrollRunner`` rollout with backward: its loss value, the rollout tensors and every parameter
gradient.  ``--diag-zero-temporal`` sets the BRAKE_DIAG inference override ``model.diag_zero_temporal``
(the zerotemp counterfactual) before running.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dump(args) -> None:
    sys.path[:0] = [str(Path(args.src).resolve()), str(ROOT / "tests"), str(ROOT)]
    import torch

    torch.set_num_threads(1)
    from test_r11_courtesy import crowded  # synthetic batch only

    from wms.cli.assemble import build_model, build_physics
    from wms.config.load import load_train_config
    from wms.train.inputs import state_from_kin
    from wms.train.losses.compose import build_loss
    from wms.train.settings import TrainSettings
    from wms.train.unroll import UnrollRunner

    cfg = load_train_config(ROOT / args.config, ["runtime.device=cpu", *args.set])
    physics = build_physics(cfg)
    torch.manual_seed(1234)
    model = build_model(cfg, physics).eval()
    torch.manual_seed(5)
    with torch.no_grad():
        for p in model.follower_courtesy.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
        model.temporal[-1].weight.copy_(torch.randn_like(model.temporal[-1].weight) * 0.3)
        model.temporal[-1].bias.copy_(torch.randn_like(model.temporal[-1].bias) * 0.3)
    if args.diag_zero_temporal:
        model.diag_zero_temporal = True
    from wms.model.history import LONG_HISTORY_F

    batch = crowded(n=9)
    k_slots = batch["states"].shape[2]
    history = torch.zeros(1, k_slots, model.driver_history_len, LONG_HISTORY_F)
    history[..., 0] = 10 / 30
    history[..., 2:5] = batch["states"][:, 0, :, None, 1:4]
    history_mask = torch.zeros(history.shape[:-1], dtype=torch.bool)
    history_mask[..., -2:] = True                  # causal observations before t0
    batch["driver_hist"], batch["driver_hist_mask"] = history, history_mask
    out = {}
    with torch.no_grad():
        for t in range(3):
            state = {**state_from_kin(batch["states"][:, t], batch["present"][:, t]),
                     "is_cav": batch["is_cav"][:, t]}
            res = model(state, driver_history=history, driver_history_mask=history_mask)
            for key, value in res.items():
                if torch.is_tensor(value):
                    out[f"forward/{t}/{key}"] = value.detach().clone()
    model.train()
    settings = TrainSettings(truncation_steps=6)
    runner = UnrollRunner(model, settings, build_loss(settings), projection=None, device=torch.device("cpu"))
    res = runner.run(batch, unroll_steps=6, backward=True)
    out["rollout/loss"] = torch.tensor(float(res["loss"]))
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[f"grad/{name}"] = p.grad.detach().clone()
    for name, value in model.state_dict().items():
        out[f"param/{name}"] = value.detach().clone()
    torch.save(out, args.out)
    print(f"{args.out}: {len(out)} tensors, loss {float(res['loss']):.6f}")


def compare(args) -> int:
    import torch

    a, b = torch.load(args.a, weights_only=True), torch.load(args.b, weights_only=True)
    keys = sorted(set(a) | set(b))
    missing = [k for k in keys if k not in a or k not in b]
    differ = [k for k in keys if k in a and k in b
              and (a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k]))]
    same = len(keys) - len(missing) - len(differ)
    print(f"{same}/{len(keys)} tensors bit-identical; missing {missing[:8]}; differ {differ[:8]}")
    return 0 if not missing and not differ else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--src", required=True)
    d.add_argument("--config", default="configs/c4_wm_v10round12.toml")
    d.add_argument("--set", action="append", default=[])
    d.add_argument("--diag-zero-temporal", action="store_true")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args(argv)
    if args.cmd == "dump":
        dump(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
