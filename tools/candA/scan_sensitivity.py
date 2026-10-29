"""S-ZEROSTART diagnostic (read-only): does the gate-panel scorer respond to the actor?

Runs the real ``policy14_cli`` start-up (same argv as the production launch), then at the first
``Policy14Trainer.update`` call it does NOT train: it scores the first K frozen gate-panel starts
(critic-free ``independent_score``, deterministic, the exact path of rescores.jsonl and
gate_panel_result.json) for several actor variants and exits.

Variants: ``loaded`` (whatever the CLI resumed, e.g. C1), ``ckpt:<path>`` (policy state from a
checkpoint), ``const:<m/s^2>`` (all-zero head weights with a bias giving that constant
residual on every CAV), ``zero`` (residual head output layer exactly 0).  While scoring, the
shared ``sumo_idm.executed_accel`` is wrapped to tally, over entries whose requested residual is
non-zero: share r>0, share where ``issued >= cap`` (residual has NO effect and zero gradient),
share where the cap is below the unconstrained base, mean residual.  No training, no SUMO.

usage: python scan_sensitivity.py OUT.json K VARIANT[,VARIANT...] -- <policy14_cli argv>
"""
import json
import math
import sys

import torch


def main():
    out_path, k, variants = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
    assert sys.argv[4] == "--"
    argv = sys.argv[5:]
    import wms.control.policy14_cli as cli
    from wms.control import sumo_idm

    tally = {}
    current = {"name": None}
    orig_exec = sumo_idm.executed_accel

    def spy(speed, leader_speed, bumper_gap, allowed_speed, residual, lo, hi, *, has_leader, **kw):
        out = orig_exec(speed, leader_speed, bumper_gap, allowed_speed, residual, lo, hi,
                        has_leader=has_leader, **kw)
        with torch.no_grad():
            has_leader_t = torch.as_tensor(has_leader)
            base = sumo_idm.base_accel(speed, leader_speed, bumper_gap, allowed_speed,
                                       has_leader=has_leader_t)
            cap = sumo_idm.safe_speed_cap(speed, leader_speed, bumper_gap, allowed_speed,
                                          has_leader=has_leader_t, lane=kw.get("lane"),
                                          s_m=kw.get("s_m"))
            r = torch.as_tensor(residual).to(base)
            upper = torch.maximum(base, torch.full_like(base, min(float(hi), 1.2)))
            lower = torch.minimum(base, torch.full_like(base, float(lo)))
            issued = torch.minimum(torch.maximum(base + r, lower), upper)
            act = (r != 0) & torch.isfinite(issued) & torch.isfinite(cap)
            t = tally.setdefault(current["name"], dict(n=0, pos=0, neg=0, pos_dead=0, neg_dead=0,
                                                       cap_below_base=0, sum_r=0.0, exec_eq_base=0,
                                                       n_all=0, n_all_cap_lt_base=0))
            t["n_all"] += int(torch.isfinite(base).sum())
            t["n_all_cap_lt_base"] += int(((cap < base - 1e-9) & torch.isfinite(base)).sum())
            n = int(act.sum())
            if n:
                dead = issued >= cap
                t["n"] += n
                t["pos"] += int((act & (r > 0)).sum())
                t["neg"] += int((act & (r < 0)).sum())
                t["pos_dead"] += int((act & (r > 0) & dead).sum())
                t["neg_dead"] += int((act & (r < 0) & dead).sum())
                t["cap_below_base"] += int((act & (cap < base - 1e-9)).sum())
                t["exec_eq_base"] += int((act & ((out - base).abs() < 1e-9)).sum())
                t["sum_r"] += float(r[act].sum())
        return out

    sumo_idm.executed_accel = spy
    box = {}

    def panel_spy(trainer, val, panel, device):
        box.update(trainer=trainer, val=val, panel=panel, device=device)
        return [0.0] * len(panel["starts"])  # skip the 16-start baseline: never used (we abort)

    def native_stub(trainer, val, panel, device):
        box.update(trainer=trainer, val=val, panel=panel, device=device)
        return {"delta_per_start": [0.0] * len(panel["starts"]), "passed": False,
                "update": trainer.update_count, "schema": "scan_stub"}

    cli.score_gate_panel = panel_spy
    cli.score_native_panel = native_stub

    def scan(self, *args, **kwargs):
        trainer, val, panel, device = (box[x] for x in ("trainer", "val", "panel", "device"))
        pol = trainer.policy
        base_state = {n: p.detach().clone() for n, p in pol.state_dict().items()}
        starts = panel["starts"][:k]
        results = {}
        for name in variants:
            pol.load_state_dict(base_state)
            if name == "loaded":
                pass
            elif name.startswith("ckpt:"):
                payload = torch.load(name[5:], map_location="cpu", weights_only=True)
                pol.load_state_dict(payload["policy"])
            elif name == "zero":
                with torch.no_grad():
                    pol.residual_head.weight.zero_()
                    pol.residual_head.bias.zero_()
            elif name.startswith("const:"):
                c = float(name[6:])
                radius = min(-pol.accel_min, pol.accel_max)
                with torch.no_grad():
                    pol.residual_head.weight.zero_()
                    pol.residual_head.bias.fill_(math.atanh(c / radius))
            else:
                raise ValueError(name)
            current["name"] = name
            js = [cli.independent_score(trainer, val._batch([val._load(int(s["index"]))], device))["J"]
                  for s in starts]
            results[name] = {"J": js, "mean_J": sum(js) / len(js), "exec_tally": tally.get(name)}
            print(json.dumps({name: results[name]}), flush=True)
        with open(out_path, "w") as f:
            json.dump({"k": k, "starts": [s["index"] for s in starts], "results": results}, f, indent=1)
        raise SystemExit("scan done")

    cli.Policy14Trainer.update = scan
    sys.exit(cli.main(argv))


if __name__ == "__main__":
    main()
