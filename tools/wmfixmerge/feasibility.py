"""Small CPU-only feasibility probe. Synthetic evidence never authorizes training."""
from __future__ import annotations

import argparse
import json
import math

from tools.wmfixmerge.data import digest, implementation_hashes, install_guard, safe


def synthetic():
    import torch
    from wms.model.merge_response import freeze_except_response
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.train.merge_response import paired_objective
    torch.manual_seed(140014)
    model = WmSModel(ModelSpec(merge_response=True)).eval()
    freeze_except_response(model)
    state = dict(s_m=torch.tensor([[-100., -60.], [-100., -60.]]),
        v_mps=torch.tensor([[10., 12.], [8., 8.]]), a_mps2=torch.zeros(2, 2),
        lane=torch.ones(2, 2, dtype=torch.long), length_m=torch.ones(2, 2) * 5,
        present=torch.ones(2, 2, dtype=torch.bool), is_cav=torch.zeros(2, 2, dtype=torch.bool))
    def rollout():
        r = model.rollout(state, n_steps=4)
        return (.5 * (1 - r["speeds"] / 31.29).clamp_min(0)).sum((1, 2))[:, None]
    with torch.no_grad():
        # Measured fixture-speed centering gives an actual conditional teacher response;
        # the earlier uniform-weight teacher mostly tested a common-mode offset and failed J.
        first, last = model.merge_response.net[0], model.merge_response.net[-1]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 1] = 1.
        first.bias[0] = -state["v_mps"].mean() / 31.29
        last.weight.zero_()
        last.weight[0, 0] = 1.
        target = rollout()
        last.weight.zero_()
    # The synthetic stack has one differentiable column: gradient on its paired
    # contrast only, no absolute anchors (anchors need TRAIN absolute scales).
    return probe(model, rollout, target, [0, 0], torch.ones(1),
                 lambda p, r, refs, s: paired_objective(p, r, refs, s,
                                                        gradient_columns=(0,), anchor_columns=()))


def probe(model, rollout, target, refs, scales, objective):
    import torch
    from wms.model.merge_response import freeze_except_response
    freeze_except_response(model)
    frozen = {n: v.detach().clone() for n, v in model.state_dict().items()
              if not n.startswith("merge_response.")}
    before = rollout()
    loss = objective(before, target, refs, scales)
    refs_tensor = torch.as_tensor(refs, device=before.device)
    active = refs_tensor != torch.arange(len(refs), device=before.device)
    paired_j = lambda x: float(abs((x - x[refs_tensor])[:, 0] -
                                   (target - target[refs_tensor])[:, 0])[active].mean().detach())
    before_j = paired_j(before)
    parameters = [p for p in model.parameters() if p.requires_grad]
    component_norms = {}
    for name, column in (("J", 0), ("source", 2), ("crossing", 3)):
        if column >= before.shape[1]:
            continue
        component = (before - before[refs_tensor])[:, column][active].sum()
        gradient = torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)
        squares = [g.detach().square().sum() for g in gradient if g is not None]
        component_norms[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.
    loss.backward()
    candidates = [(float(p.grad.abs().max()), n, p) for n, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None]
    if not candidates or not all(torch.isfinite(p.grad).all() for _, _, p in candidates):
        raise ValueError("no finite new-head gradient")
    magnitude, name, parameter = max(candidates, key=lambda t: t[0])
    if magnitude == 0:
        raise ValueError("new term cannot move paired objective")
    index = int(parameter.grad.abs().argmax())
    analytic = float(parameter.grad.flatten()[index])
    original = float(parameter.detach().flatten()[index])
    eps = .001  # numerical central-difference step, not a behavioural threshold
    values = []
    with torch.no_grad():
        for sign in (1, -1):
            parameter.flatten()[index] = original + sign * eps
            values.append(float(objective(rollout(), target, refs, scales)))
        parameter.flatten()[index] = original
    fd = (values[0] - values[1]) / (2 * eps)
    # A failed FD check cannot be rescued by a synthetic teacher ceiling.
    fd_ok = abs(fd - analytic) <= .05 * max(abs(fd), abs(analytic)) + 1e-5
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    optimizer.step()
    with torch.no_grad():
        after = rollout()
        final = float(objective(after, target, refs, scales))
    unchanged = all(torch.equal(v, model.state_dict()[n]) for n, v in frozen.items())
    return dict(initial_loss=float(loss.detach()), after_step_loss=final,
        initial_J_contrast_MAE=before_j, after_J_contrast_MAE=paired_j(after),
        gradient_parameter=name, gradient=analytic, finite_difference=fd, finite_difference_ok=fd_ok,
        frozen_bitwise=unchanged, optimizer_reduces=final < float(loss.detach()),
        predicted=before.detach().tolist(), recorded=target.detach().tolist(),
        component_gradient_norms=component_norms,
        passed=fd_ok and unchanged and final < float(loss.detach()) and paired_j(after) < before_j and
            all(math.isfinite(component_norms[n]) and component_norms[n] > 0
                for n in ("J", "crossing") if n in component_norms))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--prepared")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--controllers")
    parser.add_argument("--corrected-report")
    parser.add_argument("--harness-contract", help="independent corrected-scorer retarget/clock receipt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    install_guard()
    import torch
    torch.set_num_threads(1)
    result = dict(schema="wmfix_feasibility/1", mode="synthetic" if args.synthetic else "real_train",
                  device="cpu", launch_ready=False, implementation_hashes=implementation_hashes())
    try:
        if args.synthetic:
            result.update(synthetic())
            result["status"] = "SYNTHETIC_ONLY" if result["passed"] else "FAIL"
        else:
            from tools.wmfixmerge.runtime import Runtime
            from wms.train.merge_response import paired_objective
            for key in ("prepared", "config", "checkpoint", "controllers", "corrected_report"):
                if not getattr(args, key):
                    raise ValueError("missing --" + key.replace("_", "-"))
                result[key + "_sha256"] = digest(getattr(args, key))
            rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers)
            initial, target, refs, receipt = rt.rollout(0, 60)
            with torch.no_grad():
                rt.learned_admission = False
                ledger_only = rt.rollout(0, 60)[0]
                rt.learned_admission = True
                result["admission_ablation"] = dict(
                    ledger_only=ledger_only.tolist(), inferred_admission_zero_head=initial.detach().tolist(),
                    components=rt.doc["components"])
            # Actual TRAIN teacher ceiling retains WM downstream and source errors.
            with torch.no_grad():
                corrected = initial[:, 0] + target[:, 1] - initial[:, 1]
                result["teacher_upstream_J_paired_MAE"] = float(abs(
                    (corrected - corrected[0])[1:] - (target[:, 0] - target[0, 0])[1:]).mean())
                result["initial_J_paired_MAE"] = float(abs(
                    (initial[:, 0] - initial[0, 0])[1:] - (target[:, 0] - target[0, 0])[1:]).mean())
                result["source_contrast_error"] = ((initial[:, 2] - initial[0, 2]) -
                                                       (target[:, 2] - target[0, 2])).tolist()
            del initial
            result.update(probe(rt.model, lambda: rt.rollout(0, 60)[0], target.detach(), refs,
                                rt.scales, lambda p, r, refs, s:
                                    paired_objective(p, r, refs, s, anchor_scales=rt.anchor_scales)))
            result["receipt"] = receipt
            contract = {} if not args.harness_contract else json.loads(safe(args.harness_contract).read_text())
            matched = (contract.get("schema") == "wmfix_harness_contract/1" and
                contract.get("corrected_report_sha256") == result["corrected_report_sha256"] and
                contract.get("prepared_sha256") == result["prepared_sha256"] and
                contract.get("source_clock_match") is True and contract.get("residual_remains") is True and
                contract.get("native_lanes_sha256") == digest("tools/wmrequal/native_lanes.py") and
                contract.get("scorer_sha256") == digest("tools/wmrequal/score.py") and
                contract.get("runtime_sha256") == digest("tools/wmfixmerge/runtime.py"))
            result["source_clock_match"] = matched
            result["launch_ready"] = bool(result["passed"] and matched and
                result["teacher_upstream_J_paired_MAE"] < result["initial_J_paired_MAE"])
            result["status"] = "PASS" if result["launch_ready"] else "BLOCKED"
            if args.harness_contract:
                result["harness_contract_sha256"] = digest(args.harness_contract)
            if not result["launch_ready"]:
                result["blocker"] = "Real paired improvement, teacher ceiling and independent corrected source-clock/retarget gates must all pass"
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as exc:
        result.update(status="BLOCKED", blocker=f"{type(exc).__name__}: {exc}")
    out = safe(args.out)
    if not out.parent.is_dir():
        raise ValueError("output parent must exist")
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("launch_ready") or (args.synthetic and result.get("passed")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
