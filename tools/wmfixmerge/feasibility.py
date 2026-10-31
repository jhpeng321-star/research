"""Small CPU-only feasibility probe. Synthetic evidence never authorizes training."""
from __future__ import annotations

import argparse
import json
import math
import time

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


def probe(model, rollout, target, refs, scales, objective, fd_eps=None,
          fd_sweep=(1e-3, 1e-4, 1e-5)):
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
    sweeps = {}
    for eps in fd_sweep if fd_eps is None else (fd_eps,):
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                parameter.flatten()[index] = original + sign * eps
                values.append(float(objective(rollout(), target, refs, scales)))
            parameter.flatten()[index] = original
        sweeps[f"{eps:g}"] = (values[0] - values[1]) / (2 * eps)
    eps = .001 if fd_eps is None else float(fd_eps)
    fd = sweeps[f"{eps:g}"]
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
        finite_difference_sweep=sweeps, frozen_bitwise=unchanged, optimizer_reduces=final < float(loss.detach()),
        predicted=before.detach().tolist(), recorded=target.detach().tolist(),
        component_gradient_norms=component_norms,
        passed=fd_ok and unchanged and final < float(loss.detach()) and paired_j(after) < before_j and
            all(math.isfinite(component_norms[n]) and component_norms[n] > 0
                for n in ("J", "crossing") if n in component_norms))


def panel_gate(rt, horizons=(120, 300)):
    """WMFIX_REVIEW blocker 5: all-group/all-arm paired-J panel at verdict horizons.

    Reports the zero-head inferred-admission paired J MAE and the upstream
    teacher ceiling (recorded upstream substituted) per horizon, plus the
    frozen-rerun noise of the shared-CRN pipeline.  launch additionally
    requires the post-step panel to beat the pre-step panel beyond that noise.
    """
    import numpy as np
    import torch
    began = time.monotonic()
    rows = []
    for h in horizons:
        for group in range(len(rt.source.groups)):
            for arm in (0, 1):
                with torch.no_grad():
                    predicted, recorded, _, _ = rt.rollout(group, h, arm_index=arm)
                initial = ((predicted[1, 0] - predicted[0, 0]) - (recorded[1, 0] - recorded[0, 0]))
                corrected = predicted[:, 0] + recorded[:, 1] - predicted[:, 1]
                teacher = (corrected[1] - corrected[0]) - (recorded[1, 0] - recorded[0, 0])
                rows.append(dict(horizon=h, group=group, arm=arm,
                                 initial=float(initial), teacher=float(teacher)))
    with torch.no_grad():
        first = rt.rollout(0, horizons[0], arm_index=0)[0]
        second = rt.rollout(0, horizons[0], arm_index=0)[0]
    noise = float((first - second).abs().max())
    panel = {}
    for h in horizons:
        part = [r for r in rows if r["horizon"] == h]
        if len(part) != 2 * len(rt.source.groups):
            raise ValueError("incomplete gate panel")
        panel[str(h)] = dict(
            initial_paired_J_MAE=float(np.mean([abs(r["initial"]) for r in part])),
            teacher_paired_J_MAE=float(np.mean([abs(r["teacher"]) for r in part])),
            per_group={str(g): dict(
                initial=float(np.mean([abs(r["initial"]) for r in part if r["group"] == g])),
                teacher=float(np.mean([abs(r["teacher"]) for r in part if r["group"] == g])))
                for g in range(len(rt.source.groups))},
            rows=len(part))
    return panel, rows, dict(frozen_rerun_noise=noise, seconds=round(time.monotonic() - began, 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--prepared")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--controllers")
    parser.add_argument("--corrected-report")
    parser.add_argument("--harness-contract", help="independent corrected-scorer retarget/clock receipt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="48-row gate panel is slow on CPU; cuda allowed for MAIN")
    # S-WMFIXMERGE PILOT v2 (MAIN 13:50 TRACK 1): the smoothed-admission training
    # objective must re-produce a PASSING FD check before any v2 launch.
    parser.add_argument("--smooth-events", action="store_true",
                        help="training-only smoothed admission surrogate (S-WMFIX-C fallback 1)")
    parser.add_argument("--admission-width", type=float, default=None)
    parser.add_argument("--skip-panel-gate", action="store_true",
                        help="record the one-step panel sub-gate as superseded by the "
                             "registered k-step pilot protocol (MAIN 10:40 amendment)")
    parser.add_argument("--fd-eps", type=float, default=None,
                        help="override the FD step; an eps sweep is recorded either way")
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
            if args.smooth_events:
                from wms.train.merge_response import SMOOTH_GRADIENT_COLUMNS
            for key in ("prepared", "config", "checkpoint", "controllers", "corrected_report"):
                if not getattr(args, key):
                    raise ValueError("missing --" + key.replace("_", "-"))
                result[key + "_sha256"] = digest(getattr(args, key))
            rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, args.device,
                         smooth_events=args.smooth_events, admission_width_m=args.admission_width)
            result["device"] = args.device
            result["smooth_events"] = bool(args.smooth_events)
            if args.smooth_events:
                from wms.train.merge_response import ADMISSION_WIDTH_M as DEFAULT_WIDTH
                result["admission_width_m"] = (DEFAULT_WIDTH if args.admission_width is None
                                               else float(args.admission_width))
                result["gradient_columns"] = list(SMOOTH_GRADIENT_COLUMNS)
            before_panel = None
            if args.skip_panel_gate:
                if not args.smooth_events:
                    raise ValueError("--skip-panel-gate is only defined for the registered "
                                     "pilot-superseded smooth-events feasibility")
                result["gate_panel_ok"] = None
                result["gate_panel_superseded_by"] = (
                    "S-WMFIX-PILOT panel-descent protocol (MAIN REVIEW DECISION 10:40; "
                    "PILOT v2 MAIN 13:50 TRACK 1)")
            else:
                before_panel, _, panel_receipt = panel_gate(rt)
            initial, target, refs, receipt = rt.rollout(0, 60)
            with torch.no_grad():
                rt.learned_admission = False
                ledger_only = rt.rollout(0, 60)[0]
                rt.learned_admission = True
                result["admission_ablation"] = dict(
                    ledger_only=ledger_only.tolist(), inferred_admission_zero_head=initial.detach().tolist(),
                    components=rt.doc["components"])
                result["source_contrast_error"] = ((initial[:, 2] - initial[0, 2]) -
                                                       (target[:, 2] - target[0, 2])).tolist()
            del initial
            result.update(probe(rt.model, lambda: rt.rollout(0, 60)[0], target.detach(), refs,
                                rt.scales, lambda p, r, refs, s:
                                    paired_objective(p, r, refs, s, anchor_scales=rt.anchor_scales,
                                                     **({} if not args.smooth_events else
                                                        dict(gradient_columns=SMOOTH_GRADIENT_COLUMNS))),
                                fd_eps=args.fd_eps))
            if args.skip_panel_gate:
                after_panel = None
                noise = 0.0
            else:
                after_panel, _, _ = panel_gate(rt)
                noise = panel_receipt["frozen_rerun_noise"]
            if before_panel is not None:
                result["gate_panel"] = dict(before=before_panel, after=after_panel,
                                            **panel_receipt)
                # Strengthened gate: teacher ceiling below initial at BOTH horizons and a
                # post-step paired-J improvement beyond frozen-rerun noise at BOTH horizons.
                result["gate_panel_ok"] = all(
                    before_panel[str(h)]["teacher_paired_J_MAE"] < before_panel[str(h)]["initial_paired_J_MAE"]
                    and after_panel[str(h)]["initial_paired_J_MAE"] <
                    before_panel[str(h)]["initial_paired_J_MAE"] - noise for h in (120, 300))
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
            # In --skip-panel-gate mode the one-step panel sub-gate is superseded
            # by the registered k-step pilot; every other gate still applies.
            panel_gate_satisfied = True if args.skip_panel_gate else bool(result["gate_panel_ok"])
            result["launch_ready"] = bool(result["passed"] and matched and panel_gate_satisfied)
            result["status"] = "PASS" if result["launch_ready"] else "BLOCKED"
            if args.harness_contract:
                result["harness_contract_sha256"] = digest(args.harness_contract)
            if not result["launch_ready"]:
                result["blocker"] = ("FD agreement, optimizer step, frozen parity and the "
                                     "corrected source-clock/retarget gates must all pass"
                                     if args.skip_panel_gate else
                                     "Real paired improvement, teacher ceiling and independent "
                                     "corrected source-clock/retarget gates must all pass")
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
