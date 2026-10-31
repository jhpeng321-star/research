"""S-PILOTDIAG CPU repro: dense 60/90 s pathwise-gradient instrumentation.

Read-only diagnosis of the saved exploratory pilot checkpoints (abs192/abs193,
zero phase start) on the frozen panel starts. No training, no GPU, no SUMO.
The trainer is constructed by the unchanged policy14 CLI with stubbed startup
scoring; this driver intercepts the constructed trainer and runs instrumented
eager rollouts (the LONGMEM memory machinery is bitwise-equivalent to the
plain eager path, which the existing CPU suites verify).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parents[2]
INPUTS = Path(os.environ.get("PILOTDIAG_INPUTS", "E:/Temp/opencode/pilotdiag/inputs"))
CKPTS = Path(os.environ.get("PILOTDIAG_CKPTS", "E:/Temp/opencode/pilotdiag"))

PANEL_STARTS = [  # frozen lean panel prefix (authority lean_recipe.panel_starts)
    ("w963552m1900p05t300__generic_random", 600),
    ("w963625m1900p10t300__generic_random", 2240),
    ("w963495m2400p20t300__generic_random", 900),
    ("w963904m2400p40t300__generic_random", 1700),
]


def environment():
    os.environ.setdefault("C4_ALLOWED_ROOTS",
                          f"{INPUTS}{os.pathsep}{CKPTS}{os.pathsep}{HERE}")
    os.environ["WMS_DATA_GENRAND_PLAIN_Q"] = str(INPUTS / "genrand_plain_q")
    os.environ["WMS_DATA_GENRAND_PLAIN"] = str(INPUTS / "genrand_plain_q")
    os.environ["WMS_DATA_V5_NATIVE_Q"] = str(INPUTS / "v5_native_q")
    os.environ["C4_EXECUTION_PARITY"] = "2"
    os.environ["C4_SUMO_LEADER_RULE"] = "1"
    os.environ["C4_EXECUTION_DRIVER"] = str(INPUTS / "execparity_cav_idm_v1.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "4")


class Ready(Exception):
    def __init__(self, trainer, val):
        super().__init__("pilotdiag trainer captured")
        self.trainer, self.val = trainer, val


def exploratory_long_credit(trainer, cfg, wm_sha256s, receipt):
    """The pilot's EXPLORATORY waiver (tools.candB.pilot_cli.install_exploratory):
    force H_long=300 without re-qualifying CCT-2 evidence; never a PASS claim."""
    from tools.distill.runtime import sha256
    if (list(wm_sha256s) != [WM_SHA256] or not cfg.ctrl0_reference or cfg.k16
            or cfg.estimator != 'antithetic' or cfg.es_directions != 2
            or trainer.config.future_samples != 8
            or not trainer.config.pessimistic_credit or not trainer.config.deterministic_crn
            or not trainer.config.entrant_control or not trainer.config.entrant_learned_mix
            or trainer.policy.residual_reference != 'ctrl0'):
        raise ValueError('pilotdiag long-credit identity mismatch')
    receipt.update(H_long=300, H_G=300, label='EXPLORATORY', policy_authorized=False,
        qualification_passed=False, reason='pilotdiag diagnosis reuses the owner exploratory waiver',
        artifact_sha256=sha256(cfg.cct2_file.format(wm_sha256=WM_SHA256)))


WM_SHA256 = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"


def restore_schedule(trainer, ckpt: Path):
    """Restore the looplong schedule pieces the plain-resume CLI path skips.

    Exactly what policy14_recovery.restore_looplong restores from the same
    payload fields, minus the remote boundary artifacts (never read here).
    """
    import torch
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    saved = payload.get("looplong") or {}
    horizon = dict((saved.get("horizon") or {}))
    allowed = horizon.pop("allowed_stage", None)
    if horizon.get("gate_file"):  # local copy of the saved measured authority
        horizon["gate_file"] = str(INPUTS / "WM2_MEASURED_AUTHORITY.json")
    from wms.control.coopw import HorizonConfig as HC, ExplorationConfig as EC
    if horizon:
        trainer.horizon_curriculum = HC(**horizon)
        trainer.horizon_curriculum.bind([WM_SHA256])
        if allowed is not None and trainer.horizon_curriculum.allowed_stage != allowed:
            raise ValueError("restored trusted horizon gate mismatch")
    trainer.exploration = EC(**saved["exploration"]) if saved.get("exploration") else None
    trainer.long_credit_requested = True
    trainer.long_credit_trusted_s = 60
    return {"horizon": trainer.horizon_curriculum.as_dict() if trainer.horizon_curriculum else None,
            "exploration": trainer.exploration.as_dict() if trainer.exploration else None,
            "update_count": payload["update_count"]}


def build_trainer(ckpt: Path, out: Path):
    """Run the unchanged CLI until its first update; capture the trainer."""
    from wms.control import policy14_cli as cli, gated300
    from wms.control import eslong
    from wms.control.dreamer import DreamerTrainer
    from wms.control.start_states import checked_path

    def stub_score(trainer, batch, *a, **kw):
        if not hasattr(trainer, "_gatemeas_score_position"):
            trainer._gatemeas_score_position = trainer._curriculum_position(advance=False)
        return dict(J=0., road_delay=0., source_wait=0., native_relative_gain=None,
                    hdv_hard_brake_seconds=0., critic_tail_included=False, J_dense=0.,
                    J_raw_dense=0., CTRL0_J_dense=0., raw_gain_vs_CTRL0_dense=0.,
                    raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0., score_horizon_s=90.,
                    composite_gated_score=0.)

    def stub_panel(trainer, val, panel, device):
        return dict(update=trainer.update_count, delta_per_start=[0.] * len(panel["starts"]),
                    passed=True, horizon_s=10., scope="pilotdiag stub (production receipts retained)")

    def stub_gate_panel(path, val):
        text = checked_path(path).read_text(encoding="utf-8")
        return json.loads(text)

    def update_ready(trainer, *a, **kw):
        raise Ready(trainer, None)

    def es_gradient(trainer, *a, **kw):
        return None, {"pilotdiag_es_skipped": True}

    cli_out = out / "cli"
    if cli_out.exists():
        import shutil
        shutil.rmtree(cli_out)

    argv = ["--config", "configs/c4_policy14_candB_pilotdiag.toml",
            "--wm-ckpt", str(INPUTS / "wm_s.pt"),
            "--wm-config", str(INPUTS / "c4_wm_loop14_r16_ft.toml"),
            "--real-returns", str(INPUTS / "returns_c03.a1.npz"),
            "--longcritic-native-returns", str(INPUTS / "native_returns_c03.a1.npz"),
            "--paircrit-receipt", str(INPUTS / "pairing_c03.a1.json"),
            "--paircrit-policy-sha256",
            "0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e",
            "--gate-panel", str(INPUTS / "gate_panel.json"),
            "--horizon-gate", str(INPUTS / "HORIZON_GATE_WM2_REG.json"),
            "--critic-warmup-updates", "0", "--native-init", "--phase-relative-signal-gate",
            "--resume", str(ckpt), "--device", "cpu", "--seed", "965140",
            "--max-updates", "10", "--output-dir", str(out / "cli"), "--compact"]
    with patch.object(cli, "independent_score", stub_score), patch.object(
            cli, "score_native_panel", stub_panel), patch.object(
            cli, "score_gate_panel", stub_panel), patch.object(
            cli, "load_gate_panel", stub_gate_panel), patch.object(
            DreamerTrainer, "update", update_ready), patch.object(
            eslong, "gradient", es_gradient), patch.object(
            gated300, "_install_gain_only", exploratory_long_credit):
        try:
            cli.main(argv)
        except Ready as ready:
            trainer = ready.trainer
            restore_schedule(trainer, ckpt)
            return trainer
    summary = json.loads((cli_out / "RUN_SUMMARY.json").read_text())
    raise RuntimeError(f"trainer was not captured: {summary.get('error')}\n"
                       f"{summary.get('traceback', '')[-3000:]}")


def panel_batch(trainer, which: int):
    """Fixed panel start (world, tick) -> StartBatch via the real corpus loader."""
    dataset = trainer.start_states.dataset
    want_world, want_tick = PANEL_STARTS[which]
    bare = want_world.split("__")[0]
    hits = []
    for index in range(len(dataset)):
        meta = dataset.start_meta(index) if hasattr(dataset, "start_meta") else {}
        found_world = str(meta.get("world", meta.get("world_id", ""))) == bare
        tick = meta.get("tick", meta.get("t0_tick", meta.get("start_tick")))
        if found_world:
            hits.append((index, tick))
    if not hits:
        raise ValueError(f"panel start {want_world}@{want_tick} not in mini corpus")
    exact = [index for index, tick in hits if tick is None or int(tick) == want_tick]
    index = exact[0] if exact else hits[0][0]
    return trainer.start_states._batch([trainer.start_states._load(index)], trainer.device)


def instrumented_rollout(trainer, batch, *, horizon_s, deterministic, trace, seed=710003,
                         futures=8):
    """One training-style imagination at a chosen dense horizon with full autograd.

    ``rollout_config`` is neutralised so the rollout runs at ``horizon_s`` (the
    dense cotangent beyond the 91 s replay boundary is exactly zero in the
    production dense-prefix VJP, so the prefix gradient is faithful). ``futures``
    may be below the production K8: per-future credits/graduates are identical
    (future tapes are CRN per future index), so a prefix of futures reproduces
    the exact partial parameter-gradient sum for those futures.
    """
    from dataclasses import replace as dc_replace
    from wms.control import gated300
    from wms.control.counterfactual import build_variants

    steps = int(round(horizon_s))  # STEP_S = 1 s per control step
    old_cfg, old_curriculum = trainer.config, trainer.horizon_curriculum
    old_scoring = getattr(trainer, "_gated300_scoring", False)
    cfg = dc_replace(old_cfg, horizon=steps, future_samples=futures, future_driver_scale=1.,
                     uncertainty_weight=0., imagination_microbatch=0,
                     imagination_checkpoint_segment=0, imagination_offload_futures=False,
                     imagination_offload_saved_tensors=False, low_penetration_horizon_cap=False)
    variants = build_variants(batch.state, trainer.counterfactual,
                              torch.Generator().manual_seed(seed))
    trainer._longgrad_state_trace = trace
    from wms.control import eslong
    try:
        trainer.config, trainer.horizon_curriculum = cfg, None
        trainer._training_imagination = not deterministic
        with patch.object(gated300, "rollout_config", lambda t, c: c), patch.object(
                eslong, "execution_config", lambda t, c, b: c):
            imagined = trainer.imagine(batch, variants=variants,
                                       deterministic=deterministic, future_seed=seed)
    finally:
        trainer.config, trainer.horizon_curriculum = old_cfg, old_curriculum
        trainer._gated300_scoring = old_scoring
        trainer._training_imagination = False
        trainer._longgrad_state_trace = None
    return imagined


def dense_difference(trainer, imagined):
    """Replicate actor_objective's dense path exactly; return (difference, coefficients)."""
    from wms.control.antiexploit import actor_terms
    from wms.control.counterfactual import baseline_index, row_coefficients
    returns = imagined["returns"]
    n_base, n_rows = imagined["n_base"], returns.shape[-1]
    variants = imagined.get("variants")
    weights = trainer.counterfactual.weights(trainer._curriculum_position(advance=False))
    coefficients = row_coefficients(variants, weights, n_rows, n_base,
                                    device=returns.device, dtype=returns.dtype)
    base = baseline_index(variants, n_rows, device=returns.device)
    has_base = base != torch.arange(n_rows, device=returns.device)
    difference = returns - torch.where(has_base, returns[..., base].detach(),
                                       torch.zeros_like(returns))
    paired = difference
    if getattr(trainer, "long_credit", None) is not None:
        from wms.control.long_credit import pessimistic_difference
        difference = pessimistic_difference(trainer, imagined, difference, base, has_base,
                                            coefficients)
    if trainer.config.pessimistic_credit or trainer.config.native_anchor:
        difference, anchor, _ = actor_terms(trainer, imagined, difference, coefficients,
                                            trainer._return_scale(returns[..., :n_base]),
                                            raw_difference=paired)
    return difference, coefficients


def executed_stats(callback_delta):
    """Aggregate controller-level executed-residual statistics."""
    return callback_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--start", type=int, default=3, help="panel start index 0..3")
    parser.add_argument("--horizons", type=int, nargs="+", default=[61, 91])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--rows", action="store_true", help="per-row decomposition")
    parser.add_argument("--score", action="store_true",
                        help="deterministic no-grad dJ vs CTRL0 at the listed horizons")
    parser.add_argument("--futures", type=int, default=8)
    parser.add_argument("--normalize-steps", action="store_true",
                        help="unit-normalise per-step state cotangents (option d) during backward")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    environment()
    out = Path(args.out or (HERE / "_out" / "pilotdiag"))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(965140)

    from time import perf_counter
    t_build = perf_counter()
    trainer = build_trainer(Path(args.ckpt), out)
    print(json.dumps({"build_s": round(perf_counter() - t_build, 1)}), flush=True)
    batch = panel_batch(trainer, args.start)
    from wms.control.gated300 import reference_index
    print(json.dumps({"ckpt": str(args.ckpt), "update_count": trainer.update_count,
                      "start": PANEL_STARTS[args.start],
                      "residual_reference": trainer.policy.residual_reference}))

    if args.score:
        report = {"ckpt": str(args.ckpt), "start": PANEL_STARTS[args.start], "runs": []}
        for horizon in args.horizons:
            from dataclasses import replace as dc_replace
            cfg = dc_replace(trainer.config, horizon=horizon,
                             future_samples=args.futures, future_driver_scale=1.,
                             uncertainty_weight=0., imagination_microbatch=0,
                             imagination_checkpoint_segment=0,
                             imagination_offload_futures=False,
                             imagination_offload_saved_tensors=False,
                             low_penetration_horizon_cap=False)
            old_cfg, old_curriculum = trainer.config, trainer.horizon_curriculum
            from wms.control import gated300, eslong
            from wms.control.counterfactual import build_variants
            variants = build_variants(batch.state, trainer.counterfactual,
                                      torch.Generator().manual_seed(710003))
            with torch.no_grad(), patch.object(gated300, "rollout_config", lambda t, c: c), \
                    patch.object(eslong, "execution_config", lambda t, c, b: c):
                trainer.config, trainer.horizon_curriculum = cfg, None
                trainer._training_imagination = False
                trainer._gated300_scoring = True
                try:
                    imagined = trainer.imagine(batch, deterministic=True, variants=variants,
                                               future_seed=710003)
                finally:
                    trainer.config, trainer.horizon_curriculum = old_cfg, old_curriculum
                    trainer._gated300_scoring = False
            components = imagined["components"]
            n = imagined["n_base"]
            refs = reference_index(imagined["variants"])
            j = (components["delay"] + components["source_wait"]).sum(1).mean(0)
            entry = {"horizon_s": horizon,
                     "J_rows": [float(v) for v in j],
                     "raw_dJ_vs_CTRL0": float((j[:n] - j[refs[:n]]).mean()),
                     "per_row_dJ": [float(v) for v in (j - j[refs])]}
            report["runs"].append(entry)
            print(json.dumps(entry)[:1200])
        (out / "repro_score.json").write_text(json.dumps(report, indent=1))
        return 0

    report = {"ckpt": str(args.ckpt), "start": PANEL_STARTS[args.start],
              "deterministic": bool(args.deterministic), "futures": args.futures,
              "normalize_steps": bool(args.normalize_steps),
              "runs": []}
    from time import perf_counter
    from wms.control.longgrad import StateTrace, norm as gnorm
    for horizon in args.horizons:
        trace = StateTrace(normalize=args.normalize_steps)
        t0 = perf_counter()
        imagined = instrumented_rollout(trainer, batch, horizon_s=horizon,
                                        deterministic=args.deterministic, trace=trace,
                                        futures=args.futures)
        t1 = perf_counter()
        difference, coefficients = dense_difference(trainer, imagined)
        row_terms = -(difference.mean((0, 1)) * coefficients)
        params = [p for p in trainer.actor_parameters if p.requires_grad]
        scale = trainer._return_scale(imagined["returns"][..., :imagined["n_base"]])
        dense_loss = row_terms.sum() / scale
        grads = torch.autograd.grad(dense_loss, params, retain_graph=True, allow_unused=True)
        t2 = perf_counter()
        entry = {"horizon_s": horizon, "rollout_s": round(t1 - t0, 1),
                 "backward_s": round(t2 - t1, 1),
                 "dense_credit_veh_s": float(row_terms.sum().detach()),
                 "dense_loss": float(dense_loss.detach()),
                 "grad_norm_total": gnorm(grads),
                 "row_credit_veh_s": [float(v) for v in row_terms.detach()],
                 "row_coefficients": [float(v) for v in coefficients.detach()],
                 "row_kind": imagined["variants"].row_kind.detach().cpu().tolist(),
                 "row_start": imagined["variants"].row_start.detach().cpu().tolist()}
        if args.rows:
            per_row = []
            for r in range(row_terms.shape[0]):
                if float(coefficients[r]) == 0.:
                    per_row.append(None)
                    continue
                g = torch.autograd.grad(row_terms[r] / scale, params, retain_graph=True,
                                        allow_unused=True)
                per_row.append(gnorm(g))
                print(json.dumps({"row_grad": r, "norm": per_row[-1]}), flush=True)
            entry["row_grad_norms"] = per_row
        entry["state_cotangent"] = trace.summary()
        # executed-residual / emergency-floor proximity from the audit record
        audit = imagined.get("action_audit")
        if audit is not None:
            acting = audit["audit_acting"][..., :imagined["n_base"], :]
            action = audit["audit_action"][..., :imagined["n_base"], :][acting]
            entry["residual_stats"] = {"count": int(action.numel()),
                "mean": float(action.mean()) if action.numel() else 0.,
                "std": float(action.std()) if action.numel() else 0.,
                "min": float(action.min()) if action.numel() else 0.,
                "frac_negative": float((action < 0).float().mean()) if action.numel() else 0.}
        report["runs"].append(entry)
        (out / "repro_dense.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(entry)[:1500], flush=True)
    del trainer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
