"""Pre-run checks (replaces the shell chain's guards) and the two-update CPU dry run.

``preflight`` is side-effect free: it validates the resolved configuration and the
filesystem contract of a run before any data or GPU is touched, and returns a report
instead of raising on the first problem.  ``dry_run`` performs two CPU optimizer steps
at the 10 s stage plus 10 s / 30 s timing profiles on one batch.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from torch.utils.data import DataLoader, Dataset

from wms.data.collate import collate

from .curriculum import stage_plan
from .gates import required_terms
from .logging import write_json
from .settings import TrainSettings

#: Path tokens of restricted (sealed / calibration / holdout) material.
RESTRICTED = re.compile(
    r"(?:^|[^a-z0-9])(cal(?:ibration)?|holdout|sealed|chunk[-_]?5|j4b|inc1)(?:$|[^a-z0-9])", re.I)
MIN_FREE_BYTES = 10 * 1024 ** 3


@dataclass
class PreflightReport:
    """Collected failures plus the resolved plan (printed like the chain's last line)."""

    failures: list[str] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if not self.ok:
            return "preflight FAILED: " + "; ".join(self.failures)
        p = self.plan
        return ("preflight passed stages={stages} steps={steps} workers=0 max_hours={hours} "
                "terms={terms}").format(
            stages=",".join("%g" % s for s in p["stages_s"]),
            steps=",".join(str(n) for n in p["stage_steps"]), hours=p["max_hours"],
            terms=",".join(p["terms"]))


def preflight(cfg: Any, *, max_hours: float = 10.0, out_dir: Optional[Path] = None,
              inputs: Iterable[Path] = (), scripts: Iterable[Path] = (),
              min_free_bytes: int = MIN_FREE_BYTES) -> PreflightReport:
    """Validate ``cfg`` and the run's filesystem contract.

    ``inputs`` must exist and contain no restricted token; ``scripts`` must be LF-only;
    ``out_dir`` must not exist (failed runs are preserved) and its volume needs
    ``min_free_bytes``.
    """
    report = PreflightReport()
    try:
        settings = TrainSettings.from_config(cfg)
    except ValueError as exc:
        report.failures.append(f"invalid configuration: {exc}")
        return report
    if settings.uses_hard_cap:
        report.failures.append("hard residual clamp is REJECTED for training (change-log #3)")
    if settings.num_workers != 0:
        report.failures.append("num_workers must be 0 (lazy closures are not picklable)")
    if settings.threads < 1:
        report.failures.append("threads must be a positive integer")
    if settings.stage_steps is None:
        report.failures.append("explicit stage_steps are required for a registered run")
    if not (math.isfinite(max_hours) and max_hours > 0):
        report.failures.append("max_hours must be finite and positive")
    if settings.select_by_long_horizon and not settings.val_window_list:
        report.failures.append("select_by_long_horizon requires a validation window list")
    for path in map(Path, inputs):
        if RESTRICTED.search(str(path).replace("\\", "/")):
            report.failures.append(f"restricted input location: {path}")
        elif not path.exists():
            report.failures.append(f"missing input {path}")
    for script in map(Path, scripts):
        if script.is_file() and b"\r" in script.read_bytes():
            report.failures.append(f"{script} has CR bytes; require LF")
    if out_dir is not None:
        out_dir = Path(out_dir)
        if out_dir.exists():
            report.failures.append(f"output exists: {out_dir}; choose a fresh directory")
        else:
            probe = out_dir
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            if shutil.disk_usage(probe).free < min_free_bytes:
                report.failures.append(f"insufficient disk under {probe}")
    plans = stage_plan(settings, 0.5)
    report.plan = {"stages_s": [p.stage_s for p in plans], "stage_steps": [p.steps for p in plans],
                   "max_hours": max_hours, "terms": list(required_terms(settings))}
    return report


def dry_run(cfg: Any, model: Any, dataset: Dataset, *, out_dir: Path) -> dict:
    """Two CPU updates at the 10 s stage plus 10 s / 30 s forward-backward timing profiles."""
    from .trainer import WmSTrainer

    flat = dict(cfg.to_legacy_flat())
    flat.update(device="cpu", steps_per_epoch=2, stage_steps=None, unroll_curriculum_s=[10.0])
    # The single 10 s dry-run stage keeps only its own per-stage settings (round-5
    # configs carry per-stage truncation / learning rates; the full lists would not
    # validate against the collapsed one-stage curriculum).
    for per_stage in ("truncation_steps_per_stage", "lr_per_stage"):
        if flat.get(per_stage) is not None:
            flat[per_stage] = list(flat[per_stage])[:1]
    trainer = WmSTrainer(type(cfg).from_legacy_flat(flat), model, dataset, out_dir=out_dir)
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False,
                        collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    if batch["states"].shape[1] < 21:
        raise ValueError("CPU dry run requires 10 s (21 state boundaries)")
    records = [trainer.train_batch(batch, unroll_steps=20) for _ in range(2)]
    profiles = []
    for seconds in (10.0, 30.0):
        steps = round(seconds / trainer.model.physics.dt_s)
        if batch["states"].shape[1] <= steps:
            continue
        trainer.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        trainer.forward_backward(batch, unroll_steps=steps)
        elapsed = time.perf_counter() - started
        profiles.append({"unroll_s": seconds, "steps": steps, "fwd_bwd_time_s": elapsed,
                         "fwd_bwd_time_per_sim_step_ms": 1000.0 * elapsed / steps})
    receipt = {"schema": "wm_s_train_receipt/1", "status": "passed", "updates": records,
               "unroll_s": 10.0, "batch_size": int(batch["states"].shape[0]),
               "threads": trainer.settings.threads, "split": "TRAIN", "optimizer_steps": 2,
               "profiles": profiles, "data": getattr(dataset, "receipt", {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "TRAIN_RECEIPT.json", receipt)
    (out_dir / "train_log.jsonl").write_text(
        "".join(json.dumps({"step": i + 1, **row}, sort_keys=True) + "\n"
                for i, row in enumerate(records)), encoding="utf-8")
    return receipt

