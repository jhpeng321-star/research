"""TRAIN_RECEIPT.json writer (schema ``wm_s_train_receipt/1``, unchanged)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .logging import write_json
from .selection import LongHorizonSelection, MultiHorizonSelection
from .settings import TrainSettings

RECEIPT_SCHEMA = "wm_s_train_receipt/1"
RECEIPT_NAME = "TRAIN_RECEIPT.json"


def write_train_receipt(out_dir: Path, *, settings: TrainSettings, flat_config: Mapping[str, Any],
                        physics: Any, device: str, checkpoint: Path, log_path: Path,
                        steps: int, val_by_stage: list[dict], val_log_path: Path,
                        has_val: bool, data_receipt: Mapping[str, Any],
                        val_data_receipt: Mapping[str, Any],
                        selection: Optional[Union[LongHorizonSelection, MultiHorizonSelection]] = None,
                        best_checkpoint: Optional[Path] = None,
                        config_fingerprint: Optional[str] = None) -> dict:
    """Write and return the training receipt of a finished run."""
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "split": "TRAIN", "device": device,
        "threads": settings.threads, "curriculum_s": list(settings.unroll_curriculum_s),
        "checkpoint": str(checkpoint), "steps": steps, "val_by_stage": val_by_stage,
        "log": str(log_path), "physics": asdict(physics), "config": dict(flat_config),
        "val_monitoring": (None if not has_val else {
            "mode": "monitor-only", "early_stopping": False, "checkpoint_selection": False,
            "log": str(val_log_path), "by_stage": val_by_stage, "data": dict(val_data_receipt)}),
        "data": dict(data_receipt)}
    if config_fingerprint is not None:
        receipt["config_fingerprint"] = config_fingerprint
    if settings.select_by_long_horizon:
        receipt["long_horizon_selection"] = {
            "enabled": True, "metric": "mean_position", "minimum_stage_s": 60.0,
            "policy": "longest-eligible-horizon-then-lowest-mean-position",
            "val_every": settings.val_every, "best": selection.best,
            "checkpoint": str(best_checkpoint) if selection.best else None,
            "longest_horizon_s": selection.horizon_s,
            "final_checkpoint_policy": "last-optimizer-step"}
        receipt["val_monitoring"].update(mode="long-horizon-selection", checkpoint_selection=True)
        if getattr(settings, "selection_mode", "long_horizon") == "multi_horizon":
            # SCHED: one decision per validation event, mean reference-normalised position.
            receipt["long_horizon_selection"].update(
                metric="mean_position/reference (mean over validated horizons)",
                minimum_stage_s=selection.eligible_stage_s,
                policy="multi-horizon-reference-normalised-mean",
                val_all_horizons=bool(settings.val_all_horizons),
                multi_horizon=selection.summary())
            receipt["val_monitoring"].update(mode="multi-horizon-selection")
        elif getattr(settings, "selection_mode", "long_horizon") == "multi_horizon_v2":
            # INFRAV5: online same-horizon score + end-of-run same-horizon comparison.
            receipt["long_horizon_selection"].update(
                metric="mean_position/normaliser (mean over ONE common horizon set)",
                minimum_stage_s=selection.eligible_stage_s,
                policy="same-horizon-final-comparison-of-stage-candidates",
                val_all_horizons=bool(settings.val_all_horizons),
                multi_horizon_v2=selection.summary())
            receipt["val_monitoring"].update(mode="same-horizon-selection")
    write_json(Path(out_dir) / RECEIPT_NAME, receipt)
    return receipt
