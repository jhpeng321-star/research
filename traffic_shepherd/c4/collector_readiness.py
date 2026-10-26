"""Check reset-time controller assets without a seed, scenario build, or SUMO."""
from __future__ import annotations

import hashlib


def check_controller_assets(config, controller_factory=None):
    path = config.model_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != config.model_sha256:
        raise ValueError("CIRCLES model checksum mismatch")
    if controller_factory is None:
        from traffic_shepherd.damping import CirclesController
        controller_factory = CirclesController
    controller_factory(config)  # also validates ONNX runtime/session creation, no inference tuning
    return {"model_path": str(path), "model_sha256": actual, "onnx_session": "ready"}
