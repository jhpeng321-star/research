"""Initial checkpoint of the wz3to2 fine-tune: WM2 with the merge-map-geometry tensors re-initialised.

Only tensors tied to the on-ramp merge geometry are reset, to the values a FRESHLY BUILT model of
the fine-tune config holds (the model's own initialisation: ``merge_w_init``, the 2 m/s^2 / 60 m
merge-yield initialisation, the ramp speed-adaptation initialisation file, default first-passage
pivot values), under a fixed torch seed:

* ``merge_pivot.*``      first-passage merge acceptance (gap/time/decel thresholds, timing tau/w)
                         fitted to the 75 m acceleration lane (R9);
* ``merge_core.*``       the merge-core v2 heads and zone scalars (``init_s0``/``s0``: merge-zone
                         positions; inactive under ``merge_decision = first_passage`` but kept
                         consistent);
* ``merge_yield.*``      mainline yield to lane-end mergers (lane-end distance scale, R13Y);
* ``ramp_speed_adapt.*`` ramp-HDV speed adaptation inside the acceleration lane (R16).

Everything else - driver encoder (driver settings stay learned from prediction error), residual
network, lane_v5, gates, courtesy - is copied from the source checkpoint byte-for-byte; the receipt
proves it per tensor.  No data, labels or driver truth are read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

DEFAULT_PREFIXES = ("merge_pivot.", "merge_core.", "merge_yield.", "ramp_speed_adapt.")
CHECKPOINT_FORMAT = "wm_s/1"
SCHEMA = "wzwm.reinit/1"
FRESH_SEED = 20261106


class ReinitError(RuntimeError):
    """The source checkpoint and the fine-tune model do not line up (fail closed)."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(t) -> str:
    """Content hash of one tensor (dtype, shape and raw bytes)."""
    a = t.detach().to("cpu").contiguous()
    h = hashlib.sha256(f"{a.dtype}|{tuple(a.shape)}|".encode())
    h.update(a.numpy().tobytes() if a.numel() else b"")
    return h.hexdigest()


def geometry_prefixes() -> tuple[str, ...]:
    """The port's declared geometry-tied prefixes (``wms.core.road.GEOMETRY_TIED_PREFIXES``) if
    WM-A publishes them, else :data:`DEFAULT_PREFIXES`."""
    try:
        from wms.core import road  # noqa: PLC0415  (optional, WM-A port)
    except Exception:
        return DEFAULT_PREFIXES
    declared = getattr(road, "GEOMETRY_TIED_PREFIXES", None)
    return tuple(declared) if declared else DEFAULT_PREFIXES


def _selected(key: str, prefixes: Sequence[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


def reinit_state(source: Mapping, fresh: Mapping, prefixes: Sequence[str] = DEFAULT_PREFIXES, *,
                 require_every_prefix: bool = True) -> tuple[dict, dict]:
    """``(state, report)``: ``source`` with the ``prefixes`` tensors taken from ``fresh``.

    Fails closed when the source has a tensor the fine-tune model lacks, when a shared tensor
    differs in shape/dtype, or (``require_every_prefix``) when a prefix matches nothing.
    Tensors new in the model (absent from the source) take the fresh value and are reported.
    """
    import torch

    prefixes = tuple(prefixes)
    if not prefixes:
        raise ReinitError("no re-initialisation prefixes")
    unused = sorted(set(source) - set(fresh))
    if unused:
        raise ReinitError(f"source tensors absent from the fine-tune model: {unused[:8]}")
    bad = sorted(k for k in set(source) & set(fresh)
                 if tuple(source[k].shape) != tuple(fresh[k].shape) or source[k].dtype != fresh[k].dtype)
    if bad:
        raise ReinitError(f"shape/dtype mismatch: {bad[:8]}")
    empty = [p for p in prefixes if not any(k.startswith(p) for k in fresh)]
    if empty and require_every_prefix:
        raise ReinitError(f"prefixes match no model tensor: {empty}")
    state, reinit, kept, new = {}, [], [], []
    for key in fresh:                       # model order
        if _selected(key, prefixes):
            state[key] = fresh[key].detach().clone()
            reinit.append(key)
        elif key in source:
            state[key] = source[key].detach().clone()
            kept.append(key)
        else:
            state[key] = fresh[key].detach().clone()
            new.append(key)
    for key in kept:                        # proof: every kept tensor is the source tensor
        if not torch.equal(state[key], source[key]):
            raise ReinitError(f"kept tensor {key} differs from the source")
    report = {
        "prefixes": list(prefixes), "n_tensors": len(state), "n_reinit": len(reinit),
        "n_kept": len(kept), "n_new_in_model": len(new),
        "reinit": {k: {"source_sha": tensor_sha(source[k]) if k in source else None,
                       "fresh_sha": tensor_sha(state[k]),
                       "changed": (k not in source) or not torch.equal(state[k], source[k])}
                   for k in reinit},
        "new_in_model": new,
        "kept_sha": {k: tensor_sha(state[k]) for k in kept},
        "unmatched_prefixes": empty,
    }
    return state, report


def fresh_state_from_config(config: str, overrides: Iterable[str] = (), *, seed: int = FRESH_SEED,
                            repo: Optional[Path] = None) -> dict:
    """State dict of a freshly built WM-S model of the fine-tune config (fixed torch seed)."""
    import torch

    from wms.cli.assemble import build_model, build_physics
    from wms.config.load import load_train_config

    path = Path(config)
    if repo is not None and not path.is_absolute():
        path = Path(repo) / path
    cfg = load_train_config(str(path), list(overrides))
    torch.manual_seed(int(seed))
    model = build_model(cfg, build_physics(cfg))
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def reinit_checkpoint(src: Path, out: Path, *, expected_sha256: str,
                      fresh: Mapping | Callable[[], Mapping], prefixes: Optional[Sequence[str]] = None,
                      receipt_path: Optional[Path] = None, fresh_seed: int = FRESH_SEED,
                      config: Optional[str] = None) -> dict:
    """Verify ``src`` (sha256), write the re-initialised ``wm_s/1`` checkpoint ``out`` and a receipt.

    ``out`` is write-once.  The payload keeps the source ``config`` (flat) and adds ``wz_reinit``
    metadata; ``history``/``val_by_stage`` of the source run are dropped (they describe WM2's run).
    """
    import torch

    src, out = Path(src), Path(out)
    got = sha256_file(src)
    if got != expected_sha256:
        raise ReinitError(f"{src}: sha256 {got} != expected {expected_sha256}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    payload = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT or "model" not in payload:
        raise ReinitError(f"{src} is not a {CHECKPOINT_FORMAT} checkpoint")
    fresh_state = fresh() if callable(fresh) else fresh
    prefixes = tuple(prefixes or geometry_prefixes())
    state, report = reinit_state(payload["model"], fresh_state, prefixes)
    new_payload = {k: v for k, v in payload.items() if k not in ("model", "history", "val_by_stage")}
    new_payload["model"] = state
    new_payload["wz_reinit"] = {"source_sha256": got, "prefixes": list(prefixes),
                                "reinit_keys": sorted(report["reinit"]), "fresh_seed": int(fresh_seed),
                                "fresh_config": config}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(new_payload, tmp)
    tmp.replace(out)
    receipt = {"schema": SCHEMA, "source": str(src), "source_sha256": got, "out": str(out),
               "out_sha256": sha256_file(out), "fresh_seed": int(fresh_seed), "fresh_config": config,
               "dropped_payload_keys": sorted(set(payload) & {"history", "val_by_stage"}), **report}
    if receipt_path is not None:
        Path(receipt_path).write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")
    return receipt


def verify_init_report(init_from: Mapping, *, expected_sha256: str, n_model_tensors: int) -> list[str]:
    """Problems in a trainer ``INIT_FROM.json`` (empty = every model tensor came from the init file)."""
    out = []
    if init_from.get("sha256") != expected_sha256:
        out.append(f"init sha {init_from.get('sha256')} != {expected_sha256}")
    for key in ("shape_mismatch", "unused_from_checkpoint", "new_in_model", "expanded"):
        if init_from.get(key):
            out.append(f"{key}: {list(init_from[key])[:5]}")
    if len(init_from.get("loaded", ())) != n_model_tensors:
        out.append(f"loaded {len(init_from.get('loaded', ()))} != model tensors {n_model_tensors}")
    return out
