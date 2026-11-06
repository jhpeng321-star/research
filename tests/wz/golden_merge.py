"""WM-A merge-map golden: the WM-S pipeline on the ORIGINAL merge map, before vs after the map port.

Captured on c4/wz-wm-port at the merge of c4/mainstart (bd9bd5d6), BEFORE any map-abstraction
change, then recomputed by ``test_merge_golden.py`` after every change: with the merge-map road
every value must be bit-identical (sha256 of the raw bytes of every array; floats included).

Covered path (CPU, deterministic, no SUMO, no data_cache):
  synthetic merge-map ``c4_sumo_episode/1`` episodes (tests/wz/synth.py)
  -> ``wms.data.c4_episodes.convert`` (pack arrays, counters, manifest)
  -> ``build_c4_datasets`` windows (60 s driver history, entries_v2, junction ffill, pivot history)
  -> collate -> the round-16 loop FT model (configs/c4_wm_loop14_r16_ft.toml, fresh seed-101
     weights) through ``WmSTrainer.forward_backward`` (free HDV rollout with CAV teacher, entries,
     exits, every configured loss term) and ONE optimizer step (``train_batch``)
  -> physics primitives on the window's t0 state (mobil/merge logits, physical envelope,
     lane-v5 base features, section ids, exit bound, entry process placement).

    python tests/wz/golden_merge.py --capture      # writes tests/wz/data/merge_golden.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

GOLDEN = HERE / "data" / "merge_golden.json"
CONFIG = ROOT / "configs" / "c4_wm_loop14_r16_ft.toml"
SEEDS = (970001, 970002, 970003)
HORIZON_STEPS = 20


def sha(value) -> str:
    import torch
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256(str(arr.dtype).encode() + str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _flatten(prefix, obj, out):
    import torch
    if isinstance(obj, dict):
        for k in sorted(obj):
            _flatten(f"{prefix}.{k}" if prefix else str(k), obj[k], out)
    elif isinstance(obj, (list, tuple)) and obj and all(
            torch.is_tensor(x) or isinstance(x, np.ndarray) for x in obj):
        for i, x in enumerate(obj):
            _flatten(f"{prefix}[{i}]", x, out)
    elif torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        out[prefix] = sha(obj)
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        out[prefix] = obj if not isinstance(obj, float) else float.hex(obj)
    return out


def build_source(tmp: Path, geo, seeds=SEEDS, extra_meta=None):
    from tests.wz import synth
    from wms.data import c4_episodes as C4
    collect = tmp / "collect" / "episodes" / "native"
    for seed in seeds:
        synth.write_episode(collect, geo, seed=seed, extra_meta=extra_meta)
    out = tmp / "source"
    doc = C4.convert([tmp / "collect"], out, val_frac=0.0, split_seed=0)
    return out, doc


def load_cfg(overrides=()):
    from wms.config.load import load_train_config
    return load_train_config(CONFIG, ["runtime.device=\"cpu\"", "runtime.threads=1",
                                      *overrides])


def compute(tmp: Path, geo=None, *, cfg_overrides=(), seeds=SEEDS, extra_meta=None) -> dict:
    import torch
    from tests.wz import synth
    from wms.cli.assemble import build_model, build_physics
    from wms.data import c4_episodes as C4
    from wms.data.collate import collate

    geo = geo or synth.MERGE
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    out: dict = {}
    src, doc = build_source(tmp, geo, seeds, extra_meta)
    for e in doc["episodes"]:
        ep = C4.EpisodeArrays(src / e["dir"])
        for name in C4.ARRAYS:
            out[f"pack.{e['episode']}.{name}"] = sha(np.asarray(ep.array(name)))
        out[f"pack.{e['episode']}.entry"] = {k: e[k] for k in sorted(e)
                                             if k not in ("source", "dir")}
    out["manifest"] = {k: doc[k] for k in sorted(doc) if k not in ("roots", "episodes")}

    cfg = load_cfg(cfg_overrides)
    built = C4.build_c4_datasets(
        src, horizon_steps=HORIZON_STEPS, history_s=cfg.data.history_s,
        history_stride_s=cfg.data.history_stride_s, entries_v2=bool(cfg.data.entries_v2),
        stride_s=10.0, junction_fix=cfg.data.junction_fix, command_known=True,
        merge_pivot_history=True)
    train = built["train"]
    out["n_windows"] = len(train)
    picks = [0, len(train) // 2, len(train) - 1]
    batch = collate([train[i] for i in picks])
    _flatten("batch", {k: v for k, v in batch.items() if k != "meta"}, out)

    physics = build_physics(cfg)
    torch.manual_seed(101)
    model = build_model(cfg, physics)
    out["init_params"] = sha(torch.cat([p.detach().reshape(-1) for p in model.parameters()]))
    from wms.train.trainer import WmSTrainer

    class _Trainer(WmSTrainer):            # fresh weights: no checkpoint initialisation
        def _init_from_checkpoint(self, name):
            return None

    trainer = _Trainer(cfg, model, train, out_dir=tmp / "run")
    torch.manual_seed(7)
    rec = trainer.forward_backward(batch, unroll_steps=HORIZON_STEPS, backward=False)
    _flatten("fb", {k: v for k, v in rec.items() if not k.startswith("_")}, out)
    torch.manual_seed(8)
    step = trainer.train_batch(batch, unroll_steps=HORIZON_STEPS)
    _flatten("step", {k: v for k, v in step.items() if k not in ("seconds",)
                      and not k.startswith("_")}, out)
    out["params_after_step"] = sha(torch.cat([p.detach().reshape(-1)
                                              for p in trainer.model.parameters()]))
    out.update(physics_primitives(trainer.model, batch))
    return out


def physics_primitives(model, batch) -> dict:
    import torch
    from wms.model.entries import section_ids
    from wms.physics.batched import road_end_exit_bound
    from wms.physics.mobil import mobil_merge_logits, physical_merge_envelope
    out = {}
    states = torch.as_tensor(np.asarray(batch["states"]))
    present = torch.as_tensor(np.asarray(batch["present"])).bool()
    kin = states[:, 0].float()
    pres = present[:, 0]
    p = model.physics
    with torch.no_grad():
        out["prim.mobil_logits"] = sha(mobil_merge_logits(kin, pres, p))
        env = physical_merge_envelope(kin, pres, p)
        for k in sorted(env):
            out[f"prim.envelope.{k}"] = sha(env[k])
        out["prim.sections"] = sha(section_ids(kin[..., 0], pres))
        out["prim.exit_bound"] = float.hex(float(road_end_exit_bound(p)))
        if getattr(model, "lane_v5", None) is not None:
            out["prim.lane_v5_base"] = sha(model.lane_v5_base(kin, pres))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        result = compute(Path(tmp))
    if args.capture:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {GOLDEN} ({len(result)} keys)")
    else:
        print(json.dumps({k: result[k] for k in list(result)[:20]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
